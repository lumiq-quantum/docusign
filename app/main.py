from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks # Added BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict # Added Dict
import PyPDF2
import io
import os
import uuid # Added for generating application_number
import google.generativeai as genai
import httpx
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from fastapi.responses import HTMLResponse, Response # Added Response
import boto3 # Added for AWS Textract
from pdf2image import convert_from_bytes # Added for PDF to image conversion
import json # Added for JSON processing
from PIL import Image # Added for image manipulation

from . import models
# Updated model imports to reflect new structure and Pydantic schemas
from .models import (
    SessionLocal, engine, get_db,
    Project, ProjectCreate, ProjectResponse,
    Document, DocumentCreate, DocumentResponse,
    Page, PageCreate, PageResponse,
    Stakeholder, StakeholderCreate, StakeholderResponse,
    SignatureInstance, SignatureInstanceCreate, SignatureInstanceResponse, # Uncommented
    GeneratedHtmlResponse
)

# models.Base.metadata.create_all(bind=engine) # Creates tables if they don't exist (dev only, Alembic should manage schema in prod)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize Gemini client
# IMPORTANT: Set your GOOGLE_API_KEY environment variable before running the app.
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # User updated to gemini-2.5-flash-preview-05-20 or similar
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    # Potentially exit or disable Gemini-dependent features
    gemini_model = None 
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    gemini_model = None

# HTTP client for calling external services
http_client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

# --- Helper Functions ---
def generate_application_number() -> str:
    """Generates a unique application number."""
    return str(uuid.uuid4())

# --- API Endpoints ---

# I. Proposal (Project) Management
@app.post("/proposals/", response_model=models.ProjectResponse, status_code=201)
async def create_proposal(
    project_in: models.ProjectCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new proposal (project).
    An application_number will be automatically generated.
    """
    application_number = generate_application_number()
    # Ensure uniqueness, though UUIDs are highly unlikely to collide
    while db.query(models.Project).filter(models.Project.application_number == application_number).first():
        application_number = generate_application_number()

    db_project = models.Project(
        name=project_in.name,
        application_number=application_number,
        chat_session_id=project_in.chat_session_id # If provided
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@app.post("/proposals/{proposal_id}/documents/", response_model=List[models.DocumentResponse])
async def upload_documents_to_proposal(
    proposal_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload one or more PDF documents to an existing proposal.
    """
    db_proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not db_proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")

    created_documents = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            # Or collect errors and report, for now, skip non-PDFs or raise error for the batch
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF. Only PDF files are allowed.")

        pdf_content = await file.read()
        file_name = file.filename

        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            num_pages = len(pdf_reader.pages)

            db_document = models.Document(
                file_name=file_name,
                pdf_file=pdf_content,
                total_pages=num_pages,
                project_id=db_proposal.id
            )
            db.add(db_document)
            db.commit() # Commit per document to get its ID for pages
            db.refresh(db_document)

            # Extract text and create page entries for this document
            for i in range(num_pages):
                page_text = pdf_reader.pages[i].extract_text()
                db_page = models.Page(
                    page_number=i + 1,
                    text_content=page_text if page_text else "",
                    document_id=db_document.id,
                    generated_form_html=None # Initialize with no form
                )
                db.add(db_page)
            db.commit() # Commit pages for the current document
            db.refresh(db_document) # Refresh to get pages loaded in the response if needed by schema
            created_documents.append(db_document)

        except Exception as e:
            db.rollback()
            # Log the full exception for debugging
            print(f"Error processing file {file_name} for proposal {proposal_id}: {e}")
            # Decide if to continue with other files or raise immediately
            raise HTTPException(status_code=500, detail=f"Error processing file {file_name}: {str(e)}")
    
    return created_documents

@app.get("/proposals/", response_model=List[models.ProjectResponse])
def list_proposals(db: Session = Depends(get_db)):
    return db.query(models.Project).all()

@app.get("/proposals/{proposal_id}/", response_model=models.ProjectResponse)
def get_proposal(proposal_id: int, db: Session = Depends(get_db)):
    proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")
    return proposal

@app.delete("/proposals/{proposal_id}/", status_code=204)
def delete_proposal(proposal_id: int, db: Session = Depends(get_db)):
    proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")
    db.delete(proposal)
    db.commit()
    return

# PDF Interaction & Viewing - These need to be adapted to the new structure
# The old /projects/{project_id}/pages/... endpoints will be replaced by
# /proposals/{proposal_id}/documents/{document_id}/pages/...

# Placeholder for the old list_project_pages - this logic is now implicitly handled by getting a Document and its pages.
# @app.get("/projects/{project_id}/pages/", response_model=List[PageResponse]) ...

@app.get("/proposals/{proposal_id}/documents/{document_id}/pages/{page_number}/pdf")
async def get_document_page_pdf(
    proposal_id: int, 
    document_id: int, 
    page_number: int, 
    db: Session = Depends(get_db)
):
    db_doc = db.query(models.Document).filter(
        models.Document.id == document_id,
        models.Document.project_id == proposal_id
    ).first()

    if not db_doc:
        raise HTTPException(status_code=404, detail=f"Document with id {document_id} in proposal {proposal_id} not found")
    if not db_doc.pdf_file:
        raise HTTPException(status_code=404, detail="PDF file not found for this document")

    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(db_doc.pdf_file))
        if not (0 < page_number <= len(pdf_reader.pages)):
            raise HTTPException(status_code=404, detail=f"Page number {page_number} out of range. PDF has {len(pdf_reader.pages)} pages.")

        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_number - 1])  # PyPDF2 pages are 0-indexed

        output_pdf_buffer = io.BytesIO()
        pdf_writer.write(output_pdf_buffer)
        output_pdf_buffer.seek(0)

        return Response(content=output_pdf_buffer.read(), media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF page: {str(e)}")

# @app.get("/projects/{project_id}/pages/{page_number}/text", response_model=PageResponse)
# This should now be:
@app.get("/proposals/{proposal_id}/documents/{document_id}/pages/{page_number}/text", response_model=models.PageResponse)
def get_document_page_text_content(
    proposal_id: int,
    document_id: int,
    page_number: int,
    db: Session = Depends(get_db)
):
    page = db.query(models.Page).join(models.Document).filter(
        models.Document.project_id == proposal_id,
        models.Page.document_id == document_id,
        models.Page.page_number == page_number
    ).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    return page

# Dynamic Digital Form System - Needs significant refactoring for new structure
# The old /projects/{project_id}/pages/{page_number}/form/generate will become
# POST /proposals/{proposal_id}/documents/{document_id}/extract-html (for all pages of a document)
# and GET /proposals/{proposal_id}/documents/{document_id}/pages/{page_number}/html

# @app.post("/projects/{project_id}/pages/{page_number}/form/generate", response_model=dict)
# async def generate_form_fields(project_id: int, page_number: int, db: Session = Depends(get_db)):
# This endpoint logic will be moved into a background task triggered by /extract-html

# New endpoint to trigger HTML generation for all pages of a specific document
@app.post("/proposals/{proposal_id}/documents/{document_id}/extract-html", response_model=dict)
async def trigger_document_html_extraction(
    proposal_id: int,
    document_id: int,
    background_tasks: BackgroundTasks, # Added BackgroundTasks
    db: Session = Depends(get_db)
):
    db_doc = db.query(models.Document).filter(
        models.Document.id == document_id,
        models.Document.project_id == proposal_id
    ).first()
    if not db_doc:
        raise HTTPException(status_code=404, detail=f"Document with id {document_id} in proposal {proposal_id} not found")

    # Here, we'll define a background task to process all pages.
    # The actual AI call logic will be in a helper function.
    background_tasks.add_task(generate_html_for_document_pages, document_id, db_doc.project_id, db) # Pass db or necessary components

    return {"message": f"HTML extraction initiated for all pages of document {document_id} in proposal {proposal_id}."}

async def generate_html_for_document_pages(document_id: int, proposal_id: int, db: Session = Depends(get_db)):
    # This function will run in the background.
    # It needs its own DB session or careful handling if db is passed.
    # For simplicity, let's assume we can re-query or use a new session if needed.
    # A more robust solution might involve a task queue like Celery.

    # Re-fetch the document and its pages within the background task context
    # This requires careful session management. For now, we'll assume 'db' is usable
    # or we create a new session. Let's try to use the passed 'db' but be mindful.
    # A better way for background tasks is to pass IDs and let the task create its own session.
    # Create a new session for the background task to avoid session conflicts
    # This is a simplified approach. Production systems might use a dedicated task runner with its own session management.
    db_bg = SessionLocal() # Create a new session for this background task
    try:
        doc = db_bg.query(models.Document).filter(models.Document.id == document_id).first()
        if not doc or not doc.pdf_file:
            print(f"Background task: Document {document_id} or its PDF file not found.")
            return
        pages_to_process = db_bg.query(models.Page).filter(models.Page.document_id == document_id).all()
        if not pages_to_process:
            print(f"Background task: No pages found for document {document_id}.")
            return
        print(f"Background task: Starting HTML generation for document {document_id}, {len(pages_to_process)} pages.")
        for page in pages_to_process:
            if page.generated_form_html: # Skip if already generated
                print(f"Background task: HTML already exists for page {page.page_number} of document {document_id}. Skipping.")
                continue

            try:
                pdf_reader_bg = PyPDF2.PdfReader(io.BytesIO(doc.pdf_file))
                if not (0 < page.page_number <= len(pdf_reader_bg.pages)):
                    print(f"Background task: Page number {page.page_number} out of range for document {document_id}. Skipping.")
                    continue
                pdf_writer_bg = PyPDF2.PdfWriter()
                pdf_writer_bg.add_page(pdf_reader_bg.pages[page.page_number - 1])
                single_page_pdf_buffer_bg = io.BytesIO()
                pdf_writer_bg.write(single_page_pdf_buffer_bg)
                single_page_pdf_buffer_bg.seek(0)
                pdf_page_bytes_bg = single_page_pdf_buffer_bg.read()
                # Ensure Gemini client is configured (it's configured globally but good to be aware)
                if not gemini_model:
                    print("Background task: Gemini model not initialized. Skipping AI call.")
                    continue
                prompt = """You are Expert in reading complex documents.
                Task for you: Extract the information from the document, You need to convert physical document into a digital verion which imitates the physical form , keep information prefilled and editable.Rememeber the accuracy of the information extracted specially filled information is absolutely important.You need to take care of multilingual , checkboxes and handwritten complexity within document. give me the html with good stylinng for review, if you are not confident on any field or section enough mark that area as red so that Human can rectify that easily, there might be signatures on the documents, if there are multiple signatures on the document you have to match those documents and if there is dissimilarity between those signatures you need to highlight in red for those. There can be multiple singatures of a set of person, you need to also categorise the group of signature by the same person. The output should be the html page content without suffix or prefix."""
                pdf_blob_bg = {
                    'mime_type': 'application/pdf',
                    'data': pdf_page_bytes_bg
                }

                print(f"Background task: Calling Gemini for page {page.page_number} of document {document_id}...")
                ai_response = await gemini_model.generate_content_async([prompt, pdf_blob_bg])
                if ai_response.parts:
                    html_content = ai_response.text
                    # Clean up potential markdown code block fences
                    if html_content.startswith("```html"):
                        html_content = html_content[7:]
                    if html_content.startswith("```"):
                        html_content = html_content[3:]
                    if html_content.endswith("```"):
                        html_content = html_content[:-3]
                    html_content = html_content.strip()

                    page.generated_form_html = html_content
                    db_bg.commit() # Commit change for this page
                    print(f"Background task: Successfully generated and saved HTML for page {page.page_number} of document {document_id}.")
                else:
                    error_detail = "AI model did not return expected content."
                    if ai_response.prompt_feedback and ai_response.prompt_feedback.block_reason:
                        error_detail += f" Reason: {ai_response.prompt_feedback.block_reason_message or ai_response.prompt_feedback.block_reason}"
                    print(f"Background task: Error generating HTML for page {page.page_number} of document {document_id}: {error_detail}")

            except Exception as e_page:
                db_bg.rollback()
                print(f"Background task: Exception processing page {page.page_number} of document {document_id}: {str(e_page)}")

        print(f"Background task: Finished HTML generation for document {document_id}.")
    except Exception as e_doc:
        print(f"Background task: General error for document {document_id}: {str(e_doc)}")
    finally:
        db_bg.close() # Ensure the session is closed

@app.get("/proposals/{proposal_id}/documents/{document_id}/pages/{page_number}/html", response_model=models.GeneratedHtmlResponse)
async def get_document_page_html(
    proposal_id: int,
    document_id: int,
    page_number: int,
    db: Session = Depends(get_db)
):
    db_page = db.query(models.Page).join(models.Document).filter(
        models.Document.project_id == proposal_id,
        models.Page.document_id == document_id,
        models.Page.page_number == page_number
    ).first()

    if not db_page:
        raise HTTPException(status_code=404, detail="Page not found")
    if db_page.generated_form_html is not None:
        return models.GeneratedHtmlResponse(html_content=db_page.generated_form_html)
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"HTML form for page {page_number} of document {document_id} has not been generated yet. Use the POST /proposals/{proposal_id}/documents/{document_id}/extract-html endpoint to create it."
        )
    
# @app.get("/projects/{project_id}/pages/{page_number}/html_view", response_class=HTMLResponse)
# This should now be:
@app.get("/proposals/{proposal_id}/documents/{document_id}/pages/{page_number}/html_view", response_class=HTMLResponse)
async def view_document_page_generated_html(
    proposal_id: int,
    document_id: int,
    page_number: int, 
    db: Session = Depends(get_db)
):
    db_page = db.query(models.Page).join(models.Document).filter(
        models.Document.project_id == proposal_id,
        models.Page.document_id == document_id,
        models.Page.page_number == page_number
    ).first()
    
    if not db_page:
        raise HTTPException(status_code=404, detail="Page not found")

    if db_page.generated_form_html is not None:
        return HTMLResponse(content=db_page.generated_form_html)
    else:
        error_html = f"""
        <html>
            <head>
                <title>HTML Not Generated</title>
            </head>
            <body>
                <h1>HTML form for proposal {proposal_id}, document {document_id}, page {page_number} has not been generated yet.</h1>
                <p>Use the <code>POST /proposals/{'{proposal_id}'}/documents/{'{document_id}'}/extract-html</code> endpoint to trigger generation for the document.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=404)

# The old /projects/generate-all-forms/ endpoint is superseded by the per-document HTML extraction.
# If project-wide generation is still needed, it would loop through documents and call the per-document endpoint.
# @app.get("/projects/generate-all-forms/", response_model=dict)
# async def generate_all_forms_for_project(project_id: int, db: Session = Depends(get_db)):

# --- Stakeholder Management ---
@app.post("/proposals/{proposal_id}/stakeholders/", response_model=models.StakeholderResponse, status_code=201)
def create_stakeholder_for_proposal(
    proposal_id: int,
    stakeholder_in: models.StakeholderCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new stakeholder for a specific proposal.
    The project_id in the stakeholder_in payload MUST match the proposal_id in the URL.
    """
    db_proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not db_proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")

    # StakeholderCreate schema includes project_id. Validate it against the path.
    if stakeholder_in.project_id != proposal_id:
        raise HTTPException(
            status_code=400,
            detail=f"The project_id in the request body ({stakeholder_in.project_id}) must match the proposal_id in the URL path ({proposal_id})."
        )

    # Create the stakeholder instance using the validated name and the project_id from the path.
    db_stakeholder = models.Stakeholder(name=stakeholder_in.name, project_id=proposal_id)
    db.add(db_stakeholder)
    db.commit()
    db.refresh(db_stakeholder)
    return db_stakeholder

@app.get("/proposals/{proposal_id}/stakeholders/", response_model=List[models.StakeholderResponse])
def list_stakeholders_for_proposal(
    proposal_id: int,
    db: Session = Depends(get_db)
):
    """
    List all stakeholders associated with a specific proposal.
    """
    db_proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not db_proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")
    
    return db_proposal.stakeholders

@app.get("/proposals/{proposal_id}/stakeholders/{stakeholder_id}/", response_model=models.StakeholderResponse)
def get_stakeholder_for_proposal(
    proposal_id: int,
    stakeholder_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve a specific stakeholder for a specific proposal.
    """
    db_stakeholder = db.query(models.Stakeholder).filter(
        models.Stakeholder.id == stakeholder_id,
        models.Stakeholder.project_id == proposal_id
    ).first()
    
    if not db_stakeholder:
        raise HTTPException(status_code=404, detail=f"Stakeholder with id {stakeholder_id} not found in proposal {proposal_id}")
    return db_stakeholder

@app.put("/proposals/{proposal_id}/stakeholders/{stakeholder_id}/", response_model=models.StakeholderResponse)
def update_stakeholder_for_proposal(
    proposal_id: int,
    stakeholder_id: int,
    stakeholder_in: models.StakeholderCreate, # Using Create schema for update. Name is the primary field to update.
    db: Session = Depends(get_db)
):
    """
    Update a stakeholder's details (primarily name) for a specific proposal.
    The project_id of the stakeholder cannot be changed.
    """
    db_stakeholder = db.query(models.Stakeholder).filter(
        models.Stakeholder.id == stakeholder_id,
        models.Stakeholder.project_id == proposal_id
    ).first()

    if not db_stakeholder:
        raise HTTPException(status_code=404, detail=f"Stakeholder with id {stakeholder_id} not found in proposal {proposal_id}")

    # The StakeholderCreate schema has 'name' and 'project_id'.
    # We only want to update 'name'. 'project_id' from payload must match existing.
    if stakeholder_in.project_id != db_stakeholder.project_id:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot change the project_id of a stakeholder. Expected project_id {db_stakeholder.project_id} but got {stakeholder_in.project_id} in payload."
        )

    db_stakeholder.name = stakeholder_in.name
    
    db.commit()
    db.refresh(db_stakeholder)
    return db_stakeholder

@app.delete("/proposals/{proposal_id}/stakeholders/{stakeholder_id}/", status_code=204)
def delete_stakeholder_from_proposal(
    proposal_id: int,
    stakeholder_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a stakeholder from a specific proposal.
    """
    db_stakeholder = db.query(models.Stakeholder).filter(
        models.Stakeholder.id == stakeholder_id,
        models.Stakeholder.project_id == proposal_id
    ).first()

    if not db_stakeholder:
        raise HTTPException(status_code=404, detail=f"Stakeholder with id {stakeholder_id} not found in proposal {proposal_id}")

    db.delete(db_stakeholder)
    db.commit()
    return

# --- AWS Textract Client ---
# Configure your AWS region
AWS_REGION = os.getenv("AWS_REGION", "us-east-1") # Default to us-east-1 if not set
try:
    textract_client = boto3.client("textract", region_name=AWS_REGION)
except Exception as e:
    print(f"Error initializing AWS Textract client: {e}")
    textract_client = None

# --- Signature Analysis ---

@app.post("/proposals/{proposal_id}/signature-analysis/start", response_model=Dict[str, str])
async def start_signature_analysis_for_proposal(
    proposal_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start the signature analysis process for all documents in a proposal.
    This process runs in the background.
    """
    db_proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not db_proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")

    if not textract_client:
        raise HTTPException(status_code=500, detail="AWS Textract client is not configured. Cannot start analysis.")
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini client is not configured. Cannot start analysis.")

    # Update status
    db_proposal.signature_analysis_status = "processing_started"
    db.commit()

    background_tasks.add_task(perform_signature_analysis, proposal_id)

    return {"message": f"Signature analysis initiated for proposal {proposal_id}."}

async def perform_signature_analysis(proposal_id: int):
    """
    Background task to perform signature analysis using AWS Textract and Google Gemini.
    """
    db_bg = SessionLocal()
    try:
        proposal = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if not proposal:
            print(f"Signature Analysis Task: Proposal {proposal_id} not found.")
            return

        proposal.signature_analysis_status = "processing_textract"
        db_bg.commit()

        all_signature_instances_data = []

        for doc in proposal.documents:
            if not doc.pdf_file:
                print(f"Signature Analysis Task: PDF file missing for document {doc.id} in proposal {proposal_id}.")
                continue

            try:
                # Convert whole PDF to images once, then process pages
                images_from_pdf = convert_from_bytes(doc.pdf_file)
            except Exception as e_conv:
                print(f"Signature Analysis Task: Failed to convert PDF to images for doc {doc.id}: {e_conv}")
                proposal.signature_analysis_status = f"failed_pdf_conversion_doc_{doc.id}"
                db_bg.commit()
                continue

            for i, page_image_pil in enumerate(images_from_pdf): # page_image_pil is a PIL Image object
                page_number = i + 1
                db_page = db_bg.query(models.Page).filter(
                    models.Page.document_id == doc.id,
                    models.Page.page_number == page_number
                ).first()

                if not db_page:
                    print(f"Signature Analysis Task: Page entry not found for doc {doc.id}, page {page_number}. Skipping Textract.")
                    continue

                # Convert PIL image to bytes for Textract
                img_byte_arr = io.BytesIO()
                page_image_pil.save(img_byte_arr, format='PNG') # Or JPEG
                img_byte_arr = img_byte_arr.getvalue()

                try:
                    print(f"Signature Analysis Task: Calling Textract for doc {doc.id}, page {page_number}...")
                    textract_response = textract_client.analyze_document(
                        Document={'Bytes': img_byte_arr},
                        FeatureTypes=['SIGNATURES']
                    )
                    
                    # Process Textract response to find signatures
                    signatures_found_on_page = []
                    for block in textract_response.get('Blocks', []):
                        if block['BlockType'] == 'SIGNATURE':
                            signatures_found_on_page.append(block)
                            
                            # --- Crop and Store Signature Image ---
                            cropped_image_bytes = None
                            try:
                                geometry = block['Geometry']
                                bbox = geometry['BoundingBox']
                                img_width, img_height = page_image_pil.size
                                
                                # Calculate absolute pixel coordinates
                                left = int(bbox['Left'] * img_width)
                                top = int(bbox['Top'] * img_height)
                                right = int((bbox['Left'] + bbox['Width']) * img_width)
                                bottom = int((bbox['Top'] + bbox['Height']) * img_height)
                                
                                # Crop the image
                                cropped_pil_image = page_image_pil.crop((left, top, right, bottom))
                                
                                # Convert cropped PIL image to bytes
                                cropped_img_byte_arr = io.BytesIO()
                                cropped_pil_image.save(cropped_img_byte_arr, format='PNG') # Save as PNG
                                cropped_image_bytes = cropped_img_byte_arr.getvalue()
                                print(f"Signature Analysis Task: Successfully cropped signature for doc {doc.id}, page {page_number}")
                            except Exception as e_crop:
                                print(f"Signature Analysis Task: Error cropping signature for doc {doc.id}, page {page_number}: {e_crop}")
                            # --- End Crop and Store ---

                            # Create SignatureInstance
                            db_signature_instance = models.SignatureInstance(
                                page_id=db_page.id,
                                document_id=doc.id,
                                # stakeholder_id will be linked later
                                bounding_box_json=block['Geometry']['BoundingBox'], # Store the BoundingBox dict
                                textract_response_json=block, # Store the whole signature block
                                cropped_signature_image=cropped_image_bytes # Store the cropped image
                            )
                            db_bg.add(db_signature_instance)
                            all_signature_instances_data.append({
                                "document_id": doc.id,
                                "page_number": page_number,
                                "signature_id_in_db": db_signature_instance.id, # Will be available after commit
                                "textract_data": block 
                            })
                    
                    if signatures_found_on_page:
                        db_bg.commit() # Commit after processing each page to save signatures
                        db_bg.refresh(proposal) # Refresh proposal to get updated signature instances if needed later in this task
                        print(f"Signature Analysis Task: Found and stored {len(signatures_found_on_page)} signatures for doc {doc.id}, page {page_number}.")
                    else:
                        print(f"Signature Analysis Task: No signatures found by Textract for doc {doc.id}, page {page_number}.")

                except Exception as e_textract:
                    print(f"Signature Analysis Task: Error calling Textract for doc {doc.id}, page {page_number}: {e_textract}")
                    # Potentially mark page/doc as failed for Textract
                    proposal.signature_analysis_status = f"failed_textract_doc_{doc.id}_page_{page_number}"
                    db_bg.commit()
                    # Continue to next page/document or fail entirely? For now, continue.

        # After processing all documents and pages:
        proposal.signature_analysis_status = "processing_gemini_report"
        db_bg.commit()

        # Construct prompt for Gemini based on all_signature_instances_data
        # This part remains largely the same, but now Gemini has access to Textract's raw output for each signature
        if not all_signature_instances_data:
            print(f"Signature Analysis Task: No signatures found in any document for proposal {proposal_id}. Skipping Gemini report.")
            proposal.signature_analysis_status = "completed_no_signatures"
            proposal.signature_analysis_report_html = "<p>No signatures were detected in the submitted documents.</p>"
            db_bg.commit()
            return

        # Re-fetch all signature instances with their DB IDs for the prompt
        # This ensures we have the IDs if they were committed inside the loop
        # A more optimized way might be to collect db_signature_instance objects directly
        # For now, let's re-query based on proposal to ensure data consistency
        
        # Simplified: Assume all_signature_instances_data contains enough info or re-query if needed.
        # For the prompt, we need to structure the Textract data.
        
        textract_summary_for_gemini = []
        # We need to query the stakeholders for this proposal to include them in the prompt
        stakeholders = db_bg.query(models.Stakeholder).filter(models.Stakeholder.project_id == proposal_id).all()
        stakeholder_names = [s.name for s in stakeholders] if stakeholders else ["Unknown Stakeholder"]


        # Fetch all signature instances for the proposal to build the prompt
        # This ensures we have the latest data, including IDs.
        current_signature_instances = db_bg.query(models.SignatureInstance).join(models.Document).filter(models.Document.project_id == proposal_id).all()

        for sig_instance in current_signature_instances:
            textract_summary_for_gemini.append({
                "signature_database_id": sig_instance.id,
                "document_id": sig_instance.document_id,
                "page_id": sig_instance.page_id,
                "textract_bounding_box": sig_instance.bounding_box_json,
                # Add any other relevant parts of textract_response_json if needed for the prompt
            })


        gemini_prompt = f"""
        You are an expert in signature analysis for financial and legal documents.
        You have been provided with data from AWS Textract for signatures found across multiple pages of various documents within a single proposal (Proposal ID: {proposal_id}).
        The proposal involves the following stakeholders: {', '.join(stakeholder_names) if stakeholder_names else 'Not specified'}.

        Your task is to generate a comprehensive HTML report that analyzes these signatures. The report should focus on:
        1.  **Intra-Stakeholder Consistency**: For each stakeholder (if signatures can be attributed to them later), how consistent are their signatures across all documents?
        2.  **Inter-Stakeholder Uniqueness**: How unique are the signatures when compared between different stakeholders? (e.g., are Applicant 1\'s signatures clearly different from Applicant 2\'s?)
        3.  **Overall Observations**: Any anomalies, low-confidence detections, or other points of interest.

        IMPORTANT: Currently, the signatures are NOT YET LINKED to specific stakeholders in the data provided.
        Your report should acknowledge this limitation. For now, analyze the signatures based on their appearance and Textract data.
        You can make general observations about groups of similar-looking signatures if they appear.
        The HTML report should be well-structured, easy to read, and use appropriate styling (e.g., tables, highlights for important findings).

        Data for detected signatures:
        {json.dumps(textract_summary_for_gemini, indent=2)}

        Please generate only the HTML content for the report body. Do not include <html>, <head>, or <body> tags.
        Highlight any areas of concern (e.g., very low confidence, significant variations that might belong to the same person but look different) in red or with strong emphasis.
        If no stakeholders are specified, make general observations about the consistency and uniqueness of the detected signature patterns.
        """
        try:
            print(f"Signature Analysis Task: Calling Gemini for proposal {proposal_id} report...")
            # Note: Gemini API here is called with text only. If images are needed, the call needs to be structured differently.
            # For now, Gemini will work off the descriptions and Textract data.
            ai_response = await gemini_model.generate_content_async(gemini_prompt)

            if ai_response.parts:
                html_report = ai_response.text
                # Clean up
                if html_report.startswith("```html"): html_report = html_report[7:]
                if html_report.startswith("```"): html_report = html_report[3:]
                if html_report.endswith("```"): html_report = html_report[:-3]
                html_report = html_report.strip()
                
                proposal.signature_analysis_report_html = html_report
                proposal.signature_analysis_status = "completed"
                print(f"Signature Analysis Task: Successfully generated signature analysis report for proposal {proposal_id}.")
            else:
                error_detail = "Gemini did not return expected content for signature report."
                if ai_response.prompt_feedback and ai_response.prompt_feedback.block_reason:
                    error_detail += f" Reason: {ai_response.prompt_feedback.block_reason_message or ai_response.prompt_feedback.block_reason}"
                print(f"Signature Analysis Task: Error generating report for proposal {proposal_id}: {error_detail}")
                proposal.signature_analysis_status = "failed_gemini_report_generation"
                proposal.signature_analysis_report_html = f"<p>Error generating report: {error_detail}</p>"
            db_bg.commit()

        except Exception as e_gemini:
            print(f"Signature Analysis Task: Exception calling Gemini for proposal {proposal_id} report: {e_gemini}")
            proposal.signature_analysis_status = "failed_gemini_exception"
            proposal.signature_analysis_report_html = f"<p>Exception during Gemini report generation: {e_gemini}</p>"
            db_bg.commit()

    except Exception as e_task:
        print(f"Signature Analysis Task: General error for proposal {proposal_id}: {e_task}")
        if db_bg.is_active:
            try:
                proposal = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
                if proposal and proposal.signature_analysis_status not in ["completed", "failed_gemini_report_generation", "failed_gemini_exception", "completed_no_signatures_found"]:
                    proposal.signature_analysis_status = "failed_unknown_task_error"
                    proposal.signature_analysis_report_html = f"<p>An unexpected error occurred during analysis: {e_task}</p>"
                db_bg.commit()
            except Exception as e_final_commit:
                 print(f"Signature Analysis Task: Error during final error commit for proposal {proposal_id}: {e_final_commit}")
                 db_bg.rollback() # Rollback if commit fails
    finally:
        if db_bg.is_active:
            db_bg.close()

@app.get("/proposals/{proposal_id}/signature-analysis/status", response_model=Dict[str, str])
def get_signature_analysis_status(proposal_id: int, db: Session = Depends(get_db)):
    """
    Get the current status of the signature analysis for a proposal.
    """
    proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")
    return {"proposal_id": str(proposal_id), "status": proposal.signature_analysis_status}

@app.get("/proposals/{proposal_id}/signature-analysis/report", response_class=HTMLResponse)
async def get_signature_analysis_report(proposal_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the generated HTML signature analysis report for a proposal.
    """
    proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")

    if proposal.signature_analysis_status == "completed" and proposal.signature_analysis_report_html:
        return HTMLResponse(content=proposal.signature_analysis_report_html)
    elif proposal.signature_analysis_report_html: # If there's any report HTML (e.g. error message)
        return HTMLResponse(content=proposal.signature_analysis_report_html, status_code=202) # Accepted, but not final
    else:
        return HTMLResponse(
            content=f"<h1>Signature analysis report for proposal {proposal_id}</h1><p>Status: {proposal.signature_analysis_status}. Report not yet available or generation failed.</p>",
            status_code=202 if proposal.signature_analysis_status not in ["completed", "failed_unknown_task_error", "failed_textract", "failed_gemini_report_generation", "failed_gemini_exception", "completed_no_signatures_found"] else 404
        )

# Main application entry point for Uvicorn
# To run: uvicorn app.main:app --reload

