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
    Page, # PageCreate and PageResponse removed
    Stakeholder, 
    SignatureInstance, SignatureInstanceCreate, SignatureInstanceResponse,
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

# New endpoint to serve cropped signature images
@app.get("/proposals/{proposal_id}/signatures/{signature_instance_id}/image")
async def get_cropped_signature_image(
    proposal_id: int,
    signature_instance_id: int,
    db: Session = Depends(get_db)
):
    # Query for the signature instance and verify its association with the proposal
    db_signature = db.query(models.SignatureInstance)\
        .join(models.Page, models.SignatureInstance.page_id == models.Page.id)\
        .join(models.Document, models.Page.document_id == models.Document.id)\
        .filter(
            models.SignatureInstance.id == signature_instance_id,
            models.Document.project_id == proposal_id
        ).first()

    if not db_signature:
        raise HTTPException(status_code=404, detail=f"Signature instance with id {signature_instance_id} not found in proposal {proposal_id}")

    if not db_signature.cropped_signature_image:
        raise HTTPException(status_code=404, detail=f"Cropped signature image not available for signature instance {signature_instance_id}")

    # Assuming the image was stored as PNG bytes (as per previous implementation)
    return Response(content=db_signature.cropped_signature_image, media_type="image/png")

@app.get("/proposals/{proposal_id}/signature-analysis/report", response_class=HTMLResponse)
async def get_signature_analysis_report(
    proposal_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve the HTML signature analysis report for a proposal.
    """
    db_proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()

    if not db_proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")

    if not db_proposal.signature_analysis_report_html:
        error_html = f"""
        <html>
            <head>
                <title>Signature Analysis Report Not Available</title>
            </head>
            <body>
                <h1>Signature analysis report for proposal {proposal_id} is not available or not yet generated.</h1>
                <p>Current status: {db_proposal.signature_analysis_status}</p>
                <p>Please ensure the analysis has been started and completed successfully using the <code>POST /proposals/{'{proposal_id}'}/signature-analysis/start</code> endpoint.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=404)

    return HTMLResponse(content=db_proposal.signature_analysis_report_html)

# The old /projects/generate-all-forms/ endpoint is superseded by the per-document HTML extraction.
# If project-wide generation is still needed, it would loop through documents and call the per-document endpoint.
# @app.get("/projects/generate-all-forms/", response_model=dict)
# async def generate_all_forms_for_project(project_id: int, db: Session = Depends(get_db)):

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
            # No further action possible if proposal doesn't exist
            return

        proposal.signature_analysis_status = "processing_textract_started"
        db_bg.commit()

        stakeholders_for_proposal = db_bg.query(models.Stakeholder).filter(models.Stakeholder.project_id == proposal_id).all()
        stakeholder_names = [s.name for s in stakeholders_for_proposal]

        for db_doc in proposal.documents:
            if not db_doc.pdf_file:
                print(f"Signature Analysis Task: PDF file missing for document {db_doc.id} in proposal {proposal_id}.")
                continue
            
            # Fetch proposal again to ensure it's current for status updates within loop
            current_proposal_in_loop = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
            if not current_proposal_in_loop: # Should not happen if initial check passed
                print(f"Signature Analysis Task: Proposal {proposal_id} disappeared during processing doc {db_doc.id}.")
                return
            current_proposal_in_loop.signature_analysis_status = f"processing_textract_doc_{db_doc.id}"
            db_bg.commit()

            try:
                images_from_pdf = convert_from_bytes(db_doc.pdf_file, dpi=200)
            except Exception as e_conv:
                db_bg.rollback() # Rollback on PDF conversion error
                print(f"Signature Analysis Task: Failed to convert PDF to images for doc {db_doc.id}: {e_conv}")
                proposal_after_conv_error = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
                if proposal_after_conv_error:
                    proposal_after_conv_error.signature_analysis_status = f"failed_pdf_conversion_doc_{db_doc.id}"
                    db_bg.commit()
                continue # Continue to next document

            for i, page_image_pil in enumerate(images_from_pdf):
                page_number = i + 1
                db_page = db_bg.query(models.Page).filter(
                    models.Page.document_id == db_doc.id,
                    models.Page.page_number == page_number
                ).first()

                if not db_page:
                    print(f"Signature Analysis Task: Page entry not found for doc {db_doc.id}, page {page_number}. Skipping Textract for this page.")
                    continue

                img_byte_arr = io.BytesIO()
                page_image_pil.save(img_byte_arr, format='PNG')
                img_byte_arr_val = img_byte_arr.getvalue()

                try:
                    textract_response = textract_client.analyze_document(
                        Document={'Bytes': img_byte_arr_val},
                        FeatureTypes=['SIGNATURES']
                    )

                    page_width, page_height = page_image_pil.size
                    found_signatures_on_page = False
                    for block in textract_response.get('Blocks', []):
                        if block['BlockType'] == 'SIGNATURE':
                            found_signatures_on_page = True
                            bbox = block['Geometry']['BoundingBox']
                            abs_left = int(bbox['Left'] * page_width)
                            abs_top = int(bbox['Top'] * page_height)
                            abs_width = int(bbox['Width'] * page_width)
                            abs_height = int(bbox['Height'] * page_height)
                            abs_right = abs_left + abs_width
                            abs_bottom = abs_top + abs_height

                            cropped_pil_image = page_image_pil.crop((abs_left, abs_top, abs_right, abs_bottom))
                            
                            cropped_img_byte_arr = io.BytesIO()
                            cropped_pil_image.save(cropped_img_byte_arr, format='PNG')
                            cropped_signature_bytes = cropped_img_byte_arr.getvalue()

                            db_sig_instance = models.SignatureInstance(
                                page_id=db_page.id,
                                document_id=db_doc.id,  # *** FIX: Populate document_id ***
                                bounding_box_json=json.dumps(bbox),
                                textract_response_json=json.dumps(block),
                                cropped_signature_image=cropped_signature_bytes,
                            )
                            db_bg.add(db_sig_instance)
                    
                    if found_signatures_on_page:
                        db_bg.commit() # Commit signatures for this page

                except Exception as e_textract:
                    db_bg.rollback() # *** FIX: Rollback on Textract error ***
                    print(f"Signature Analysis Task: Error during Textract processing for doc {db_doc.id}, page {page_number}: {e_textract}")
                    proposal_after_textract_error = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
                    if proposal_after_textract_error:
                        proposal_after_textract_error.signature_analysis_status = f"failed_textract_doc_{db_doc.id}_page_{page_number}"
                        db_bg.commit()
                    # Continue to the next page or document

        # Fetch proposal again before Gemini step
        proposal_before_gemini = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if not proposal_before_gemini:
            print(f"Signature Analysis Task: Proposal {proposal_id} not found before Gemini report stage.")
            return
        proposal_before_gemini.signature_analysis_status = "processing_gemini_report"
        db_bg.commit()

        content_parts_for_gemini = []
        
        initial_prompt_text = f"""
You are an expert in signature analysis for financial and legal documents.
You will be provided with one or more PDF documents for a single proposal (Proposal ID: {proposal_id}).
The proposal involves the following stakeholders: {', '.join(stakeholder_names) if stakeholder_names else 'Not specified'}.

For each PDF document provided, you will also receive a list of page numbers and, for each signature detected on those pages:
- Its unique Signature ID.
- The bounding box coordinates (as detected by AWS Textract) within the PDF page.
- An embedded HTML `<img>` tag that directly displays the cropped signature image. You MUST render this image in your report.

Your task is to generate a comprehensive HTML report that analyzes these signatures. The report should focus on:
1.  **Intra-Stakeholder Consistency**: For each stakeholder (if signatures can be attributed to them later by a human), how consistent are their signatures across all documents and pages? (Visually assess the signatures using the provided cropped images and their context in the PDFs at the given locations).
2.  **Inter-Stakeholder Uniqueness**: How unique are the signatures when compared between different stakeholders? (e.g., are Applicant 1's signatures clearly different from Applicant 2's based on visual assessment of the cropped images and their PDF context?).
3.  **Signature Quality**: For each detected signature (shown via `<img>` tag), assess its clarity and quality (e.g., clear, faint, rushed, potentially part of a stamp, obscured). Refer to the signature by its ID.
4.  **Overall Observations**: Any anomalies, irregularities, or other points of interest regarding the signatures (e.g., signatures appearing in unexpected places, signs of tampering near signature areas).

IMPORTANT: You are analyzing the visual signatures. Use the cropped images extensively in your analysis and refer to them by their Signature ID.
Currently, the signatures are NOT YET programmatically LINKED to specific stakeholders in the data provided.
Your report should acknowledge this limitation. For now, analyze the signatures based on their appearance.
You can make general observations about groups of similar-looking signatures if they appear.

The HTML report should be well-structured, easy to read, and use appropriate styling (e.g., tables, highlights for important findings).
Please generate only the HTML content for the report body. Do not include <html>, <head>, or <body> tags.
Highlight any areas of concern (e.g., significant variations that might belong to the same person but look different, very faint or unclear signatures, suspicious marks) in red or with strong emphasis.
If no stakeholders are specified, make general observations about the consistency, uniqueness, and quality of the detected signature patterns.

The following sections will detail the PDF documents and their associated signature locations, including the cropped images.
"""
        content_parts_for_gemini.append(initial_prompt_text)

        for db_doc_for_gemini in proposal.documents:
            if not db_doc_for_gemini.pdf_file:
                continue

            content_parts_for_gemini.append(f"--- Document: {db_doc_for_gemini.file_name} (ID: {db_doc_for_gemini.id}) ---")
            content_parts_for_gemini.append({'mime_type': 'application/pdf', 'data': db_doc_for_gemini.pdf_file})
            
            signatures_on_this_pdf_details_parts = []
            pages_with_signatures_on_this_doc = db_bg.query(models.Page)\
                .join(models.SignatureInstance, models.SignatureInstance.page_id == models.Page.id)\
                .filter(models.Page.document_id == db_doc_for_gemini.id)\
                .distinct()\
                .order_by(models.Page.page_number)\
                .all()

            if not pages_with_signatures_on_this_doc:
                signatures_on_this_pdf_details_parts.append("No signatures were detected by Textract in this document.")
            else:
                for page_with_sig in pages_with_signatures_on_this_doc:
                    page_sig_instances = db_bg.query(models.SignatureInstance)\
                        .filter(models.SignatureInstance.page_id == page_with_sig.id)\
                        .all()
                    if not page_sig_instances:
                        continue

                    sig_details_for_page_text = [f"Page {page_with_sig.page_number}:"]
                    for sig in page_sig_instances:
                        # IMPORTANT: Construct the URL carefully. This assumes your FastAPI app is running at the root.
                        # If it's behind a proxy or has a prefix, this URL needs to be adjusted or made absolute.
                        image_url = f"/proposals/{proposal_id}/signatures/{sig.id}/image" 
                        sig_detail = (
                            f"  - Signature ID: {sig.id}\n"
                            f"    Bounding Box (relative to page): {sig.bounding_box_json}\n"
                            f"    Cropped Image: <img src='{image_url}' alt='Signature {sig.id}' style='max-width:200px; max-height:100px; border:1px solid #ddd; vertical-align: middle; margin: 5px;' />"
                        )
                        sig_details_for_page_text.append(sig_detail)
                    
                    signatures_on_this_pdf_details_parts.append("\n".join(sig_details_for_page_text))
            
            if signatures_on_this_pdf_details_parts:
                text_part_for_signatures_on_pdf = "Signature Details for this Document:\n" + "\n".join(signatures_on_this_pdf_details_parts)
                content_parts_for_gemini.append(text_part_for_signatures_on_pdf)
            content_parts_for_gemini.append(f"--- End of Document: {db_doc_for_gemini.file_name} ---")

        if not gemini_model:
            db_bg.rollback() # Ensure clean state if we are to commit status
            print("Signature Analysis Task: Gemini model not initialized. Cannot generate report.")
            proposal_gemini_init_fail = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
            if proposal_gemini_init_fail:
                proposal_gemini_init_fail.signature_analysis_status = "failed_gemini_not_initialized"
                db_bg.commit()
            return

        try:
            print(f"Signature Analysis Task: Calling Gemini for proposal {proposal_id} with {len(content_parts_for_gemini)} parts.")
            ai_response = await gemini_model.generate_content_async(content_parts_for_gemini)
            
            proposal_after_gemini_call = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
            if not proposal_after_gemini_call:
                 print(f"Signature Analysis Task: Proposal {proposal_id} not found after Gemini call.")
                 return # Cannot update status

            if ai_response.parts:
                html_report_content = ai_response.text
                if html_report_content.startswith("```html"):
                    html_report_content = html_report_content[7:]
                if html_report_content.startswith("```"):
                    html_report_content = html_report_content[3:]
                if html_report_content.endswith("```"):
                    html_report_content = html_report_content[:-3]
                html_report_content = html_report_content.strip()

                proposal_after_gemini_call.signature_analysis_report_html = html_report_content
                proposal_after_gemini_call.signature_analysis_status = "completed_success"
                print(f"Signature Analysis Task: Successfully generated and saved signature analysis report for proposal {proposal_id}.")
            else:
                error_detail = "Gemini model did not return expected content for signature report."
                if ai_response.prompt_feedback and ai_response.prompt_feedback.block_reason:
                    error_detail += f" Reason: {ai_response.prompt_feedback.block_reason_message or ai_response.prompt_feedback.block_reason}"
                print(f"Signature Analysis Task: Error generating signature report for proposal {proposal_id}: {error_detail}")
                proposal_after_gemini_call.signature_analysis_status = f"failed_gemini_no_content ({error_detail[:100]})"
            db_bg.commit() # Commit Gemini results/status

        except Exception as e_gemini:
            db_bg.rollback() # *** FIX: Rollback on Gemini error ***
            print(f"Signature Analysis Task: Exception during Gemini call for proposal {proposal_id}: {str(e_gemini)}")
            proposal_gemini_exception = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
            if proposal_gemini_exception:
                proposal_gemini_exception.signature_analysis_status = f"failed_gemini_exception ({str(e_gemini)[:100]})"
                db_bg.commit()
        
    except Exception as e_task:
        db_bg.rollback() # *** FIX: Rollback on general task error ***
        print(f"Signature Analysis Task: General error for proposal {proposal_id}: {str(e_task)}")
        # Attempt to update status if proposal object is available
        proposal_final_error = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if proposal_final_error:
            try:
                proposal_final_error.signature_analysis_status = f"failed_task_exception ({str(e_task)[:100]})"
                db_bg.commit()
            except Exception as e_commit_fail:
                # If even this fails, rollback and log
                db_bg.rollback()
                print(f"Signature Analysis Task: Failed to commit final error status for proposal {proposal_id}: {e_commit_fail}")
    finally:
        db_bg.close()

# --- Document and Page Retrieval (Adjusted for new structure) ---
# I. Proposal (Project) Management
@app.get("/proposals/{proposal_id}/", response_model=models.ProjectResponse)
def get_proposal(proposal_id: int, db: Session = Depends(get_db)):
    proposal = db.query(models.Project).filter(models.Project.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal with id {proposal_id} not found")
    return proposal

# --- Health Check Endpoint ---
@app.get("/health/")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "ok"}

# Main application entry point for Uvicorn
# To run: uvicorn app.main:app --reload

