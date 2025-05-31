from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks # Added BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict # Added Dict
import uuid # Added for generating application_number
import random # Added for numerical application_number
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response # Added Response
import boto3 # Added for AWS Textract
from pdf2image import convert_from_bytes # Added for PDF to image conversion
import io
import json # Added for JSON processing
from PIL import Image # Added for image manipulation
import httpx # Added for HTTP client
import os
import PyPDF2 # For PDF processing
import google.generativeai as genai # Corrected import
from datetime import datetime # Added for report generation date
import base64 # Add this import

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
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise KeyError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # Using a common model
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    gemini_model = None
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    gemini_model = None

# HTTP client for calling external services
http_client = httpx.AsyncClient()

CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8090/chat/new") # Added
CHAT_MESSAGE_API_URL_TEMPLATE = os.getenv("CHAT_MESSAGE_API_URL_TEMPLATE", "http://localhost:8090/chat/{session_id}/message") # Added

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

# --- Helper Functions ---
def generate_application_number() -> str:
    """Generates a unique 10-digit numerical application number starting with 1."""
    # Generate a 9-digit random number
    random_digits = random.randint(0, 999999999)
    # Format it as a 9-digit string, padding with leading zeros if necessary
    nine_digits = str(random_digits).zfill(9)
    # Prepend '1' to make it a 10-digit number starting with 1
    application_number = "1" + nine_digits
    return application_number

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
    A chat session will be created for the proposal.
    """
    application_number = generate_application_number()
    # Ensure uniqueness, though UUIDs are highly unlikely to collide
    while db.query(models.Project).filter(models.Project.application_number == application_number).first():
        application_number = generate_application_number()

    chat_session_id = None
    try:
        response = await http_client.post(CHAT_API_URL)
        response.raise_for_status() # Raise an exception for bad status codes
        chat_session_id = response.json().get("id") # Assuming the API returns { "session_id": "..." }
    except httpx.RequestError as e:
        print(f"Error creating chat session for new proposal: {e}")
        # Decide if this should be a fatal error or if proposal can be created without it
        # For now, let's allow creation without it, but log the error.
        pass # Or raise HTTPException
    except Exception as e:
        print(f"An unexpected error occurred when creating chat session: {e}")
        pass

    db_project = models.Project(
        name=project_in.name,
        application_number=application_number,
        chat_session_id=chat_session_id # Store the new session ID
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

        chat_session_id_doc = None
        try:
            response = await http_client.post(CHAT_API_URL)
            response.raise_for_status()
            chat_session_id_doc = response.json().get("id")
        except httpx.RequestError as e:
            print(f"Error creating chat session for document {file_name}: {e}")
            # Decide if this should be a fatal error or if document can be created without it
            pass
        except Exception as e:
            print(f"An unexpected error occurred when creating chat session for document {file_name}: {e}")
            pass

        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            num_pages = len(pdf_reader.pages)

            db_document = models.Document(
                file_name=file_name,
                pdf_file=pdf_content,
                total_pages=num_pages,
                project_id=db_proposal.id,
                chat_session_id=chat_session_id_doc # Store the new session ID
            )
            db.add(db_document)
            db.commit() # Commit per document to get its ID for pages
            db.refresh(db_document)

            # Add document to its own chat session
            if db_document.chat_session_id:
                try:
                    pdf_content_for_upload_doc_session = io.BytesIO(pdf_content)
                    message_payload_doc_session = {"message": f"Context for document '{db_document.file_name}' (ID: {db_document.id})."}
                    doc_session_message_url = CHAT_MESSAGE_API_URL_TEMPLATE.format(session_id=db_document.chat_session_id)
                    
                    response_doc_chat = await http_client.post(
                        doc_session_message_url,
                        files={'file': (db_document.file_name, pdf_content_for_upload_doc_session, 'application/pdf')},
                        data=message_payload_doc_session
                    )
                    response_doc_chat.raise_for_status()
                    print(f"Successfully sent document {db_document.file_name} to its own chat session {db_document.chat_session_id}")
                except httpx.RequestError as exc:
                    print(f"Error sending PDF to document's chat session {db_document.chat_session_id}: {exc}")
                except httpx.HTTPStatusError as exc:
                    print(f"Chat service error for document's session {db_document.chat_session_id} during PDF upload: {exc.response.status_code} - {exc.response.text}")
                except Exception as e:
                    print(f"Unexpected error during PDF upload to document's chat session {db_document.chat_session_id}: {str(e)}")

            # Add document to proposal's chat session
            if db_proposal.chat_session_id:
                try:
                    pdf_content_for_upload_proposal_session = io.BytesIO(pdf_content)
                    message_payload_proposal_session = {
                        "message": f"New document '{db_document.file_name}' (ID: {db_document.id}) added to proposal '{db_proposal.name}' (ID: {db_proposal.id})."
                    }
                    proposal_session_message_url = CHAT_MESSAGE_API_URL_TEMPLATE.format(session_id=db_proposal.chat_session_id)

                    response_proposal_chat = await http_client.post(
                        proposal_session_message_url,
                        files={'file': (db_document.file_name, pdf_content_for_upload_proposal_session, 'application/pdf')},
                        data=message_payload_proposal_session
                    )
                    response_proposal_chat.raise_for_status()
                    print(f"Successfully sent document {db_document.file_name} to proposal's chat session {db_proposal.chat_session_id}")
                except httpx.RequestError as exc:
                    print(f"Error sending PDF to proposal's chat session {db_proposal.chat_session_id}: {exc}")
                except httpx.HTTPStatusError as exc:
                    print(f"Chat service error for proposal's session {db_proposal.chat_session_id} during PDF upload: {exc.response.status_code} - {exc.response.text}")
                except Exception as e:
                    print(f"Unexpected error during PDF upload to proposal's chat session {db_proposal.chat_session_id}: {str(e)}")

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
                Task for you: Extract the information from the document, You need to convert physical document into a digital verion which imitates the physical form , keep information prefilled and editable.Rememeber the accuracy of the information extracted specially filled information is absolutely important.You need to take care of multilingual , checkboxes and handwritten complexity within document. give me the html with good stylinng for review, if you are not confident on any field or section enough mark that area as red so that Human can rectify that easily. The output should be the html page content without suffix or prefix."""
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
            detail=f"HTML form for page {page_number} of document {document_id} has not been generated yet."
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

async def perform_signature_analysis(proposal_id: int, db_bg: Session = Depends(get_db)): # Assuming get_db_bg is your dependency
    """
    Background task to perform signature analysis using AWS Textract and Google Gemini.
    """
    db_bg = SessionLocal() # Create a new session for this background task
    try:
        proposal_before_textract = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if not proposal_before_textract:
            print(f"Signature Analysis Task: Proposal {proposal_id} not found.")
            return

        proposal_before_textract.signature_analysis_status = "processing_textract_started"
        db_bg.commit()
        db_bg.refresh(proposal_before_textract)

        for db_doc in proposal_before_textract.documents:
            if not db_doc.pdf_file:
                print(f"Signature Analysis Task: PDF file missing for document {db_doc.id} in proposal {proposal_id}.")
                continue
            
            # Fetch proposal again inside loop to get most recent status if other tasks modified it (though unlikely here)
            current_proposal_in_loop = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
            if not current_proposal_in_loop:
                print(f"Signature Analysis Task: Proposal {proposal_id} disappeared during processing doc {db_doc.id}.")
                return 
            current_proposal_in_loop.signature_analysis_status = f"processing_textract_doc_{db_doc.id}"
            db_bg.commit()
            db_bg.refresh(current_proposal_in_loop)

            try:
                images_from_pdf = convert_from_bytes(db_doc.pdf_file, dpi=200)
            except Exception as e_conv:
                db_bg.rollback() # Rollback any potential partial changes from this iteration
                print(f"Signature Analysis Task: Failed to convert PDF to images for doc {db_doc.id}: {e_conv}")
                proposal_after_conv_error = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
                if proposal_after_conv_error:
                    proposal_after_conv_error.signature_analysis_status = f"failed_pdf_conversion_doc_{db_doc.id}"
                    db_bg.commit()
                    db_bg.refresh(proposal_after_conv_error)
                continue

            for i, page_image_pil in enumerate(images_from_pdf):
                page_number = i + 1
                db_page = db_bg.query(models.Page).filter(
                    models.Page.document_id == db_doc.id,
                    models.Page.page_number == page_number
                ).first()

                if not db_page:
                    print(f"Signature Analysis Task: Page object not found for doc {db_doc.id}, page {page_number}. Skipping.")
                    # Potentially create the page if it's missing and expected to exist
                    # For now, we skip to avoid errors.
                    continue # Added continue

                img_byte_arr = io.BytesIO()
                page_image_pil.save(img_byte_arr, format='PNG')
                img_byte_arr_val = img_byte_arr.getvalue()

                try:
                    # Placeholder for Textract call and signature instance creation
                    # This is where you'd call textract_client.analyze_document
                    # and then process the response to find signatures, crop them,
                    # and create models.SignatureInstance objects.
                    # Example:
                    # response = textract_client.analyze_document(...)
                    # for block in response['Blocks']:
                    #   if block['BlockType'] == 'SIGNATURE':
                    #     # ... process signature ...
                    #     # db_signature_instance = models.SignatureInstance(...)
                    #     # db_bg.add(db_signature_instance)
                    #     pass # Replace with actual logic
                    print(f"Signature Analysis Task: Placeholder for Textract processing for doc {db_doc.id}, page {page_number}")
                    pass # Added pass

                except Exception as e_textract:
                    print(f"Signature Analysis Task: Error during Textract processing for doc {db_doc.id}, page {page_number}: {e_textract}")
                    # Update proposal status to reflect this specific error
                    proposal_after_textract_error = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
                    if proposal_after_textract_error:
                        proposal_after_textract_error.signature_analysis_status = f"error_textract_page_{db_doc.id}_{page_number}"
                        db_bg.commit()
                        db_bg.refresh(proposal_after_textract_error)
                    # Continue to the next page or document, or handle error more critically
                    pass # Added pass
            # After processing all pages of a document
            db_bg.commit() # Commit signature instances for the document
        
        proposal_before_gemini = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if not proposal_before_gemini:
            print(f"Signature Analysis Task: Proposal {proposal_id} not found before Gemini report generation.")
            # Update status if possible, though proposal object is None
            # This case should ideally be caught earlier.
            return # Added return
            
        proposal_before_gemini.signature_analysis_status = "processing_gemini_report"
        db_bg.commit()
        db_bg.refresh(proposal_before_gemini)

        # Prepare data for the HTML template
        documents_html_parts = []
        if hasattr(proposal_before_gemini, 'documents') and proposal_before_gemini.documents:
            for doc_idx, db_doc_report in enumerate(proposal_before_gemini.documents): # Renamed db_doc to db_doc_report to avoid conflict
                signatures_html_parts = []
                # Ensure db_doc_report.signature_instances is accessible and populated
                # This might require refreshing db_doc_report or ensuring relationships are loaded
                # For now, assuming it's available.
                # Example: db_bg.refresh(db_doc_report, ['signature_instances'])
                
                # Correctly query signature instances associated with the current document (db_doc_report) and its pages
                # This part was missing the actual query for signature instances for the report.
                # We need to iterate through pages of db_doc_report and then their signature_instances.
                
                # Fetch pages for the current document
                pages_for_report = db_bg.query(models.Page).filter(models.Page.document_id == db_doc_report.id).all()
                for page_for_report in pages_for_report:
                    # Fetch signature instances for the current page
                    signature_instances_for_page = db_bg.query(models.SignatureInstance).filter(models.SignatureInstance.page_id == page_for_report.id).all()

                    for sig_idx, sig_instance in enumerate(signature_instances_for_page):
                        cropped_signature_image_base64 = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" # Default placeholder
                        if hasattr(sig_instance, 'cropped_signature_image') and sig_instance.cropped_signature_image and isinstance(sig_instance.cropped_signature_image, bytes):
                            img_b64 = base64.b64encode(sig_instance.cropped_signature_image).decode('utf-8')
                            cropped_signature_image_base64 = f"data:image/png;base64,{img_b64}"
                        
                        textract_response_display = "{}"
                        if hasattr(sig_instance, 'textract_response_json') and sig_instance.textract_response_json:
                            try:
                                loaded_json = json.loads(sig_instance.textract_response_json)
                                textract_response_display = json.dumps(loaded_json, indent=2)
                            except (json.JSONDecodeError, TypeError):
                                textract_response_display = "Error decoding/loading Textract JSON."
                        
                        bounding_box_display = "{}"
                        if hasattr(sig_instance, 'bounding_box_json') and sig_instance.bounding_box_json:
                            try:
                                loaded_json_bbox = json.loads(sig_instance.bounding_box_json)
                                bounding_box_display = json.dumps(loaded_json_bbox, indent=2)
                            except (json.JSONDecodeError, TypeError):
                                bounding_box_display = "Error decoding/loading Bounding Box JSON."
                        
                        page_num_display = sig_instance.page_number if hasattr(sig_instance, 'page_number') else page_for_report.page_number # Fallback to page_for_report

                        signatures_html_parts.append(f"""
                            <div class="signature-block" style="margin-bottom: 15px; padding: 10px; border: 1px solid #eee;">
                                <h4>Signature {sig_idx + 1} (Page {page_num_display})</h4>
                                <img src="{cropped_signature_image_base64}" alt="Signature Image {sig_idx + 1}" style="max-width: 200px; max-height:100px; border: 1px solid #ccc; margin-bottom:5px;"/>
                                <p><strong>Bounding Box:</strong><pre>{bounding_box_display}</pre></p>
                                <p><strong>Textract Details (summary):</strong><pre>{textract_response_display}</pre></p>
                            </div>
                        """)
                signatures_html_content = "".join(signatures_html_parts) if signatures_html_parts else "<p>No signatures processed for this document.</p>"
                
                doc_name_display = db_doc_report.file_name if hasattr(db_doc_report, 'file_name') and db_doc_report.file_name else f"Unnamed Document (ID: {db_doc_report.id if hasattr(db_doc_report, 'id') else 'N/A'})"
                documents_html_parts.append(f"""
                    <div class="document-block" style="margin-top: 20px; padding:15px; background-color:#f9f9f9;">
                        <h3>Document: {doc_name_display}</h3>
                        {signatures_html_content}
                    </div>
                """)
        documents_html_content_final = "".join(documents_html_parts) if documents_html_parts else "<p>No documents found or processed for this proposal.</p>"
        
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        project_name_display = proposal_before_gemini.name if hasattr(proposal_before_gemini, 'name') and proposal_before_gemini.name else "N/A" # Corrected attribute to 'name'
        stakeholder_names_display = proposal_before_gemini.stakeholder_names if hasattr(proposal_before_gemini, 'stakeholder_names') and proposal_before_gemini.stakeholder_names else "N/A"
        proposal_id_display = proposal_before_gemini.id if hasattr(proposal_before_gemini, 'id') else "N/A"
        current_year = datetime.now().year

        SIGNATURE_ANALYSIS_REPORT_PROMPT_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signature Analysis Report for Proposal ID: {proposal_id_display}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; }}
        .container {{ max-width: 800px; margin: 20px auto; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1, h2, h3, h4 {{ color: #333; }}
        h1 {{ text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-bottom: 20px; color: #007bff;}}
        h2 {{ background-color: #e9ecef; padding: 10px; border-left: 4px solid #007bff; margin-top: 30px; }}
        h3 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
        pre {{ background-color: #e9ecef; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; }}
        .footer {{ text-align: center; margin-top: 30px; font-size: 0.9em; color: #777; }}
        .section {{ margin-bottom: 25px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Signature Analysis Report</h1>
        <div class="section" id="proposal-overview">
            <h2>Proposal Overview</h2>
            <table>
                <tr><th>Proposal ID</th><td>{proposal_id_display}</td></tr>
                <tr><th>Project Name</th><td>{project_name_display}</td></tr>
                <tr><th>Analysis Date</th><td>{current_time_str}</td></tr>
                <tr><th>Stakeholders</th><td>{stakeholder_names_display}</td></tr>
            </table>
        </div>
        <div class="section" id="document-analysis">
            <h2>Document & Signature Details</h2>
            {documents_html_content_final}
        </div>
        <div class="section" id="ai-summary">
            <h2>AI-Powered Insights & Summary</h2>
            <div id="gemini-analysis-content">
                <p><em>(AI analysis to be provided by Gemini based on the data presented. Gemini, please review the signature data, note any inconsistencies, patterns, or anomalies, and provide a concise summary of findings and a recommendation.)</em></p>
                {'''<!-- Example of what Gemini might fill in:
                <p><strong>Overall Consistency:</strong> High</p>
                <p><strong>Anomalies Detected:</strong> One signature on Document 'XYZ.pdf', page 3, appears rushed compared to others.</p>
                <p><strong>Recommendation:</strong> Manual review of the flagged signature is advised. Otherwise, documents appear consistent.</p>
                -->'''}
            </div>
        </div>
        <div class="footer">
            Generated by Signature Analysis System &copy; {current_year}
        </div>
    </div>
</body>
</html>
"""
        content_parts_for_gemini = [SIGNATURE_ANALYSIS_REPORT_PROMPT_TEMPLATE]
        
        global gemini_model 
        if not gemini_model:
            proposal_before_gemini.signature_analysis_status = "error_gemini_model_not_initialized"
            db_bg.commit()
            raise RuntimeError("Gemini model is not initialized.")

        ai_report_response = await gemini_model.generate_content_async(content_parts_for_gemini)
        
        html_report_content = ""
        if ai_report_response.prompt_feedback and ai_report_response.prompt_feedback.block_reason:
            error_message = f"Gemini content generation blocked: {ai_report_response.prompt_feedback.block_reason}"
            print(f"Signature Analysis Task: {error_message}")
            proposal_before_gemini.signature_analysis_status = f"error_gemini_blocked_{ai_report_response.prompt_feedback.block_reason}"
        elif not ai_report_response.parts:
            error_message = "Gemini response empty or invalid."
            print(f"Signature Analysis Task: {error_message}")
            proposal_before_gemini.signature_analysis_status = "error_gemini_empty_response"
        else:
            html_report_content = ai_report_response.text
        
        if html_report_content.strip().startswith("```html"):
            html_report_content = html_report_content.strip()[7:] 
            if html_report_content.strip().endswith("```"):
                html_report_content = html_report_content.strip()[:-3] 
        
        proposal_before_gemini.signature_analysis_report_html = html_report_content
        if not proposal_before_gemini.signature_analysis_status.startswith("error_"): 
            proposal_before_gemini.signature_analysis_status = "completed"
        db_bg.commit()
        db_bg.refresh(proposal_before_gemini)

    except Exception as e_gemini_report:
        print(f"Signature Analysis Task: Error during Gemini report generation: {e_gemini_report}")
        # Ensure proposal_before_gemini is available if it was fetched before the error
        # This might need to be re-fetched or handled carefully if the error occurred before its assignment
        proposal_at_error_time = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if proposal_at_error_time: 
            proposal_at_error_time.signature_analysis_status = f"error_gemini_report_generation_failed:_{str(e_gemini_report)}"
            db_bg.commit()
            db_bg.refresh(proposal_at_error_time)
        db_bg.rollback() 
    finally:
        if db_bg:
            db_bg.close()
# ... rest of your main.py file ...

# --- Health Check Endpoint ---
@app.get("/health/")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "ok"}

# Main application entry point for Uvicorn
# To run: uvicorn app.main:app --reload

