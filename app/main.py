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
from google import genai as thegenai
from datetime import datetime # Added for report generation date
import base64 # Add this import
from pathlib import Path
import asyncio
import threading 

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

# from signature import perform_signature_analysis
from . import signature

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
    # background_tasks.add_task(generate_html_for_document_pages, document_id, db_doc.project_id, db) # Pass db or necessary components
    background_tasks.add_task(generate_html_for_document_pages_parallel, document_id, db_doc.project_id) # Pass db or necessary components

    

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



async def process_single_page_concurrently(
    page_id: int,
    page_number: int,
    document_pdf_bytes: bytes, # Pass the whole PDF bytes
    # gemini_model_instance # Pass the model instance if not truly global in worker context
):
    """
    Processes a single page: extracts it, calls Gemini, and updates the DB.
    This function will manage its own database session.
    """
    print(f"Task - Starting processing for page ID {page_id}, page number {page_number}")
    # Create a new session for this specific task to ensure DB operation isolation
    db_task_session = SessionLocal()
    try:
        # --- 1. PDF Page Extraction (using the passed document_pdf_bytes) ---
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(document_pdf_bytes))
        if not (0 < page_number <= len(pdf_reader.pages)):
            print(f"Task - Page number {page_number} out of range for page ID {page_id}. Skipping.")
            return {"page_id": page_id, "status": "error", "message": "Page number out of range"}

        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_number - 1])
        single_page_pdf_buffer = io.BytesIO()
        pdf_writer.write(single_page_pdf_buffer)
        single_page_pdf_buffer.seek(0)
        pdf_page_bytes_for_ai = single_page_pdf_buffer.read()

        # --- 2. Gemini API Call ---
        if not gemini_model: # Assuming gemini_model is accessible (e.g., global or passed)
            print(f"Task - Gemini model not initialized for page ID {page_id}. Skipping AI call.")
            return {"page_id": page_id, "status": "error", "message": "Gemini model not initialized"}

        prompt = """You are Expert in reading complex documents.
        Task for you: Extract the information from the document, You need to convert physical document into a digital verion which imitates the physical form , keep information prefilled and editable.Rememeber the accuracy of the information extracted specially filled information is absolutely important.You need to take care of multilingual , checkboxes and handwritten complexity within document. give me the html with good stylinng for review, if you are not confident on any field or section enough mark that area as red so that Human can rectify that easily. The output should be the html page content without suffix or prefix."""
        pdf_blob = {
            'mime_type': 'application/pdf',
            'data': pdf_page_bytes_for_ai
        }

        print(f"Task - Calling Gemini for page ID {page_id}, page number {page_number}...")
        # Ensure gemini_model is the actual initialized model client
        ai_response = await gemini_model.generate_content_async([prompt, pdf_blob])

        html_content = None
        if ai_response.parts:
            html_content = ai_response.text
            if html_content.startswith("```html"):
                html_content = html_content[7:]
            if html_content.startswith("```"):
                html_content = html_content[3:]
            if html_content.endswith("```"):
                html_content = html_content[:-3]
            html_content = html_content.strip()
        else:
            error_detail = "AI model did not return expected content."
            if ai_response.prompt_feedback and ai_response.prompt_feedback.block_reason:
                error_detail += f" Reason: {ai_response.prompt_feedback.block_reason_message or ai_response.prompt_feedback.block_reason}"
            print(f"Task - Error generating HTML for page ID {page_id}: {error_detail}")
            return {"page_id": page_id, "status": "error", "message": error_detail}

        # --- 3. Database Update ---
        if html_content:
            page_to_update = db_task_session.query(models.Page).filter(models.Page.id == page_id).first()
            if page_to_update:
                page_to_update.generated_form_html = html_content
                db_task_session.commit()
                print(f"Task - Successfully generated and saved HTML for page ID {page_id}.")
                return {"page_id": page_id, "status": "success"}
            else:
                # This case should ideally not happen if page_id is valid
                print(f"Task - Page ID {page_id} not found in DB for update.")
                return {"page_id": page_id, "status": "error", "message": "Page not found in DB for update"}
        
        # Should have returned earlier if html_content was None due to AI error
        return {"page_id": page_id, "status": "error", "message": "Unknown error before DB update"}

    except Exception as e_page:
        db_task_session.rollback()
        print(f"Task - Exception processing page ID {page_id}: {str(e_page)}")
        return {"page_id": page_id, "status": "exception", "message": str(e_page)}
    finally:
        db_task_session.close()


async def generate_html_for_document_pages_parallel(document_id: int, proposal_id: int): # proposal_id still unused
    # This main function will manage fetching initial doc/page data with one session
    # but each concurrent task will use its own session for its specific update.
    db_main = SessionLocal()
    try:
        doc = db_main.query(models.Document).filter(models.Document.id == document_id).first()
        if not doc or not doc.pdf_file:
            print(f"Main Task: Document {document_id} or its PDF file not found.")
            return

        # Fetch all page models/metadata at once
        pages_models_to_process = db_main.query(models.Page).filter(models.Page.document_id == document_id).all()
        if not pages_models_to_process:
            print(f"Main Task: No pages found for document {document_id}.")
            return

        print(f"Main Task: Starting HTML generation for document {document_id}, {len(pages_models_to_process)} pages in parallel.")

        # Store the PDF bytes in memory to pass to each task
        # This avoids each task trying to access doc.pdf_file which might involve lazy loading issues
        # or repeated reads if it's not just a simple byte array in the 'doc' object.
        document_pdf_bytes = doc.pdf_file # Assuming doc.pdf_file holds the raw bytes

        tasks = []
        for page_model in pages_models_to_process:
            if page_model.generated_form_html: # Skip if already generated
                print(f"Main Task: HTML already exists for page {page_model.page_number} (ID: {page_model.id}). Skipping.")
                continue
            
            # Create a task for each page
            tasks.append(
                process_single_page_concurrently(
                    page_id=page_model.id,
                    page_number=page_model.page_number,
                    document_pdf_bytes=document_pdf_bytes
                    # gemini_model_instance=gemini_model # if needed to pass explicitly
                )
            )

        if not tasks:
            print(f"Main Task: No pages require processing for document {document_id}.")
            return

        # Run all tasks concurrently and wait for them to complete
        # return_exceptions=True ensures that if one task fails, others continue,
        # and exceptions are returned as results instead of stopping asyncio.gather.
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception): # An unexpected error in a task not caught by its own try/except
                print(f"Main Task: A page processing task failed with an unhandled exception: {result}")
            elif isinstance(result, dict): # Expected dictionary from process_single_page_concurrently
                print(f"Main Task: Result for page ID {result.get('page_id')}: {result.get('status')}, Message: {result.get('message', '')}")
            else:
                print(f"Main Task: Received an unexpected result type from a page task: {result}")


        print(f"Main Task: Finished parallel HTML generation attempt for document {document_id}.")

    except Exception as e_doc:
        # db_main.rollback() # Not strictly necessary here as db_main is read-only in this revised structure
        print(f"Main Task: General error for document {document_id}: {str(e_doc)}")
    finally:
        db_main.close()
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
    if not gemini_model: # Ensure gemini_model is checked here as well
        raise HTTPException(status_code=500, detail="Gemini client is not configured. Cannot start analysis.")

    # Update status
    db_proposal.signature_analysis_status = "processing_started"
    db.commit()

    # Pass proposal_id to the background task
    background_tasks.add_task(perform_signature_analysis, proposal_id)

    return {"message": f"Signature analysis initiated for proposal {proposal_id}."}


async def _process_page_with_textract_concurrently(
    page_image_pil,  # PIL image object for the page
    page_id: int,    # Database ID of the page
    document_id: int,
    page_number: int,  # 1-based page number for logging/context
    textract_client_instance  # Pass the initialized Textract client
):
    """
    Processes a single page with Textract in a way suitable for concurrent execution.
    Returns data for SignatureInstance objects or error information.
    """
    # Convert PIL image to bytes for Textract
    img_byte_arr_io = io.BytesIO()
    page_image_pil.save(img_byte_arr_io, format='PNG')  # Or JPEG, ensure consistency
    img_bytes_for_textract = img_byte_arr_io.getvalue()

    signature_data_for_db = []  # To hold dicts for creating SignatureInstance objects
    try:
        current_thread_id = asyncio.get_running_loop().call_soon_threadsafe(lambda: threading.get_ident()) # For debug
        print(f"Textract Task (Doc: {document_id}, Page: {page_number}, Thread: {current_thread_id}): Calling Textract...")

        # AWS SDK calls are synchronous (blocking). Run in a separate thread
        # to avoid blocking the asyncio event loop.
        textract_response = await asyncio.to_thread(
            textract_client_instance.analyze_document,
            Document={'Bytes': img_bytes_for_textract},
            FeatureTypes=['SIGNATURES']
        )

        signatures_found_on_page_count = 0
        for block in textract_response.get('Blocks', []):
            if block['BlockType'] == 'SIGNATURE':
                signatures_found_on_page_count += 1
                cropped_image_bytes = None
                try:
                    # Cropping logic (PIL operations are CPU-bound but generally fast for small crops)
                    # If this part becomes a bottleneck, it could also be wrapped in asyncio.to_thread
                    geometry = block['Geometry']
                    bbox = geometry['BoundingBox']
                    img_width, img_height = page_image_pil.size

                    left = int(bbox['Left'] * img_width)
                    top = int(bbox['Top'] * img_height)
                    right = int((bbox['Left'] + bbox['Width']) * img_width)
                    bottom = int((bbox['Top'] + bbox['Height']) * img_height)

                    cropped_pil_image = page_image_pil.crop((left, top, right, bottom))

                    cropped_img_byte_io = io.BytesIO()
                    cropped_pil_image.save(cropped_img_byte_io, format='PNG')
                    cropped_image_bytes = cropped_img_byte_io.getvalue()
                    # print(f"Textract Task (Doc: {document_id}, Page: {page_number}): Successfully cropped signature.")
                except Exception as e_crop:
                    print(f"Textract Task (Doc: {document_id}, Page: {page_number}): Error cropping signature: {e_crop}")

                # Prepare data for SignatureInstance. We don't create the SQLAlchemy model instance here
                # to avoid passing session-bound objects or the session itself into sub-tasks.
                signature_data_for_db.append({
                    "page_id": page_id,
                    "document_id": document_id,
                    "bounding_box_json": json.dumps(block['Geometry']['BoundingBox']),
                    "textract_response_json": json.dumps(block),
                    "cropped_signature_image": cropped_image_bytes,
                })

        if signatures_found_on_page_count > 0:
            print(f"Textract Task (Doc: {document_id}, Page: {page_number}): Found {signatures_found_on_page_count} signatures.")
        # else:
            # print(f"Textract Task (Doc: {document_id}, Page: {page_number}): No signatures found by Textract.")
        
        return {"status": "success", "page_id": page_id, "page_number": page_number, "data": signature_data_for_db}

    except Exception as e_textract:
        print(f"Textract Task (Doc: {document_id}, Page: {page_number}): Error during Textract API call or processing: {e_textract}")
        return {"status": "error", "page_id": page_id, "page_number": page_number, "message": str(e_textract)}


async def perform_signature_analysis(proposal_id: int):
    """
    Background task to perform signature analysis using AWS Textract (in parallel for pages)
    and Google Gemini.
    """
    db_bg = SessionLocal()  # Create a new session for this background task
    # textract_client should be initialized and available here
    # gemini_model (global) and thegenai.Client() are used in the Gemini part

    try:
        proposal = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if not proposal:
            print(f"Signature Analysis Task: Proposal {proposal_id} not found.")
            return

        proposal.signature_analysis_status = "processing_textract"
        db_bg.commit()
        db_bg.refresh(proposal)

        # This list will be populated after all Textract processing for all documents is done.
        # The original `all_signature_instances_data_for_gemini_prompt` was populated inside the doc loop,
        # which is fine if it's cleared or managed per proposal.
        # For this version, we'll fetch all signatures fresh before the Gemini stage.

        for doc in proposal.documents:
            if not doc.pdf_file:
                print(f"Signature Analysis Task: PDF file missing for document {doc.id} in proposal {proposal_id}.")
                continue
            
            # This re-query pattern was in your original code.
            current_proposal_status_doc = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
            if not current_proposal_status_doc:
                print(f"Signature Analysis Task: Proposal {proposal_id} disappeared during doc {doc.id} processing.")
                return # Or handle error appropriately
            current_proposal_status_doc.signature_analysis_status = f"processing_textract_doc_{doc.id}"
            db_bg.commit()
            # db_bg.refresh(current_proposal_status_doc) # Refreshing 'proposal' object is usually preferred

            try:
                images_from_pdf = convert_from_bytes(doc.pdf_file, dpi=200)
            except Exception as e_conv:
                print(f"Signature Analysis Task: Failed to convert PDF to images for doc {doc.id}: {e_conv}")
                proposal.signature_analysis_status = f"failed_pdf_conversion_doc_{doc.id}" # Update main proposal object
                db_bg.commit()
                db_bg.refresh(proposal)
                continue  # Skip to next document

            # --- Parallel Textract Processing for Pages of current document ---
            page_processing_tasks = []
            # Pre-fetch page DB objects for this document to get their IDs efficiently
            page_db_objects_map = {
                p.page_number: p for p in db_bg.query(models.Page).filter(models.Page.document_id == doc.id).all()
            }

            for i, page_image_pil in enumerate(images_from_pdf):
                page_number = i + 1
                db_page_obj = page_db_objects_map.get(page_number)

                if not db_page_obj:
                    print(f"Signature Analysis Task: Page DB entry not found for doc {doc.id}, page {page_number}. Skipping Textract task creation.")
                    continue

                page_processing_tasks.append(
                    _process_page_with_textract_concurrently(
                        page_image_pil,
                        db_page_obj.id,
                        doc.id,
                        page_number,
                        textract_client  # Pass your initialized textract_client
                    )
                )
            
            new_signature_instances_for_this_document = []
            document_textract_processing_had_errors = False
            if page_processing_tasks:
                print(f"Signature Analysis Task (Doc: {doc.id}): Launching {len(page_processing_tasks)} Textract page tasks in parallel.")
                # Note: Consider asyncio.Semaphore here if you have many pages per doc to limit concurrency
                results = await asyncio.gather(*page_processing_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):  # Unhandled exception from gather itself
                        print(f"Signature Analysis Task (Doc: {doc.id}): A page task failed unexpectedly (gather exception): {result}")
                        document_textract_processing_had_errors = True
                    elif isinstance(result, dict): # Expected result from helper
                        if result.get("status") == "success":
                            for sig_data_dict in result.get("data", []):
                                new_signature_instances_for_this_document.append(models.SignatureInstance(**sig_data_dict))
                        elif result.get("status") == "error":
                            print(f"Signature Analysis Task (Doc: {doc.id}, Page: {result.get('page_number')}): Failed Textract processing for page: {result.get('message')}")
                            document_textract_processing_had_errors = True
                    else: # Unexpected result type
                        print(f"Signature Analysis Task (Doc: {doc.id}): A page task returned an unexpected result type: {result}")
                        document_textract_processing_had_errors = True
            
            if new_signature_instances_for_this_document:
                db_bg.add_all(new_signature_instances_for_this_document)
                # Commit all signatures found in this document by Textract in one transaction
                db_bg.commit() 
                print(f"Signature Analysis Task (Doc: {doc.id}): Stored {len(new_signature_instances_for_this_document)} new signatures from Textract.")
                # No need to refresh proposal here unless adding signatures affects a direct relationship used immediately
            
            # Update document processing status
            current_proposal_status_doc_done = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first() # Re-query as per original pattern
            if current_proposal_status_doc_done:
                if document_textract_processing_had_errors:
                    current_proposal_status_doc_done.signature_analysis_status = f"completed_textract_doc_{doc.id}_with_errors"
                else:
                    current_proposal_status_doc_done.signature_analysis_status = f"completed_textract_doc_{doc.id}"
                db_bg.commit()
                # db_bg.refresh(current_proposal_status_doc_done)
            # --- End of Parallel Textract Processing for current document ---

        # --- Gemini Processing Stage (largely as per your original structure) ---
        proposal_for_gemini = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if not proposal_for_gemini:
            print(f"Signature Analysis Task: Proposal {proposal_id} not found before Gemini stage.")
            return

        proposal_for_gemini.signature_analysis_status = "processing_gemini_report"
        db_bg.commit()
        db_bg.refresh(proposal_for_gemini)

        # Fetch all signature instances for the proposal to build the prompt
        # This query will fetch all instances committed from the Textract stage above.
        current_signature_instances = db_bg.query(models.SignatureInstance)\
            .join(models.Page).join(models.Document)\
            .filter(models.Document.project_id == proposal_id).all()

        if not current_signature_instances:
            print(f"Signature Analysis Task: No signatures found in any document for proposal {proposal_id} after Textract. Skipping Gemini report.")
            proposal_for_gemini.signature_analysis_status = "completed_no_signatures"
            proposal_for_gemini.signature_analysis_report_json = {"message": "No signatures were detected in the submitted documents by Textract."} # Store JSON as string or use JSON type if DB supports
            db_bg.commit()
            db_bg.refresh(proposal_for_gemini)
            return

        all_signature_instances_data_for_gemini_prompt = []
        for sig_instance in current_signature_instances: # This loop has N+1 query potential
            # The N+1 query problem was in your original code here. For minimal changes, I'm leaving it.
            # To fix, use joinedload when fetching current_signature_instances or query documents/pages separately once.
            document_fetch_for_gemini = db_bg.query(models.Document).filter(models.Document.id == sig_instance.document_id).first()
            page_fetch_for_gemini = db_bg.query(models.Page).filter(models.Page.id == sig_instance.page_id).first()
            
            doc_name = document_fetch_for_gemini.file_name if document_fetch_for_gemini else "Unknown Document"
            page_num = page_fetch_for_gemini.page_number if page_fetch_for_gemini else "Unknown Page"

            all_signature_instances_data_for_gemini_prompt.append({
                "document_name": doc_name,
                "page_number": page_num,
                "textract_bounding_box": json.loads(sig_instance.bounding_box_json) if isinstance(sig_instance.bounding_box_json, str) else sig_instance.bounding_box_json,
                "signature_image_url": f"http://localhost:8000/proposals/{proposal_id}/signatures/{sig_instance.id}/image", # Make base URL configurable
            })
        
        # Fetching document URLs for Gemini (as per your original)
        documents_for_gemini_urls = db_bg.query(models.Document).filter(models.Document.project_id == proposal_id).all()
        docUrls = []
        for doc_obj in documents_for_gemini_urls: # Renamed 'doc' to 'doc_obj' to avoid conflict with outer loop 'doc'
            docUrls.append(f"http://localhost:8000/proposals/{doc_obj.project_id}/documents/{doc_obj.id}/pdf") # Make base URL configurable

        print(f"Document URLs for Gemini: {docUrls}")
        
        # Using thegenai.Client as per your original Gemini section
        # Ensure 'thegenai' is imported and client configured if needed
        try:
            client = thegenai.Client() # Ensure this client is configured for auth if needed
        except Exception as e_thegenai_client:
            print(f"Signature Analysis Task: Failed to initialize thegenai.Client: {e_thegenai_client}")
            proposal_for_gemini.signature_analysis_status = "failed_gemini_client_init"
            db_bg.commit()
            return


        docData = []
        # The httpx.get calls here are synchronous. For a fully async function, these should be async.
        # To keep changes minimal to "existing code", this is left as is from your snippet.
        print(f"Signature Analysis Task: Fetching PDF bytes for Gemini via HTTP...")
        try:
            for doc_url in docUrls:
                print(f"Fetching content from: {doc_url}")
                response = httpx.get(doc_url) # Synchronous call
                response.raise_for_status() # Check for HTTP errors
                doc_data_bytes = io.BytesIO(response.content)
                # The file upload to Gemini SDK might also be synchronous or async
                # Assuming synchronous here based on client.models.generate_content later
                uploaded_file = client.files.upload( # This might be sync or async depending on thegenai SDK
                    file=doc_data_bytes,
                    config=dict(mime_type='application/pdf') # Ensure this config is correct for the SDK
                )
                docData.append(uploaded_file)
                print(f"Uploaded {doc_url} to Gemini files.")
        except Exception as e_http_fetch_upload:
            print(f"Signature Analysis Task: Error fetching or uploading PDF for Gemini: {e_http_fetch_upload}")
            proposal_for_gemini.signature_analysis_status = "failed_gemini_pdf_fetch_upload"
            db_bg.commit()
            return
        
        gemini_prompt_text = f""" 
        Hi i am building an application where i will upload multiple pdf documents and there can be multiple signature of different stake holders
        We need to do the signature matching of all the stakeholders
        Basically for 1 type of stakeholder all their signature shoud be matching
        And no two stake holder can be same person so no two stakeholder signature should match
        I have given you all the documents also for the processing, after doing the signature analysis please build the final report, how should be the UI of 
        the final report. Please generate the report very intituitive

        Along with the documents provided are are some of the signatures we have detected using the amazon textract service. use this information if needed. But not necessary.
        {json.dumps(all_signature_instances_data_for_gemini_prompt, indent=2)}
        Here we have provided the image URLs for the detected signatures. You can use these URLs to include images in your report using the <img> tag in HTML.

        Task: Generate a comprehensive HTML report analyzing the signatures detected in the provided documents.

        Focus on:
        1. Intra-Stakeholder Consistency (Where all the signature of a particular stakeholder is matched).
        2. Inter-Stakeholder Uniqueness (No two stakeholder should have matching signatures).
        3. Overall Observations: Anomalies, low-confidence detections.

        Have the following sections in the report:
        1. Overall Summary
            1.1 Document Analysed
            1.2 Stakeholders Identified
            1.3 Overall Status (e.g., "All signatures match", "Some signatures do not match", etc.)
        2. Detailed Stakeholder Analysis
            List of all stakeholders with their signatures, status, and any anomalies with the analysis result.
        3. Cross-Stakeholder Uniqueness Verification
            Table confirms that no two distinct stakeholders share the same signature

        Provide the report in a well-structured HTML format use javascript and css to make it more interactive and user friendly.
        Ensure the report is suitable for review by a human analyst.
        """
        
        # global gemini_model # This was in your code, but 'client' from thegenai is used for the call
        # Check if the 'gemini_model' global is still relevant or if 'client' is the sole Gemini interface here.
        # The code below uses `client.models.generate_content`, not the global `gemini_model`.
        # If `gemini_model` was for a different async client, this part needs reconciliation.
        # Assuming `thegenai.Client().models.generate_content` is the intended synchronous call as per your snippet.

        try:
            print(f"Signature Analysis Task: Calling Gemini for proposal {proposal_id} HTML report...")
            # This call `client.models.generate_content` is likely synchronous.
            # In an async function, this will block the event loop.
            # To keep changes minimal to "existing code", this is left as is.
            ai_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20", # Model name from your snippet, ensure it's current
                contents=docData + [gemini_prompt_text] # Ensure docData format is compatible
            )

            print(f"AI Response received. Type: {type(ai_response)}, Parts: {hasattr(ai_response, 'parts')}, Text: {hasattr(ai_response, 'text')}")


            if hasattr(ai_response, 'text') and ai_response.text: # Check if text attribute exists and is not empty
                html_report = ai_response.text
                if html_report.startswith("```html"): html_report = html_report[7:]
                if html_report.startswith("```"): html_report = html_report[3:]
                if html_report.endswith("```"): html_report = html_report[:-3]
                html_report = html_report.strip()
                
                proposal_for_gemini.signature_analysis_report_html = html_report
                proposal_for_gemini.signature_analysis_status = "completed"
                print(f"Signature Analysis Task: Successfully generated signature analysis HTML report for proposal {proposal_id}.")
            else:
                error_detail = "Gemini did not return expected text content for signature report."
                # The original code had `ai_response.prompt_feedback` which might not exist on `thegenai` client's response object
                # Adjust error feedback parsing based on actual `thegenai` response structure for errors.
                # For now, a generic message:
                # if hasattr(ai_response, 'prompt_feedback') and ai_response.prompt_feedback and ai_response.prompt_feedback.block_reason:
                #    error_detail += f" Reason: {ai_response.prompt_feedback.block_reason_message or ai_response.prompt_feedback.block_reason}"
                print(f"Signature Analysis Task: Error generating report for proposal {proposal_id}: {error_detail}. AI Response: {ai_response}")
                proposal_for_gemini.signature_analysis_status = "failed_gemini_report_generation"
                # Storing error in JSON field, ensure it's a dict or JSON string
                proposal_for_gemini.signature_analysis_report_json = {"error": error_detail, "ai_response_debug": str(ai_response)} 
            
            db_bg.commit()
            db_bg.refresh(proposal_for_gemini)

        except Exception as e_gemini:
            print(f"Signature Analysis Task: Exception calling Gemini for proposal {proposal_id} report: {e_gemini}")
            proposal_for_gemini.signature_analysis_status = "failed_gemini_exception"
            proposal_for_gemini.signature_analysis_report_json = {"error": f"Exception during Gemini report generation: {str(e_gemini)}"}
            db_bg.commit()
            db_bg.refresh(proposal_for_gemini)

    except Exception as e_task:
        print(f"Signature Analysis Task: General error for proposal {proposal_id}: {e_task}")
        import traceback
        traceback.print_exc()
        if db_bg.is_active:
            try:
                proposal_at_error = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
                if proposal_at_error and proposal_at_error.signature_analysis_status not in [
                    "completed", "failed_gemini_report_generation", "failed_gemini_exception", 
                    "completed_no_signatures" # Removed "failed_gemini_invalid_json" as JSON parsing was removed
                ]:
                    proposal_at_error.signature_analysis_status = "failed_unknown_task_error"
                    proposal_at_error.signature_analysis_report_json = {"error": f"An unexpected error occurred during analysis: {str(e_task)}"}
                    db_bg.commit()
            except Exception as e_final_commit:
                 print(f"Signature Analysis Task: Error during final error commit for proposal {proposal_id}: {e_final_commit}")
                 db_bg.rollback() 
    finally:
        if db_bg.is_active:
            db_bg.close()
# --- Health Check Endpoint ---
@app.get("/health/")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "ok"}



@app.get("/proposals/{proposal_id}/documents/{document_id}/pdf")
async def get_document_page_pdf(
    proposal_id: int, 
    document_id: int, 
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
        # print(pdf_reader, "Total pages:", len(pdf_reader.pages))
        pdf_writer = PyPDF2.PdfWriter()
        for page_number in range(1, len(pdf_reader.pages) + 1):
            pdf_writer.add_page(pdf_reader.pages[page_number - 1])  # PyPDF2 pages are 0-indexed

        output_pdf_buffer = io.BytesIO()
        pdf_writer.write(output_pdf_buffer)
        output_pdf_buffer.seek(0)

        return Response(content=output_pdf_buffer.read(), media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF page: {str(e)}")

# Main application entry point for Uvicorn
# To run: uvicorn app.main:app --reload

