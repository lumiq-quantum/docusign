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
from pathlib import Path

# --- AWS Textract Client ---
# Configure your AWS region
AWS_REGION = os.getenv("AWS_REGION", "us-east-1") # Default to us-east-1 if not set
try:
    textract_client = boto3.client("textract", region_name=AWS_REGION)
except Exception as e:
    print(f"Error initializing AWS Textract client: {e}")
    textract_client = None

from .models import (
    SessionLocal, engine, get_db,
    Project, ProjectCreate, ProjectResponse,
    Document, DocumentCreate, DocumentResponse,
    Page, # PageCreate and PageResponse removed
    Stakeholder, 
    SignatureInstance, SignatureInstanceCreate, SignatureInstanceResponse,
    GeneratedHtmlResponse
)

from . import models
from pdf2image import convert_from_bytes # Added for PDF to image conversion


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


async def perform_signature_analysis(proposal_id: int): # Removed db_bg: Session = Depends(get_db) as SessionLocal is used
    """
    Background task to perform signature analysis using AWS Textract and Google Gemini.
    """
    db_bg = SessionLocal() # Create a new session for this background task
    try:
        proposal = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if not proposal:
            print(f"Signature Analysis Task: Proposal {proposal_id} not found.")
            return

        proposal.signature_analysis_status = "processing_textract"
        db_bg.commit()
        db_bg.refresh(proposal) # Refresh to ensure status is updated before proceeding

        all_signature_instances_data_for_gemini_prompt = [] # To collect data for Gemini

        for doc in proposal.documents:
            if not doc.pdf_file:
                print(f"Signature Analysis Task: PDF file missing for document {doc.id} in proposal {proposal_id}.")
                continue
            
            current_proposal_status_doc = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
            if not current_proposal_status_doc: # Should not happen if proposal existed initially
                print(f"Signature Analysis Task: Proposal {proposal_id} disappeared during doc {doc.id} processing.")
                return
            current_proposal_status_doc.signature_analysis_status = f"processing_textract_doc_{doc.id}"
            db_bg.commit()
            db_bg.refresh(current_proposal_status_doc)


            try:
                # Convert whole PDF to images once, then process pages
                images_from_pdf = convert_from_bytes(doc.pdf_file, dpi=200) # Added dpi as in previous working version
            except Exception as e_conv:
                print(f"Signature Analysis Task: Failed to convert PDF to images for doc {doc.id}: {e_conv}")
                # Update proposal status directly on the 'proposal' object we are working with
                proposal.signature_analysis_status = f"failed_pdf_conversion_doc_{doc.id}"
                db_bg.commit()
                db_bg.refresh(proposal)
                continue # Skip to next document

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
                img_byte_arr_io = io.BytesIO() # Renamed to avoid confusion
                page_image_pil.save(img_byte_arr_io, format='PNG') # Or JPEG
                img_bytes_for_textract = img_byte_arr_io.getvalue()

                try:
                    print(f"Signature Analysis Task: Calling Textract for doc {doc.id}, page {page_number}...")
                    textract_response = textract_client.analyze_document(
                        Document={'Bytes': img_bytes_for_textract},
                        FeatureTypes=['SIGNATURES']
                    )
                    
                    signatures_found_on_page_count = 0
                    for block in textract_response.get('Blocks', []):
                        if block['BlockType'] == 'SIGNATURE':
                            signatures_found_on_page_count += 1
                            
                            cropped_image_bytes = None
                            try:
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
                                print(f"Signature Analysis Task: Successfully cropped signature for doc {doc.id}, page {page_number}")
                            except Exception as e_crop:
                                print(f"Signature Analysis Task: Error cropping signature for doc {doc.id}, page {page_number}: {e_crop}")

                            db_signature_instance = models.SignatureInstance(
                                page_id=db_page.id,
                                document_id=doc.id, # Storing document_id directly as well
                                # stakeholder_id will be linked later
                                bounding_box_json=json.dumps(block['Geometry']['BoundingBox']), # Store as JSON string
                                textract_response_json=json.dumps(block), # Store as JSON string
                                cropped_signature_image=cropped_image_bytes
                            )
                            db_bg.add(db_signature_instance)
                            # We need to commit here to get the ID for all_signature_instances_data_for_gemini_prompt
                            # However, committing frequently can be slow. Alternative: append object and query later.
                            # For now, let's commit after processing all signatures on a page.
                    
                    if signatures_found_on_page_count > 0:
                        db_bg.commit() # Commit after processing all signatures on the page
                        db_bg.refresh(proposal) # Refresh proposal to get updated relations if needed
                        print(f"Signature Analysis Task: Found and stored {signatures_found_on_page_count} signatures for doc {doc.id}, page {page_number}.")
                    else:
                        print(f"Signature Analysis Task: No signatures found by Textract for doc {doc.id}, page {page_number}.")

                except Exception as e_textract:
                    print(f"Signature Analysis Task: Error calling Textract for doc {doc.id}, page {page_number}: {e_textract}")
                    proposal.signature_analysis_status = f"failed_textract_doc_{doc.id}_page_{page_number}"
                    db_bg.commit()
                    db_bg.refresh(proposal)
            
            # After processing all pages of a document, update status
            current_proposal_status_doc_done = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
            if current_proposal_status_doc_done:
                 current_proposal_status_doc_done.signature_analysis_status = f"completed_textract_doc_{doc.id}"
                 db_bg.commit()
                 db_bg.refresh(current_proposal_status_doc_done)


        # After processing all documents and pages:
        proposal_for_gemini = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
        if not proposal_for_gemini:
            print(f"Signature Analysis Task: Proposal {proposal_id} not found before Gemini stage.")
            return

        proposal_for_gemini.signature_analysis_status = "processing_gemini_report"
        db_bg.commit()
        db_bg.refresh(proposal_for_gemini)

        # Fetch all signature instances for the proposal to build the prompt
        current_signature_instances = db_bg.query(models.SignatureInstance).join(models.Page).join(models.Document).filter(models.Document.project_id == proposal_id).all()

        if not current_signature_instances:
            print(f"Signature Analysis Task: No signatures found in any document for proposal {proposal_id}. Skipping Gemini report.")
            proposal_for_gemini.signature_analysis_status = "completed_no_signatures"
            # Storing JSON for consistency, even if it's a simple message
            proposal_for_gemini.signature_analysis_report_json = {"message": "No signatures were detected in the submitted documents."}
            db_bg.commit()
            db_bg.refresh(proposal_for_gemini)
            return

        for sig_instance in current_signature_instances:
            all_signature_instances_data_for_gemini_prompt.append({
                "signature_database_id": sig_instance.id,
                "document_id": sig_instance.document_id,
                "page_id": sig_instance.page_id,
                "textract_bounding_box": json.loads(sig_instance.bounding_box_json) if isinstance(sig_instance.bounding_box_json, str) else sig_instance.bounding_box_json,
                "textract_confidence": json.loads(sig_instance.textract_response_json).get('Confidence', 'N/A') if isinstance(sig_instance.textract_response_json, str) else 'N/A'
                # Add other relevant parts of textract_response_json if needed
            })

        stakeholders = db_bg.query(models.Stakeholder).filter(models.Stakeholder.project_id == proposal_id).all()
        stakeholder_names = [s.name for s in stakeholders] if stakeholders else []


        # Updated Gemini Prompt for JSON output
        gemini_prompt_text = f"""
        You are an expert in signature analysis for financial and legal documents.
        Proposal ID: {proposal_id}.
        Stakeholders: {', '.join(stakeholder_names) if stakeholder_names else 'Not specified'}.

        Data for detected signatures:
        {json.dumps(all_signature_instances_data_for_gemini_prompt, indent=2)}

        Task: Generate a JSON report with the following structure:
        {{
          "proposal_id": {proposal_id},
          "stakeholders": [{', '.join(f'"{name}"' for name in stakeholder_names)}],
          "overall_summary": {{
            "total_signatures_detected": <integer>,
            "key_observations": ["<observation1>", "<observation2>"],
            "recommendations": ["<recommendation1>"]
          }},
          "signature_details": [
            {{
              "signature_database_id": <integer>,
              "document_id": <integer>,
              "page_id": <integer>,
              "textract_confidence": <float_or_string>,
              "analysis": {{
                "consistency_with_stakeholder_pattern": "<Pending/Not Applicable/Low/Medium/High - if stakeholder known and patterns exist>",
                "potential_anomalies": ["<anomaly1>", "<anomaly2_if_any>"],
                "comments": "<General comments about this specific signature>"
              }}
            }}
            // ... more signature entries
          ]
        }}

        Focus on:
        1. Intra-Stakeholder Consistency (General patterns if stakeholder not linked yet).
        2. Inter-Stakeholder Uniqueness (General patterns if stakeholder not linked yet).
        3. Overall Observations: Anomalies, low-confidence detections.
        Acknowledge that signatures are not yet linked to specific stakeholders.
        Provide concise, factual analysis.
        """
        
        global gemini_model # Ensure it's accessible
        if not gemini_model:
            proposal_for_gemini.signature_analysis_status = "error_gemini_model_not_initialized"
            db_bg.commit()
            raise RuntimeError("Gemini model is not initialized for background task.")

        try:
            print(f"Signature Analysis Task: Calling Gemini for proposal {proposal_id} JSON report...")
            
            ai_response = await gemini_model.generate_content_async(gemini_prompt_text)

            if ai_response.parts:
                generated_text = ai_response.text
                # Clean up potential markdown code block fences for JSON
                if generated_text.strip().startswith("```json"):
                    generated_text = generated_text.strip()[7:]
                    if generated_text.strip().endswith("```"):
                        generated_text = generated_text.strip()[:-3]
                elif generated_text.strip().startswith("```"): # More generic cleanup
                    generated_text = generated_text.strip()[3:]
                    if generated_text.strip().endswith("```"):
                        generated_text = generated_text.strip()[:-3]
                
                generated_text = generated_text.strip()

                try:
                    json_report = json.loads(generated_text)
                    proposal_for_gemini.signature_analysis_report_json = json_report
                    proposal_for_gemini.signature_analysis_status = "completed"
                    print(f"Signature Analysis Task: Successfully generated JSON signature analysis report for proposal {proposal_id}.")
                except json.JSONDecodeError as e_json:
                    print(f"Signature Analysis Task: Gemini returned non-JSON response for proposal {proposal_id}: {e_json}")
                    print(f"Received text: {generated_text}")
                    proposal_for_gemini.signature_analysis_status = "failed_gemini_invalid_json"
                    proposal_for_gemini.signature_analysis_report_json = {
                        "error": "Failed to parse Gemini response as JSON.", 
                        "details": str(e_json),
                        "received_text": generated_text
                    }
            else:
                error_detail = "Gemini did not return expected content for signature report."
                if ai_response.prompt_feedback and ai_response.prompt_feedback.block_reason:
                    error_detail += f" Reason: {ai_response.prompt_feedback.block_reason_message or ai_response.prompt_feedback.block_reason}"
                print(f"Signature Analysis Task: Error generating report for proposal {proposal_id}: {error_detail}")
                proposal_for_gemini.signature_analysis_status = "failed_gemini_report_generation"
                proposal_for_gemini.signature_analysis_report_json = {"error": error_detail}
            
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
        # Ensure db_bg is active and proposal object is available for update
        if db_bg.is_active:
            try:
                # Re-fetch proposal within this exception block to ensure it's current
                proposal_at_error = db_bg.query(models.Project).filter(models.Project.id == proposal_id).first()
                if proposal_at_error and proposal_at_error.signature_analysis_status not in [
                    "completed", "failed_gemini_report_generation", "failed_gemini_exception", 
                    "completed_no_signatures", "failed_gemini_invalid_json" # Added new states
                ]:
                    proposal_at_error.signature_analysis_status = "failed_unknown_task_error"
                    proposal_at_error.signature_analysis_report_json = {"error": f"An unexpected error occurred during analysis: {str(e_task)}"}
                    db_bg.commit()
            except Exception as e_final_commit:
                 print(f"Signature Analysis Task: Error during final error commit for proposal {proposal_id}: {e_final_commit}")
                 db_bg.rollback() 
    finally:
        if db_bg.is_active: # Check if session is active before closing
            db_bg.close()

