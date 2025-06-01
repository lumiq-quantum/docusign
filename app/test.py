from .models import (
    SessionLocal, engine, get_db,
    Project, ProjectCreate, ProjectResponse,
    Document, DocumentCreate, DocumentResponse,
    Page, # PageCreate and PageResponse removed
    Stakeholder, 
    SignatureInstance, SignatureInstanceCreate, SignatureInstanceResponse,
    GeneratedHtmlResponse
)


from google import genai as thegenai
import io
import httpx


db_bg = SessionLocal()

proposal_id = 10
proposal = db_bg.query(Project).filter(Project.id == proposal_id).first()

document = db_bg.query(Document).filter(Document.project_id == proposal_id).all()

docUrls = []
docData = []

for doc in document:
    docUrls.append("http://localhost:8000/proposals/"+str(doc.project_id)+"/documents/"+str(doc.id)+"/pdf")


print(docUrls)
client = thegenai.Client()


for doc_url in docUrls:
    print(f"Document URL: {doc_url}")
    doc_data = io.BytesIO(httpx.get(doc_url).content)
    doc_pdf = client.files.upload(
        file=doc_data,
        config=dict(mime_type='application/pdf')
    )
    docData.append(doc_pdf)


prompt = "How many documents you know about and how many pages are there in each document? Please provide the information in a JSON format with keys as 'document_name' and 'page_count'."
# print(f"Prompt: {prompt}")

response = client.models.generate_content(
  model="gemini-2.5-flash-preview-05-20",
  contents=docData+[prompt])
print(response.text)