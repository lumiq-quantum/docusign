from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, LargeBinary, Boolean, JSON
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, field_validator # Updated import
from datetime import datetime
import os
import json
from typing import Optional, List # Added List

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5433/mydatabase")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    application_number = Column(String, unique=True, index=True, nullable=True) # Added
    chat_session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    signature_analysis_report_html = Column(Text, nullable=True) # Added
    signature_analysis_status = Column(String, nullable=True, default="pending") # Added

    # Relationships
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    stakeholders = relationship("Stakeholder", back_populates="project", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    pdf_file = Column(LargeBinary, nullable=False)
    total_pages = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    chat_session_id = Column(String, nullable=True) # Added

    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    project = relationship("Project", back_populates="documents")
    
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    signatures = relationship("SignatureInstance", back_populates="document", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"
    id = Column(Integer, primary_key=True, index=True)
    page_number = Column(Integer, nullable=False)
    text_content = Column(Text, nullable=True)
    generated_form_html = Column(Text, nullable=True)
    
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    document = relationship("Document", back_populates="pages")
    
    signatures = relationship("SignatureInstance", back_populates="page", cascade="all, delete-orphan")

class Stakeholder(Base):
    __tablename__ = "stakeholders"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False) # e.g., "Applicant 1", "Guarantor A"
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    
    project = relationship("Project", back_populates="stakeholders")
    signatures = relationship("SignatureInstance", back_populates="stakeholder")

class SignatureInstance(Base):
    __tablename__ = "signature_instances"
    id = Column(Integer, primary_key=True, index=True)
    
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False) # For easier querying
    stakeholder_id = Column(Integer, ForeignKey("stakeholders.id"), nullable=True) # Can be null initially
    
    bounding_box_json = Column(JSON, nullable=True) # Added
    textract_response_json = Column(JSON, nullable=True) # Added
    cropped_signature_image = Column(LargeBinary, nullable=True) # Added
    
    # Details from AI - consider storing bounding box as JSON
    # e.g., bounding_box = Column(JSON) # {'x1': % , 'y1': %, 'x2': %, 'y2': %}
    ai_signature_id = Column(String, nullable=True) # An ID given by the AI if it groups signatures
    ai_confidence = Column(String, nullable=True) # e.g. "high", "medium", "low", "red_flag"
    # signature_image_embedding = Column(LargeBinary, nullable=True) # For advanced comparison

    # Analysis results
    is_consistent_with_stakeholder_group = Column(Boolean, nullable=True)
    is_unique_among_stakeholders = Column(Boolean, nullable=True)
    analysis_notes = Column(Text, nullable=True)

    page = relationship("Page", back_populates="signatures")
    document = relationship("Document", back_populates="signatures")
    stakeholder = relationship("Stakeholder", back_populates="signatures")


# Pydantic models (Schemas) for API requests and responses

# --- SignatureInstance Schemas ---
class SignatureInstanceBase(BaseModel):
    page_id: int
    document_id: int
    stakeholder_id: Optional[int] = None
    ai_signature_id: Optional[str] = None
    ai_confidence: Optional[str] = None
    bounding_box_json: Optional[dict] = None 
    textract_response_json: Optional[dict] = None
    # cropped_signature_image will not be directly in base/create, usually handled differently if sent via API

class SignatureInstanceCreate(SignatureInstanceBase):
    pass

class SignatureInstanceResponse(SignatureInstanceBase):
    id: int
    is_consistent_with_stakeholder_group: Optional[bool] = None
    is_unique_among_stakeholders: Optional[bool] = None
    analysis_notes: Optional[str] = None
    # cropped_signature_image: Optional[bytes] = None # Decide if this should be sent in response; can be large

    @field_validator('bounding_box_json', 'textract_response_json', mode='before')
    @classmethod
    def parse_json_fields(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Handle error or return value as is, depending on desired behavior
                return None # Or raise an error, or return an empty dict
        return value

    class Config:
        from_attributes = True

# --- Stakeholder Schemas ---
class StakeholderBase(BaseModel):
    name: str
    project_id: int

class StakeholderCreate(StakeholderBase):
    pass

class StakeholderResponse(StakeholderBase):
    id: int
    signatures: List[SignatureInstanceResponse] = []
    class Config:
        from_attributes = True

# --- Page Schemas ---
class PageBase(BaseModel):
    page_number: int
    text_content: Optional[str] = None
    generated_form_html: Optional[str] = None

class PageCreate(PageBase):
    document_id: int

class PageResponse(PageBase):
    id: int
    document_id: int
    signatures: List[SignatureInstanceResponse] = []
    class Config:
        from_attributes = True

# --- Document Schemas ---
class DocumentBase(BaseModel):
    file_name: str
    total_pages: int
    chat_session_id: Optional[str] = None # Added

class DocumentCreate(DocumentBase):
    # pdf_file will be handled via UploadFile in endpoint
    project_id: int

class DocumentResponse(DocumentBase):
    id: int
    project_id: int
    created_at: datetime
    pages: List[PageResponse] = []
    # Optionally include signatures at document level if needed for quick overview
    # signatures: List[SignatureInstanceResponse] = [] 
    class Config:
        from_attributes = True

# --- Project Schemas ---
class ProjectBase(BaseModel):
    name: str
    chat_session_id: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: int
    application_number: Optional[str] = None # Added
    created_at: datetime
    signature_analysis_report_html: Optional[str] = None # Added
    signature_analysis_status: Optional[str] = None # Added
    documents: List[DocumentResponse] = []
    stakeholders: List[StakeholderResponse] = []
    class Config:
        from_attributes = True

# --- General Utility Schemas ---
class GeneratedHtmlResponse(BaseModel):
    html_content: str
    class Config:
        from_attributes = True

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables if they don't exist (for development without Alembic)
# In production, use Alembic migrations
# Base.metadata.create_all(bind=engine)
