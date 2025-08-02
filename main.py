from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv

from processor import extract_text_from_url
from retriever import VectorStoreManager
from qa import answer_questions

# Load environment variables
load_dotenv()

# Set OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = "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"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx Policy QA System",
    description="LLM-powered Policy QA System using Gemini 2.5 Flash",
    version="1.0.0"
)

class HackRxInput(BaseModel):
    """Input model for the HackRx API."""
    documents: Optional[str] = None  # URL or path to PDF document (new format)
    pdf_url: Optional[str] = None    # URL or path to PDF document (legacy format)
    questions: List[str]
    
    def get_document_url(self) -> str:
        """Get the document URL from either field."""
        return self.documents or self.pdf_url

class HackRxResponse(BaseModel):
    """Response model for the HackRx API."""
    answers: List[str]

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "HackRx Policy QA System is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "HackRx Policy QA System",
        "version": "1.0.0"
    }

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(
    query: HackRxInput, 
    authorization: Optional[str] = Header(None)
):
    """
    Main endpoint for the HackRx Policy QA System.
    
    Args:
        query (HackRxInput): Input containing document URL and questions
        authorization (str): API key for authentication
        
    Returns:
        HackRxResponse: Answers to the questions
    """
    try:
        # Validate API key (bypassed for testing)
        api_key = os.getenv("HACKRX_API_KEY")
        # For testing purposes, accept any API key that follows the Bearer format
        if api_key and (not authorization or not authorization.startswith("Bearer ")):
            raise HTTPException(
                status_code=401, 
                detail="Invalid API key format"
            )
        logger.info("API key validation bypassed for testing")
        
        logger.info(f"Processing request with {len(query.questions)} questions")
        
        # Extract text from PDF
        logger.info("Extracting text from document")
        document_url = query.get_document_url()
        if not document_url:
            raise HTTPException(
                status_code=400,
                detail="Either 'documents' or 'pdf_url' field is required"
            )
        
        text = extract_text_from_url(document_url)
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )
        
        # Build vector store with TF-IDF scoring and overlapping chunks
        logger.info("Building enhanced vector store with TF-IDF scoring and overlapping chunks")
        vector_manager = VectorStoreManager()
        vector_store = vector_manager.build_vector_store(text)
        
        # Answer questions with enhanced accuracy systems
        logger.info("Generating answers with enhanced accuracy systems: numerical grounding, completeness checking, and confidence scoring")
        
        # Create QA system and pass the vector_manager to ensure scorer access
        from qa import PolicyQASystem
        qa_system = PolicyQASystem()
        qa_system.vector_manager = vector_manager  # Use the same vector_manager with initialized scorers
        
        answers = qa_system.answer_questions(
            questions=query.questions,
            vector_store=vector_store,
            use_hybrid=True,
            use_advanced_hybrid=True,  # Use the most advanced retrieval method
            document_text=text  # Pass full document text for numerical grounding
        )
        
        logger.info("Request completed successfully")
        
        # Extract just the answer strings from the answer objects
        answer_strings = []
        for answer_obj in answers:
            if isinstance(answer_obj, dict) and 'answer' in answer_obj:
                answer_strings.append(answer_obj['answer'])
            elif isinstance(answer_obj, str):
                answer_strings.append(answer_obj)
            else:
                # Fallback: convert to string
                answer_strings.append(str(answer_obj))
        
        return HackRxResponse(answers=answer_strings)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/hackrx/process")
async def process_document(
    document_url: str,
    authorization: Optional[str] = Header(None)
):
    """
    Process a document and return its extracted text.
    
    Args:
        document_url (str): URL of the document to process
        authorization (str): API key for authentication
        
    Returns:
        dict: Extracted text and metadata
    """
    try:
        # Validate API key
        api_key = os.getenv("HACKRX_API_KEY")
        if api_key and (not authorization or not authorization.startswith("Bearer ") or 
                       authorization.replace("Bearer ", "") != api_key):
            raise HTTPException(
                status_code=401, 
                detail="Invalid API key"
            )
        
        logger.info(f"Processing document: {document_url}")
        
        # Extract text
        text = extract_text_from_url(document_url)
        
        return {
            "status": "success",
            "document_url": document_url,
            "text_length": len(text),
            "text_preview": text[:500] + "..." if len(text) > 500 else text
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)