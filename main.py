from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv

from processor import extract_text_from_url
from pinecone_retriever import PineconeVectorStoreManager
from qa import answer_questions

# Load environment variables from .env file
load_dotenv()

# Set API keys - use environment variables only
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Validate required API keys
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required. Please set it in your .env file or environment.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required. Please set it in your .env file or environment.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx Policy QA System",
    description="LLM-powered Policy QA System using OpenRouter",
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
@app.head("/")
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
        # Use authorization header as OpenRouter API key if provided
        if authorization and authorization.startswith("Bearer "):
            openrouter_api_key = authorization.replace("Bearer ", "")
            logger.info("Using OpenRouter API key from authorization header")
        else:
            openrouter_api_key = OPENROUTER_API_KEY
            logger.info("Using default OpenRouter API key")
        
        # Validate HackRx API key (bypassed for testing)
        hackrx_api_key = os.getenv("HACKRX_API_KEY")
        if hackrx_api_key and (not authorization or not authorization.startswith("Bearer ")):
            raise HTTPException(
                status_code=401, 
                detail="Invalid API key format"
            )
        logger.info("HackRx API key validation bypassed for testing")
        
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
        logger.info("Building vector store with Pinecone and overlapping chunks")
        vector_manager = PineconeVectorStoreManager(api_key=PINECONE_API_KEY)
        vector_store = vector_manager.build_vector_store(text)
        
        # Answer questions with simplified system for faster response
        logger.info("Generating answers with simplified system for faster response")
        
        # Create QA system and reuse the existing vector manager
        from qa import PolicyQASystem
        qa_system = PolicyQASystem(api_key=openrouter_api_key)
        qa_system.vector_manager = vector_manager  # Reuse the existing vector manager with working Pinecone connection
        
        answers = qa_system.answer_questions(
            questions=query.questions,
            vector_store=vector_store,
            use_hybrid=True,
            document_text=text
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
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)