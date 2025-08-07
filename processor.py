import pdfplumber
import requests
from io import BytesIO
from typing import Optional
import logging
import urllib3

# Suppress SSL warnings for development/deployment
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

def extract_text_from_url(pdf_url: str) -> str:
    """
    Extract text from a PDF file given its URL.
    
    Args:
        pdf_url (str): URL of the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        logger.info(f"Downloading PDF from: {pdf_url}")
        # Disable SSL certificate verification to handle expired certificates
        response = requests.get(pdf_url, timeout=30, verify=False)
        response.raise_for_status()
        
        logger.info("Extracting text from PDF")
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            text_parts = []
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                logger.debug(f"Processed page {page_num + 1}")
            
            full_text = "\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
    except requests.RequestException as e:
        logger.error(f"Failed to download PDF: {e}")
        raise Exception(f"Failed to download PDF from {pdf_url}: {e}")
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise Exception(f"Failed to extract text from PDF: {e}")

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a local PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        logger.info(f"Extracting text from local file: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            text_parts = []
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                logger.debug(f"Processed page {page_num + 1}")
            
            full_text = "\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
    except Exception as e:
        logger.error(f"Failed to extract text from PDF file: {e}")
        raise Exception(f"Failed to extract text from PDF file {file_path}: {e}")