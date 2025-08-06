from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
import numpy as np
from collections import defaultdict
from tfidf_scorer import TFIDFScorer, create_tfidf_scorer
from pinecone import Pinecone
import uuid

logger = logging.getLogger(__name__)

class PineconeVectorStoreManager:
    def __init__(self, api_key: str = None, index_name: str = "hackrx"):
        """
        Initialize the Pinecone vector store manager.
        
        Args:
            api_key (str): Pinecone API key
            index_name (str): Name of the Pinecone index
        """
        self.pinecone_api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable.")
        
        self.index_name = index_name
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Using HuggingFace embeddings (no API key required)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased for better context
            chunk_overlap=1000,  # Increased to 50% for better context continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Added sentence boundary
        )
        
        # Initialize TF-IDF scorer only
        self.tfidf_scorer = None
        self.chunks = []
        # Store original document chunks for context expansion
        self.original_chunks = []
        # Map chunk IDs to their positions in the document
        self.chunk_positions = {}
        
        # Initialize or connect to Pinecone index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Pinecone index '{self.index_name}' not found. Please create it manually in the Pinecone console.")
                raise Exception(f"Pinecone index '{self.index_name}' not found. Please create it manually in the Pinecone console.")
            else:
                logger.info(f"Connecting to existing Pinecone index: {self.index_name}")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise Exception(f"Failed to initialize Pinecone index: {e}")
    
    def build_vector_store(self, text: str) -> 'PineconeIndex':
        """
        Build a Pinecone vector store from text with TF-IDF scoring.
        
        Args:
            text (str): The text to process and store
            
        Returns:
            PineconeIndex: The Pinecone index
        """
        try:
            logger.info("Splitting text into chunks")
            chunks = self.text_splitter.split_text(text)
            self.chunks = chunks
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Store original chunks for context expansion
            self.original_chunks = chunks
            
            # Map chunk positions for context expansion
            for i, chunk in enumerate(chunks):
                self.chunk_positions[i] = {
                    'index': i,
                    'content': chunk,
                    'prev': i-1 if i > 0 else None,
                    'next': i+1 if i < len(chunks)-1 else None
                }
            
            # Initialize TF-IDF scorer with chunks
            logger.info("Initializing TF-IDF scorer")
            self.tfidf_scorer = create_tfidf_scorer(chunks)
            
            # Generate embeddings and upload to Pinecone
            logger.info("Generating embeddings and uploading to Pinecone")
            self._upload_chunks_to_pinecone(chunks)
            
            logger.info(f"Vector store created with {len(chunks)} documents in Pinecone")
            return self.index
            
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            raise Exception(f"Failed to build vector store: {e}")
    
    def _upload_chunks_to_pinecone(self, chunks: List[str]):
        """Upload chunks to Pinecone with embeddings."""
        try:
            # Clear existing data (optional - remove if you want to append)
            try:
                self.index.delete(delete_all=True)
                logger.info("Cleared existing data from Pinecone index")
            except Exception as e:
                logger.info(f"Could not clear existing data (this is normal for empty index): {e}")
            
            # Generate embeddings for all chunks
            embeddings = self.embeddings.embed_documents(chunks)
            
            # Prepare vectors for upload
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"chunk_{i}"
                metadata = {
                    "chunk_index": i,
                    "content": chunk,
                    "source": "pdf"
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upload in batches (Pinecone has limits on batch size)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace="")
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to upload chunks to Pinecone: {e}")
            raise Exception(f"Failed to upload chunks to Pinecone: {e}")
    
    def expand_context(self, chunk_ids: List[int], window_size: int = 1) -> List[Document]:
        """
        Expand context by including surrounding chunks for each retrieved chunk.
        
        Args:
            chunk_ids (List[int]): List of chunk IDs to expand context for
            window_size (int): Number of chunks to include on each side
            
        Returns:
            List[Document]: Expanded list of documents with surrounding context
        """
        expanded_chunks = set()
        
        for chunk_id in chunk_ids:
            # Add the original chunk
            expanded_chunks.add(chunk_id)
            
            # Add surrounding chunks
            for offset in range(-window_size, window_size + 1):
                neighbor_id = chunk_id + offset
                if 0 <= neighbor_id < len(self.original_chunks):
                    expanded_chunks.add(neighbor_id)
        
        # Convert to documents
        expanded_docs = []
        for chunk_id in sorted(expanded_chunks):
            chunk_content = self.original_chunks[chunk_id]
            doc = Document(
                page_content=chunk_content,
                metadata={
                    "source": "pdf",
                    "chunk_index": chunk_id,
                    "position": chunk_id,
                    "expanded": True
                }
            )
            expanded_docs.append(doc)
        
        return expanded_docs
    
    def get_hybrid_retriever(self, query: str, top_k: int = 8, 
                           tfidf_weight: float = 0.4, embedding_weight: float = 0.6) -> List[Document]:
        """
        Get documents using hybrid retrieval combining TF-IDF and Pinecone similarity.
        
        Args:
            query (str): The query
            top_k (int): Number of documents to retrieve
            tfidf_weight (float): Weight for TF-IDF scores
            embedding_weight (float): Weight for embedding similarity scores
            
        Returns:
            List[Document]: Retrieved documents
        """
        try:
            if not self.tfidf_scorer:
                logger.warning("TF-IDF scorer not initialized, using embedding-only retrieval")
                return self._get_pinecone_only_retrieval(query, top_k)
            
            # Get TF-IDF scores
            tfidf_scores = self.tfidf_scorer.calculate_similarity_scores(query)
            
            # Get Pinecone similarity scores
            pinecone_results = self._get_pinecone_similarity(query, top_k * 2)
            
            # Create a mapping of chunk content to Pinecone score
            pinecone_scores = {}
            for result in pinecone_results:
                chunk_id = result.metadata.get("chunk_index", 0)
                pinecone_scores[chunk_id] = result.score
            
            # Combine scores
            combined_scores = []
            for chunk_id, tfidf_score in tfidf_scores:
                if chunk_id < len(self.chunks):
                    chunk_content = self.chunks[chunk_id]
                    pinecone_score = pinecone_scores.get(chunk_id, 0.0)
                    
                    # Combine scores with weights
                    combined_score = (tfidf_weight * tfidf_score) + (embedding_weight * pinecone_score)
                    
                    combined_scores.append((chunk_id, combined_score, chunk_content))
            
            # Sort by combined score and get top_k
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = combined_scores[:top_k]
            
            # Convert to documents
            documents = []
            for chunk_id, combined_score, chunk_content in top_results:
                doc = Document(
                    page_content=chunk_content,
                    metadata={
                        "source": "pdf",
                        "chunk_index": chunk_id,
                        "position": chunk_id,
                        "combined_score": combined_score,
                        "tfidf_score": tfidf_scores[chunk_id][1] if chunk_id < len(tfidf_scores) else 0.0,
                        "pinecone_score": pinecone_scores.get(chunk_id, 0.0)
                    }
                )
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents using hybrid retrieval")
            return documents
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            # Fallback to Pinecone-only retrieval
            return self._get_pinecone_only_retrieval(query, top_k)
    
    def _get_pinecone_similarity(self, query: str, top_k: int) -> List:
        """Get similarity search results from Pinecone."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=""
            )
            
            return results.matches
            
        except Exception as e:
            logger.error(f"Error in Pinecone similarity search: {e}")
            return []
    
    def _get_pinecone_only_retrieval(self, query: str, top_k: int) -> List[Document]:
        """Get documents using only Pinecone similarity search."""
        try:
            pinecone_results = self._get_pinecone_similarity(query, top_k)
            
            documents = []
            for result in pinecone_results:
                chunk_content = result.metadata.get("content", "")
                chunk_id = result.metadata.get("chunk_index", 0)
                
                doc = Document(
                    page_content=chunk_content,
                    metadata={
                        "source": "pdf",
                        "chunk_index": chunk_id,
                        "position": chunk_id,
                        "pinecone_score": result.score
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in Pinecone-only retrieval: {e}")
            return []
    
    def get_tfidf_analysis(self, query: str) -> Dict[str, Any]:
        """
        Get TF-IDF analysis for debugging.
        
        Args:
            query (str): The query to analyze
            
        Returns:
            Dict[str, Any]: TF-IDF analysis results
        """
        try:
            if not self.tfidf_scorer:
                return {"error": "TF-IDF scorer not initialized"}
            
            scores = self.tfidf_scorer.calculate_similarity_scores(query)
            top_scores = scores[:5]  # Top 5 scores
            
            return {
                "query": query,
                "top_scores": [(i, score) for i, score in top_scores],
                "total_chunks": len(self.chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in TF-IDF analysis: {e}")
            return {"error": str(e)}
    
    def delete_index(self):
        """Delete the Pinecone index."""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index: {e}")

def build_pinecone_vector_store(text: str, api_key: str = None, index_name: str = "hackrx") -> PineconeVectorStoreManager:
    """
    Convenience function to build a Pinecone vector store.
    
    Args:
        text (str): The text to process
        api_key (str): Pinecone API key
        index_name (str): Name of the Pinecone index
        
    Returns:
        PineconeVectorStoreManager: The vector store manager
    """
    manager = PineconeVectorStoreManager(api_key, index_name)
    return manager.build_vector_store(text) 