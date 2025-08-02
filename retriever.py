from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
import numpy as np
from collections import defaultdict
from tfidf_scorer import TFIDFScorer, create_tfidf_scorer
from cross_encoder_scorer import CrossEncoderScorer, create_cross_encoder_scorer

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, api_key: str = None):
        """
        Initialize the vector store manager.
        
        Args:
            api_key (str): Google API key for embeddings
        """
        # Using HuggingFace embeddings (no API key required)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased for better context
            chunk_overlap=1000,  # Increased to 50% for better context continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Added sentence boundary
        )
        # Initialize scorers
        self.tfidf_scorer = None
        self.cross_encoder_scorer = None
        self.chunks = []
        # Store original document chunks for context expansion
        self.original_chunks = []
        # Map chunk IDs to their positions in the document
        self.chunk_positions = {}
    
    def build_vector_store(self, text: str) -> FAISS:
        """
        Build a FAISS vector store from text with TF-IDF scoring and two-stage retrieval process.
        
        Args:
            text (str): The text to process and store
            
        Returns:
            FAISS: The vector store
        """
        try:
            logger.info("Splitting text into chunks")
            chunks = self.text_splitter.split_text(text)
            self.chunks = chunks
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Store original chunks for context expansion
            self.original_chunks = chunks
            
            # Map chunk positions for two-stage retrieval
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
            
            # Initialize cross-encoder scorer with chunks
            logger.info("Initializing cross-encoder scorer")
            self.cross_encoder_scorer = create_cross_encoder_scorer(chunks)
            
            # Create documents from chunks with position metadata
            docs = [Document(page_content=chunk, metadata={"source": "pdf", "chunk_index": i, "position": i}) for i, chunk in enumerate(chunks)]
            
            logger.info("Creating vector store with embeddings")
            db = FAISS.from_documents(docs, self.embeddings)
            logger.info(f"Vector store created with {len(docs)} documents")
            
            return db
            
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            raise Exception(f"Failed to build vector store: {e}")
    
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
            # Add the current chunk
            expanded_chunks.add(chunk_id)
            
            # Add previous chunks based on window size
            current = chunk_id
            for _ in range(window_size):
                if current in self.chunk_positions and self.chunk_positions[current]['prev'] is not None:
                    prev_id = self.chunk_positions[current]['prev']
                    expanded_chunks.add(prev_id)
                    current = prev_id
            
            # Add next chunks based on window size
            current = chunk_id
            for _ in range(window_size):
                if current in self.chunk_positions and self.chunk_positions[current]['next'] is not None:
                    next_id = self.chunk_positions[current]['next']
                    expanded_chunks.add(next_id)
                    current = next_id
        
        # Create documents from expanded chunks
        expanded_docs = []
        for chunk_id in sorted(expanded_chunks):
            if chunk_id < len(self.chunks):
                doc = Document(
                    page_content=self.chunks[chunk_id],
                    metadata={
                        "source": "pdf", 
                        "chunk_index": chunk_id, 
                        "is_context_expansion": chunk_id not in chunk_ids
                    }
                )
                expanded_docs.append(doc)
        
        return expanded_docs
        
    def get_hybrid_retriever(self, db: FAISS, query: str, top_k: int = 10, 
                            tfidf_weight: float = 0.3, embedding_weight: float = 0.7) -> List[Document]:
        """
        Get documents using hybrid retrieval combining TF-IDF and embedding scores.
        
        Args:
            db (FAISS): The vector store
            query (str): The query text
            top_k (int): Number of top documents to retrieve
            tfidf_weight (float): Weight for TF-IDF scores (0-1)
            embedding_weight (float): Weight for embedding scores (0-1)
            
        Returns:
            List[Document]: List of retrieved documents
        """
        try:
            if self.tfidf_scorer is None:
                logger.warning("TF-IDF scorer not initialized, using embedding-only retrieval")
                retriever = db.as_retriever(search_kwargs={"k": top_k})
                return retriever.get_relevant_documents(query)
            
            # Get TF-IDF scores
            tfidf_scores = self.tfidf_scorer.calculate_similarity_scores(query)
            
            # Get embedding scores
            retriever = db.as_retriever(search_kwargs={"k": len(self.chunks)})
            embedding_docs = retriever.get_relevant_documents(query)
            
            # Create a mapping of chunk index to embedding score
            embedding_scores = {}
            for i, doc in enumerate(embedding_docs):
                chunk_index = doc.metadata.get("chunk_index", i)
                embedding_scores[chunk_index] = 1.0 - (i / len(embedding_docs))  # Normalize scores
            
            # Combine scores
            combined_scores = []
            for chunk_idx, tfidf_score in tfidf_scores:
                embedding_score = embedding_scores.get(chunk_idx, 0.0)
                combined_score = (tfidf_weight * tfidf_score) + (embedding_weight * embedding_score)
                combined_scores.append((chunk_idx, combined_score))
            
            # Sort by combined score and get top-k
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in combined_scores[:top_k]]
            
            # Create documents for top chunks
            top_docs = []
            for idx in top_indices:
                if idx < len(self.chunks):
                    doc = Document(
                        page_content=self.chunks[idx],
                        metadata={"source": "pdf", "chunk_index": idx, "combined_score": combined_scores[idx][1]}
                    )
                    top_docs.append(doc)
            
            # Expand context with surrounding chunks (two-stage retrieval)
            if top_indices:
                expanded_docs = self.expand_context(top_indices, window_size=1)
                # Add expanded docs to results if not already present
                for doc in expanded_docs:
                    if doc.metadata.get("is_context_expansion", False):
                        top_docs.append(doc)
            
            logger.info(f"Retrieved {len(top_docs)} documents using hybrid scoring with context expansion")
            return top_docs
            
        except Exception as e:
            logger.error(f"Failed to get hybrid retriever: {e}")
            # Fallback to embedding-only retrieval
            retriever = db.as_retriever(search_kwargs={"k": top_k})
            return retriever.get_relevant_documents(query)
    
    def get_cross_encoder_retriever(self, db: FAISS, query: str, top_k: int = 10, 
                                   use_reranking: bool = True) -> List[Document]:
        """
        Get documents using cross-encoder for more accurate relevance scoring.
        
        Args:
            db (FAISS): The vector store
            query (str): The query text
            top_k (int): Number of top documents to retrieve
            use_reranking (bool): Whether to use cross-encoder for reranking
            
        Returns:
            List[Document]: List of retrieved documents
        """
        try:
            if self.cross_encoder_scorer is None:
                logger.warning("Cross-encoder scorer not initialized, using embedding-only retrieval")
                retriever = db.as_retriever(search_kwargs={"k": top_k})
                return retriever.get_relevant_documents(query)
            
            if use_reranking:
                # Use cross-encoder for reranking (more accurate but slower)
                logger.info("Using cross-encoder for reranking")
                cross_encoder_scores = self.cross_encoder_scorer.calculate_similarity_scores(query)
                
                # Get top-k documents based on cross-encoder scores
                top_indices = [idx for idx, _ in cross_encoder_scores[:top_k]]
                
                # Create documents for top chunks
                top_docs = []
                for idx in top_indices:
                    if idx < len(self.chunks):
                        doc = Document(
                            page_content=self.chunks[idx],
                            metadata={
                                "source": "pdf", 
                                "chunk_index": idx, 
                                "cross_encoder_score": cross_encoder_scores[idx][1],
                                "retrieval_method": "cross_encoder"
                            }
                        )
                        top_docs.append(doc)
                
                logger.info(f"Retrieved {len(top_docs)} documents using cross-encoder reranking")
                return top_docs
            else:
                # Use cross-encoder for initial retrieval (faster but less accurate)
                logger.info("Using cross-encoder for initial retrieval")
                top_docs = self.cross_encoder_scorer.get_top_documents(query, top_k)
                
                # Convert to Document objects
                documents = []
                for idx, score, text in top_docs:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": "pdf", 
                            "chunk_index": idx, 
                            "cross_encoder_score": score,
                            "retrieval_method": "cross_encoder"
                        }
                    )
                    documents.append(doc)
                
                logger.info(f"Retrieved {len(documents)} documents using cross-encoder")
                return documents
            
        except Exception as e:
            logger.error(f"Failed to get cross-encoder retriever: {e}")
            # Fallback to embedding-only retrieval
            retriever = db.as_retriever(search_kwargs={"k": top_k})
            return retriever.get_relevant_documents(query)
    
    def get_advanced_hybrid_retriever(self, db: FAISS, query: str, top_k: int = 10,
                                     tfidf_weight: float = 0.2, 
                                     embedding_weight: float = 0.3,
                                     cross_encoder_weight: float = 0.5) -> List[Document]:
        """
        Get documents using advanced hybrid retrieval combining TF-IDF, embedding, and cross-encoder scores.
        
        Args:
            db (FAISS): The vector store
            query (str): The query text
            top_k (int): Number of top documents to retrieve
            tfidf_weight (float): Weight for TF-IDF scores (0-1)
            embedding_weight (float): Weight for embedding scores (0-1)
            cross_encoder_weight (float): Weight for cross-encoder scores (0-1)
            
        Returns:
            List[Document]: List of retrieved documents
        """
        try:
            if self.tfidf_scorer is None or self.cross_encoder_scorer is None:
                logger.warning("Scorers not initialized, using embedding-only retrieval")
                retriever = db.as_retriever(search_kwargs={"k": top_k})
                return retriever.get_relevant_documents(query)
            
            # Get TF-IDF scores
            tfidf_scores = self.tfidf_scorer.calculate_similarity_scores(query)
            
            # Get embedding scores
            retriever = db.as_retriever(search_kwargs={"k": len(self.chunks)})
            embedding_docs = retriever.get_relevant_documents(query)
            
            # Create a mapping of chunk index to embedding score
            embedding_scores = {}
            for i, doc in enumerate(embedding_docs):
                chunk_index = doc.metadata.get("chunk_index", i)
                embedding_scores[chunk_index] = 1.0 - (i / len(embedding_docs))  # Normalize scores
            
            # Get cross-encoder scores
            cross_encoder_scores = self.cross_encoder_scorer.calculate_similarity_scores(query)
            
            # Combine all three scores
            combined_scores = []
            for chunk_idx, tfidf_score in tfidf_scores:
                embedding_score = embedding_scores.get(chunk_idx, 0.0)
                cross_encoder_score = next((score for idx, score in cross_encoder_scores if idx == chunk_idx), 0.0)
                
                combined_score = (tfidf_weight * tfidf_score) + \
                               (embedding_weight * embedding_score) + \
                               (cross_encoder_weight * cross_encoder_score)
                combined_scores.append((chunk_idx, combined_score))
            
            # Sort by combined score and get top-k
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in combined_scores[:top_k]]
            
            # Create documents for top chunks
            top_docs = []
            for idx in top_indices:
                if idx < len(self.chunks):
                    doc = Document(
                        page_content=self.chunks[idx],
                        metadata={
                            "source": "pdf", 
                            "chunk_index": idx, 
                            "combined_score": combined_scores[idx][1],
                            "retrieval_method": "advanced_hybrid"
                        }
                    )
                    top_docs.append(doc)
            
            logger.info(f"Retrieved {len(top_docs)} documents using advanced hybrid scoring")
            return top_docs
            
        except Exception as e:
            logger.error(f"Failed to get advanced hybrid retriever: {e}")
            # Fallback to embedding-only retrieval
            retriever = db.as_retriever(search_kwargs={"k": top_k})
            return retriever.get_relevant_documents(query)
    
    def get_tfidf_analysis(self, query: str) -> Dict[str, Any]:
        """
        Get TF-IDF analysis for a query.
        
        Args:
            query (str): The query text
            
        Returns:
            Dict[str, Any]: TF-IDF analysis results
        """
        try:
            if self.tfidf_scorer is None:
                return {"error": "TF-IDF scorer not initialized"}
            
            # Get important terms for the query
            important_terms = self.tfidf_scorer.get_important_terms(query, top_n=10)
            
            # Get top documents with TF-IDF scores
            top_docs = self.tfidf_scorer.get_top_documents(query, top_k=5)
            
            return {
                "important_terms": important_terms,
                "top_documents": [(idx, score, text[:200] + "...") for idx, score, text in top_docs],
                "query_processed": self.tfidf_scorer.preprocess_text(query)
            }
            
        except Exception as e:
            logger.error(f"Failed to get TF-IDF analysis: {e}")
            return {"error": str(e)}
    
    def get_cross_encoder_analysis(self, query: str) -> Dict[str, Any]:
        """
        Get cross-encoder analysis for a query.
        
        Args:
            query (str): The query text
            
        Returns:
            Dict[str, Any]: Cross-encoder analysis results
        """
        try:
            if self.cross_encoder_scorer is None:
                return {"error": "Cross-encoder scorer not initialized"}
            
            # Get relevance analysis from cross-encoder
            relevance_analysis = self.cross_encoder_scorer.get_relevance_analysis(query, top_k=5)
            
            return {
                "relevance_analysis": relevance_analysis,
                "query_processed": self.cross_encoder_scorer.preprocess_text(query)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cross-encoder analysis: {e}")
            return {"error": str(e)}
    
    def save_vector_store(self, db: FAISS, path: str = "vector_store"):
        """
        Save the vector store to disk.
        
        Args:
            db (FAISS): The vector store to save
            path (str): Path to save the vector store
        """
        try:
            logger.info(f"Saving vector store to {path}")
            db.save_local(path)
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise Exception(f"Failed to save vector store: {e}")
    
    def load_vector_store(self, path: str = "vector_store") -> FAISS:
        """
        Load the vector store from disk.
        
        Args:
            path (str): Path to load the vector store from
            
        Returns:
            FAISS: The loaded vector store
        """
        try:
            logger.info(f"Loading vector store from {path}")
            db = FAISS.load_local(path, self.embeddings)
            logger.info("Vector store loaded successfully")
            return db
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise Exception(f"Failed to load vector store: {e}")

def build_vector_store(text: str, api_key: str = None) -> FAISS:
    """
    Convenience function to build a vector store from text.
    
    Args:
        text (str): The text to process
        api_key (str): Google API key for embeddings
        
    Returns:
        FAISS: The vector store
    """
    manager = VectorStoreManager(api_key)
    return manager.build_vector_store(text)