import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple, Any
import logging
import torch
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
    
class CrossEncoderScorer:
    """
    Cross-encoder based scoring system for improved document relevance.
    Cross-encoders provide more accurate relevance scores by jointly encoding
    query-document pairs rather than encoding them separately.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder scorer.
        
        Args:
            model_name (str): Name of the cross-encoder model to use
        """
        try:
            logger.info(f"Loading cross-encoder model: {model_name}")
            self.cross_encoder = CrossEncoder(model_name, max_length=512)
            self.scaler = MinMaxScaler()
            self.documents = []
            self.is_fitted = False
            
            logger.info("Cross-encoder model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise Exception(f"Failed to load cross-encoder model: {e}")
    
    def preprocess_text(self, text: str, max_length: int = 500) -> str:
        """
        Preprocess text for cross-encoder analysis.
        
        Args:
            text (str): Raw text
            max_length (int): Maximum length for text truncation
            
        Returns:
            str: Preprocessed text
        """
        # Clean and truncate text
        text = text.strip()
        if len(text) > max_length:
            text = text[:max_length] + "..."
        return text
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit the cross-encoder scorer with documents.
        
        Args:
            documents (List[str]): List of document texts
        """
        try:
            logger.info(f"Fitting cross-encoder with {len(documents)} documents")
            
            # Preprocess documents
            self.documents = [self.preprocess_text(doc) for doc in documents]
            self.is_fitted = True
            
            logger.info("Cross-encoder fitted successfully")
            
        except Exception as e:
            logger.error(f"Failed to fit cross-encoder: {e}")
            raise Exception(f"Failed to fit cross-encoder: {e}")
    
    def calculate_similarity_scores(self, query: str, batch_size: int = 32) -> List[Tuple[int, float]]:
        """
        Calculate similarity scores between query and all documents using cross-encoder.
        
        Args:
            query (str): Query text
            batch_size (int): Batch size for processing
            
        Returns:
            List[Tuple[int, float]]: List of (document_index, similarity_score) tuples
        """
        try:
            if not self.is_fitted:
                raise ValueError("Cross-encoder not fitted. Call fit first.")
            
            processed_query = self.preprocess_text(query)
            
            # Create query-document pairs
            pairs = [(processed_query, doc) for doc in self.documents]
            
            # Calculate scores in batches
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.cross_encoder.predict(batch_pairs)
                scores.extend(batch_scores)
            
            # Normalize scores to 0-1 range
            scores = np.array(scores).reshape(-1, 1)
            normalized_scores = self.scaler.fit_transform(scores).flatten()
            
            # Create list of (index, score) tuples and sort by score
            score_tuples = [(i, float(score)) for i, score in enumerate(normalized_scores)]
            score_tuples.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Calculated cross-encoder similarity scores for query: {query[:50]}...")
            
            return score_tuples
            
        except Exception as e:
            logger.error(f"Failed to calculate cross-encoder similarity scores: {e}")
            raise Exception(f"Failed to calculate cross-encoder similarity scores: {e}")
    
    def get_top_documents(self, query: str, top_k: int = 10, batch_size: int = 32) -> List[Tuple[int, float, str]]:
        """
        Get top-k most similar documents for a query using cross-encoder.
        
        Args:
            query (str): Query text
            top_k (int): Number of top documents to return
            batch_size (int): Batch size for processing
            
        Returns:
            List[Tuple[int, float, str]]: List of (index, score, document_text) tuples
        """
        try:
            scores = self.calculate_similarity_scores(query, batch_size)
            
            # Get top-k documents
            top_docs = []
            for i, score in scores[:top_k]:
                if score > 0:  # Only include documents with positive similarity
                    doc_text = self.documents[i] if i < len(self.documents) else ""
                    top_docs.append((i, score, doc_text))
            
            logger.info(f"Retrieved {len(top_docs)} top documents using cross-encoder")
            
            return top_docs
            
        except Exception as e:
            logger.error(f"Failed to get top documents with cross-encoder: {e}")
            raise Exception(f"Failed to get top documents with cross-encoder: {e}")
    
    def calculate_pairwise_scores(self, query: str, documents: List[str], batch_size: int = 32) -> List[float]:
        """
        Calculate pairwise similarity scores between query and specific documents.
        
        Args:
            query (str): Query text
            documents (List[str]): List of documents to score
            batch_size (int): Batch size for processing
            
        Returns:
            List[float]: List of similarity scores
        """
        try:
            processed_query = self.preprocess_text(query)
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            
            # Create query-document pairs
            pairs = [(processed_query, doc) for doc in processed_docs]
            
            # Calculate scores in batches
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.cross_encoder.predict(batch_pairs)
                scores.extend(batch_scores)
            
            # Normalize scores
            scores = np.array(scores).reshape(-1, 1)
            normalized_scores = self.scaler.fit_transform(scores).flatten()
            
            return normalized_scores.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate pairwise scores: {e}")
            raise Exception(f"Failed to calculate pairwise scores: {e}")
    
    def get_relevance_analysis(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Get detailed relevance analysis for a query.
        
        Args:
            query (str): Query text
            top_k (int): Number of top documents to analyze
            
        Returns:
            Dict[str, Any]: Relevance analysis results
        """
        try:
            top_docs = self.get_top_documents(query, top_k)
            
            # Calculate confidence based on score distribution
            scores = [score for _, score, _ in top_docs]
            avg_score = np.mean(scores) if scores else 0.0
            score_std = np.std(scores) if len(scores) > 1 else 0.0
            
            confidence = "high" if avg_score > 0.7 else "medium" if avg_score > 0.4 else "low"
            
            return {
                "top_documents": [(idx, score, text[:200] + "...") for idx, score, text in top_docs],
                "average_score": float(avg_score),
                "score_std": float(score_std),
                "confidence": confidence,
                "query_processed": self.preprocess_text(query)
            }
            
        except Exception as e:
            logger.error(f"Failed to get relevance analysis: {e}")
            return {"error": str(e)}

def create_cross_encoder_scorer(documents: List[str], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderScorer:
    """
    Convenience function to create and fit a cross-encoder scorer.
    
    Args:
        documents (List[str]): List of document texts
        model_name (str): Name of the cross-encoder model to use
        
    Returns:
        CrossEncoderScorer: Fitted cross-encoder scorer
    """
    scorer = CrossEncoderScorer(model_name)
    scorer.fit(documents)
    return scorer 