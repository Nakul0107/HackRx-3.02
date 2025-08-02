import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

class TFIDFScorer:
    """
    TF-IDF based scoring system for improved document relevance.
    """
    
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the TF-IDF scorer.
        
        Args:
            max_features (int): Maximum number of features to consider
            ngram_range (tuple): Range of n-grams to consider (unigrams and bigrams)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9_]*\b'
        )
        self.tfidf_matrix = None
        self.feature_names = None
        self.documents = []
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TF-IDF analysis.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
        return text.lower()
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit the TF-IDF vectorizer and transform documents.
        
        Args:
            documents (List[str]): List of document texts
            
        Returns:
            np.ndarray: TF-IDF matrix
        """
        try:
            logger.info(f"Fitting TF-IDF vectorizer on {len(documents)} documents")
            
            # Preprocess documents
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            self.documents = processed_docs
            
            # Fit and transform
            self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            logger.info(f"Number of features: {len(self.feature_names)}")
            
            return self.tfidf_matrix
            
        except Exception as e:
            logger.error(f"Failed to fit TF-IDF vectorizer: {e}")
            raise Exception(f"Failed to fit TF-IDF vectorizer: {e}")
    
    def transform_query(self, query: str) -> np.ndarray:
        """
        Transform a query using the fitted vectorizer.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Query TF-IDF vector
        """
        try:
            processed_query = self.preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            return query_vector
            
        except Exception as e:
            logger.error(f"Failed to transform query: {e}")
            raise Exception(f"Failed to transform query: {e}")
    
    def calculate_similarity_scores(self, query: str) -> List[Tuple[int, float]]:
        """
        Calculate similarity scores between query and all documents.
        
        Args:
            query (str): Query text
            
        Returns:
            List[Tuple[int, float]]: List of (document_index, similarity_score) tuples
        """
        try:
            if self.tfidf_matrix is None:
                raise ValueError("TF-IDF matrix not fitted. Call fit_transform first.")
            
            # Transform query
            query_vector = self.transform_query(query)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Create list of (index, score) tuples and sort by score
            scores = [(i, float(score)) for i, score in enumerate(similarities)]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Calculated similarity scores for query: {query[:50]}...")
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity scores: {e}")
            raise Exception(f"Failed to calculate similarity scores: {e}")
    
    def get_top_documents(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Get top-k most similar documents for a query.
        
        Args:
            query (str): Query text
            top_k (int): Number of top documents to return
            
        Returns:
            List[Tuple[int, float, str]]: List of (index, score, document_text) tuples
        """
        try:
            scores = self.calculate_similarity_scores(query)
            
            # Get top-k documents
            top_docs = []
            for i, score in scores[:top_k]:
                if score > 0:  # Only include documents with positive similarity
                    doc_text = self.documents[i] if i < len(self.documents) else ""
                    top_docs.append((i, score, doc_text))
            
            logger.info(f"Retrieved {len(top_docs)} top documents for query")
            
            return top_docs
            
        except Exception as e:
            logger.error(f"Failed to get top documents: {e}")
            raise Exception(f"Failed to get top documents: {e}")
    
    def get_important_terms(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most important terms for a query based on TF-IDF scores.
        
        Args:
            query (str): Query text
            top_n (int): Number of top terms to return
            
        Returns:
            List[Tuple[str, float]]: List of (term, tfidf_score) tuples
        """
        try:
            query_vector = self.transform_query(query)
            
            # Get feature names and their scores
            feature_scores = []
            for i, score in enumerate(query_vector.toarray()[0]):
                if score > 0:
                    feature_scores.append((self.feature_names[i], float(score)))
            
            # Sort by score and return top terms
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            return feature_scores[:top_n]
            
        except Exception as e:
            logger.error(f"Failed to get important terms: {e}")
            raise Exception(f"Failed to get important terms: {e}")
    
    def get_document_keywords(self, doc_index: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most important keywords for a specific document.
        
        Args:
            doc_index (int): Index of the document
            top_n (int): Number of top keywords to return
            
        Returns:
            List[Tuple[str, float]]: List of (keyword, tfidf_score) tuples
        """
        try:
            if doc_index >= self.tfidf_matrix.shape[0]:
                raise ValueError(f"Document index {doc_index} out of range")
            
            # Get document vector
            doc_vector = self.tfidf_matrix[doc_index].toarray()[0]
            
            # Get feature names and their scores
            feature_scores = []
            for i, score in enumerate(doc_vector):
                if score > 0:
                    feature_scores.append((self.feature_names[i], float(score)))
            
            # Sort by score and return top keywords
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            return feature_scores[:top_n]
            
        except Exception as e:
            logger.error(f"Failed to get document keywords: {e}")
            raise Exception(f"Failed to get document keywords: {e}")

def create_tfidf_scorer(documents: List[str]) -> TFIDFScorer:
    """
    Convenience function to create and fit a TF-IDF scorer.
    
    Args:
        documents (List[str]): List of document texts
        
    Returns:
        TFIDFScorer: Fitted TF-IDF scorer
    """
    scorer = TFIDFScorer()
    scorer.fit_transform(documents)
    return scorer 