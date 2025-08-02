import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceFactors:
    """Factors that contribute to confidence scoring."""
    retrieval_score: float = 0.0
    numerical_verification: float = 0.0
    completeness_score: float = 0.0
    context_relevance: float = 0.0
    answer_specificity: float = 0.0
    uncertainty_indicators: float = 0.0

@dataclass
class ConfidenceResult:
    """Results of confidence analysis."""
    overall_confidence: str  # 'high', 'medium', 'low'
    confidence_score: float  # 0.0 to 1.0
    factors: ConfidenceFactors
    uncertainty_level: str  # 'certain', 'likely', 'uncertain'
    recommendations: List[str]

class EnhancedConfidenceScorer:
    """
    Enhanced confidence scoring system that provides better calibration for answers,
    especially for numerical values and uncertainty expression.
    """
    
    def __init__(self):
        """Initialize the enhanced confidence scorer."""
        # Uncertainty indicators in answers
        self.uncertainty_indicators = {
            'high_uncertainty': [
                'i cannot find', 'not specified', 'not mentioned', 'unclear',
                'may vary', 'typically', 'generally', 'usually', 'might be',
                'could be', 'appears to', 'seems to', 'possibly', 'likely'
            ],
            'medium_uncertainty': [
                'approximately', 'around', 'about', 'roughly', 'estimated',
                'up to', 'at least', 'minimum', 'maximum', 'subject to'
            ],
            'low_uncertainty': [
                'exactly', 'precisely', 'specifically', 'clearly stated',
                'defined as', 'is', 'are', 'will be', 'must be'
            ]
        }
        
        # Specificity indicators
        self.specificity_indicators = {
            'high_specificity': [
                r'\d+\s*%', r'\d+\s*days?', r'\d+\s*months?', r'\d+\s*years?',
                r'rs\.?\s*\d+', r'\$\s*\d+', r'\d+\s*rupees?',
                'section \d+', 'clause \d+', 'page \d+'
            ],
            'medium_specificity': [
                'waiting period', 'grace period', 'coverage limit', 'sum insured',
                'pre-existing', 'maternity', 'exclusion', 'condition'
            ],
            'low_specificity': [
                'policy', 'coverage', 'benefit', 'claim', 'insurance',
                'treatment', 'medical', 'hospital'
            ]
        }
        
        # Context relevance keywords
        self.relevance_keywords = [
            'policy', 'coverage', 'benefit', 'claim', 'premium', 'deductible',
            'waiting period', 'exclusion', 'condition', 'limit', 'cap',
            'sum insured', 'maternity', 'pre-existing', 'hospital'
        ]
    
    def calculate_confidence(self, question: str, answer: str, context: str,
                           retrieval_metadata: Dict[str, Any] = None,
                           numerical_verification: Dict[str, Any] = None,
                           completeness_result: Dict[str, Any] = None) -> ConfidenceResult:
        """
        Calculate comprehensive confidence score for an answer.
        
        Args:
            question (str): Original question
            answer (str): Generated answer
            context (str): Context used for generation
            retrieval_metadata (Dict): Metadata from retrieval process
            numerical_verification (Dict): Results from numerical verification
            completeness_result (Dict): Results from completeness check
            
        Returns:
            ConfidenceResult: Comprehensive confidence analysis
        """
        factors = ConfidenceFactors()
        
        # 1. Retrieval Score (25% weight)
        factors.retrieval_score = self._calculate_retrieval_confidence(retrieval_metadata)
        
        # 2. Numerical Verification (25% weight)
        factors.numerical_verification = self._calculate_numerical_confidence(
            answer, numerical_verification
        )
        
        # 3. Completeness Score (20% weight)
        factors.completeness_score = self._calculate_completeness_confidence(completeness_result)
        
        # 4. Context Relevance (15% weight)
        factors.context_relevance = self._calculate_context_relevance(question, context)
        
        # 5. Answer Specificity (10% weight)
        factors.answer_specificity = self._calculate_answer_specificity(answer)
        
        # 6. Uncertainty Indicators (5% weight)
        factors.uncertainty_indicators = self._calculate_uncertainty_score(answer)
        
        # Calculate weighted overall confidence
        weights = {
            'retrieval_score': 0.25,
            'numerical_verification': 0.25,
            'completeness_score': 0.20,
            'context_relevance': 0.15,
            'answer_specificity': 0.10,
            'uncertainty_indicators': 0.05
        }
        
        confidence_score = (
            factors.retrieval_score * weights['retrieval_score'] +
            factors.numerical_verification * weights['numerical_verification'] +
            factors.completeness_score * weights['completeness_score'] +
            factors.context_relevance * weights['context_relevance'] +
            factors.answer_specificity * weights['answer_specificity'] +
            factors.uncertainty_indicators * weights['uncertainty_indicators']
        )
        
        # Determine confidence level
        if confidence_score >= 0.8:
            overall_confidence = "high"
        elif confidence_score >= 0.6:
            overall_confidence = "medium"
        else:
            overall_confidence = "low"
        
        # Determine uncertainty level
        uncertainty_level = self._determine_uncertainty_level(answer, confidence_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(factors, confidence_score)
        
        logger.info(f"Confidence analysis: {confidence_score:.3f} ({overall_confidence})")
        
        return ConfidenceResult(
            overall_confidence=overall_confidence,
            confidence_score=confidence_score,
            factors=factors,
            uncertainty_level=uncertainty_level,
            recommendations=recommendations
        )
    
    def _calculate_retrieval_confidence(self, retrieval_metadata: Dict[str, Any] = None) -> float:
        """Calculate confidence based on retrieval quality."""
        if not retrieval_metadata:
            return 0.5  # Default medium confidence
        
        confidence = 0.5
        
        # Check retrieval scores
        if 'combined_score' in retrieval_metadata:
            score = retrieval_metadata['combined_score']
            if isinstance(score, (int, float)):
                confidence = min(1.0, max(0.0, score))
        
        # Boost for multiple retrieval methods
        if retrieval_metadata.get('retrieval_method') == 'advanced_hybrid':
            confidence += 0.1
        elif retrieval_metadata.get('retrieval_method') == 'hybrid_tfidf':
            confidence += 0.05
        
        # Check number of sources
        num_sources = len(retrieval_metadata.get('sources', []))
        if num_sources >= 3:
            confidence += 0.1
        elif num_sources >= 2:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _calculate_numerical_confidence(self, answer: str, 
                                      numerical_verification: Dict[str, Any] = None) -> float:
        """Calculate confidence based on numerical value verification."""
        if not numerical_verification:
            # Check if answer contains numerical values
            if re.search(r'\d+', answer):
                return 0.3  # Low confidence for unverified numbers
            else:
                return 0.8  # High confidence for non-numerical answers
        
        verification_score = numerical_verification.get('confidence_score', 0.0)
        
        # Penalty for hallucinated values
        hallucinated_count = len(numerical_verification.get('hallucinated_values', []))
        if hallucinated_count > 0:
            penalty = min(0.5, hallucinated_count * 0.2)
            verification_score = max(0.0, verification_score - penalty)
        
        return verification_score
    
    def _calculate_completeness_confidence(self, completeness_result: Dict[str, Any] = None) -> float:
        """Calculate confidence based on answer completeness."""
        if not completeness_result:
            return 0.6  # Default medium confidence
        
        return completeness_result.get('completeness_score', 0.6)
    
    def _calculate_context_relevance(self, question: str, context: str) -> float:
        """Calculate confidence based on context relevance to question."""
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        
        # Calculate word overlap
        overlap = len(question_words.intersection(context_words))
        if len(question_words) == 0:
            return 0.5
        
        basic_relevance = overlap / len(question_words)
        
        # Boost for policy-specific terms
        policy_term_boost = 0.0
        for keyword in self.relevance_keywords:
            if keyword in question.lower() and keyword in context.lower():
                policy_term_boost += 0.1
        
        relevance_score = min(1.0, basic_relevance + policy_term_boost)
        return relevance_score
    
    def _calculate_answer_specificity(self, answer: str) -> float:
        """Calculate confidence based on answer specificity."""
        answer_lower = answer.lower()
        specificity_score = 0.0
        
        # Check for high specificity indicators
        for pattern in self.specificity_indicators['high_specificity']:
            if re.search(pattern, answer_lower):
                specificity_score += 0.3
        
        # Check for medium specificity indicators
        for term in self.specificity_indicators['medium_specificity']:
            if term in answer_lower:
                specificity_score += 0.2
        
        # Check for low specificity indicators
        for term in self.specificity_indicators['low_specificity']:
            if term in answer_lower:
                specificity_score += 0.1
        
        return min(1.0, specificity_score)
    
    def _calculate_uncertainty_score(self, answer: str) -> float:
        """Calculate confidence based on uncertainty indicators in answer."""
        answer_lower = answer.lower()
        uncertainty_score = 0.8  # Start with high confidence
        
        # Check for high uncertainty indicators
        for indicator in self.uncertainty_indicators['high_uncertainty']:
            if indicator in answer_lower:
                uncertainty_score -= 0.3
        
        # Check for medium uncertainty indicators
        for indicator in self.uncertainty_indicators['medium_uncertainty']:
            if indicator in answer_lower:
                uncertainty_score -= 0.1
        
        # Boost for low uncertainty indicators
        for indicator in self.uncertainty_indicators['low_uncertainty']:
            if indicator in answer_lower:
                uncertainty_score += 0.1
        
        return max(0.0, min(1.0, uncertainty_score))
    
    def _determine_uncertainty_level(self, answer: str, confidence_score: float) -> str:
        """Determine the uncertainty level of the answer."""
        answer_lower = answer.lower()
        
        # Check for explicit uncertainty expressions
        high_uncertainty_count = sum(1 for indicator in self.uncertainty_indicators['high_uncertainty'] 
                                   if indicator in answer_lower)
        
        if high_uncertainty_count > 0 or confidence_score < 0.4:
            return "uncertain"
        elif confidence_score >= 0.8:
            return "certain"
        else:
            return "likely"
    
    def _generate_recommendations(self, factors: ConfidenceFactors, 
                                confidence_score: float) -> List[str]:
        """Generate recommendations for improving confidence."""
        recommendations = []
        
        if factors.retrieval_score < 0.6:
            recommendations.append("Consider using more advanced retrieval methods or expanding the search context.")
        
        if factors.numerical_verification < 0.6:
            recommendations.append("Verify numerical values against the source document to prevent hallucination.")
        
        if factors.completeness_score < 0.7:
            recommendations.append("Include more comprehensive details such as conditions, exclusions, and limitations.")
        
        if factors.context_relevance < 0.6:
            recommendations.append("Ensure the retrieved context is more relevant to the specific question asked.")
        
        if factors.answer_specificity < 0.5:
            recommendations.append("Provide more specific details and concrete information in the answer.")
        
        if confidence_score < 0.5:
            recommendations.append("Consider expressing uncertainty when information is not clearly available in the source.")
        
        return recommendations
    
    def suggest_uncertainty_expression(self, confidence_score: float, 
                                     missing_info: List[str] = None) -> str:
        """Suggest appropriate uncertainty expression based on confidence."""
        if confidence_score >= 0.8:
            return ""  # No uncertainty expression needed
        
        if confidence_score >= 0.6:
            expressions = [
                "Based on the available information",
                "According to the policy document",
                "The policy indicates that"
            ]
        elif confidence_score >= 0.4:
            expressions = [
                "The policy appears to indicate",
                "Based on the available sections",
                "From what is specified in the document"
            ]
        else:
            expressions = [
                "The exact details are not clearly specified",
                "The policy document does not provide complete information about",
                "Additional clarification may be needed regarding"
            ]
        
        # Add specific missing information context
        if missing_info:
            missing_str = ", ".join(missing_info)
            expressions.append(f"Note that specific details about {missing_str} are not clearly stated")
        
        return expressions[0]  # Return the first appropriate expression

def create_confidence_scorer() -> EnhancedConfidenceScorer:
    """
    Convenience function to create an enhanced confidence scorer.
    
    Returns:
        EnhancedConfidenceScorer: Initialized confidence scorer
    """
    return EnhancedConfidenceScorer()