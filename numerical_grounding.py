import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class NumericalEntity:
    """Represents a numerical entity found in text."""
    value: str
    entity_type: str  # 'percentage', 'currency', 'days', 'months', 'years', 'number'
    context: str  # surrounding context
    position: int  # position in text
    confidence: float = 1.0

class NumericalGroundingSystem:
    """
    System for extracting, storing, and verifying numerical values from documents.
    Helps prevent hallucination of non-existent numerical values.
    """
    
    def __init__(self):
        """Initialize the numerical grounding system."""
        self.numerical_entities = []
        self.entity_patterns = {
            'percentage': [
                r'(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*percent',
                r'(\d+(?:\.\d+)?)\s*per\s*cent'
            ],
            'currency': [
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'Rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'INR\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'USD\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rupees?|dollars?)'
            ],
            'days': [
                r'(\d+)\s*days?',
                r'(\d+)\s*day\s*period',
                r'within\s*(\d+)\s*days?'
            ],
            'months': [
                r'(\d+)\s*months?',
                r'(\d+)\s*month\s*period',
                r'(\d+)\s*monthly',
                r'(\d+)\s*mo\b'
            ],
            'years': [
                r'(\d+)\s*years?',
                r'(\d+)\s*year\s*period',
                r'(\d+)\s*yearly',
                r'(\d+)\s*yr\b'
            ],
            'number': [
                r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b'
            ]
        }
        
        # Common policy-related contexts to look for
        self.policy_contexts = [
            'waiting period', 'grace period', 'coverage limit', 'sum insured',
            'premium', 'deductible', 'co-payment', 'room rent', 'icu charges',
            'maternity', 'pre-existing', 'cataract', 'claim', 'discount',
            'benefit', 'maximum', 'minimum', 'cap', 'limit'
        ]
    
    def extract_numerical_entities(self, text: str, context_window: int = 50) -> List[NumericalEntity]:
        """
        Extract all numerical entities from text with their context.
        
        Args:
            text (str): Text to extract from
            context_window (int): Number of characters around the entity for context
            
        Returns:
            List[NumericalEntity]: List of extracted numerical entities
        """
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)
                    position = match.start()
                    
                    # Extract context around the match
                    start_ctx = max(0, position - context_window)
                    end_ctx = min(len(text), position + len(match.group(0)) + context_window)
                    context = text[start_ctx:end_ctx].strip()
                    
                    # Calculate confidence based on policy context
                    confidence = self._calculate_context_confidence(context.lower())
                    
                    entity = NumericalEntity(
                        value=value,
                        entity_type=entity_type,
                        context=context,
                        position=position,
                        confidence=confidence
                    )
                    entities.append(entity)
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.position)
        
        logger.info(f"Extracted {len(entities)} numerical entities from text")
        return entities
    
    def _calculate_context_confidence(self, context: str) -> float:
        """
        Calculate confidence score based on policy-related context.
        
        Args:
            context (str): Context around the numerical entity
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        base_confidence = 0.5
        context_bonus = 0.0
        
        # Check for policy-related terms in context
        for policy_term in self.policy_contexts:
            if policy_term in context:
                context_bonus += 0.1
        
        # Bonus for specific formatting patterns
        if any(pattern in context for pattern in ['%', 'percent', 'per cent']):
            context_bonus += 0.2
        
        if any(pattern in context for pattern in ['rs.', 'inr', 'rupees', '$', 'usd', 'dollars']):
            context_bonus += 0.2
        
        if any(pattern in context for pattern in ['days', 'months', 'years', 'period']):
            context_bonus += 0.15
        
        return min(1.0, base_confidence + context_bonus)
    
    def _deduplicate_entities(self, entities: List[NumericalEntity]) -> List[NumericalEntity]:
        """
        Remove duplicate entities based on value and proximity.
        
        Args:
            entities (List[NumericalEntity]): List of entities to deduplicate
            
        Returns:
            List[NumericalEntity]: Deduplicated list
        """
        if not entities:
            return entities
        
        # Sort by position first
        entities.sort(key=lambda x: x.position)
        
        deduplicated = []
        for entity in entities:
            # Check if this entity is too close to an existing one with same value
            is_duplicate = False
            for existing in deduplicated:
                if (entity.value == existing.value and 
                    abs(entity.position - existing.position) < 20):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    break
            
            if not is_duplicate:
                deduplicated.append(entity)
        
        return deduplicated
    
    def build_entity_database(self, text: str) -> Dict[str, List[NumericalEntity]]:
        """
        Build a database of numerical entities from document text.
        
        Args:
            text (str): Document text to process
            
        Returns:
            Dict[str, List[NumericalEntity]]: Database organized by entity type
        """
        self.numerical_entities = self.extract_numerical_entities(text)
        
        # Organize by type
        entity_db = {}
        for entity in self.numerical_entities:
            if entity.entity_type not in entity_db:
                entity_db[entity.entity_type] = []
            entity_db[entity.entity_type].append(entity)
        
        logger.info(f"Built numerical entity database with {len(self.numerical_entities)} entities")
        for entity_type, entities in entity_db.items():
            logger.info(f"  {entity_type}: {len(entities)} entities")
        
        return entity_db
    
    def verify_numerical_values(self, answer: str) -> Dict[str, Any]:
        """
        Verify numerical values in an answer against the entity database.
        
        Args:
            answer (str): Generated answer to verify
            
        Returns:
            Dict[str, Any]: Verification results
        """
        # Extract numerical entities from the answer
        answer_entities = self.extract_numerical_entities(answer)
        
        verification_results = {
            'total_numerical_values': len(answer_entities),
            'verified_values': 0,
            'unverified_values': 0,
            'hallucinated_values': [],
            'verified_entities': [],
            'confidence_score': 1.0
        }
        
        for answer_entity in answer_entities:
            is_verified = False
            
            # Check against database entities
            for db_entity in self.numerical_entities:
                if self._values_match(answer_entity.value, db_entity.value, answer_entity.entity_type):
                    verification_results['verified_values'] += 1
                    verification_results['verified_entities'].append({
                        'value': answer_entity.value,
                        'type': answer_entity.entity_type,
                        'context': db_entity.context[:100] + "..." if len(db_entity.context) > 100 else db_entity.context
                    })
                    is_verified = True
                    break
            
            if not is_verified:
                verification_results['unverified_values'] += 1
                verification_results['hallucinated_values'].append({
                    'value': answer_entity.value,
                    'type': answer_entity.entity_type,
                    'context': answer_entity.context
                })
        
        # Calculate confidence score
        if verification_results['total_numerical_values'] > 0:
            verification_results['confidence_score'] = (
                verification_results['verified_values'] / verification_results['total_numerical_values']
            )
        
        return verification_results
    
    def _values_match(self, value1: str, value2: str, entity_type: str) -> bool:
        """
        Check if two numerical values match, considering type-specific rules.
        
        Args:
            value1 (str): First value
            value2 (str): Second value
            entity_type (str): Type of numerical entity
            
        Returns:
            bool: True if values match
        """
        try:
            # Clean and normalize values
            clean_val1 = re.sub(r'[,\s]', '', value1.strip())
            clean_val2 = re.sub(r'[,\s]', '', value2.strip())
            
            # Direct string match
            if clean_val1 == clean_val2:
                return True
            
            # Numerical comparison for numbers
            if entity_type in ['number', 'percentage', 'currency']:
                try:
                    num1 = float(clean_val1)
                    num2 = float(clean_val2)
                    # Allow small floating point differences
                    return abs(num1 - num2) < 0.01
                except ValueError:
                    pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error comparing values {value1} and {value2}: {e}")
            return False
    
    def get_entity_suggestions(self, query: str, entity_type: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant numerical entities based on query context.
        
        Args:
            query (str): Query to find relevant entities for
            entity_type (str): Specific entity type to filter by
            top_k (int): Number of top suggestions to return
            
        Returns:
            List[Dict[str, Any]]: List of relevant entities with context
        """
        query_lower = query.lower()
        relevant_entities = []
        
        for entity in self.numerical_entities:
            if entity_type and entity.entity_type != entity_type:
                continue
            
            # Calculate relevance score based on context overlap
            relevance_score = 0.0
            entity_context_lower = entity.context.lower()
            
            # Check for query terms in entity context
            query_words = set(query_lower.split())
            context_words = set(entity_context_lower.split())
            
            overlap = len(query_words.intersection(context_words))
            if overlap > 0:
                relevance_score = overlap / len(query_words)
            
            # Boost score for policy-related terms
            for policy_term in self.policy_contexts:
                if policy_term in query_lower and policy_term in entity_context_lower:
                    relevance_score += 0.3
            
            if relevance_score > 0:
                relevant_entities.append({
                    'entity': entity,
                    'relevance_score': relevance_score * entity.confidence
                })
        
        # Sort by relevance and return top-k
        relevant_entities.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        suggestions = []
        for item in relevant_entities[:top_k]:
            entity = item['entity']
            suggestions.append({
                'value': entity.value,
                'type': entity.entity_type,
                'context': entity.context,
                'confidence': entity.confidence,
                'relevance_score': item['relevance_score']
            })
        
        return suggestions

def create_numerical_grounding_system(document_text: str) -> NumericalGroundingSystem:
    """
    Convenience function to create and initialize a numerical grounding system.
    
    Args:
        document_text (str): Document text to build entity database from
        
    Returns:
        NumericalGroundingSystem: Initialized grounding system
    """
    grounding_system = NumericalGroundingSystem()
    grounding_system.build_entity_database(document_text)
    return grounding_system