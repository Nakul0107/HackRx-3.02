import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PolicyElement:
    """Represents a policy element that should be included in answers."""
    element_type: str
    keywords: List[str]
    required_for: List[str]  # Question types that require this element
    weight: float = 1.0

@dataclass
class CompletenessResult:
    """Results of completeness analysis."""
    is_complete: bool
    completeness_score: float
    missing_elements: List[str]
    present_elements: List[str]
    suggestions: List[str]
    confidence: str

class EnhancedCompletenessChecker:
    """
    Enhanced completeness checker that ensures answers include all relevant policy details.
    Addresses the issue of incomplete answers missing important conditions or exceptions.
    """
    
    def __init__(self):
        """Initialize the enhanced completeness checker."""
        self.policy_elements = {
            "coverage_limits": PolicyElement(
                element_type="coverage_limits",
                keywords=[
                    "limit", "cap", "maximum", "minimum", "sum insured", "coverage amount",
                    "benefit limit", "ceiling", "threshold", "up to", "not exceeding"
                ],
                required_for=["coverage", "benefit", "claim", "reimburse", "pay"],
                weight=1.0
            ),
            "waiting_periods": PolicyElement(
                element_type="waiting_periods",
                keywords=[
                    "waiting period", "wait", "after", "from inception", "continuous coverage",
                    "months", "years", "days", "period", "before", "eligibility period"
                ],
                required_for=["coverage", "eligible", "when", "pre-existing", "maternity"],
                weight=1.0
            ),
            "exclusions": PolicyElement(
                element_type="exclusions",
                keywords=[
                    "exclusion", "excluded", "not covered", "does not cover", "except",
                    "exception", "limitation", "restriction", "not applicable", "barred"
                ],
                required_for=["coverage", "cover", "benefit", "eligible"],
                weight=0.9
            ),
            "conditions": PolicyElement(
                element_type="conditions",
                keywords=[
                    "condition", "requirement", "criteria", "eligible", "qualify",
                    "provided", "subject to", "terms", "clause", "stipulation", "must"
                ],
                required_for=["eligible", "qualify", "claim", "benefit", "coverage"],
                weight=0.9
            ),
            "procedures": PolicyElement(
                element_type="procedures",
                keywords=[
                    "process", "procedure", "steps", "submit", "claim", "documentation",
                    "form", "application", "approval", "notification", "intimation"
                ],
                required_for=["claim", "process", "submit", "how to", "procedure"],
                weight=0.8
            ),
            "time_limits": PolicyElement(
                element_type="time_limits",
                keywords=[
                    "within", "before", "after", "days", "months", "years", "deadline",
                    "time limit", "period", "duration", "expiry", "validity"
                ],
                required_for=["claim", "submit", "notify", "process", "deadline"],
                weight=0.8
            ),
            "definitions": PolicyElement(
                element_type="definitions",
                keywords=[
                    "defined as", "means", "refers to", "includes", "definition",
                    "interpretation", "shall mean", "understood as"
                ],
                required_for=["what is", "define", "meaning", "definition"],
                weight=0.7
            ),
            "sub_limits": PolicyElement(
                element_type="sub_limits",
                keywords=[
                    "sub-limit", "sub limit", "daily limit", "per day", "room rent",
                    "icu charges", "specific limit", "separate limit", "individual limit"
                ],
                required_for=["room rent", "icu", "daily", "charges", "sub-limit"],
                weight=1.0
            )
        }
        
        # Question type patterns to identify what type of question is being asked
        self.question_patterns = {
            "coverage": [
                r"\b(?:cover|coverage|covered|benefit|reimburse|pay)\b",
                r"\b(?:what.*covered|does.*cover|coverage.*include)\b"
            ],
            "eligible": [
                r"\b(?:eligible|qualify|qualification|criteria|who can)\b",
                r"\b(?:eligibility|requirements for)\b"
            ],
            "claim": [
                r"\b(?:claim|submit|process|procedure|how to)\b",
                r"\b(?:claiming|submission|processing)\b"
            ],
            "waiting": [
                r"\b(?:waiting period|wait|when.*covered|after how long)\b"
            ],
            "definition": [
                r"\b(?:what is|define|definition|meaning|means)\b"
            ],
            "limits": [
                r"\b(?:limit|cap|maximum|minimum|sub-limit|room rent|icu)\b"
            ]
        }
    
    def analyze_question_type(self, question: str) -> List[str]:
        """
        Analyze the question to determine what types of information should be included.
        
        Args:
            question (str): The question being asked
            
        Returns:
            List[str]: List of question types identified
        """
        question_lower = question.lower()
        identified_types = []
        
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    identified_types.append(q_type)
                    break
        
        # If no specific type identified, assume it's a general coverage question
        if not identified_types:
            identified_types.append("coverage")
        
        return identified_types
    
    def extract_present_elements(self, text: str) -> Dict[str, List[str]]:
        """
        Extract policy elements present in the given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, List[str]]: Dictionary of present elements with matching keywords
        """
        text_lower = text.lower()
        present_elements = defaultdict(list)
        
        for element_name, element in self.policy_elements.items():
            for keyword in element.keywords:
                if keyword in text_lower:
                    present_elements[element_name].append(keyword)
        
        return dict(present_elements)
    
    def check_completeness(self, question: str, answer: str, context: str) -> CompletenessResult:
        """
        Check if the answer is complete based on the question type and available context.
        
        Args:
            question (str): The original question
            answer (str): The generated answer
            context (str): The context used to generate the answer
            
        Returns:
            CompletenessResult: Detailed completeness analysis
        """
        # Analyze question type
        question_types = self.analyze_question_type(question)
        
        # Extract elements present in answer and context
        answer_elements = self.extract_present_elements(answer)
        context_elements = self.extract_present_elements(context)
        
        # Determine required elements based on question type
        required_elements = set()
        for q_type in question_types:
            for element_name, element in self.policy_elements.items():
                if q_type in element.required_for:
                    required_elements.add(element_name)
        
        # Check which required elements are missing
        missing_elements = []
        present_elements = []
        element_scores = []
        
        for element_name in required_elements:
            element = self.policy_elements[element_name]
            
            # Check if element is present in answer
            if element_name in answer_elements:
                present_elements.append(element_name)
                element_scores.append(element.weight)
            else:
                # Check if element is available in context but missing from answer
                if element_name in context_elements:
                    missing_elements.append(element_name)
                    element_scores.append(0.0)
                else:
                    # Element not available in context, so not penalized heavily
                    element_scores.append(0.5)
        
        # Calculate completeness score
        if required_elements:
            completeness_score = sum(element_scores) / len(required_elements)
        else:
            completeness_score = 1.0
        
        # Determine if answer is complete
        is_complete = len(missing_elements) == 0 and completeness_score >= 0.8
        
        # Generate suggestions for missing elements
        suggestions = self._generate_suggestions(missing_elements, context_elements)
        
        # Determine confidence level
        if completeness_score >= 0.9:
            confidence = "high"
        elif completeness_score >= 0.7:
            confidence = "medium"
        else:
            confidence = "low"
        
        logger.info(f"Completeness check: {completeness_score:.2f}, Missing: {missing_elements}")
        
        return CompletenessResult(
            is_complete=is_complete,
            completeness_score=completeness_score,
            missing_elements=missing_elements,
            present_elements=present_elements,
            suggestions=suggestions,
            confidence=confidence
        )
    
    def _generate_suggestions(self, missing_elements: List[str], context_elements: Dict[str, List[str]]) -> List[str]:
        """
        Generate suggestions for improving answer completeness.
        
        Args:
            missing_elements (List[str]): List of missing element types
            context_elements (Dict[str, List[str]]): Elements available in context
            
        Returns:
            List[str]: List of suggestions
        """
        suggestions = []
        
        element_suggestions = {
            "coverage_limits": "Include specific coverage limits, caps, or maximum amounts mentioned in the policy.",
            "waiting_periods": "Mention any waiting periods or time requirements before coverage begins.",
            "exclusions": "Include any exclusions, exceptions, or limitations that apply.",
            "conditions": "Specify the conditions, requirements, or criteria that must be met.",
            "procedures": "Describe the process, steps, or procedures that need to be followed.",
            "time_limits": "Include any time limits, deadlines, or duration requirements.",
            "definitions": "Provide the specific definition or meaning as stated in the policy.",
            "sub_limits": "Mention any sub-limits, daily limits, or specific restrictions that apply."
        }
        
        for missing_element in missing_elements:
            if missing_element in context_elements:
                # Element is available in context but missing from answer
                suggestion = element_suggestions.get(missing_element, f"Include information about {missing_element}")
                suggestions.append(suggestion)
        
        return suggestions
    
    def generate_followup_prompt(self, question: str, answer: str, context: str, 
                                completeness_result: CompletenessResult) -> str:
        """
        Generate a follow-up prompt to improve answer completeness.
        
        Args:
            question (str): Original question
            answer (str): Initial answer
            context (str): Available context
            completeness_result (CompletenessResult): Results of completeness check
            
        Returns:
            str: Follow-up prompt to improve completeness
        """
        if completeness_result.is_complete:
            return ""
        
        missing_elements_str = ", ".join(completeness_result.missing_elements)
        suggestions_str = " ".join(completeness_result.suggestions)
        
        followup_prompt = f"""
You previously provided this answer to a policy question, but it's missing some important details.

Original Question: {question}

Your Initial Answer: {answer}

The answer is missing information about: {missing_elements_str}

Specific improvements needed: {suggestions_str}

Please provide a more complete answer that includes these missing details if they are mentioned in the context below. If the context doesn't contain information about these elements, state that explicitly.

Context: {context}

Complete Answer:"""
        
        return followup_prompt
    
    def validate_structured_answer(self, answer: str, required_structure: Dict[str, bool]) -> Dict[str, Any]:
        """
        Validate that an answer follows the required structured format.
        
        Args:
            answer (str): Answer to validate
            required_structure (Dict[str, bool]): Required sections and whether they're mandatory
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            "is_structured": False,
            "present_sections": [],
            "missing_sections": [],
            "structure_score": 0.0
        }
        
        # Define section patterns
        section_patterns = {
            "main_answer": r"(?:^|\n)(?:Answer|Main Answer|Coverage):\s*(.+?)(?=\n[A-Z]|$)",
            "conditions": r"(?:^|\n)(?:Conditions|Requirements|Criteria):\s*(.+?)(?=\n[A-Z]|$)",
            "exclusions": r"(?:^|\n)(?:Exclusions|Exceptions|Limitations):\s*(.+?)(?=\n[A-Z]|$)",
            "limitations": r"(?:^|\n)(?:Limitations|Caps|Limits):\s*(.+?)(?=\n[A-Z]|$)"
        }
        
        present_count = 0
        total_required = sum(1 for required in required_structure.values() if required)
        
        for section, pattern in section_patterns.items():
            if re.search(pattern, answer, re.IGNORECASE | re.DOTALL):
                validation_results["present_sections"].append(section)
                if required_structure.get(section, False):
                    present_count += 1
            elif required_structure.get(section, False):
                validation_results["missing_sections"].append(section)
        
        validation_results["structure_score"] = present_count / total_required if total_required > 0 else 1.0
        validation_results["is_structured"] = validation_results["structure_score"] >= 0.8
        
        return validation_results

def create_completeness_checker() -> EnhancedCompletenessChecker:
    """
    Convenience function to create an enhanced completeness checker.
    
    Returns:
        EnhancedCompletenessChecker: Initialized completeness checker
    """
    return EnhancedCompletenessChecker()