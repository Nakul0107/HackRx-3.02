import os
import logging
import re
from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from retriever import VectorStoreManager
from openrouter_integration import OpenRouterChatModel
from numerical_grounding import create_numerical_grounding_system, NumericalGroundingSystem
from completeness_checker import create_completeness_checker, EnhancedCompletenessChecker
from confidence_scorer import create_confidence_scorer, EnhancedConfidenceScorer

logger = logging.getLogger(__name__)

class PolicyQASystem:
    def __init__(self, api_key: str = None):
        """
        Initialize the Policy QA System with OpenRouter API and TF-IDF scoring.
        
        Args:
            api_key (str): OpenRouter API key
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or "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"
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        # Initialize OpenRouter with a free model
        self.llm = OpenRouterChatModel(
            api_key=self.api_key,
            model="openai/gpt-3.5-turbo",  # Using a free model
            temperature=0.1,
            max_tokens=2048
        )
        
        # Enhanced prompt template for policy QA with structured completeness checking
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a policy document expert. Answer questions about insurance policies with comprehensive, structured responses.
            
            CRITICAL INSTRUCTIONS:
            - Provide complete, factual answers based ONLY on the context provided
            - NEVER invent or hallucinate numerical values, dates, or amounts not in the context
            - Include ALL relevant details: waiting periods, coverage limits, exclusions, conditions, and procedures
            - Use a structured format when appropriate:
              * Main Answer: [Direct response to the question]
              * Conditions/Requirements: [Any conditions that must be met]
              * Exclusions/Limitations: [Any exclusions or limitations that apply]
              * Additional Details: [Other relevant information]
            
            NUMERICAL VALUES:
            - Only cite numerical values (amounts, percentages, time periods) that are explicitly stated in the context
            - If exact values are not in the context, state "The exact amount/percentage is not specified in the provided sections"
            - Always include the unit (days, months, years, %, Rs., etc.) when citing numbers
            
            COMPLETENESS:
            - For coverage questions: Include limits, conditions, exclusions, and waiting periods
            - For eligibility questions: Include all criteria and requirements
            - For claim questions: Include process steps, documentation, and time limits
            - For definition questions: Provide the complete definition as stated
            
            If information is not available in the context, explicitly state: "This information is not provided in the available policy sections."
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Define follow-up prompt template for missing details
        self.followup_prompt_template = PromptTemplate(
            input_variables=["context", "question", "initial_answer", "missing_elements"],
            template="""
            You are a policy document expert. You previously provided this answer to a question, but it's missing some important details.
            
            Context: {context}
            
            Original Question: {question}
            
            Your Initial Answer: {initial_answer}
            
            The answer is missing information about: {missing_elements}
            
            Please provide a more complete answer that includes these missing details if they are mentioned in the context.
            If the context doesn't contain information about these elements, state that explicitly.
            
            Complete Answer:"""
        )
        
        # Define common policy elements to check for completeness
        self.policy_elements = {
            "coverage": ["limits", "cap", "maximum", "amount", "coverage", "covered"],
            "time": ["waiting period", "days", "months", "years", "duration", "term"],
            "exclusions": ["exclusion", "excluded", "not covered", "does not cover", "exception"],
            "conditions": ["condition", "requirement", "criteria", "eligible", "qualify"],
            "process": ["process", "procedure", "steps", "submit", "claim", "documentation"]
        }
        
        # Initialize vector store manager for hybrid retrieval
        self.vector_manager = VectorStoreManager(api_key)
        
        # Initialize enhancement systems
        self.numerical_grounding = None
        self.completeness_checker = create_completeness_checker()
        self.confidence_scorer = create_confidence_scorer()
    
    def check_completeness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Check if the answer is complete by looking for missing policy elements.
        
        Args:
            question (str): The original question
            answer (str): The generated answer
            context (str): The context used to generate the answer
            
        Returns:
            Dict[str, Any]: Results of completeness check with missing elements
        """
        # Determine which policy elements should be in the answer based on the question
        required_elements = []
        missing_elements = []
        
        # Check if question is about coverage
        if any(term in question.lower() for term in ["cover", "coverage", "benefit", "pay", "reimburse"]):
            required_elements.extend(["coverage", "exclusions"])
            # Check for time-related terms if it's about coverage
            if any(term in context.lower() for term in self.policy_elements["time"]):
                required_elements.append("time")
        
        # Check if question is about process or claims
        if any(term in question.lower() for term in ["claim", "process", "submit", "how to", "procedure"]):
            required_elements.append("process")
        
        # Check if question is about eligibility
        if any(term in question.lower() for term in ["eligible", "qualify", "who", "criteria"]):
            required_elements.append("conditions")
        
        # Check for missing elements in the answer
        for element in required_elements:
            # Check if any terms from this element category are in the context
            element_in_context = any(term in context.lower() for term in self.policy_elements[element])
            
            # Check if any terms from this element category are in the answer
            element_in_answer = any(term in answer.lower() for term in self.policy_elements[element])
            
            # If element is in context but not in answer, it's missing
            if element_in_context and not element_in_answer:
                missing_elements.append(element)
        
        return {
            "is_complete": len(missing_elements) == 0,
            "missing_elements": missing_elements,
            "required_elements": required_elements
        }
    
    def generate_followup_question(self, missing_elements: List[str]) -> str:
        """
        Generate a follow-up question based on missing elements.
        
        Args:
            missing_elements (List[str]): List of missing policy elements
            
        Returns:
            str: Follow-up question
        """
        if "coverage" in missing_elements:
            return "What are the coverage limits or caps mentioned in the policy?"
        elif "time" in missing_elements:
            return "Are there any waiting periods or time limitations mentioned in the policy?"
        elif "exclusions" in missing_elements:
            return "What exclusions or exceptions are mentioned in the policy?"
        elif "conditions" in missing_elements:
            return "What conditions or requirements need to be met according to the policy?"
        elif "process" in missing_elements:
            return "What is the process or procedure mentioned in the policy?"
        else:
            return "Can you provide more specific details from the policy?"

    def initialize_document_systems(self, document_text: str):
        """
        Initialize document-specific systems for numerical grounding.
        
        Args:
            document_text (str): Full document text for analysis
        """
        logger.info("Initializing numerical grounding system")
        self.numerical_grounding = create_numerical_grounding_system(document_text)

    def answer_question(self, question: str, vector_store, use_hybrid: bool = True,
                       use_cross_encoder: bool = False, use_advanced_hybrid: bool = False,
                       document_text: str = None) -> Dict[str, Any]:
        """
        Answer a single question using enhanced RAG with comprehensive accuracy improvements.
        
        Args:
            question (str): The question to answer
            vector_store: The FAISS vector store
            use_hybrid (bool): Whether to use hybrid retrieval with TF-IDF scoring
            use_cross_encoder (bool): Whether to use cross-encoder for more accurate scoring
            use_advanced_hybrid (bool): Whether to use advanced hybrid with all three methods
            document_text (str): Full document text for numerical grounding (optional)
            
        Returns:
            Dict[str, Any]: Answer with comprehensive metadata and verification
        """
        try:
            logger.info(f"Processing question with enhanced accuracy systems: {question}")
            
            # Initialize numerical grounding if document text is provided
            if document_text and not self.numerical_grounding:
                self.initialize_document_systems(document_text)
            
            # Initialize variables
            source_docs = []
            tfidf_analysis = None
            cross_encoder_analysis = None
            retrieval_metadata = {}
            
            if use_advanced_hybrid and hasattr(self.vector_manager, 'cross_encoder_scorer') and self.vector_manager.cross_encoder_scorer:
                # Use advanced hybrid retrieval with all three methods
                logger.info("Using advanced hybrid retrieval with TF-IDF, embedding, and cross-encoder")
                source_docs = self.vector_manager.get_advanced_hybrid_retriever(
                    vector_store, question, top_k=10, 
                    tfidf_weight=0.2, embedding_weight=0.3, cross_encoder_weight=0.5
                )
                
                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in source_docs])
                
                # Get analysis for debugging
                tfidf_analysis = self.vector_manager.get_tfidf_analysis(question)
                cross_encoder_analysis = self.vector_manager.get_cross_encoder_analysis(question)
                
                # Use LLM directly with context
                prompt = self.prompt_template.format(context=context, question=question)
                result = self.llm.invoke(prompt)
                initial_answer = result.content
                
            elif use_cross_encoder and hasattr(self.vector_manager, 'cross_encoder_scorer') and self.vector_manager.cross_encoder_scorer:
                # Use cross-encoder retrieval for more accurate scoring
                logger.info("Using cross-encoder retrieval for more accurate scoring")
                source_docs = self.vector_manager.get_cross_encoder_retriever(
                    vector_store, question, top_k=10, use_reranking=True
                )
                
                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in source_docs])
                
                # Get cross-encoder analysis for debugging
                cross_encoder_analysis = self.vector_manager.get_cross_encoder_analysis(question)
                
                # Use LLM directly with context
                prompt = self.prompt_template.format(context=context, question=question)
                result = self.llm.invoke(prompt)
                initial_answer = result.content
                tfidf_analysis = None
                
            elif use_hybrid and hasattr(self.vector_manager, 'tfidf_scorer') and self.vector_manager.tfidf_scorer:
                # Use hybrid retrieval with TF-IDF scoring
                logger.info("Using hybrid retrieval with TF-IDF scoring")
                source_docs = self.vector_manager.get_hybrid_retriever(
                    vector_store, question, top_k=10, tfidf_weight=0.4, embedding_weight=0.6
                )
                
                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in source_docs])
                
                # Get TF-IDF analysis for debugging
                tfidf_analysis = self.vector_manager.get_tfidf_analysis(question)
                cross_encoder_analysis = None
                
                # Use LLM directly with context
                prompt = self.prompt_template.format(context=context, question=question)
                result = self.llm.invoke(prompt)
                initial_answer = result.content
                
            else:
                # Fallback to standard retrieval
                logger.info("Using standard embedding-based retrieval")
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 10,
                        "score_threshold": 0.5
                    }
                )
                
                # Instead of using RetrievalQA, we'll manually retrieve documents and use our LLM
                docs = retriever.get_relevant_documents(question)
                source_docs = docs
                
                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in source_docs])
                
                # Use LLM directly with context
                prompt = self.prompt_template.format(context=context, question=question)
                result = self.llm.invoke(prompt)
                initial_answer = result.content
                tfidf_analysis = None
                cross_encoder_analysis = None
            
            # Store retrieval metadata for confidence scoring
            retrieval_metadata = {
                'sources': [doc.page_content[:200] + "..." for doc in source_docs],
                'retrieval_method': 'advanced_hybrid' if use_advanced_hybrid else
                                  'cross_encoder' if use_cross_encoder else
                                  'hybrid_tfidf' if use_hybrid else 'embedding_only',
                'num_sources': len(source_docs)
            }
            
            # Add retrieval scores if available
            if source_docs and hasattr(source_docs[0], 'metadata'):
                scores = [doc.metadata.get('combined_score', 0.0) for doc in source_docs
                         if hasattr(doc, 'metadata') and 'combined_score' in doc.metadata]
                if scores:
                    retrieval_metadata['combined_score'] = sum(scores) / len(scores)
            
            # Enhanced completeness checking
            logger.info("Performing enhanced completeness analysis")
            completeness_result = self.completeness_checker.check_completeness(
                question, initial_answer, context
            )
            
            # Numerical verification if grounding system is available
            numerical_verification = None
            if self.numerical_grounding:
                logger.info("Performing numerical value verification")
                numerical_verification = self.numerical_grounding.verify_numerical_values(initial_answer)
            
            # Generate improved answer if needed
            if not completeness_result.is_complete and completeness_result.suggestions:
                logger.info(f"Answer is incomplete. Missing: {completeness_result.missing_elements}")
                
                # Generate enhanced follow-up prompt
                followup_prompt = self.completeness_checker.generate_followup_prompt(
                    question, initial_answer, context, completeness_result
                )
                
                if followup_prompt:
                    logger.info("Generating improved answer with enhanced completeness")
                    improved_answer = self.llm.invoke(followup_prompt)
                    answer = improved_answer.content
                    
                    # Re-verify numerical values in improved answer
                    if self.numerical_grounding:
                        numerical_verification = self.numerical_grounding.verify_numerical_values(answer)
                else:
                    answer = initial_answer
            else:
                logger.info("Answer meets completeness requirements")
                answer = initial_answer
            
            # Enhanced confidence scoring
            logger.info("Calculating enhanced confidence score")
            confidence_result = self.confidence_scorer.calculate_confidence(
                question=question,
                answer=answer,
                context=context,
                retrieval_metadata=retrieval_metadata,
                numerical_verification=numerical_verification,
                completeness_result=completeness_result.__dict__ if hasattr(completeness_result, '__dict__') else {
                    'completeness_score': completeness_result.completeness_score if hasattr(completeness_result, 'completeness_score') else 0.8,
                    'is_complete': completeness_result.is_complete if hasattr(completeness_result, 'is_complete') else True
                }
            )
            
            # Add uncertainty expression if confidence is low
            if confidence_result.confidence_score < 0.6:
                uncertainty_expr = self.confidence_scorer.suggest_uncertainty_expression(
                    confidence_result.confidence_score,
                    completeness_result.missing_elements if hasattr(completeness_result, 'missing_elements') else []
                )
                if uncertainty_expr:
                    answer = f"{uncertainty_expr}. {answer}"
            
            logger.info(f"Generated enhanced answer for question: {question[:50]}...")
            
            # Build comprehensive response
            response = {
                "question": question,
                "answer": answer,
                "sources": retrieval_metadata.get('sources', []),
                "confidence": confidence_result.overall_confidence,
                "confidence_score": confidence_result.confidence_score,
                "uncertainty_level": confidence_result.uncertainty_level,
                "retrieval_method": retrieval_metadata.get('retrieval_method', 'unknown'),
                "completeness": {
                    "is_complete": completeness_result.is_complete if hasattr(completeness_result, 'is_complete') else True,
                    "completeness_score": completeness_result.completeness_score if hasattr(completeness_result, 'completeness_score') else 0.8,
                    "missing_elements": completeness_result.missing_elements if hasattr(completeness_result, 'missing_elements') else [],
                    "suggestions": completeness_result.suggestions if hasattr(completeness_result, 'suggestions') else []
                },
                "confidence_factors": {
                    "retrieval_score": confidence_result.factors.retrieval_score,
                    "numerical_verification": confidence_result.factors.numerical_verification,
                    "completeness_score": confidence_result.factors.completeness_score,
                    "context_relevance": confidence_result.factors.context_relevance,
                    "answer_specificity": confidence_result.factors.answer_specificity,
                    "uncertainty_indicators": confidence_result.factors.uncertainty_indicators
                },
                "recommendations": confidence_result.recommendations
            }
            
            # Add numerical verification results if available
            if numerical_verification:
                response["numerical_verification"] = {
                    "total_values": numerical_verification.get('total_numerical_values', 0),
                    "verified_values": numerical_verification.get('verified_values', 0),
                    "hallucinated_values": numerical_verification.get('hallucinated_values', []),
                    "verification_confidence": numerical_verification.get('confidence_score', 1.0)
                }
            
            # Add TF-IDF analysis if available
            if tfidf_analysis and not tfidf_analysis.get("error"):
                response["tfidf_analysis"] = tfidf_analysis
            
            # Add cross-encoder analysis if available
            if cross_encoder_analysis and not cross_encoder_analysis.get("error"):
                response["cross_encoder_analysis"] = cross_encoder_analysis
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "confidence": "error",
                "retrieval_method": "error"
            }
    
    def answer_questions(self, questions: List[str], vector_store, use_hybrid: bool = True,
                        use_cross_encoder: bool = False, use_advanced_hybrid: bool = False,
                        document_text: str = None) -> List[Dict[str, Any]]:
        """
        Answer multiple questions using enhanced RAG with comprehensive accuracy improvements.
        
        Args:
            questions (List[str]): List of questions to answer
            vector_store: The FAISS vector store
            use_hybrid (bool): Whether to use hybrid retrieval with TF-IDF scoring
            use_cross_encoder (bool): Whether to use cross-encoder for more accurate scoring
            use_advanced_hybrid (bool): Whether to use advanced hybrid with all three methods
            document_text (str): Full document text for numerical grounding (optional)
            
        Returns:
            List[Dict[str, Any]]: List of answers with comprehensive metadata
        """
        logger.info(f"Processing {len(questions)} questions with enhanced accuracy systems")
        
        # Initialize document systems once for all questions
        if document_text and not self.numerical_grounding:
            self.initialize_document_systems(document_text)
        
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            answer = self.answer_question(
                question, vector_store, use_hybrid, use_cross_encoder, use_advanced_hybrid, document_text
            )
            answers.append(answer)
        
        logger.info(f"Completed processing {len(questions)} questions with enhanced accuracy")
        return answers

def answer_questions(questions: List[str], vector_store, api_key: str = None, use_hybrid: bool = True,
                    use_cross_encoder: bool = False, use_advanced_hybrid: bool = False,
                    document_text: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to answer questions using the enhanced Policy QA System.
    
    Args:
        questions (List[str]): List of questions to answer
        vector_store: The FAISS vector store
        api_key (str): OpenRouter API key
        use_hybrid (bool): Whether to use hybrid retrieval with TF-IDF scoring
        use_cross_encoder (bool): Whether to use cross-encoder for more accurate scoring
        use_advanced_hybrid (bool): Whether to use advanced hybrid with all three methods
        document_text (str): Full document text for numerical grounding (optional)
        
    Returns:
        List[Dict[str, Any]]: List of answers with comprehensive metadata and verification
    """
    qa_system = PolicyQASystem(api_key)
    return qa_system.answer_questions(questions, vector_store, use_hybrid, use_cross_encoder, use_advanced_hybrid, document_text)