import os
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from pinecone_retriever import PineconeVectorStoreManager
from openrouter_integration import OpenRouterChatModel

logger = logging.getLogger(__name__)

class PolicyQASystem:
    def __init__(self, api_key: str = None):
        """
        Initialize the Policy QA System with OpenRouter API and simplified scoring.
        
        Args:
            api_key (str): OpenRouter API key
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        # Initialize OpenRouter
        self.llm = OpenRouterChatModel(
            api_key=self.api_key,
            model="openai/gpt-3.5-turbo",  # Using a more stable model
            temperature=0.1,
            max_tokens=512  # Reduced for more concise responses
        )
        
        # Simplified prompt template for policy QA
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a precise summarizer for insurance policy Q&A. Return ONLY a concise answer.
            
            CRITICAL INSTRUCTIONS:
            - Return answers in 1-2 sentences maximum (30 words max)
            - Keep only essential details: numbers, waiting periods, limits, and yes/no where applicable
            - Remove all unnecessary explanations, exclusions, definitions, and disclaimers
            - If numeric values or limits are specified in the policy, include them exactly (e.g., "30 days", "36 months", "1% of Sum Insured")
            - Provide CONCISE, factual answers based ONLY on the context provided
            - NEVER invent or hallucinate numerical values, dates, or amounts not in the context
            
            ANSWER FORMAT:
            - Lead with the direct answer to the question
            - Include only the most relevant conditions and limitations
            - Use clear, direct language
            - Focus on actionable information
            - If information is not available in the context, state: "Not specified in the policy"
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Initialize vector store manager for hybrid retrieval
        # Use the same Pinecone API key that was used for uploading
        pinecone_api_key = api_key or os.getenv("PINECONE_API_KEY")
        # Don't create a new vector manager here - it will be set from main.py
        self.vector_manager = None
    
    def answer_question(self, question: str, vector_store, use_hybrid: bool = True,
                       document_text: str = None) -> Dict[str, Any]:
        """
        Answer a single question using simplified RAG for faster response times.
        
        Args:
            question (str): The question to answer
            vector_store: The FAISS vector store
            use_hybrid (bool): Whether to use hybrid retrieval with TF-IDF scoring
            document_text (str): Full document text (optional)
            
        Returns:
            Dict[str, Any]: Answer with basic metadata
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Initialize variables
            source_docs = []
            
            # Use simplified Pinecone-only retrieval with fewer documents for speed
            logger.info("Using optimized Pinecone retrieval")
            source_docs = self.vector_manager._get_pinecone_only_retrieval(question, top_k=4)  # Reduced from 8 to 4
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in source_docs])
            
            # Use LLM directly with context
            prompt = self.prompt_template.format(context=context, question=question)
            result = self.llm.invoke(prompt)
            answer = result.content
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            
            # Build simplified response
            response = {
                "question": question,
                "answer": answer,
                "sources": [doc.page_content[:200] + "..." for doc in source_docs],
                "retrieval_method": 'hybrid_tfidf' if use_hybrid else 'pinecone_only',
                "num_sources": len(source_docs)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "retrieval_method": "error"
            }
    
    def answer_questions(self, questions: List[str], vector_store, use_hybrid: bool = True,
                        document_text: str = None) -> List[Dict[str, Any]]:
        """
        Answer multiple questions using parallel processing for faster response times.
        
        Args:
            questions (List[str]): List of questions to answer
            vector_store: The FAISS vector store
            use_hybrid (bool): Whether to use hybrid retrieval with TF-IDF scoring
            document_text (str): Full document text (optional)
            
        Returns:
            List[Dict[str, Any]]: List of answers with basic metadata
        """
        logger.info(f"Processing {len(questions)} questions with parallel processing")
        
        import concurrent.futures
        import threading
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all questions for parallel processing
            future_to_question = {
                executor.submit(self.answer_question, question, vector_store, use_hybrid, document_text): question 
                for question in questions
            }
            
            answers = []
            for future in concurrent.futures.as_completed(future_to_question):
                try:
                    answer = future.result()
                    answers.append(answer)
                except Exception as e:
                    question = future_to_question[future]
                    logger.error(f"Error processing question '{question}': {e}")
                    answers.append({
                        "question": question,
                        "answer": f"Error processing question: {str(e)}",
                        "sources": [],
                        "retrieval_method": "error"
                    })
        
        logger.info(f"Completed processing {len(questions)} questions in parallel")
        return answers

def answer_questions(questions: List[str], vector_store, api_key: str = None, use_hybrid: bool = True,
                    document_text: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to answer questions using the simplified Policy QA System.
    
    Args:
        questions (List[str]): List of questions to answer
        vector_store: The FAISS vector store
        api_key (str): OpenRouter API key
        use_hybrid (bool): Whether to use hybrid retrieval with TF-IDF scoring
        document_text (str): Full document text (optional)
        
    Returns:
        List[Dict[str, Any]]: List of answers with basic metadata
    """
    qa_system = PolicyQASystem(api_key)
    return qa_system.answer_questions(questions, vector_store, use_hybrid, document_text)