# Enhanced Policy QA System

A high-accuracy LLM-powered Policy Question Answering System with advanced retrieval and verification capabilities.

## Features

- **Advanced Hybrid Retrieval**: Combines TF-IDF, embedding similarity, and cross-encoder reranking
- **Numerical Grounding**: Prevents hallucination by verifying numerical values against source documents
- **Completeness Checking**: Ensures comprehensive answers with all relevant policy details
- **Confidence Scoring**: Multi-factor confidence assessment with uncertainty expression
- **Enhanced Context**: 50% overlapping chunks for better context continuity

## Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API key

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual API keys
# Set your OpenRouter API key and other configuration
```

Or set environment variables directly:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### Running the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Usage

**Endpoint**: `POST /hackrx/run`

**Request**:
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

**Response**:
```json
{
  "answers": [
    {
      "question": "What is the grace period for premium payment?",
      "answer": "The grace period for premium payment is 30 days...",
      "confidence": "high",
      "confidence_score": 0.85,
      "retrieval_method": "advanced_hybrid",
      "completeness": {
        "is_complete": true,
        "completeness_score": 0.9
      },
      "numerical_verification": {
        "verified_values": 1,
        "total_values": 1,
        "hallucinated_values": []
      }
    }
  ],
  "status": "success",
  "message": "Successfully processed 2 questions"
}
```

## Core Components

- **`main.py`**: FastAPI server and main application
- **`qa.py`**: Enhanced QA system with accuracy improvements
- **`retriever.py`**: Advanced hybrid retrieval system
- **`numerical_grounding.py`**: Numerical value verification
- **`completeness_checker.py`**: Answer completeness analysis
- **`confidence_scorer.py`**: Multi-factor confidence scoring
- **`processor.py`**: PDF text extraction
- **`openrouter_integration.py`**: LLM integration

## Configuration

The system uses OpenAI GPT-3.5-turbo via OpenRouter by default. You can modify the model in `qa.py`:

```python
self.llm = OpenRouterChatModel(
    api_key=self.api_key,
    model="openai/gpt-3.5-turbo",  # Change model here
    temperature=0.1,
    max_tokens=2048
)
```



 
