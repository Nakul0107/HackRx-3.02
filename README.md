# HackRx Policy QA System

A high-performance question-answering system for insurance policy documents using OpenRouter LLM and Pinecone vector storage.

## 🚀 Features

- **Fast Response**: Optimized for sub-30 second response times
- **Hybrid Retrieval**: Combines TF-IDF scoring with Pinecone similarity search
- **Cloud Vector Storage**: Pinecone for scalable document embeddings
- **Multiple LLM Support**: OpenRouter integration for various AI models
- **PDF Processing**: Automatic text extraction from PDF documents
- **RESTful API**: FastAPI-based web service

## 🏗️ Architecture

### Core Components

- **`main.py`**: FastAPI application with endpoints
- **`qa.py`**: Policy QA system with LLM integration
- **`pinecone_retriever.py`**: Pinecone vector store manager
- **`tfidf_scorer.py`**: TF-IDF scoring for hybrid retrieval
- **`openrouter_integration.py`**: OpenRouter LLM client
- **`processor.py`**: PDF text extraction

### Technology Stack

- **Backend**: FastAPI (Python)
- **LLM**: OpenRouter (Claude, GPT models)
- **Vector DB**: Pinecone
- **Embeddings**: Sentence Transformers
- **PDF Processing**: PDFPlumber
- **Deployment**: Render (Docker support)

## 🛠️ Setup

### Prerequisites

- Python 3.11+
- OpenRouter API key
- Pinecone API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd HACKRX
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   export PINECONE_API_KEY="your_pinecone_api_key"
   export HACKRX_API_KEY="your_hackrx_api_key"  # Optional
   ```

4. **Run the server**
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Main QA Endpoint
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_openrouter_api_key" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the grace period?",
      "What are the coverage limits?"
    ]
  }'
```

### Response Format
```json
{
  "answers": [
    "The grace period is 30 days from the due date.",
    "Coverage limits are $100,000 per occurrence."
  ]
}
```

## 🚀 Deployment

### Render (Recommended)

1. **Connect repository** to Render
2. **Create Web Service** with:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. **Set environment variables** in Render dashboard
4. **Deploy** automatically

### Docker

```bash
# Build image
docker build -t hackrx-policy-qa .

# Run container
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  hackrx-policy-qa
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM access | Yes |
| `PINECONE_API_KEY` | Pinecone API key for vector storage | Yes |
| `HACKRX_API_KEY` | API key for authentication | No |

### Performance Tuning

- **Chunk Size**: Adjust in `pinecone_retriever.py` (default: 2000)
- **Retrieval Count**: Modify `top_k` in QA system (default: 8)
- **Model Selection**: Change OpenRouter model in `qa.py`

## 📊 Performance

- **Response Time**: < 30 seconds
- **Vector Storage**: Pinecone cloud database
- **Memory Usage**: Optimized for Render free tier (512MB)
- **Scalability**: Horizontal scaling support

## 🔒 Security

- **API Key Protection**: Environment variables only
- **Input Validation**: Pydantic models
- **HTTPS**: Automatic on Render deployment
- **Rate Limiting**: Configurable (not implemented)

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify environment variables are set
   - Check API key permissions

2. **Memory Issues**
   - Reduce chunk size
   - Upgrade Render plan

3. **Timeout Errors**
   - Optimize retrieval parameters
   - Check network connectivity

### Logs

- **Local**: Console output
- **Render**: Dashboard logs
- **Health Check**: `/health` endpoint

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review deployment logs
3. Open an issue on GitHub



 
