# HackRx Policy QA System - Deployment Guide

## Overview
HackRx Policy QA System is a FastAPI-based application that provides intelligent question-answering capabilities for insurance policy documents using OpenRouter LLM and Pinecone vector storage.

## Features
- **FastAPI Web Framework**: High-performance async API
- **OpenRouter Integration**: Access to multiple LLM models
- **Pinecone Vector Storage**: Cloud-based vector database for document embeddings
- **Hybrid Retrieval**: TF-IDF + Pinecone similarity search
- **PDF Processing**: Automatic text extraction from PDF documents
- **Fast Response**: Optimized for sub-30 second response times

## Deployment on Render

### Prerequisites
- Render account
- OpenRouter API key
- Pinecone API key

### Environment Variables
Set these environment variables in your Render dashboard:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
HACKRX_API_KEY=your_hackrx_api_key_here  # Optional
```

### Deployment Steps

1. **Connect Repository**
   - Connect your GitHub repository to Render
   - Select the repository containing this code

2. **Create Web Service**
   - Choose "Web Service" as the service type
   - Set the following configuration:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Environment**: Python 3.11

3. **Set Environment Variables**
   - Add the required environment variables in the Render dashboard
   - Ensure all API keys are properly configured

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Main QA Endpoint
```bash
POST /hackrx/run
Content-Type: application/json
Authorization: Bearer your_openrouter_api_key

{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period?",
    "What are the coverage limits?"
  ]
}
```

#### Document Processing
```bash
POST /hackrx/process
Authorization: Bearer your_hackrx_api_key

{
  "document_url": "https://example.com/policy.pdf"
}
```

### Docker Deployment (Alternative)

If you prefer Docker deployment:

1. **Build Image**
   ```bash
   docker build -t hackrx-policy-qa .
   ```

2. **Run Container**
   ```bash
   docker run -p 8000:8000 \
     -e OPENROUTER_API_KEY=your_key \
     -e PINECONE_API_KEY=your_key \
     hackrx-policy-qa
   ```

### Monitoring and Logs

- **Render Dashboard**: Monitor application health and logs
- **Health Check**: Use `/health` endpoint for monitoring
- **Logs**: Access logs through Render dashboard

### Troubleshooting

#### Common Issues

1. **API Key Errors**
   - Verify all environment variables are set correctly
   - Check API key permissions and quotas

2. **Memory Issues**
   - Render provides up to 512MB RAM for free tier
   - Consider upgrading for larger documents

3. **Timeout Issues**
   - Default timeout is 30 seconds
   - Optimize for faster response times

#### Performance Optimization

- **Vector Caching**: Pinecone provides fast vector retrieval
- **Chunk Optimization**: Adjust chunk size in `pinecone_retriever.py`
- **Model Selection**: Choose appropriate OpenRouter model for your needs

### Security Considerations

- **API Key Protection**: Never commit API keys to version control
- **Input Validation**: All inputs are validated using Pydantic
- **Rate Limiting**: Consider implementing rate limiting for production

### Support

For deployment issues:
1. Check Render logs in the dashboard
2. Verify environment variables
3. Test locally before deploying
4. Monitor application health endpoints 