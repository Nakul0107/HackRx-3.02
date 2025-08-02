# HackRx API Deployment Guide

## Pre-Submission Checklist ✅

- [x] API endpoint: `/hackrx/run`
- [x] Request format matches requirements
- [x] Response format matches requirements
- [x] HTTPS enabled (after deployment)
- [x] Handles POST requests
- [x] Returns JSON response
- [x] Response time < 30s
- [x] Tested with sample data

## Deployment Options

### Option 1: Railway (Recommended - Free & Easy)

1. **Sign up** at [railway.app](https://railway.app)
2. **Connect your GitHub repository**:
   - Go to Railway Dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `fxhxdxd/hackrx_chatbot`
3. **Deploy automatically**:
   - Railway will detect the Python app
   - It will install dependencies from `requirements.txt`
   - Use the `Procfile` to run the app
4. **Get your URL**:
   - Railway provides HTTPS URL automatically
   - Format: `https://your-app-name.railway.app`

### Option 2: Render (Free Tier Available)

1. **Sign up** at [render.com](https://render.com)
2. **Create new Web Service**:
   - Connect GitHub repository
   - Select `fxhxdxd/hackrx_chatbot`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. **Deploy** and get HTTPS URL

### Option 3: Heroku (Paid)

1. **Install Heroku CLI**:
   ```bash
   brew install heroku/brew/heroku
   ```
2. **Login and create app**:
   ```bash
   heroku login
   heroku create your-app-name
   ```
3. **Deploy**:
   ```bash
   git push heroku main
   ```

### Option 4: Vercel (Free)

1. **Sign up** at [vercel.com](https://vercel.com)
2. **Import GitHub repository**
3. **Configure** as Python app
4. **Deploy** automatically

## Testing Your Deployed API

### 1. Health Check
```bash
curl https://your-app-url.railway.app/health
```

### 2. Test the Main Endpoint
```bash
curl -X POST https://your-app-url.railway.app/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "documents": "https://example.com/sample.pdf",
    "questions": ["What is this document about?"]
  }'
```

### 3. Using the Test Script
Update `test_api.py` with your deployed URL:
```python
base_url = "https://your-app-url.railway.app"
```

Then run:
```bash
python test_api.py
```

## Expected Response Format

Your API should return:
```json
{
  "answers": [
    "Answer to question 1",
    "Answer to question 2"
  ]
}
```

## Submission Details

Once deployed, your submission should include:

- **Webhook URL**: `https://your-app-url.railway.app/hackrx/run`
- **Description**: "FastAPI + Gemini 2.5 Flash + Enhanced Vector Search with TF-IDF scoring, numerical grounding, and confidence scoring"

## Environment Variables

Make sure to set these in your deployment platform:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `HACKRX_API_KEY`: Your API key for authentication (optional for testing)

## Troubleshooting

### Common Issues:
1. **Timeout errors**: Increase timeout limits in deployment platform
2. **Memory issues**: Upgrade to paid tier if needed
3. **Dependency issues**: Check `requirements.txt` is complete
4. **Port issues**: Use `$PORT` environment variable

### Local Testing:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then test with: `python test_api.py` 