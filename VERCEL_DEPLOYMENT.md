# Vercel Deployment Guide for HackRx

## Prerequisites
- GitHub account with your `hackrx_chatbot` repository
- Vercel account (free)

## Step-by-Step Deployment

### 1. Sign Up for Vercel
1. Go to [vercel.com](https://vercel.com)
2. Click "Sign Up" and choose "Continue with GitHub"
3. Authorize Vercel to access your GitHub account

### 2. Import Your Repository
1. In Vercel Dashboard, click "New Project"
2. Select "Import Git Repository"
3. Find and select `fxhxdxd/hackrx_chatbot`
4. Click "Import"

### 3. Configure Project Settings
1. **Project Name**: `hackrx-chatbot` (or your preferred name)
2. **Framework Preset**: Select "Other" (Vercel will auto-detect Python)
3. **Root Directory**: Leave as `./` (default)
4. **Build Command**: Leave empty (Vercel will auto-detect)
5. **Output Directory**: Leave empty (Vercel will auto-detect)
6. **Install Command**: Leave empty (Vercel will use `pip install -r requirements.txt`)

### 4. Set Environment Variables
Click "Environment Variables" and add:
- **Name**: `OPENROUTER_API_KEY`
- **Value**: Your OpenRouter API key
- **Environment**: Production, Preview, Development (select all)

### 5. Deploy
1. Click "Deploy"
2. Wait for build to complete (usually 2-3 minutes)
3. Your app will be live at: `https://your-project-name.vercel.app`

## Testing Your Deployed API

### 1. Health Check
```bash
curl https://your-project-name.vercel.app/health
```

### 2. Test Main Endpoint
```bash
curl -X POST https://your-project-name.vercel.app/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "documents": "https://example.com/sample.pdf",
    "questions": ["What is this document about?"]
  }'
```

### 3. Update Test Script
Edit `test_api.py`:
```python
base_url = "https://your-project-name.vercel.app"
```

## Vercel-Specific Features

### Automatic Deployments
- Every push to `main` branch triggers automatic deployment
- Preview deployments for pull requests
- Easy rollback to previous versions

### Custom Domain (Optional)
1. Go to Project Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed

### Environment Variables Management
- Set different values for Production/Preview/Development
- Secure storage of API keys
- Easy updates through Vercel dashboard

## Troubleshooting

### Common Issues:

1. **Build Failures**
   - Check `requirements.txt` for compatibility
   - Ensure all dependencies are listed
   - Check Vercel build logs for errors

2. **Timeout Issues**
   - Vercel has 30-second timeout for serverless functions
   - Your API is optimized for this limit
   - Consider upgrading to Pro plan for longer timeouts

3. **Memory Issues**
   - Vercel provides 1024MB RAM by default
   - Sufficient for your application
   - Upgrade to Pro plan for more memory if needed

4. **Cold Start Delays**
   - First request might be slower
   - Subsequent requests will be faster
   - Consider using Vercel Pro for better performance

### Performance Optimization:
- Your app is optimized for Vercel's serverless environment
- FastAPI provides excellent performance
- Vector operations are optimized for memory constraints

## Submission URL Format

Once deployed, your submission URL will be:
```
https://your-project-name.vercel.app/hackrx/run
```

## Monitoring

- **Vercel Analytics**: Built-in performance monitoring
- **Function Logs**: View execution logs in Vercel dashboard
- **Error Tracking**: Automatic error reporting

## Cost
- **Free Tier**: 100GB bandwidth/month, 100 serverless function executions/day
- **Pro Plan**: $20/month for unlimited usage
- **Enterprise**: Custom pricing for large-scale deployments 