# 🚀 Vercel Deployment Checklist

## ✅ Pre-Deployment Checklist

### 1. Repository Setup
- [ ] Code is in a GitHub repository
- [ ] All required files are present:
  - [ ] `app_vercel.py` (Vercel-optimized Flask app)
  - [ ] `vercel.json` (Vercel configuration)
  - [ ] `requirements.txt` (Python dependencies)
  - [ ] `runtime.txt` (Python version)
  - [ ] `api/index.py` (Serverless function entry point)
  - [ ] `templates/index.html` (Frontend template)
  - [ ] `speech_to_text.py` (Audio transcription module)

### 2. Environment Variables
- [ ] OpenAI API key is ready
- [ ] Optional: Google News API key (for enhanced fact-checking)
- [ ] Optional: Google SERP API key (for enhanced fact-checking)

### 3. Code Review
- [ ] No hardcoded API keys in the code
- [ ] Image analysis functions are removed (not supported on Vercel)
- [ ] File size limits are respected (4MB max)
- [ ] Processing time is optimized (10 seconds max)

## 🔧 Deployment Steps

### Step 1: Connect to Vercel
1. [ ] Go to [vercel.com](https://vercel.com)
2. [ ] Sign in with GitHub
3. [ ] Click "New Project"
4. [ ] Import your GitHub repository
5. [ ] Select the repository

### Step 2: Configure Environment Variables
1. [ ] Go to Project Settings → Environment Variables
2. [ ] Add the following variables:
   ```
   OPENAI_API_KEY = your_actual_openai_api_key_here
   OPENAI_MODEL = gpt-4o-mini
   MODEL_WEIGHT = 0.40
   GOOGLE_NEWS_API_KEY = your_google_news_api_key_here (optional)
   GOOGLE_SERP_API_KEY = your_google_serp_api_key_here (optional)
   ```

### Step 3: Deploy
1. [ ] Click "Deploy"
2. [ ] Wait for build to complete
3. [ ] Check for any build errors
4. [ ] Note the deployment URL

## 🧪 Post-Deployment Testing

### 1. Basic Functionality
- [ ] Homepage loads correctly
- [ ] Text analysis works
- [ ] Audio transcription works
- [ ] Document analysis works
- [ ] API endpoints respond correctly

### 2. API Testing
- [ ] Test `/api/health` endpoint
- [ ] Test `/api/config` endpoint
- [ ] Test `/api/verify-news` with text input
- [ ] Test `/api/verify-news` with file upload

### 3. Enhanced Features
- [ ] Enhanced fact-checking works (when OpenAI returns FALSE/UNCERTAIN)
- [ ] Wikipedia integration works
- [ ] Google News integration works (if configured)
- [ ] Google Search integration works (if configured)

## ⚠️ Known Limitations

### What Won't Work on Vercel
- [ ] Image analysis (OCR) - Tesseract not supported
- [ ] Large file uploads (>4MB)
- [ ] Long processing times (>10 seconds)
- [ ] System-level dependencies

### What Works on Vercel
- [ ] Text analysis and fact-checking
- [ ] Audio transcription and analysis
- [ ] Document analysis (basic)
- [ ] Enhanced fact-checking with APIs
- [ ] All API endpoints
- [ ] Beautiful UI and responsive design

## 🔍 Troubleshooting

### Common Issues
1. **Build Fails**
   - [ ] Check `requirements.txt` has all dependencies
   - [ ] Verify Python version in `runtime.txt`
   - [ ] Check for syntax errors in code

2. **Environment Variables Not Working**
   - [ ] Verify variable names are correct (case-sensitive)
   - [ ] Redeploy after adding environment variables
   - [ ] Check Vercel dashboard for variable status

3. **API Errors**
   - [ ] Verify API keys are valid
   - [ ] Check API quotas and limits
   - [ ] Test API keys locally first

4. **Function Timeout**
   - [ ] Optimize code for faster processing
   - [ ] Reduce file sizes
   - [ ] Consider Vercel Pro for longer timeouts

## 📊 Monitoring

### Vercel Dashboard
- [ ] Check Function Logs for errors
- [ ] Monitor API usage and costs
- [ ] Review performance metrics
- [ ] Set up error alerts

### Health Monitoring
- [ ] Set up health check monitoring
- [ ] Monitor API response times
- [ ] Track error rates
- [ ] Monitor user feedback

## 🔄 Updates and Maintenance

### Regular Tasks
- [ ] Monitor API usage and costs
- [ ] Update dependencies as needed
- [ ] Review and rotate API keys
- [ ] Monitor for security updates
- [ ] Backup environment variables

### Deployment Updates
- [ ] Push changes to GitHub
- [ ] Vercel will auto-deploy
- [ ] Test new deployment
- [ ] Monitor for issues
- [ ] Rollback if needed

## 📞 Support Resources

### Vercel Support
- [ ] [Vercel Documentation](https://vercel.com/docs)
- [ ] [Vercel Community](https://github.com/vercel/vercel/discussions)
- [ ] [Vercel Status](https://vercel-status.com)

### Project Support
- [ ] Check main README.md
- [ ] Review README_VERCEL.md
- [ ] Check GitHub issues
- [ ] Contact project maintainer

## 🎉 Success Criteria

Your deployment is successful when:
- [ ] App is accessible at the Vercel URL
- [ ] All core features work (text, audio, document analysis)
- [ ] Enhanced fact-checking works
- [ ] API endpoints respond correctly
- [ ] No critical errors in logs
- [ ] Performance is acceptable

---

**Deployment URL:** `https://your-project-name.vercel.app`

**Last Updated:** [Date]
**Deployed By:** [Your Name]

