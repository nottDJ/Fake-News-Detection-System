# OpenAI Integration for Enhanced Fact Checking

## Overview
This update adds OpenAI GPT-powered fact checking to your misinformation detection app, providing intelligent AI analysis alongside traditional web scraping methods.

## Setup Instructions

### 1. Get Your OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Generate a new API key
4. Copy the API key

### 2. Configure the API Key
1. Open `.streamlit/secrets.toml`
2. Replace `your_openai_api_key_here` with your actual API key:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-api-key-here"
   ```

### 3. Restart the App
After adding the API key, restart your Streamlit app for the changes to take effect.

## Features Added

### 🤖 AI-Powered Fact Checking
- **Intelligent Claim Analysis**: Automatically identifies specific factual claims in text
- **Credibility Assessment**: AI evaluates source and claim credibility
- **Misinformation Detection**: Detects potential misinformation patterns
- **Verification Recommendations**: Provides actionable steps for fact verification
- **Red/Green Flags**: Highlights concerning and positive indicators

### 📊 Enhanced Results Display
- **AI Metrics**: Shows AI-generated factual accuracy scores
- **Claim Breakdown**: Lists key claims identified by AI
- **Issue Detection**: Highlights factual problems found
- **Smart Recommendations**: AI-suggested verification steps
- **Comprehensive Assessment**: Detailed AI analysis summary

## How It Works

1. **Text Input**: When you submit text for analysis
2. **Traditional Analysis**: Web scraping and existing fact-checking runs first
3. **AI Analysis**: OpenAI GPT analyzes the same text for additional insights
4. **Integrated Results**: Both traditional and AI results are displayed together
5. **Smart Recommendations**: AI insights are integrated into overall recommendations

## API Usage

The OpenAI integration uses the `gpt-4o-mini` model with:
- **Temperature**: 0.1 (low randomness for consistent fact-checking)
- **Max Tokens**: 1000 (sufficient for detailed analysis)
- **System Prompt**: Expert fact-checker persona
- **Structured Output**: JSON format for easy parsing

## Cost Considerations

- **API Calls**: Each fact-check request makes one OpenAI API call
- **Token Usage**: Approximately 500-1000 tokens per analysis
- **Pricing**: Check [OpenAI Pricing](https://openai.com/pricing) for current rates
- **Budget Management**: Consider setting usage limits in your OpenAI account

## Error Handling

The system gracefully handles:
- Missing API keys
- API rate limits
- Network errors
- JSON parsing failures
- Invalid responses

## Security Notes

- API keys are stored in `.streamlit/secrets.toml` (not committed to git)
- Text is limited to 2000 characters for API calls
- No user data is stored or logged by OpenAI
- API calls are made securely over HTTPS

## Troubleshooting

### API Key Not Working
- Verify the key is correctly copied to `secrets.toml`
- Check if the key has sufficient credits
- Ensure the key has access to GPT-4o-mini

### No AI Analysis Displayed
- Check if the API key is properly configured
- Look for error messages in the app
- Verify the OpenAI package is installed

### Slow Response Times
- AI analysis adds 2-5 seconds to processing
- Consider the complexity of the text being analyzed
- Check your internet connection

## Future Enhancements

Potential improvements:
- Support for GPT-4 model
- Batch processing for multiple claims
- Custom fact-checking prompts
- Integration with other AI models
- Caching of similar analyses

## Support

If you encounter issues:
1. Check the error messages in the app
2. Verify your API key configuration
3. Test with simple text first
4. Check OpenAI service status
