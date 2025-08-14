# 🔍 AI Fake News Detector

A powerful Flask web application that uses OpenAI GPT-4 for accurate fact-checking and **local OCR for image analysis**. Analyze news articles, social media posts, claims, audio recordings, and images for authenticity without OpenAI quota concerns for images.

## ✨ Features

- 🤖 **AI-Powered Analysis**: Uses OpenAI GPT-4 for accurate fact-checking
- 🔍 **Enhanced Fact-Checking**: When OpenAI returns FALSE/UNCERTAIN, automatically searches Wikipedia, Google News, and Google Search for verification
- 📊 **Credibility Scoring**: Provides confidence and credibility scores (0-100%)
- 🎯 **Clear Verdicts**: Classifies content as TRUE, FALSE, or UNCERTAIN
- 📝 **Detailed Analysis**: Shows reasoning, fact-check notes, and risk factors
- 📚 **Multiple Sources**: Cross-references claims with Wikipedia, news articles, and web search results
- 🎨 **Beautiful UI**: Modern, responsive design with intuitive user experience
- 📱 **Mobile-Friendly**: Works perfectly on all devices
- ⚡ **Real-time Processing**: Instant analysis with loading indicators
- 🎵 **Audio Support**: Upload and transcribe audio files for analysis
- 🖼️ **Local Image Analysis**: Extract text from images using local OCR (no OpenAI quota needed!)
- 📄 **Document Support**: Upload text documents for fact-checking
- 🚨 **Enhanced Recent Event Detection**: Special handling for recent incidents like aircraft crashes, breaking news, and current events

## 🚀 How It Works

1. **Input Options**: Users can input content in multiple ways:
   - **Text**: Paste news articles, social media posts, or claims
   - **Audio**: Upload audio files (MP3, WAV, M4A, AAC, OGG) for transcription and analysis
   - **Images**: Upload images (PNG, JPG, GIF, BMP) for **local OCR text extraction**
   - **Documents**: Upload text files (TXT, PDF, DOC, DOCX) for analysis

2. **AI Analysis**: OpenAI GPT-4 analyzes the content using advanced fact-checking techniques:
   - For text: Direct analysis of the content
   - For audio: Automatic transcription followed by analysis
   - For images: **Local OCR text extraction → AI analysis of extracted text**
   - For documents: Text extraction followed by analysis

3. **Enhanced Verification**: If OpenAI returns FALSE or UNCERTAIN:
   - **Wikipedia Search**: Searches for relevant articles and information with enhanced recent event support
   - **Google News**: Finds recent news articles about the claim (crucial for recent events)
   - **Google Search**: Searches the web for additional sources
   - **Cross-references**: Compares findings across multiple sources

4. **Results**: The system provides:
   - Clear verdict (TRUE/FALSE/UNCERTAIN)
   - Confidence score (how sure the AI is)
   - Credibility score (overall trustworthiness)
   - Detailed reasoning and analysis
   - Fact-check notes and risk factors
   - Input type identification
   - Extracted content summary
   - **Enhanced analysis results** (when applicable)
   - **Multiple source references** (Wikipedia, News, Web search)

## 🔍 **NEW: Enhanced Recent Event Detection**

### **Recent Event Detection Features**
- ✅ **Aviation Incidents**: Special handling for aircraft crashes, emergency landings, and aviation accidents
- ✅ **Breaking News**: Enhanced detection for recent events and current incidents
- ✅ **Location-Specific Queries**: Automatic generation of location-based search queries
- ✅ **Date-Specific Searches**: Intelligent date detection and search variations
- ✅ **Comprehensive Coverage**: Searches across multiple years (2023-2025) for recent events

### **How Recent Event Detection Works**
1. **Event Type Detection**: Identifies aviation incidents, breaking news, and recent events
2. **Location Extraction**: Extracts location names (e.g., "Ahmedabad", "Gujarat", "India")
3. **Query Generation**: Creates comprehensive search queries with:
   - Location-specific variations
   - Aviation-specific terms
   - Date-specific searches
   - Recent event indicators
4. **Multi-Source Search**: Searches Wikipedia, Google News, and Google Search with enhanced queries
5. **Comprehensive Results**: Provides detailed verification from multiple sources

### **Example: Ahmedabad Aircraft Crash**
When you input: *"Aircraft crash at Ahmedabad airport in 2024"*

The system will:
- Detect it as an aviation incident
- Extract "Ahmedabad" as the location
- Generate queries like:
  - "Ahmedabad aircraft crash"
  - "Ahmedabad aviation incident 2024"
  - "Ahmedabad airport emergency landing"
  - "Ahmedabad flight incident"
- Search Wikipedia for aviation accident pages
- Search Google News for recent articles
- Search Google for web verification
- Provide comprehensive fact-checking results

## 🔍 **Enhanced Fact-Checking with Multiple Sources**

### **When Enhanced Analysis Triggers**
- ✅ **OpenAI returns FALSE**: Automatically searches additional sources for verification
- ✅ **OpenAI returns UNCERTAIN**: Searches for supporting or contradicting evidence
- ✅ **OpenAI returns TRUE**: Uses OpenAI analysis only (no additional API calls)

### **How Enhanced Analysis Works**
1. **Initial Analysis**: OpenAI GPT-4 analyzes the content
2. **Smart Query Generation**: Extracts key claims and entities for targeted searches
3. **Multi-Source Search**: Simultaneously queries Wikipedia, Google News, and Google Search
4. **Cross-Reference**: Compares findings across all sources
5. **Enhanced Results**: Provides comprehensive analysis with source links

### **Benefits**
- 🔍 **More Accurate**: Multiple sources provide better verification
- 📚 **Comprehensive**: Wikipedia for factual information, News for recent events, Web for current context
- 💰 **Cost-Effective**: Only uses additional APIs when needed
- ⚡ **Fast**: Parallel API calls for quick results

## 🖼️ **Local Image Processing**

### **Why Local OCR?**
- ✅ **No OpenAI quota needed** for images
- ✅ **Faster processing** (no API calls)
- ✅ **Privacy-focused** (images stay on your computer)
- ✅ **Cost-effective** (completely free for image processing)

### **How It Works**
1. Upload image → Local OCR extracts text → Send text to OpenAI → Get fake news analysis
2. **No images are sent to OpenAI** - only extracted text
3. Advanced image preprocessing with OpenCV for better accuracy
4. Multiple OCR attempts for optimal results

## 🛠️ Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR (Required for image analysis):**
   - **Windows**: Download from [Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`
   
   See `TESSERACT_INSTALL.md` for detailed installation instructions.

3. **Configure your API keys:**
   Edit `config.env` file:
   ```env
   OPENAI_API_KEY=your_actual_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   MODEL_WEIGHT=0.40
   
   # Optional: For enhanced fact-checking (when OpenAI returns FALSE/UNCERTAIN)
   GOOGLE_NEWS_API_KEY=your_google_news_api_key_here
   GOOGLE_SERP_API_KEY=your_google_serp_api_key_here
   ```
   
   **API Key Sources:**
   - **OpenAI**: [https://platform.openai.com/](https://platform.openai.com/)
   - **Google News**: [https://newsapi.org/](https://newsapi.org/) (free tier available - **CRUCIAL for recent events**)
   - **Google SERP**: [https://serpapi.com/](https://serpapi.com/) (free tier available)
   - **Wikipedia**: No API key needed (free)

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:5000`

## 🔧 API Endpoints

- `GET /` - Main fake news detector interface
- `POST /api/verify-news` - Analyze content (text, audio, images, documents) and return results
- `GET /api/config` - Get current configuration
- `GET /api/health` - Health check endpoint

## 📊 Analysis Features

The AI analyzes content based on multiple factors:

- **Source credibility indicators**
- **Factual consistency**
- **Emotional language usage**
- **Claim specificity**
- **Supporting evidence mentioned**
- **Common fake news patterns**
- **Context and reliability of the input source**
- **Recent event detection and verification**

## 🎯 Use Cases

- **Journalists**: Fact-check articles before publishing
- **Social Media Users**: Verify viral posts, images, and audio claims
- **Students**: Research and verify information from various sources
- **General Public**: Stay informed with verified news and content
- **Content Creators**: Ensure accuracy in their multimedia content
- **Researchers**: Analyze claims from different media formats
- **Emergency Responders**: Verify breaking news and recent incidents
- **Aviation Professionals**: Fact-check aviation incidents and accidents

## 📁 Project Structure

```
gobaldhapool/
├── app.py              # Flask application with enhanced recent event detection
├── config.env          # OpenAI API configuration
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Enhanced fake news detector interface
├── test_recent_events.py # Test script for recent events
├── test_enhanced_ocr.py # Test script for enhanced OCR
├── test_basic_functionality.py # Basic functionality tests
├── test_enhanced_fact_checking.py # Enhanced fact-checking tests
├── test_multimedia.py # Multimedia support tests
├── test_ocr.py        # OCR functionality tests
├── TESSERACT_INSTALL.md # Tesseract OCR installation guide
├── QUOTA_SOLUTIONS.md  # OpenAI quota solutions guide
└── README.md          # This file
```

## 📋 Requirements

- Python 3.7+
- Flask 3.0.3
- python-dotenv 1.0.1
- requests 2.32.3
- openai 0.28.1
- Pillow 10.0.1 (for image processing)
- SpeechRecognition 3.10.0 (for audio transcription)
- pydub 0.25.1 (for audio format conversion)
- python-multipart 0.0.6 (for file uploads)
- pytesseract 0.3.10 (for OCR text extraction)
- opencv-python 4.8.1.78 (for image preprocessing)
- beautifulsoup4 4.12.2 (for web scraping)
- wikipedia-api 0.6.0 (for Wikipedia integration)
- **Tesseract OCR** (system dependency - see installation guide)

## 🔑 API Requirements

This application uses multiple APIs for comprehensive fact-checking:

### Required APIs:
- **OpenAI API**: For initial content analysis. Get your key at [https://platform.openai.com/](https://platform.openai.com/)

### Optional APIs (for enhanced fact-checking):
- **Google News API**: For recent news articles. **FREE tier available** at [https://newsapi.org/](https://newsapi.org/) - **CRUCIAL for recent events**
- **Google SERP API**: For web search results. **FREE tier available** at [https://serpapi.com/](https://serpapi.com/)
- **Wikipedia API**: Free, no key required

**Note**: 
- **Text, Audio, and Document analysis**: Uses OpenAI API (incurs costs)
- **Image analysis**: Uses local OCR (completely free, no OpenAI quota needed)
- **Enhanced verification**: Only triggered when OpenAI returns FALSE/UNCERTAIN
- **Recent events**: Google News API is crucial for finding recent articles about current incidents

## 🚨 Important Notes

- The AI analysis is based on available information and may not catch all fake news
- Audio transcription accuracy depends on audio quality and clarity
- **Image analysis works best with clear, readable text and good image quality**
- **Recent events require Google News API for optimal verification**
- Always verify critical information from multiple sources
- This tool is meant to assist, not replace, critical thinking
- Results should be used as guidance, not absolute truth

## 📱 Example Usage

### Text Analysis
1. Copy a news article or claim
2. Paste it into the text area
3. Click "🔍 Analyze Content"

### Audio Analysis
1. Record or upload an audio file
2. Select the audio file in the file upload tab
3. Click "🔍 Analyze Content"
4. Wait for transcription and analysis

### Image Analysis (NEW: Local OCR)
1. Take a screenshot or upload an image with text
2. Select the image file in the file upload tab
3. Click "🔍 Analyze Content"
4. **Local OCR extracts text → AI analyzes the text for fake news**

### Document Analysis
1. Upload a text document (TXT, PDF, DOC, DOCX)
2. Select the document in the file upload tab
3. Click "🔍 Analyze Content"
4. Wait for text extraction and analysis

### Recent Event Analysis
1. Input recent event details (e.g., "Aircraft crash at Ahmedabad airport")
2. The system will automatically detect it as a recent event
3. Enhanced analysis will search for recent news articles and verification
4. Get comprehensive fact-checking results from multiple sources

## 🎨 UI Features

- **Tabbed Interface**: Easy switching between text input and file upload
- **Drag & Drop**: Intuitive file upload with visual feedback
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Loading Animations**: Visual feedback during analysis
- **Color-coded Results**: Green for true, red for false, orange for uncertain
- **Input Type Badge**: Shows the type of content analyzed
- **Local Processing Indicator**: Shows when images are processed locally
- **Smooth Scrolling**: Automatic navigation to results
- **Error Handling**: Clear error messages for troubleshooting
- **Enhanced Analysis Display**: Shows multiple source results for recent events

## 🔒 Privacy & Security

- **Text, Audio, Documents**: Content is sent to OpenAI for analysis
- **Images**: Processed locally with OCR - never sent to OpenAI
- No data is stored permanently on the server
- API keys are kept secure in environment variables
- HTTPS recommended for production use
- File uploads are processed temporarily and not stored

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using OpenAI GPT-4, Flask, and Local OCR**
