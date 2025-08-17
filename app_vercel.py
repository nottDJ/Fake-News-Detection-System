import os
import json
import tempfile
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import openai
import speech_recognition as sr
from pydub import AudioSegment
import requests
import wikipediaapi
from bs4 import BeautifulSoup
import urllib.parse
from speech_to_text import transcribe_audio_file

# Load environment variables - support both .env file and Vercel environment variables
load_dotenv('config.env')

app = Flask(__name__)

# Configuration - support Vercel environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-REPLACE_ME')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

def get_float_env(var_name, default):
    try:
        value = os.getenv(var_name, str(default))
        return float(value) if value else float(default)
    except ValueError:
        return float(default)

MODEL_WEIGHT = get_float_env('MODEL_WEIGHT', 0.40)

# Additional API Configuration for Enhanced Fact-Checking
GOOGLE_NEWS_API_KEY = os.getenv('GOOGLE_NEWS_API_KEY', '')
GOOGLE_SERP_API_KEY = os.getenv('GOOGLE_SERP_API_KEY', '')

# Initialize Wikipedia API
wiki_api = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyFakeNewsApp/1.0 (meshwar256@gmail.com)'
)

# Configure OpenAI
if OPENAI_API_KEY != 'sk-REPLACE_ME':
    openai.api_key = OPENAI_API_KEY

# Supported file types for Vercel (removed image types due to OCR limitations)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'mp3', 'wav', 'm4a', 'aac', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(audio_file):
    """Transcribe audio file to text using external speech_to_text module."""
    try:
        return transcribe_audio_file(audio_file)
    except Exception as e:
        raise Exception(f"Audio transcription failed: {str(e)}")

def extract_text_from_file(file):
    """Extract text from various file types"""
    try:
        filename = file.filename.lower()
        
        if filename.endswith('.txt'):
            return file.read().decode('utf-8')
        elif filename.endswith(('.pdf', '.doc', '.docx')):
            # For now, return a placeholder - you'd need additional libraries for these
            return f"[File content from {filename} - text extraction not implemented for this file type]"
        else:
            return file.read().decode('utf-8')
    except Exception as e:
        raise Exception(f"File text extraction failed: {str(e)}")

def search_wikipedia(query, max_results=5):
    """Search Wikipedia for relevant information with enhanced recent event support"""
    try:
        results = []
        
        # 1. Try direct page search first
        direct_page = wiki_api.page(query)
        if direct_page.exists():
            content = direct_page.summary
            results.append({
                'title': direct_page.title,
                'summary': content[:1000] + "..." if len(content) > 1000 else content,
                'url': direct_page.fullurl,
                'source': 'Wikipedia',
                'search_type': 'direct_match'
            })
        
        # 2. Try broader search for related pages
        search_pages = wiki_api.search(query, results=max_results)
        for page in search_pages:
            if page.exists() and len(results) < max_results:
                # Check if this page is not already in results
                if not any(r['title'] == page.title for r in results):
                    content = page.summary
                    results.append({
                        'title': page.title,
                        'summary': content[:800] + "..." if len(content) > 800 else content,
                        'url': page.fullurl,
                        'source': 'Wikipedia',
                        'search_type': 'related_search'
                    })
        
        return results if results else None
        
    except Exception as e:
        print(f"Wikipedia search error: {e}")
        return None

def search_google_news(query, max_results=5):
    """Search Google News for recent articles with enhanced recent event support"""
    try:
        if not GOOGLE_NEWS_API_KEY:
            return None
        
        results = []
        
        # 1. Search for recent articles with broader date range
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': GOOGLE_NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',  # Sort by date for recent events
            'pageSize': max_results,
            'from': '2023-01-01'  # Include articles from 2023 onwards for better recent event coverage
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get('articles', [])
        
        for article in articles:
            results.append({
                'title': article.get('title', ''),
                'description': article.get('description', '')[:400] + "..." if len(article.get('description', '')) > 400 else article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'publishedAt': article.get('publishedAt', ''),
                'source_type': 'Google News',
                'search_type': 'recent_articles'
            })
        
        return results
        
    except Exception as e:
        print(f"Google News search error: {e}")
        return None

def search_google_serp(query, max_results=5):
    """Search Google using SERP API for web results"""
    try:
        if not GOOGLE_SERP_API_KEY:
            return None
            
        url = "https://serpapi.com/search"
        params = {
            'q': query,
            'api_key': GOOGLE_SERP_API_KEY,
            'num': max_results,
            'gl': 'us',
            'hl': 'en'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        organic_results = data.get('organic_results', [])
        
        results = []
        for result in organic_results:
            results.append({
                'title': result.get('title', ''),
                'snippet': result.get('snippet', '')[:300] + "..." if len(result.get('snippet', '')) > 300 else result.get('snippet', ''),
                'url': result.get('link', ''),
                'source': result.get('displayed_link', 'Unknown'),
                'source_type': 'Google Search'
            })
        
        return results
    except Exception as e:
        return None

def extract_key_claims(text):
    """Extract key claims and entities from text for better search queries with enhanced recent event support"""
    try:
        # Simple entity extraction - look for capitalized phrases and numbers
        claims = []
        
        # Extract sentences with numbers (potential statistics)
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if re.search(r'\d+', sentence) and len(sentence.strip()) > 20:
                claims.append(sentence.strip())
        
        # Extract capitalized phrases (potential names, places, organizations)
        words = text.split()
        entities = []
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Look for multi-word entities
                entity = word
                j = i + 1
                while j < len(words) and words[j][0].isupper() and len(words[j]) > 2:
                    entity += " " + words[j]
                    j += 1
                if len(entity.split()) <= 4:  # Increased limit for recent events
                    entities.append(entity)
        
        # Create search queries
        search_queries = []
        
        # Use the first few words as a general query
        first_words = " ".join(text.split()[:12])
        search_queries.append(first_words)
        
        # Add entity-based queries
        for entity in entities[:4]:  # Increased limit for recent events
            search_queries.append(entity)
        
        # Add claim-based queries
        for claim in claims[:3]:  # Increased limit for recent events
            search_queries.append(claim[:60])  # Increased claim length
        
        # Enhanced recent event detection and query generation
        text_lower = text.lower()
        
        # Check for recent event indicators
        recent_indicators = ['recent', 'latest', 'today', 'yesterday', 'this week', 'this month', '2024', '2023', '2025', 'breaking', 'just', 'now']
        if any(indicator in text_lower for indicator in recent_indicators):
            # Add recent event variations
            for entity in entities[:2]:
                search_queries.extend([
                    f"{entity} 2024",
                    f"{entity} 2023",
                    f"{entity} 2025",
                    f"{entity} recent",
                    f"{entity} latest",
                    f"{entity} breaking news"
                ])
        
        # Enhanced incident/accident detection
        incident_indicators = ['incident', 'accident', 'crash', 'emergency', 'disaster', 'event', 'flight', 'plane', 'aircraft', 'aviation', 'airport', 'runway']
        if any(indicator in text_lower for indicator in incident_indicators):
            # Add comprehensive incident-specific queries
            for entity in entities[:2]:
                search_queries.extend([
                    f"{entity} incident",
                    f"{entity} accident",
                    f"{entity} crash",
                    f"{entity} emergency landing",
                    f"{entity} flight incident",
                    f"{entity} plane incident",
                    f"{entity} aircraft incident",
                    f"{entity} aviation accident",
                    f"{entity} runway incident",
                    f"{entity} airport incident"
                ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in search_queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        return unique_queries[:10]  # Return max 10 queries for better coverage
    except Exception as e:
        return [text[:100]]  # Fallback to first 100 characters

def analyze_with_additional_apis(content):
    """Analyze content using additional APIs when OpenAI returns FALSE/UNCERTAIN with enhanced recent event support"""
    try:
        # Extract key claims and create search queries
        search_queries = extract_key_claims(content)
        
        all_results = {
            'wikipedia': [],
            'google_news': [],
            'google_search': [],
            'summary': ''
        }
        
        # Search each API with the best queries (increased coverage)
        for query in search_queries[:8]:  # Use top 8 queries for better coverage
            # Wikipedia search
            wiki_result = search_wikipedia(query)
            if wiki_result:
                if isinstance(wiki_result, list):
                    all_results['wikipedia'].extend(wiki_result)
                else:
                    all_results['wikipedia'].append(wiki_result)
            
            # Google News search
            news_result = search_google_news(query)
            if news_result:
                all_results['google_news'].extend(news_result)
            
            # Google Search
            search_result = search_google_serp(query)
            if search_result:
                all_results['google_search'].extend(search_result)
        
        # Remove duplicates and limit results (increased limits for better coverage)
        all_results['wikipedia'] = all_results['wikipedia'][:5]  # Increased from 3 to 5
        all_results['google_news'] = all_results['google_news'][:5]  # Increased from 3 to 5
        all_results['google_search'] = all_results['google_search'][:5]  # Increased from 3 to 5
        
        # Create a comprehensive summary of findings
        summary_parts = []
        
        if all_results['wikipedia']:
            summary_parts.append(f"📚 Wikipedia: Found {len(all_results['wikipedia'])} relevant articles")
        
        if all_results['google_news']:
            summary_parts.append(f"📰 Google News: Found {len(all_results['google_news'])} recent articles")
        
        if all_results['google_search']:
            summary_parts.append(f"🔍 Google Search: Found {len(all_results['google_search'])} web results")
        
        if summary_parts:
            all_results['summary'] = " | ".join(summary_parts)
        else:
            all_results['summary'] = "No additional sources found"
        
        return all_results
        
    except Exception as e:
        return {
            'wikipedia': [],
            'google_news': [],
            'google_search': [],
            'summary': f'Enhanced analysis failed: {str(e)}'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config')
def get_config():
    return jsonify({
        'model': OPENAI_MODEL,
        'model_weight': MODEL_WEIGHT,
        'api_key_configured': OPENAI_API_KEY != 'sk-REPLACE_ME',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'deployment': 'vercel',
        'limitations': {
            'image_analysis': 'Not supported on Vercel',
            'file_size_limit': '4MB',
            'timeout': '10 seconds'
        }
    })

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Flask app is running on Vercel'})

@app.route('/api/verify-news', methods=['POST'])
def verify_news():
    try:
        # Check if it's a file upload or text input
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not supported'}), 400
            
            filename = file.filename.lower()
            input_type = 'unknown'
            extracted_content = ''
            
            # Process based on file type
            if filename.endswith(('.mp3', '.wav', '.m4a', '.aac', '.ogg')):
                input_type = 'audio'
                extracted_content = transcribe_audio(file)
            else:
                input_type = 'document'
                extracted_content = extract_text_from_file(file)
                
        else:
            # Text input
            data = request.get_json()
            extracted_content = data.get('news_text', '').strip()
            input_type = 'text'
        
        if not extracted_content:
            return jsonify({'error': 'No content to analyze'}), 400
        
        if OPENAI_API_KEY == 'sk-REPLACE_ME':
            return jsonify({'error': 'OpenAI API key not configured'}), 400
        
        # Create the prompt for fake news detection
        prompt = f"""
        Analyze the following content and determine if it's likely to be true or false. 
        This content was extracted from: {input_type.upper()} input.
        
        Provide your analysis in the following JSON format:
        {{
            "verdict": "TRUE" or "FALSE" or "UNCERTAIN",
            "confidence_score": 0.0 to 1.0,
            "credibility_score": 0.0 to 1.0,
            "reasoning": "Detailed explanation of your analysis",
            "fact_check_notes": "Key points that support or contradict the claim",
            "risk_factors": "List of factors that make this content suspicious or credible",
            "input_type": "{input_type}",
            "extracted_content": "{extracted_content[:500]}..."
        }}

        Content to analyze: "{extracted_content}"

        Be thorough in your analysis. Consider:
        1. Source credibility indicators
        2. Factual consistency
        3. Emotional language usage
        4. Claim specificity
        5. Supporting evidence mentioned
        6. Common fake news patterns
        7. Context and reliability of the {input_type} source

        Respond only with valid JSON.
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert fact-checker and fake news detector. Analyze content with high accuracy and provide detailed reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=1000
        )
        
        # Extract the response content
        ai_response = response.choices[0].message.content.strip()
        
        # Try to parse the JSON response
        try:
            # Clean the response to ensure it's valid JSON
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            
            ai_response = ai_response.strip()
            result = json.loads(ai_response)
            
            # Add metadata
            result['input_type'] = input_type
            result['extracted_content'] = extracted_content[:200] + "..." if len(extracted_content) > 200 else extracted_content
            result['model_used'] = OPENAI_MODEL
            result['timestamp'] = response.created
            
            # Check if we need enhanced analysis (FALSE or UNCERTAIN verdict)
            verdict = result.get('verdict', '').upper()
            if verdict in ['FALSE', 'UNCERTAIN']:
                # Perform enhanced analysis with additional APIs
                enhanced_results = analyze_with_additional_apis(extracted_content)
                
                # Add enhanced results to the response
                result['enhanced_analysis'] = enhanced_results
                result['analysis_type'] = 'enhanced_with_additional_apis'
                
                # Update reasoning with enhanced findings
                enhanced_summary = enhanced_results.get('summary', '')
                if enhanced_summary and enhanced_summary != "No additional sources found":
                    result['reasoning'] += f"\n\n🔍 Enhanced Analysis: {enhanced_summary}"
                    result['fact_check_notes'] += f"\n\n📚 Additional Sources: {enhanced_summary}"
                else:
                    result['reasoning'] += "\n\n🔍 Enhanced Analysis: No additional sources found to verify or contradict the claim."
            else:
                result['analysis_type'] = 'openai_only'
                result['enhanced_analysis'] = None
            
            return jsonify(result)
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured response
            result = {
                'verdict': 'UNCERTAIN',
                'confidence_score': 0.5,
                'credibility_score': 0.5,
                'reasoning': 'AI analysis completed but response format was unexpected. Raw response: ' + ai_response[:200],
                'fact_check_notes': 'Unable to parse AI response properly',
                'risk_factors': 'Response parsing error',
                'input_type': input_type,
                'extracted_content': extracted_content[:200] + "..." if len(extracted_content) > 200 else extracted_content,
                'model_used': OPENAI_MODEL,
                'raw_response': ai_response
            }
            
            # Since this is UNCERTAIN, perform enhanced analysis
            enhanced_results = analyze_with_additional_apis(extracted_content)
            result['enhanced_analysis'] = enhanced_results
            result['analysis_type'] = 'enhanced_with_additional_apis'
            
            # Update reasoning with enhanced findings
            enhanced_summary = enhanced_results.get('summary', '')
            if enhanced_summary and enhanced_summary != "No additional sources found":
                result['reasoning'] += f"\n\n🔍 Enhanced Analysis: {enhanced_summary}"
                result['fact_check_notes'] += f"\n\n📚 Additional Sources: {enhanced_summary}"
            else:
                result['reasoning'] += "\n\n🔍 Enhanced Analysis: No additional sources found to verify or contradict the claim."
            
            return jsonify(result)
            
    except openai.error.OpenAIError as e:
        return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Export the Flask app for Vercel
app.debug = False

