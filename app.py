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

# Load environment variables
load_dotenv('config.env')

app = Flask(__name__)

# Configuration
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

# Supported file types
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp3', 'wav', 'm4a', 'aac', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(audio_file):
    """Transcribe audio file to text using external speech_to_text module."""
    try:
        return transcribe_audio_file(audio_file)
    except Exception as e:
        raise Exception(f"Audio transcription failed: {str(e)}")

def analyze_image(image_file):
    """Comprehensive image analysis with text extraction, context detection, and visual analysis"""
    try:
        # Extract text and analyze image comprehensively
        analysis_result = comprehensive_image_analysis(image_file)
        
        if not analysis_result['extracted_text'] or analysis_result['extracted_text'].strip() == "":
            # If no text found, provide visual context analysis
            visual_context = analyze_visual_context(image_file)
            return {
                "extracted_text": "No text detected in image",
                "image_context": visual_context['context'],
                "claims_found": visual_context['potential_claims'],
                "suspicious_elements": visual_context['suspicious_elements'],
                "visual_elements": visual_context['elements'],
                "image_type": visual_context['image_type']
            }
        
        # Return comprehensive analysis with extracted text
        return analysis_result
        
    except Exception as e:
        raise Exception(f"Comprehensive image analysis failed: {str(e)}")

def comprehensive_image_analysis(image_file):
    """Perform comprehensive image analysis including text, context, and visual elements"""
    try:
        from PIL import Image
        import pytesseract
        import cv2
        import numpy as np
        
        # Open and process the image
        image = Image.open(image_file)
        image_file.seek(0)  # Reset file pointer
        
        # Get basic image information
        width, height = image.size
        format_info = image.format
        mode_info = image.mode
        
        # Convert PIL image to OpenCV format for advanced analysis
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Extract text using enhanced OCR
        extracted_text = extract_text_with_enhanced_ocr(image_file)
        
        # Analyze visual context and elements
        visual_analysis = analyze_visual_context(image_file)
        
        # Determine image type and context
        image_type = determine_image_type(opencv_image, extracted_text)
        
        # Generate comprehensive context
        context = generate_image_context(extracted_text, visual_analysis, image_type, width, height)
        
        # Identify potential claims and suspicious elements
        claims = identify_potential_claims(extracted_text, visual_analysis, image_type)
        suspicious = identify_suspicious_elements(extracted_text, visual_analysis, image_type)
        
        return {
            "extracted_text": extracted_text.strip() if extracted_text else "No text detected",
            "image_context": context,
            "claims_found": claims,
            "suspicious_elements": suspicious,
            "visual_elements": visual_analysis['elements'],
            "image_type": image_type,
            "image_metadata": f"{width}x{height}, {format_info}, {mode_info}"
        }
        
    except Exception as e:
        raise Exception(f"Comprehensive analysis failed: {str(e)}")

def extract_text_with_enhanced_ocr(image_file):
    """Enhanced OCR with multiple preprocessing techniques for better text extraction"""
    try:
        from PIL import Image
        import pytesseract
        import cv2
        import numpy as np
        
        # Open image
        image = Image.open(image_file)
        image_file.seek(0)
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Multiple OCR attempts with different preprocessing
        ocr_results = []
        
        # 1. Original image
        try:
            text = pytesseract.image_to_string(image, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%&*()_+-=[]{}|;:,.<>?/ ')
            if text.strip():
                ocr_results.append(text.strip())
        except:
            pass
        
        # 2. Grayscale with thresholding
        try:
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(binary, config='--psm 6')
            if text.strip():
                ocr_results.append(text.strip())
        except:
            pass
        
        # 3. Enhanced preprocessing
        try:
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(adaptive, config='--psm 6')
            if text.strip():
                ocr_results.append(text.strip())
        except:
            pass
        
        # 4. Edge detection approach
        try:
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            # Dilate edges to connect text
            kernel = np.ones((2,2), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            text = pytesseract.image_to_string(dilated, config='--psm 6')
            if text.strip():
                ocr_results.append(text.strip())
        except:
            pass
        
        # Return the best result (longest text)
        if ocr_results:
            return max(ocr_results, key=len)
        else:
            return ""
            
    except Exception as e:
        return f"OCR extraction failed: {str(e)}"

def analyze_visual_context(image_file):
    """Analyze visual elements and context of the image"""
    try:
        from PIL import Image
        import cv2
        import numpy as np
        
        image = Image.open(image_file)
        image_file.seek(0)
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Basic visual analysis
        height, width = opencv_image.shape[:2]
        aspect_ratio = width / height
        
        # Color analysis
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(hsv, axis=(0, 1))
        
        # Edge detection for structure analysis
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Determine visual characteristics
        elements = []
        
        # Analyze based on visual characteristics
        if edge_density > 0.1:
            elements.append("High detail/complex structure")
        else:
            elements.append("Simple/minimal design")
            
        if aspect_ratio > 1.5:
            elements.append("Wide/landscape format")
        elif aspect_ratio < 0.7:
            elements.append("Tall/portrait format")
        else:
            elements.append("Square/standard format")
            
        # Color analysis
        if avg_color[1] > 100:  # High saturation
            elements.append("Colorful/vibrant")
        else:
            elements.append("Muted/grayscale tones")
            
        # Determine likely image type
        if edge_density > 0.15:
            image_type = "Document/Text-heavy"
        elif avg_color[1] > 120:
            image_type = "Colorful/Graphic"
        else:
            image_type = "Photo/Image"
            
        return {
            "elements": elements,
            "image_type": image_type,
            "edge_density": edge_density,
            "color_profile": "Colorful" if avg_color[1] > 100 else "Muted",
            "aspect_ratio": aspect_ratio
        }
        
    except Exception as e:
        return {
            "elements": ["Visual analysis failed"],
            "image_type": "Unknown",
            "edge_density": 0,
            "color_profile": "Unknown",
            "aspect_ratio": 1.0
        }

def determine_image_type(opencv_image, extracted_text):
    """Determine the type of image based on content and visual characteristics"""
    try:
        # Analyze text content for clues
        text_lower = extracted_text.lower()
        
        # Check for common patterns
        if any(word in text_lower for word in ['news', 'article', 'headline', 'breaking', 'report']):
            return "News Article"
        elif any(word in text_lower for word in ['social', 'media', 'post', 'tweet', 'facebook', 'instagram']):
            return "Social Media Post"
        elif any(word in text_lower for word in ['meme', 'funny', 'joke', 'humor']):
            return "Meme/Humor"
        elif any(word in text_lower for word in ['advertisement', 'ad', 'promotion', 'sponsored']):
            return "Advertisement"
        elif any(word in text_lower for word in ['screenshot', 'capture', 'screen']):
            return "Screenshot"
        elif len(extracted_text.strip()) > 100:
            return "Document/Text-heavy"
        elif len(extracted_text.strip()) > 20:
            return "Text Image"
        else:
            return "Visual Image"
            
    except Exception as e:
        return "Unknown"

def generate_image_context(extracted_text, visual_analysis, image_type, width, height):
    """Generate comprehensive context description for the image"""
    try:
        context_parts = []
        
        # Add image type context
        context_parts.append(f"This appears to be a {image_type.lower()} image")
        
        # Add size context
        if width > 1000 or height > 1000:
            context_parts.append("high-resolution")
        else:
            context_parts.append("standard resolution")
            
        # Add visual characteristics
        if visual_analysis['elements']:
            context_parts.append(f"with {', '.join(visual_analysis['elements']).lower()}")
            
        # Add text context
        if extracted_text and extracted_text.strip():
            text_length = len(extracted_text.strip())
            if text_length > 200:
                context_parts.append("containing substantial text content")
            elif text_length > 50:
                context_parts.append("with moderate text content")
            else:
                context_parts.append("with minimal text content")
        else:
            context_parts.append("containing no readable text")
            
        return " ".join(context_parts) + "."
        
    except Exception as e:
        return f"Image analysis context generation failed: {str(e)}"

def identify_potential_claims(extracted_text, visual_analysis, image_type):
    """Identify potential claims or statements in the image"""
    try:
        claims = []
        
        if not extracted_text or not extracted_text.strip():
            return "No text content to analyze for claims"
            
        # Look for claim indicators in text
        text_lower = extracted_text.lower()
        
        # Common claim patterns
        claim_indicators = [
            'scientists discover', 'new study shows', 'research proves',
            'experts say', 'doctors recommend', 'study finds',
            'breakthrough', 'miracle', 'cure', 'treatment',
            'guaranteed', 'proven', 'effective', 'revolutionary',
            'shocking', 'amazing', 'incredible', 'unbelievable'
        ]
        
        for indicator in claim_indicators:
            if indicator in text_lower:
                claims.append(f"Contains claim indicator: '{indicator}'")
                
        # Check for sensational language
        sensational_words = ['shocking', 'amazing', 'incredible', 'unbelievable', 'miracle', 'breakthrough']
        for word in sensational_words:
            if word in text_lower:
                claims.append(f"Uses sensational language: '{word}'")
                
        # Check for specific vs vague claims
        if any(word in text_lower for word in ['all', 'every', 'never', 'always', '100%', 'guaranteed']):
            claims.append("Contains absolute or extreme claims")
            
        if claims:
            return "; ".join(claims)
        else:
            return "No obvious claims detected in the text content"
            
    except Exception as e:
        return f"Claim analysis failed: {str(e)}"

def identify_suspicious_elements(extracted_text, visual_analysis, image_type):
    """Identify potentially suspicious or misleading elements"""
    try:
        suspicious = []
        
        if not extracted_text or not extracted_text.strip():
            return "No text content to analyze for suspicious elements"
            
        text_lower = extracted_text.lower()
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'click here', 'limited time', 'act now', 'don\'t miss out',
            'secret', 'hidden', 'exposed', 'they don\'t want you to know',
            'free money', 'make money fast', 'get rich quick',
            'lose weight fast', 'miracle cure', 'one weird trick',
            'doctors hate this', 'big pharma', 'conspiracy'
        ]
        
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                suspicious.append(f"Suspicious pattern: '{pattern}'")
                
        # Check for poor grammar/spelling (potential fake content)
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        text_words = text_lower.split()
        if len(text_words) > 10:
            misspelled_count = sum(1 for word in text_words if len(word) > 3 and word not in common_words and not word.isalpha())
            if misspelled_count > len(text_words) * 0.1:  # More than 10% misspelled
                suspicious.append("Contains multiple spelling/grammar errors")
                
        # Check for excessive punctuation
        if text_lower.count('!') > 3 or text_lower.count('?') > 3:
            suspicious.append("Uses excessive punctuation (potential clickbait)")
            
        # Check for urgency/scarcity tactics
        urgency_words = ['now', 'today', 'limited', 'urgent', 'immediate', 'last chance']
        if any(word in text_lower for word in urgency_words):
            suspicious.append("Uses urgency/scarcity tactics")
            
        if suspicious:
            return "; ".join(suspicious)
        else:
            return "No obvious suspicious elements detected"
            
    except Exception as e:
        return f"Suspicious element analysis failed: {str(e)}"

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
        
        # 3. Enhanced recent event search with more variations
        if len(results) < 3:  # If we don't have enough results
            # More comprehensive recent event variations
            recent_variations = [
                f"{query} 2024",
                f"{query} 2023", 
                f"{query} 2025",
                f"{query} recent",
                f"{query} incident",
                f"{query} accident",
                f"{query} crash",
                f"{query} emergency",
                f"{query} flight",
                f"{query} plane",
                f"{query} aircraft",
                f"{query} aviation",
                f"{query} airport",
                # Add location-specific variations
                f"{query} Ahmedabad",
                f"{query} Gujarat",
                f"{query} India",
                # Add specific aviation terms
                f"{query} runway",
                f"{query} landing",
                f"{query} takeoff",
                f"{query} emergency landing",
                f"{query} flight incident",
                f"{query} plane crash",
                f"{query} aircraft accident"
            ]
            
            for variation in recent_variations:
                if len(results) >= max_results:
                    break
                    
                recent_pages = wiki_api.search(variation, results=2)
                for page in recent_pages:
                    if page.exists() and len(results) < max_results:
                        # Check if this page is not already in results
                        if not any(r['title'] == page.title for r in results):
                            content = page.summary
                            results.append({
                                'title': page.title,
                                'summary': content[:600] + "..." if len(content) > 600 else content,
                                'url': page.fullurl,
                                'source': 'Wikipedia',
                                'search_type': 'recent_event_search'
                            })
        
        # 4. Try searching for aviation/aircraft related categories
        if len(results) < 2:
            aviation_terms = ['aviation accident', 'aircraft crash', 'flight incident', 'emergency landing']
            for term in aviation_terms:
                if len(results) >= max_results:
                    break
                    
                aviation_pages = wiki_api.search(term, results=1)
                for page in aviation_pages:
                    if page.exists() and len(results) < max_results:
                        # Check if this page is not already in results
                        if not any(r['title'] == page.title for r in results):
                            content = page.summary
                            results.append({
                                'title': page.title,
                                'summary': content[:500] + "..." if len(content) > 500 else content,
                                'url': page.fullurl,
                                'source': 'Wikipedia',
                                'search_type': 'aviation_category_search'
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
        
        # 2. If we don't have enough results, try broader search without date filter
        if len(results) < 2:
            params['sortBy'] = 'relevancy'
            params.pop('from', None)  # Remove date filter for broader search
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                additional_articles = data.get('articles', [])
                
                for article in additional_articles:
                    if len(results) >= max_results:
                        break
                    # Check if this article is not already in results
                    if not any(r['url'] == article.get('url', '') for r in results):
                        results.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', '')[:400] + "..." if len(article.get('description', '')) > 400 else article.get('description', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'publishedAt': article.get('publishedAt', ''),
                            'source_type': 'Google News',
                            'search_type': 'broader_search'
                        })
        
        # 3. Try aviation-specific news sources if it's an aviation incident
        if len(results) < 2 and any(word in query.lower() for word in ['flight', 'plane', 'aircraft', 'crash', 'accident', 'aviation']):
            aviation_sources = ['aviation-week', 'flightglobal', 'airlineratings', 'aviation-safety']
            for source in aviation_sources:
                if len(results) >= max_results:
                    break
                    
                params['domains'] = source
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    aviation_articles = data.get('articles', [])
                    
                    for article in aviation_articles:
                        if len(results) >= max_results:
                            break
                        if not any(r['url'] == article.get('url', '') for r in results):
                            results.append({
                                'title': article.get('title', ''),
                                'description': article.get('description', '')[:400] + "..." if len(article.get('description', '')) > 400 else article.get('description', ''),
                                'url': article.get('url', ''),
                                'source': article.get('source', {}).get('name', 'Unknown'),
                                'publishedAt': article.get('publishedAt', ''),
                                'source_type': 'Google News',
                                'search_type': 'aviation_specialist_search'
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
        
        # Special handling for aviation incidents
        aviation_terms = ['flight', 'plane', 'aircraft', 'aviation', 'airport', 'runway', 'landing', 'takeoff']
        if any(term in text_lower for term in aviation_terms):
            # Add aviation-specific queries
            for entity in entities[:2]:
                search_queries.extend([
                    f"{entity} aviation",
                    f"{entity} flight",
                    f"{entity} aircraft",
                    f"{entity} plane crash",
                    f"{entity} aviation accident",
                    f"{entity} flight incident",
                    f"{entity} emergency landing",
                    f"{entity} runway accident"
                ])
            
            # Add location-specific aviation queries
            location_entities = [e for e in entities if any(loc in e.lower() for loc in ['ahmedabad', 'gujarat', 'india', 'mumbai', 'delhi', 'bangalore'])]
            for location in location_entities[:1]:
                search_queries.extend([
                    f"{location} aviation incident",
                    f"{location} flight crash",
                    f"{location} aircraft accident",
                    f"{location} airport incident"
                ])
        
        # Add date-specific queries if dates are mentioned
        date_patterns = re.findall(r'\b(202[3-5]|202[0-9])\b', text)
        for date in date_patterns:
            for entity in entities[:2]:
                search_queries.append(f"{entity} {date}")
        
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
        
        # Enhanced aviation incident detection and query generation
        content_lower = content.lower()
        if any(word in content_lower for word in ['flight', 'plane', 'aircraft', 'airport', 'aviation', 'crash', 'accident', 'incident']):
            # Extract location names (capitalized words)
            words = content.split()
            locations = []
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2:
                    # Look for multi-word locations
                    location = word
                    j = i + 1
                    while j < len(words) and words[j][0].isupper() and len(words[j]) > 2:
                        location += " " + words[j]
                        j += 1
                    if len(location.split()) <= 3:
                        locations.append(location)
            
            # Add comprehensive aviation-specific queries
            for location in locations[:2]:
                search_queries.extend([
                    f"{location} flight incident",
                    f"{location} airport incident",
                    f"{location} plane incident",
                    f"{location} aviation incident",
                    f"{location} emergency landing",
                    f"{location} aircraft crash",
                    f"{location} plane crash",
                    f"{location} aviation accident",
                    f"{location} runway incident",
                    f"{location} flight accident"
                ])
            
            # Add recent event specific queries for aviation
            for location in locations[:1]:
                search_queries.extend([
                    f"{location} 2024 aviation",
                    f"{location} 2023 aviation",
                    f"{location} recent aviation incident",
                    f"{location} latest aviation accident",
                    f"{location} breaking aviation news"
                ])
        
        # Enhanced recent event detection
        recent_indicators = ['recent', 'latest', 'today', 'yesterday', 'this week', 'this month', '2024', '2023', '2025', 'breaking', 'just', 'now']
        if any(indicator in content_lower for indicator in recent_indicators):
            # Add recent event specific queries
            words = content.split()
            entities = []
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2:
                    entity = word
                    j = i + 1
                    while j < len(words) and words[j][0].isupper() and len(words[j]) > 2:
                        entity += " " + words[j]
                        j += 1
                    if len(entity.split()) <= 4:
                        entities.append(entity)
            
            for entity in entities[:2]:
                search_queries.extend([
                    f"{entity} 2024",
                    f"{entity} 2023",
                    f"{entity} 2025",
                    f"{entity} recent",
                    f"{entity} latest",
                    f"{entity} breaking news",
                    f"{entity} today",
                    f"{entity} this week"
                ])
        
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
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Flask app is running'})

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
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                input_type = 'image'
                try:
                    # Extract text from image using OCR
                    image_analysis = analyze_image(file)
                    
                    # If text was extracted, use it for fake news analysis
                    if image_analysis['extracted_text'] and image_analysis['extracted_text'] != "No text detected in image":
                        extracted_content = image_analysis['extracted_text']
                    else:
                        # If no text found, use the analysis result
                        extracted_content = f"Image Analysis:\n{json.dumps(image_analysis, indent=2)}"
                        
                except Exception as e:
                    # Use fallback analysis if OCR fails
                    image_analysis = analyze_visual_context(file)  # Using analyze_visual_context as a fallback
                    extracted_content = f"Image Analysis (Fallback - OCR Failed):\n{json.dumps(image_analysis, indent=2)}"
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)