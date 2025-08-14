# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
import os
import tempfile
from urllib.parse import urlparse, quote_plus
import re
import json
from datetime import datetime, timedelta
import hashlib
from typing import Dict, List, Tuple, Optional
import time
import random
import logging  # Add at the top if not present
import pandas as pd  # Add at the top if not present
import io

# Optional imports with error handling
try:
    from transformers import pipeline
    hf_pipeline = pipeline
    HF_PIPELINE_IMPORT_ERROR = None
except ImportError as e:
    hf_pipeline = None
    HF_PIPELINE_IMPORT_ERROR = str(e)

try:
    import torch
    TORCH_IMPORT_ERROR = None
except ImportError as e:
    torch = None
    TORCH_IMPORT_ERROR = str(e)

try:
    import cv2
    CV2_IMPORT_ERROR = None
except ImportError as e:
    cv2 = None
    CV2_IMPORT_ERROR = str(e)

try:
    import numpy as np
    NUMPY_IMPORT_ERROR = None
except ImportError as e:
    np = None
    NUMPY_IMPORT_ERROR = str(e)

try:
    from PIL import Image, ImageChops
    PIL_IMPORT_ERROR = None
except ImportError as e:
    Image = None
    ImageChops = None
    PIL_IMPORT_ERROR = str(e)

try:
    import yt_dlp
    YT_DLP_IMPORT_ERROR = None
except ImportError as e:
    yt_dlp = None
    YT_DLP_IMPORT_ERROR = str(e)

try:
    import whisper
    WHISPER_IMPORT_ERROR = None
except ImportError as e:
    whisper = None
    WHISPER_IMPORT_ERROR = str(e)

# --- CONFIGURATION & MODEL LOADING ---

st.set_page_config(page_title="Veritas: Advanced Misinformation Detector", page_icon="🛡️", layout="wide")

# Enhanced model configuration
MODEL_CONFIG = {
    'primary_model': "vikram71198/distilroberta-base-finetuned-fake-news-detection",
    'secondary_models': [
        "microsoft/DialoGPT-medium",  # For conversational text
        "facebook/bart-large-mnli"    # For claim verification
    ],
    'confidence_thresholds': {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4
    }
}

# Use Streamlit's cache to load models only once
@st.cache_resource
def load_ensemble_models():
    """Loads multiple pre-trained models for ensemble analysis."""
    models = {}
    
    if HF_PIPELINE_IMPORT_ERROR:
        st.warning(f"⚠️ Transformers library not available: {HF_PIPELINE_IMPORT_ERROR}")
        return None
    
    try:
        # Primary model for fake news detection
        models['primary'] = hf_pipeline("text-classification", 
                                   model=MODEL_CONFIG['primary_model'])
        
        # Secondary model for zero-shot classification
        models['zero_shot'] = hf_pipeline("zero-shot-classification",
                                     model=MODEL_CONFIG['secondary_models'][1])
        
        # Text generation model for analysis
        models['generation'] = hf_pipeline("text-generation",
                                      model=MODEL_CONFIG['secondary_models'][0])
        
        st.success("✅ All models loaded successfully!")
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

@st.cache_resource
def load_whisper_model():
    """Loads the speech-to-text model from OpenAI."""
    if WHISPER_IMPORT_ERROR:
        st.warning(f"⚠️ Whisper library not available: {WHISPER_IMPORT_ERROR}")
        return None
    
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"❌ Error loading Whisper model: {e}")
        return None

# Load all models
models = load_ensemble_models()
whisper_model = load_whisper_model()

# Enhanced API configuration
API_CONFIG = {
    'google_fact_check': None,
    'news_api': None,
    'snopes_api': None,
    'politifact_api': None,
    'bing_search': None,
    'serpapi': None,
    'openai': None
}

# Try to load API keys from secrets
try:
    API_CONFIG['google_fact_check'] = st.secrets["GOOGLE_API_KEY"]
    API_CONFIG['news_api'] = st.secrets["NEWS_API_KEY"]
    API_CONFIG['snopes_api'] = st.secrets["SNOPES_API_KEY"]
    API_CONFIG['politifact_api'] = st.secrets["POLITIFACT_API_KEY"]
    API_CONFIG['bing_search'] = st.secrets["BING_SEARCH_KEY"]
    API_CONFIG['serpapi'] = st.secrets["SERPAPI_KEY"]
    API_CONFIG['openai'] = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.warning("⚠️ Some API keys not configured. Enhanced fact-checking will be limited.")

# --- WEB SCRAPING CONFIGURATION ---

# Fact-checking websites to scrape
FACT_CHECK_SITES = {
    'politifact': {
        'base_url': 'https://www.politifact.com',
        'search_url': 'https://www.politifact.com/search/?q=',
        'selectors': {
            'articles': '.c-textgroup__title a',
            'title': '.c-textgroup__title',
            'rating': '.c-image__original',
            'summary': '.c-textgroup__body',
            'date': '.c-textgroup__date'
        }
    },
    'snopes': {
        'base_url': 'https://www.snopes.com',
        'search_url': 'https://www.snopes.com/?s=',
        'selectors': {
            'articles': '.article-list-item h2 a',
            'title': '.article-list-item h2',
            'rating': '.rating-label',
            'summary': '.article-list-item .excerpt',
            'date': '.article-list-item .date'
        }
    },
    'factcheck_org': {
        'base_url': 'https://www.factcheck.org',
        'search_url': 'https://www.factcheck.org/?s=',
        'selectors': {
            'articles': '.entry-title a',
            'title': '.entry-title',
            'rating': '.fact-check-rating',
            'summary': '.entry-summary',
            'date': '.entry-date'
        }
    }
}

# Indian news websites to scrape
INDIAN_NEWS_SITES = {
    'the_hindu': {
        'base_url': 'https://www.thehindu.com',
        'search_url': 'https://www.thehindu.com/search/?q=',
        'selectors': {
            'articles': '.story-card a',
            'title': '.story-card h3',
            'summary': '.story-card .intro',
            'date': '.story-card .date'
        }
    },
    'times_of_india': {
        'base_url': 'https://timesofindia.indiatimes.com',
        'search_url': 'https://timesofindia.indiatimes.com/topic/',
        'selectors': {
            'articles': '.article a',
            'title': '.article h3',
            'summary': '.article .summary',
            'date': '.article .date'
        }
    },
    'hindustan_times': {
        'base_url': 'https://www.hindustantimes.com',
        'search_url': 'https://www.hindustantimes.com/search?q=',
        'selectors': {
            'articles': '.listingPage a',
            'title': '.listingPage h3',
            'summary': '.listingPage .sortDec',
            'date': '.listingPage .dateTime'
        }
    },
    'indian_express': {
        'base_url': 'https://indianexpress.com',
        'search_url': 'https://indianexpress.com/?s=',
        'selectors': {
            'articles': '.story a',
            'title': '.story h2',
            'summary': '.story .synopsis',
            'date': '.story .date'
        }
    },
    'ndtv': {
        'base_url': 'https://www.ndtv.com',
        'search_url': 'https://www.ndtv.com/search?q=',
        'selectors': {
            'articles': '.news_item a',
            'title': '.news_item h2',
            'summary': '.news_item .content_text',
            'date': '.news_item .posted_on'
        }
    }
}

# Headers for web scraping
SCRAPING_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# --- ENHANCED ANALYSIS PATTERNS ---

# Advanced pattern detection
PATTERN_CONFIG = {
    'logical_fallacies': [
        r"everyone knows that",
        r"if.*then.*therefore",
        r"either.*or.*",
        r"since.*then.*",
        r"because.*therefore.*"
    ],
    
    'emotional_manipulation': [
        r"you won't believe",
        r"shocking",
        r"outrageous",
        r"they don't want you to know",
        r"secret",
        r"exposed",
        r"revealed",
        r"breaking",
        r"urgent",
        r"warning",
        r"alert",
        r"doctors hate",
        r"big pharma",
        r"conspiracy",
        r"cover up"
    ],
    
    'credibility_indicators': [
        r"according to.*study",
        r"peer-reviewed",
        r"journal",
        r"university of",
        r"research shows",
        r"scientists found",
        r"evidence suggests",
        r"data indicates",
        r"official statement",
        r"confirmed by",
        r"verified source"
    ],
    
    'factual_contradictions': [
        ('putin', 'america', 'russian president'),
        ('modi', 'america', 'indian prime minister'),
        ('trump', 'india', 'american president'),
        ('biden', 'russia', 'american president'),
        ('china', 'usa', 'separate countries'),
        ('russia', 'european union', 'separate entities'),
        ('india', 'china', 'separate countries'),
    ],
    
    'death_hoax_indicators': [
        'is dead', 'has died', 'passed away', 'killed', 'murdered', 'assassinated',
        'found dead', 'died in', 'death of', 'obituary', 'funeral', 'memorial'
    ]
}

# --- WEB SCRAPING FUNCTIONS ---

def scrape_fact_check_site(site_name: str, query: str, max_results: int = 5) -> List[Dict]:
    """Scrapes fact-checking websites for relevant articles."""
    results = []
    
    if site_name not in FACT_CHECK_SITES:
        return results
    
    site_config = FACT_CHECK_SITES[site_name]
    search_url = site_config['search_url'] + quote_plus(query)
    
    try:
        # Add random delay to be respectful
        time.sleep(random.uniform(1, 3))
        
        response = requests.get(search_url, headers=SCRAPING_HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find articles
        article_links = soup.select(site_config['selectors']['articles'])
        
        for i, link in enumerate(article_links[:max_results]):
            try:
                article_url = link.get('href')
                if not article_url.startswith('http'):
                    article_url = site_config['base_url'] + article_url
                
                # Scrape individual article
                article_data = scrape_individual_article(article_url, site_config)
                if article_data:
                    article_data['source'] = site_name
                    article_data['url'] = article_url
                    results.append(article_data)
                    
            except Exception as e:
                logging.warning(f"Error scraping article from {site_name}: {e}")  # Log instead of showing in UI
                continue
                
    except Exception as e:
        logging.warning(f"Error scraping {site_name}: {e}")  # Log instead of showing in UI
    
    return results

def scrape_individual_article(url: str, site_config: Dict) -> Optional[Dict]:
    """Scrapes individual article content."""
    try:
        time.sleep(random.uniform(1, 2))  # Be respectful
        
        response = requests.get(url, headers=SCRAPING_HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.select_one(site_config['selectors']['title'])
        title = title_elem.get_text().strip() if title_elem else "No title found"
        
        # Extract rating (for fact-check sites)
        rating = "Unknown"
        if 'rating' in site_config['selectors']:
            rating_elem = soup.select_one(site_config['selectors']['rating'])
            if rating_elem:
                rating = rating_elem.get_text().strip()
        
        # Extract summary/content
        summary_elem = soup.select_one(site_config['selectors']['summary'])
        summary = summary_elem.get_text().strip() if summary_elem else "No summary found"
        
        # Extract date
        date_elem = soup.select_one(site_config['selectors']['date'])
        date = date_elem.get_text().strip() if date_elem else "Unknown date"
        
        return {
            'title': title,
            'rating': rating,
            'summary': summary,
            'date': date,
            'full_url': url
        }
        
    except Exception as e:
        st.warning(f"Error scraping individual article: {e}")
        return None

def scrape_indian_news_sites(query: str, max_results: int = 3) -> List[Dict]:
    """Scrapes Indian news websites for relevant articles."""
    results = []
    
    for site_name, site_config in INDIAN_NEWS_SITES.items():
        try:
            site_results = scrape_news_site(site_name, site_config, query, max_results)
            results.extend(site_results)
        except Exception as e:
            st.warning(f"Error scraping {site_name}: {e}")
            continue
    
    return results

def scrape_news_site(site_name: str, site_config: Dict, query: str, max_results: int) -> List[Dict]:
    """Scrapes individual news site."""
    results = []
    search_url = site_config['search_url'] + quote_plus(query)
    
    try:
        time.sleep(random.uniform(1, 3))
        
        response = requests.get(search_url, headers=SCRAPING_HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find articles
        article_links = soup.select(site_config['selectors']['articles'])
        
        for i, link in enumerate(article_links[:max_results]):
            try:
                article_url = link.get('href')
                if not article_url.startswith('http'):
                    article_url = site_config['base_url'] + article_url
                
                # Extract basic info from search results
                title_elem = link.find_parent().select_one(site_config['selectors']['title'])
                title = title_elem.get_text().strip() if title_elem else "No title found"
                
                summary_elem = link.find_parent().select_one(site_config['selectors']['summary'])
                summary = summary_elem.get_text().strip() if summary_elem else "No summary found"
                
                date_elem = link.find_parent().select_one(site_config['selectors']['date'])
                date = date_elem.get_text().strip() if date_elem else "Unknown date"
                
                results.append({
                    'title': title,
                    'summary': summary,
                    'date': date,
                    'url': article_url,
                    'source': site_name
                })
                
            except Exception as e:
                continue
                
    except Exception as e:
        st.warning(f"Error scraping {site_name}: {e}")
    
    return results

def extract_claims_from_text(text: str) -> List[str]:
    """Extracts specific claims from text using NLP patterns."""
    claims = []
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Look for claim indicators
    claim_indicators = [
        r'claim.*that',
        r'say.*that',
        r'report.*that',
        r'according.*to',
        r'study.*shows',
        r'research.*finds',
        r'evidence.*suggests',
        r'data.*indicates',
        r'found.*that',
        r'discovered.*that',
        r'revealed.*that',
        r'confirmed.*that',
        r'announced.*that',
        r'stated.*that',
        r'declared.*that'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Skip very short sentences
            continue
            
        # Check if sentence contains claim indicators
        for indicator in claim_indicators:
            if re.search(indicator, sentence, re.IGNORECASE):
                claims.append(sentence)
                break
    
    # If no claims found with indicators, extract key sentences
    if not claims and len(sentences) > 0:
        # Take sentences with specific details (numbers, dates, names)
        for sentence in sentences:
            if (re.search(r'\d+', sentence) or  # Contains numbers
                re.search(r'\b(202[0-9]|202[0-9])\b', sentence) or  # Contains years
                re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', sentence)):  # Contains names
                claims.append(sentence)
    
    # Limit to top 3 most relevant claims
    return claims[:3]

def search_web_for_verification(claim: str) -> Dict:
    """Searches the web for claim verification using multiple sources."""
    search_results = {
        'fact_check_sites': [],
        'news_articles': [],
        'academic_sources': [],
        'official_sources': []
    }
    
    # 1. Search fact-checking websites
    for site_name in FACT_CHECK_SITES.keys():
        try:
            site_results = scrape_fact_check_site(site_name, claim, max_results=2)
            search_results['fact_check_sites'].extend(site_results)
        except Exception as e:
            st.warning(f"Error searching {site_name}: {e}")
    
    # 2. Search Indian news sites
    try:
        indian_news_results = scrape_indian_news_sites(claim, max_results=2)
        search_results['news_articles'].extend(indian_news_results)
    except Exception as e:
        st.warning(f"Error searching Indian news sites: {e}")
    
    # 3. Use Bing Search API if available
    if API_CONFIG['bing_search']:
        try:
            bing_results = search_bing_api(claim)
            search_results['news_articles'].extend(bing_results.get('news', []))
            search_results['academic_sources'].extend(bing_results.get('academic', []))
        except Exception as e:
            st.warning(f"Bing search error: {e}")
    
    # 4. Use SerpAPI if available
    elif API_CONFIG['serpapi']:
        try:
            serp_results = search_serpapi(claim)
            search_results['news_articles'].extend(serp_results.get('news', []))
            search_results['academic_sources'].extend(serp_results.get('academic', []))
        except Exception as e:
            st.warning(f"SerpAPI error: {e}")
    
    return search_results

def search_bing_api(query: str) -> Dict:
    """Searches using Bing Search API."""
    results = {'news': [], 'academic': []}
    
    try:
        # Search for news
        news_url = f"https://api.bing.microsoft.com/v7.0/news/search"
        headers = {'Ocp-Apim-Subscription-Key': API_CONFIG['bing_search']}
        params = {'q': query, 'count': 5}
        
        response = requests.get(news_url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            news_data = response.json()
            for article in news_data.get('value', []):
                results['news'].append({
                    'title': article.get('name', ''),
                    'summary': article.get('description', ''),
                    'url': article.get('url', ''),
                    'date': article.get('datePublished', ''),
                    'source': 'Bing News API'
                })
    except Exception as e:
        st.warning(f"Bing API error: {e}")
    
    return results

def search_serpapi(query: str) -> Dict:
    """Searches using SerpAPI."""
    results = {'news': [], 'academic': []}
    
    try:
        # Search for news
        serp_url = "https://serpapi.com/search"
        params = {
            'q': query,
            'api_key': API_CONFIG['serpapi'],
            'tbm': 'nws',  # News search
            'num': 5
        }
        
        response = requests.get(serp_url, params=params, timeout=10)
        if response.status_code == 200:
            serp_data = response.json()
            for article in serp_data.get('news_results', []):
                results['news'].append({
                    'title': article.get('title', ''),
                    'summary': article.get('snippet', ''),
                    'url': article.get('link', ''),
                    'date': article.get('date', ''),
                    'source': 'SerpAPI'
                })
    except Exception as e:
        st.warning(f"SerpAPI error: {e}")
    
    return results

def analyze_claim_credibility(claim: str, search_results: Dict) -> Dict:
    """Analyzes claim credibility based on search results."""
    credibility_score = 50  # Base score
    evidence_strength = "weak"
    verification_status = "unverified"
    supporting_evidence = []
    contradicting_evidence = []
    
    # Analyze fact-check site results
    fact_check_count = len(search_results['fact_check_sites'])
    if fact_check_count > 0:
        credibility_score += 20
        evidence_strength = "moderate"
        
        for result in search_results['fact_check_sites']:
            rating = result.get('rating', '').lower()
            if 'true' in rating or 'mostly true' in rating:
                supporting_evidence.append(f"Fact-checked as TRUE by {result.get('source', 'Unknown')}")
                credibility_score += 10
            elif 'false' in rating or 'mostly false' in rating:
                contradicting_evidence.append(f"Fact-checked as FALSE by {result.get('source', 'Unknown')}")
                credibility_score -= 15
            elif 'half true' in rating or 'mixture' in rating:
                supporting_evidence.append(f"Fact-checked as PARTIALLY TRUE by {result.get('source', 'Unknown')}")
                credibility_score += 5
    
    # Analyze news coverage
    news_count = len(search_results['news_articles'])
    if news_count > 0:
        credibility_score += min(10, news_count * 2)
        if evidence_strength == "weak":
            evidence_strength = "moderate"
        
        for result in search_results['news_articles']:
            source = result.get('source', '').lower()
            if any(credible in source for credible in ['hindu', 'times', 'express', 'ndtv']):
                supporting_evidence.append(f"Covered by credible news source: {result.get('source', 'Unknown')}")
    
    # Analyze academic sources
    academic_count = len(search_results['academic_sources'])
    if academic_count > 0:
        credibility_score += 15
        evidence_strength = "strong"
        supporting_evidence.append(f"Referenced in {academic_count} academic sources")
    
    # Determine verification status
    if credibility_score >= 70:
        verification_status = "verified"
    elif credibility_score >= 50:
        verification_status = "partially verified"
    elif credibility_score >= 30:
        verification_status = "unclear"
    else:
        verification_status = "likely false"
    
    # Adjust evidence strength
    if len(supporting_evidence) + len(contradicting_evidence) >= 5:
        evidence_strength = "strong"
    elif len(supporting_evidence) + len(contradicting_evidence) >= 2:
        evidence_strength = "moderate"
    
    return {
        'score': max(0, min(100, credibility_score)),
        'evidence_strength': evidence_strength,
        'verification_status': verification_status,
        'supporting_evidence': supporting_evidence,
        'contradicting_evidence': contradicting_evidence,
        'total_sources': fact_check_count + news_count + academic_count
    }

# --- ENHANCED HELPER FUNCTIONS ---

def analyze_source_credibility(url: Optional[str]) -> float:
    """Analyzes the credibility of a source URL."""
    if not url:
        return 0.5  # Neutral score for unknown sources
    
    credible_domains = [
        'reuters.com', 'ap.org', 'bbc.com', 'npr.org', 'pbs.org',
        'nytimes.com', 'washingtonpost.com', 'wsj.com', 'economist.com',
        'nature.com', 'science.org', 'jstor.org', 'arxiv.org',
        'thehindu.com', 'timesofindia.indiatimes.com', 'hindustantimes.com',
        'indianexpress.com', 'ndtv.com', 'politifact.com', 'snopes.com',
        'factcheck.org'
    ]
    
    suspicious_domains = [
        'fake-news', 'conspiracy', 'truth-seeker', 'alternative-news',
        'real-truth', 'hidden-truth', 'exposed-truth'
    ]
    
    domain = urlparse(url).netloc.lower()
    
    for credible in credible_domains:
        if credible in domain:
            return 0.9
    
    for suspicious in suspicious_domains:
        if suspicious in domain:
            return 0.1
    
    return 0.5

def check_temporal_relevance(timestamp: Optional[str]) -> float:
    """Checks if claims are temporally relevant."""
    if not timestamp:
        return 0.7  # Neutral score
    
    try:
        claim_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        current_date = datetime.now(claim_date.tzinfo)
        days_diff = (current_date - claim_date).days
        
        if days_diff <= 7:
            return 0.9  # Very recent
        elif days_diff <= 30:
            return 0.8  # Recent
        elif days_diff <= 365:
            return 0.6  # Somewhat recent
        else:
            return 0.3  # Outdated
    except:
        return 0.5

def analyze_claim_specificity(text: str) -> float:
    """Analyzes how specific and verifiable a claim is."""
    # Count specific details
    specific_indicators = [
        r'\d{4}',  # Years
        r'\d+%',   # Percentages
        r'\$\d+',  # Dollar amounts
        r'\d+ people',  # Specific numbers
        r'according to.*',  # Citations
        r'study by.*',  # Research citations
        r'at.*university',  # Institution mentions
    ]
    
    specificity_score = 0
    for pattern in specific_indicators:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        specificity_score += matches * 0.1
    
    # Penalize vague language
    vague_indicators = [
        r'some people say',
        r'many believe',
        r'experts claim',
        r'studies show',
        r'research suggests'
    ]
    
    vagueness_penalty = 0
    for pattern in vague_indicators:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        vagueness_penalty += matches * 0.2
    
    return min(1.0, max(0.0, specificity_score - vagueness_penalty))

def detect_logical_fallacies(text: str) -> List[str]:
    """Detects logical fallacies in the text."""
    fallacies = []
    
    for pattern in PATTERN_CONFIG['logical_fallacies']:
        if re.search(pattern, text, re.IGNORECASE):
            fallacies.append(f"Logical fallacy detected: {pattern}")
    
    return fallacies

def analyze_emotional_manipulation(text: str) -> Dict[str, int]:
    """Analyzes emotional manipulation techniques."""
    emotional_score = 0
    manipulation_techniques = []
    
    for pattern in PATTERN_CONFIG['emotional_manipulation']:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        if matches > 0:
            emotional_score += matches
            manipulation_techniques.append(f"{pattern}: {matches} occurrences")
    
    return {
        'score': emotional_score,
        'techniques': manipulation_techniques
    }

def get_confidence_level(score: float, factors: Dict) -> Tuple[str, str]:
    """Determines confidence level based on multiple factors."""
    confidence_score = 0
    
    # Model confidence
    confidence_score += factors.get('model_confidence', 0.5) * 0.4
    
    # Source credibility
    confidence_score += factors.get('source_credibility', 0.5) * 0.2
    
    # Claim specificity
    confidence_score += factors.get('claim_specificity', 0.5) * 0.2
    
    # Temporal relevance
    confidence_score += factors.get('temporal_relevance', 0.5) * 0.1
    
    # Text length (longer texts provide more context)
    text_length_factor = min(1.0, factors.get('text_length', 0) / 1000)
    confidence_score += text_length_factor * 0.1
    
    if confidence_score >= MODEL_CONFIG['confidence_thresholds']['high']:
        return 'high', 'Strong evidence supports this classification'
    elif confidence_score >= MODEL_CONFIG['confidence_thresholds']['medium']:
        return 'medium', 'Moderate confidence with some uncertainty'
    else:
        return 'low', 'Limited confidence - requires manual review'

# --- ENHANCED FACT-CHECKING ---

def enhanced_fact_check(text: str) -> Dict:
    """Enhanced fact-checking using direct web scraping and multiple APIs."""
    results = {
        'extracted_claims': [],
        'claim_analysis': [],
        'overall_score': 0,
        'verification_summary': '',
        'recommendations': [],
        'scraped_sources': {
            'fact_check_sites': [],
            'indian_news': [],
            'api_results': {}
        }
    }
    
    # 1. Extract specific claims from text
    extracted_claims = extract_claims_from_text(text)
    results['extracted_claims'] = extracted_claims
    
    if not extracted_claims:
        # If no specific claims found, use the entire text
        extracted_claims = [text[:200] + "..." if len(text) > 200 else text]
        results['extracted_claims'] = extracted_claims
    
    # 2. Analyze each claim
    claim_scores = []
    total_sources = 0
    
    for i, claim in enumerate(extracted_claims[:3]):  # Limit to 3 claims
        st.info(f"🔍 Analyzing claim {i+1}: {claim[:100]}...")
        
        # Search web for verification
        search_results = search_web_for_verification(claim)
        
        # Analyze claim credibility
        credibility_analysis = analyze_claim_credibility(claim, search_results)
        
        claim_analysis = {
            'claim': claim,
            'credibility_score': credibility_analysis['score'],
            'evidence_strength': credibility_analysis['evidence_strength'],
            'verification_status': credibility_analysis['verification_status'],
            'supporting_evidence': credibility_analysis['supporting_evidence'],
            'contradicting_evidence': credibility_analysis['contradicting_evidence'],
            'search_results': search_results
        }
        
        results['claim_analysis'].append(claim_analysis)
        claim_scores.append(credibility_analysis['score'])
        total_sources += credibility_analysis['total_sources']
        
        # Aggregate scraped sources
        results['scraped_sources']['fact_check_sites'].extend(search_results['fact_check_sites'])
        results['scraped_sources']['indian_news'].extend(search_results['news_articles'])
    
    # 3. Calculate overall score
    if claim_scores:
        results['overall_score'] = sum(claim_scores) / len(claim_scores)
    else:
        results['overall_score'] = 50  # Neutral score
    
    # 4. Generate verification summary
    verified_claims = sum(1 for analysis in results['claim_analysis'] 
                         if analysis['verification_status'] in ['verified', 'partially verified'])
    total_claims = len(results['claim_analysis'])
    
    if total_claims > 0:
        verification_rate = verified_claims / total_claims
        if verification_rate >= 0.8:
            results['verification_summary'] = f"✅ {verified_claims}/{total_claims} claims verified with strong evidence"
        elif verification_rate >= 0.5:
            results['verification_summary'] = f"⚠️ {verified_claims}/{total_claims} claims partially verified"
        else:
            results['verification_summary'] = f"❌ {verified_claims}/{total_claims} claims verified - limited evidence"
    else:
        results['verification_summary'] = "No specific claims found to verify"
    
    # 5. Generate recommendations
    if total_sources == 0:
        results['recommendations'].append("🔍 No verification sources found - manual fact-checking recommended")
    elif total_sources < 3:
        results['recommendations'].append("📚 Limited sources found - seek additional verification")
    else:
        results['recommendations'].append("✅ Multiple sources consulted - good verification coverage")
    
    if any(analysis['verification_status'] == 'likely false' for analysis in results['claim_analysis']):
        results['recommendations'].append("🚨 Some claims appear to be false - verify before sharing")
    
    if any(analysis['evidence_strength'] == 'weak' for analysis in results['claim_analysis']):
        results['recommendations'].append("⚠️ Some claims lack strong evidence - seek additional sources")
    
    results['recommendations'].extend([
        "🔍 Perform reverse image search if images are involved",
        "📰 Check multiple news sources for coverage",
        "🎯 Look for official statements or press releases",
        "📊 Verify statistics with original sources",
        "⏰ Check the timeliness of information"
    ])
    
    # 6. OpenAI AI-Powered Analysis (if available)
    if API_CONFIG['openai'] and API_CONFIG['openai'] != "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":
        st.info("🤖 Running OpenAI AI-powered fact-checking analysis...")
        openai_results = openai_fact_check(text)
        
        if openai_results['openai_analysis']:
            results['openai_analysis'] = openai_results['openai_analysis']
            
            # Integrate OpenAI insights into recommendations
            if openai_results['openai_analysis'].get('red_flags'):
                for flag in openai_results['openai_analysis']['red_flags']:
                    results['recommendations'].append(f"🚨 AI Detected: {flag}")
            
            if openai_results['openai_analysis'].get('green_flags'):
                for flag in openai_results['openai_analysis']['green_flags']:
                    results['recommendations'].append(f"✅ AI Detected: {flag}")
            
            # Add AI-specific recommendations
            if openai_results['openai_analysis'].get('verification_recommendations'):
                for rec in openai_results['openai_analysis']['verification_recommendations']:
                    results['recommendations'].append(f"🤖 AI Suggests: {rec}")
        else:
            results['openai_analysis'] = {'error': openai_results['error']}
    else:
        results['openai_analysis'] = {'error': 'OpenAI API not configured'}
    
    return results

def openai_fact_check(text: str) -> Dict:
    """Enhanced fact-checking using OpenAI's GPT models for intelligent analysis."""
    if not API_CONFIG['openai'] or API_CONFIG['openai'] == "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":
        return {
            'openai_analysis': None,
            'error': 'OpenAI API key not configured'
        }
    
    try:
        # Configure OpenAI
        openai.api_key = API_CONFIG['openai']
        
        # Create a comprehensive fact-checking prompt
        prompt = f"""
        You are an expert fact-checker and misinformation analyst. Analyze the following text for factual accuracy, potential misinformation, and credibility indicators.

        TEXT TO ANALYZE:
        {text[:2000]}  # Limit to first 2000 characters

        Please provide a comprehensive analysis in the following JSON format:
        {{
            "factual_accuracy_score": 0-100,
            "credibility_assessment": "high/medium/low",
            "key_claims_identified": ["claim1", "claim2"],
            "factual_issues": ["issue1", "issue2"],
            "verification_recommendations": ["rec1", "rec2"],
            "overall_assessment": "detailed explanation",
            "confidence_level": "high/medium/low",
            "red_flags": ["flag1", "flag2"],
            "green_flags": ["flag1", "flag2"]
        }}

        Focus on:
        1. Identifying specific factual claims
        2. Assessing the credibility of sources and claims
        3. Detecting potential misinformation patterns
        4. Providing actionable verification steps
        5. Rating overall factual accuracy
        """
        
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert fact-checker and misinformation analyst. Provide accurate, unbiased analysis in the requested JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        # Parse the response
        try:
            analysis = json.loads(response.choices[0].message.content)
            return {
                'openai_analysis': analysis,
                'error': None
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, extract key information from text
            content = response.choices[0].message.content
            return {
                'openai_analysis': {
                    'raw_response': content,
                    'factual_accuracy_score': 50,
                    'credibility_assessment': 'medium',
                    'key_claims_identified': [],
                    'factual_issues': [],
                    'verification_recommendations': ['Review the OpenAI response manually'],
                    'overall_assessment': content[:200] + "..." if len(content) > 200 else content,
                    'confidence_level': 'medium',
                    'red_flags': [],
                    'green_flags': []
                },
                'error': 'JSON parsing failed, using raw response'
            }
            
    except Exception as e:
        return {
            'openai_analysis': None,
            'error': f'OpenAI API error: {str(e)}'
        }

# --- UI HELPER FUNCTIONS ---

def display_enhanced_results(credibility_score, explanation, analysis_results=None, fact_check_results=None):
    """Displays enhanced analysis results with detailed breakdown."""
    st.subheader("🔍 Enhanced Analysis Results")

    # Enhanced traffic light color coding
    if credibility_score >= 75:
        color = "#28a745"  # Green
        emoji = "✅"
        status = "Highly Credible"
        confidence_color = "#155724"
    elif 60 <= credibility_score < 75:
        color = "#17a2b8"  # Blue
        emoji = "✅"
        status = "Likely Credible"
        confidence_color = "#0c5460"
    elif 40 <= credibility_score < 60:
        color = "#ffc107"  # Yellow
        emoji = "🤔"
        status = "Uncertain"
        confidence_color = "#856404"
    elif 20 <= credibility_score < 40:
        color = "#fd7e14"  # Orange
        emoji = "⚠️"
        status = "Potentially Misleading"
        confidence_color = "#8b4513"
    else:
        color = "#dc3545"  # Red
        emoji = "❌"
        status = "Likely Misinformation"
        confidence_color = "#721c24"

    # Display the main score
    st.markdown(f"""
    <div style="padding: 25px; border-radius: 15px; background: linear-gradient(135deg, {color}, {confidence_color}); color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h1 style="font-size: 3em; margin: 0;">{emoji}</h1>
        <h2 style="font-size: 2.5em; margin: 10px 0;">{int(credibility_score)}%</h2>
        <h3 style="font-size: 1.5em; margin: 0;">{status}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for detailed breakdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display the main explanation
        with st.expander("📋 Detailed Analysis Report", expanded=True):
            st.markdown(explanation)
            
            # Display analysis results if available
            if analysis_results:
                st.subheader("🔬 Technical Analysis Breakdown")
                
                # Model confidence
                if 'model_confidence' in analysis_results:
                    st.metric("AI Model Confidence", f"{analysis_results['model_confidence']:.1%}")
                
                # Source credibility
                if 'source_credibility' in analysis_results:
                    st.metric("Source Credibility", f"{analysis_results['source_credibility']:.1%}")
                
                # Claim specificity
                if 'claim_specificity' in analysis_results:
                    st.metric("Claim Specificity", f"{analysis_results['claim_specificity']:.1%}")
                
                # Ensemble scores
                if 'ensemble_scores' in analysis_results and analysis_results['ensemble_scores']:
                    st.subheader("🤖 Ensemble Model Scores")
                    for model_name, score, confidence in analysis_results['ensemble_scores']:
                        st.metric(f"{model_name.title()} Model", f"{score:.1f}%", f"Confidence: {confidence:.1%}")

    with col2:
        # Quick insights sidebar
        st.subheader("💡 Quick Insights")
        
        if analysis_results:
            # Emotional manipulation warning
            emotional_score = analysis_results.get('emotional_manipulation', {}).get('score', 0)
            if emotional_score > 0:
                st.warning(f"⚠️ {emotional_score} emotional manipulation techniques detected")
            
            # Logical fallacies warning
            fallacy_count = len(analysis_results.get('logical_fallacies', []))
            if fallacy_count > 0:
                st.error(f"🚨 {fallacy_count} logical fallacies detected")
            
            # Credible indicators
            if analysis_results.get('source_credibility', 0) > 0.7:
                st.success("✅ Source appears credible")
            
            if analysis_results.get('claim_specificity', 0) > 0.7:
                st.success("✅ Claims are specific and verifiable")

    # Enhanced fact-checking results with web scraping
    if fact_check_results:
        st.subheader("🔍 Enhanced Fact-Checking Results")
        
        # Overall summary
        st.info(f"**Overall Fact-Check Score:** {fact_check_results.get('overall_score', 0):.1f}%")
        st.info(f"**Verification Summary:** {fact_check_results.get('verification_summary', 'No summary available')}")
        
        # Claim-by-claim analysis
        if fact_check_results.get('claim_analysis'):
            st.subheader("📋 Claim-by-Claim Analysis")
            
            for i, claim_analysis in enumerate(fact_check_results['claim_analysis']):
                with st.expander(f"Claim {i+1}: {claim_analysis['claim'][:100]}...", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Credibility Score", f"{claim_analysis['credibility_score']:.1f}%")
                        st.metric("Evidence Strength", claim_analysis['evidence_strength'].title())
                        st.metric("Verification Status", claim_analysis['verification_status'].title())
                    
                    with col2:
                        # Supporting evidence
                        if claim_analysis['supporting_evidence']:
                            st.success("✅ Supporting Evidence:")
                            for evidence in claim_analysis['supporting_evidence'][:3]:
                                st.write(f"• {evidence}")
                        
                        # Contradicting evidence
                        if claim_analysis['contradicting_evidence']:
                            st.error("❌ Contradicting Evidence:")
                            for evidence in claim_analysis['contradicting_evidence'][:3]:
                                st.write(f"• {evidence}")
                    
                    # Web verification links
                    if claim_analysis['search_results']:
                        st.subheader("🔗 Web Verification Sources")
                        
                        # Fact-check sites
                        if claim_analysis['search_results'].get('fact_check_sites'):
                            st.write("**Fact-Check Sites:**")
                            for result in claim_analysis['search_results']['fact_check_sites'][:3]:
                                st.markdown(f"• [{result.get('title', 'Unknown')}]({result.get('full_url', '#')}) - {result.get('rating', 'Unknown rating')}")
                        
                        # Indian news sites
                        if claim_analysis['search_results'].get('news_articles'):
                            st.write("**Indian News Coverage:**")
                            for result in claim_analysis['search_results']['news_articles'][:3]:
                                st.markdown(f"• [{result.get('title', 'Unknown')}]({result.get('url', '#')}) - {result.get('source', 'Unknown source')}")
        
        # Recommendations
        if fact_check_results.get('recommendations'):
            st.subheader("💡 Verification Recommendations")
            for recommendation in fact_check_results['recommendations']:
                st.write(f"• {recommendation}")
        
        # Scraped sources summary
        if fact_check_results.get('scraped_sources'):
            st.subheader("📊 Sources Consulted")
            scraped = fact_check_results['scraped_sources']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fact-Check Sites", len(scraped.get('fact_check_sites', [])))
            with col2:
                st.metric("Indian News Articles", len(scraped.get('indian_news', [])))
            with col3:
                st.metric("Total Sources", len(scraped.get('fact_check_sites', [])) + len(scraped.get('indian_news', [])))
        
        # OpenAI AI-Powered Analysis Results
        if fact_check_results.get('openai_analysis') and fact_check_results['openai_analysis'].get('error') != 'OpenAI API not configured':
            st.subheader("🤖 OpenAI AI-Powered Analysis")
            
            openai_analysis = fact_check_results['openai_analysis']
            
            if openai_analysis.get('error'):
                st.warning(f"⚠️ OpenAI Analysis Error: {openai_analysis['error']}")
            else:
                # Display OpenAI analysis metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'factual_accuracy_score' in openai_analysis:
                        st.metric("AI Factual Accuracy", f"{openai_analysis['factual_accuracy_score']:.1f}%")
                with col2:
                    if 'credibility_assessment' in openai_analysis:
                        st.metric("AI Credibility", openai_analysis['credibility_assessment'].title())
                with col3:
                    if 'confidence_level' in openai_analysis:
                        st.metric("AI Confidence", openai_analysis['confidence_level'].title())
                
                # Display key claims identified by AI
                if openai_analysis.get('key_claims_identified'):
                    st.subheader("🎯 Key Claims Identified by AI")
                    for i, claim in enumerate(openai_analysis['key_claims_identified'][:3]):
                        st.write(f"{i+1}. {claim}")
                
                # Display factual issues detected by AI
                if openai_analysis.get('factual_issues'):
                    st.subheader("🚨 Factual Issues Detected by AI")
                    for issue in openai_analysis['factual_issues'][:3]:
                        st.error(f"• {issue}")
                
                # Display AI recommendations
                if openai_analysis.get('verification_recommendations'):
                    st.subheader("🤖 AI Verification Recommendations")
                    for rec in openai_analysis['verification_recommendations'][:3]:
                        st.info(f"• {rec}")
                
                # Display overall AI assessment
                if openai_analysis.get('overall_assessment'):
                    st.subheader("📋 AI Overall Assessment")
                    st.write(openai_analysis['overall_assessment'])
                
                # Display red and green flags
                col1, col2 = st.columns(2)
                with col1:
                    if openai_analysis.get('red_flags'):
                        st.subheader("🚨 AI Red Flags")
                        for flag in openai_analysis['red_flags'][:3]:
                            st.error(f"• {flag}")
                
                with col2:
                    if openai_analysis.get('green_flags'):
                        st.subheader("✅ AI Green Flags")
                        for flag in openai_analysis['green_flags'][:3]:
                            st.success(f"• {flag}")
        elif fact_check_results.get('openai_analysis') and fact_check_results['openai_analysis'].get('error') == 'OpenAI API not configured':
            st.info("🤖 OpenAI AI-powered analysis not available. Add your OpenAI API key to .streamlit/secrets.toml to enable this feature.")

    # User feedback section
    st.subheader("📝 Help Us Improve")
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
    
    with feedback_col1:
        if st.button("👍 Accurate Analysis"):
            st.success("Thank you for your feedback!")
    
    with feedback_col2:
        if st.button("👎 Inaccurate Analysis"):
            st.error("Thank you for your feedback. We'll use this to improve our model.")
    
    with feedback_col3:
        if st.button("❓ Report Issue"):
            st.info("Please contact us with details about the issue.")

# --- CORE ANALYSIS PIPELINES ---

def enhanced_text_analysis(text: str, source_url: Optional[str] = None, timestamp: Optional[str] = None) -> Tuple[float, str, Dict]:
    """Enhanced text analysis using ensemble models and multiple validation layers."""
    if not text.strip():
        return 0, "Input text is empty.", {}
    
    # Initialize analysis results
    analysis_results = {
        'model_confidence': 0.0,
        'source_credibility': 0.5,
        'claim_specificity': 0.5,
        'temporal_relevance': 0.7,
        'text_length': len(text),
        'logical_fallacies': [],
        'emotional_manipulation': {},
        'factual_contradictions': [],
        'ensemble_scores': []
    }
    
    text_lower = text.lower()
    
    # 1. SOURCE CREDIBILITY ANALYSIS
    analysis_results['source_credibility'] = analyze_source_credibility(source_url)
    
    # 2. TEMPORAL RELEVANCE
    analysis_results['temporal_relevance'] = check_temporal_relevance(timestamp)
    
    # 3. CLAIM SPECIFICITY ANALYSIS
    analysis_results['claim_specificity'] = analyze_claim_specificity(text)
    
    # 4. LOGICAL FALLACY DETECTION
    analysis_results['logical_fallacies'] = detect_logical_fallacies(text)
    
    # 5. EMOTIONAL MANIPULATION ANALYSIS
    analysis_results['emotional_manipulation'] = analyze_emotional_manipulation(text)
    
    # 6. FACTUAL CONTRADICTION CHECKS
    for person, wrong_country, correct_role in PATTERN_CONFIG['factual_contradictions']:
        if person in text_lower and wrong_country in text_lower:
            return 5, (
                f"**🚨 FACTUAL ERROR DETECTED:**\n\n"
                f"This contains a basic factual error. {person.title()} is not the president of {wrong_country.title()}. "
                f"{person.title()} is the {correct_role}.\n\n"
                f"**Confidence Level:** HIGH\n"
                f"**Recommendation:** Verify basic facts before sharing information."
            ), analysis_results
    
    # 7. DEATH HOAX DETECTION
    death_hoax_count = sum(1 for phrase in PATTERN_CONFIG['death_hoax_indicators'] if phrase in text_lower)
    if death_hoax_count > 0 and any(name in text_lower for name in ['modi', 'putin', 'biden', 'trump', 'obama']):
        return 5, (
            f"**🚨 DEATH HOAX DETECTED:**\n\n"
            f"This appears to be a death hoax about a living public figure. "
            f"Such claims are common forms of misinformation spread on social media.\n\n"
            f"**Confidence Level:** HIGH\n"
            f"**Recommendation:** Always verify such claims with official sources before sharing."
        ), analysis_results
    
    # 8. ENSEMBLE MODEL ANALYSIS
    try:
        if models:
            # Primary model analysis
            primary_results = models['primary'](text)
            is_real = primary_results[0]['label'] == 'LABEL_1'
            primary_confidence = primary_results[0]['score']
            primary_score = primary_confidence * 100 if is_real else (1 - primary_confidence) * 100
            analysis_results['ensemble_scores'].append(('primary', primary_score, primary_confidence))
            
            # Zero-shot classification for additional validation
            try:
                zero_shot_candidates = ["fake news", "misinformation", "credible news", "factual information"]
                zero_shot_results = models['zero_shot'](text, zero_shot_candidates)
                zero_shot_score = zero_shot_results['scores'][zero_shot_results['labels'].index('credible news')] * 100
                analysis_results['ensemble_scores'].append(('zero_shot', zero_shot_score, zero_shot_results['scores'][0]))
            except Exception as e:
                st.warning(f"Zero-shot analysis failed: {e}")
            
            # Calculate ensemble score (weighted average)
            total_weight = 0
            weighted_score = 0
            
            for model_name, score, confidence in analysis_results['ensemble_scores']:
                weight = confidence if model_name == 'primary' else 0.5
                weighted_score += score * weight
                total_weight += weight
            
            ensemble_score = weighted_score / total_weight if total_weight > 0 else primary_score
            analysis_results['model_confidence'] = primary_confidence
            
        else:
            ensemble_score = 50  # Neutral score if models fail
            analysis_results['model_confidence'] = 0.5
            
    except Exception as e:
        st.error(f"Model analysis error: {e}")
        ensemble_score = 50
        analysis_results['model_confidence'] = 0.5
    
    # 9. APPLY ENHANCED ADJUSTMENTS
    final_score = ensemble_score
    
    # Emotional manipulation penalty
    emotional_score = analysis_results['emotional_manipulation'].get('score', 0)
    if emotional_score > 0:
        emotional_penalty = min(25, emotional_score * 5)
        final_score = max(0, final_score - emotional_penalty)
    
    # Logical fallacy penalty
    fallacy_penalty = len(analysis_results['logical_fallacies']) * 10
    final_score = max(0, final_score - fallacy_penalty)
    
    # Source credibility adjustment
    source_adjustment = (analysis_results['source_credibility'] - 0.5) * 20
    final_score = max(0, min(100, final_score + source_adjustment))
    
    # Claim specificity bonus
    specificity_bonus = (analysis_results['claim_specificity'] - 0.5) * 10
    final_score = max(0, min(100, final_score + specificity_bonus))
    
    # 10. GENERATE ENHANCED EXPLANATION
    confidence_level, confidence_description = get_confidence_level(final_score, analysis_results)
    
    explanation = f"""
    **🤖 AI MODEL ANALYSIS:**
    - **Primary Model Classification:** {'REAL' if is_real else 'FAKE'} ({primary_confidence:.1%} confidence)
    - **Ensemble Score:** {ensemble_score:.1f}%
    - **Final Adjusted Score:** {final_score:.1f}%
    - **Confidence Level:** {confidence_level.upper()} - {confidence_description}
    
    **🔍 DETAILED ANALYSIS:**
    """
    
    # Add source analysis
    if source_url:
        explanation += f"- **Source Credibility:** {analysis_results['source_credibility']:.1%} "
        if analysis_results['source_credibility'] > 0.7:
            explanation += "✅ (Credible source)\n"
        elif analysis_results['source_credibility'] < 0.3:
            explanation += "❌ (Suspicious source)\n"
        else:
            explanation += "⚠️ (Unknown source)\n"
    
    # Add claim specificity
    explanation += f"- **Claim Specificity:** {analysis_results['claim_specificity']:.1%} "
    if analysis_results['claim_specificity'] > 0.7:
        explanation += "✅ (Specific and verifiable)\n"
    elif analysis_results['claim_specificity'] < 0.3:
        explanation += "❌ (Vague and unverifiable)\n"
    else:
        explanation += "⚠️ (Moderately specific)\n"
    
    # Add emotional manipulation analysis
    if emotional_score > 0:
        explanation += f"- **⚠️ Emotional Manipulation Detected:** {emotional_score} manipulation techniques found\n"
        for technique in analysis_results['emotional_manipulation'].get('techniques', [])[:3]:
            explanation += f"  • {technique}\n"
    
    # Add logical fallacies
    if analysis_results['logical_fallacies']:
        explanation += f"- **⚠️ Logical Fallacies Detected:** {len(analysis_results['logical_fallacies'])} fallacies found\n"
        for fallacy in analysis_results['logical_fallacies'][:2]:
            explanation += f"  • {fallacy}\n"
    
    # Add temporal relevance
    if timestamp:
        explanation += f"- **Temporal Relevance:** {analysis_results['temporal_relevance']:.1%} "
        if analysis_results['temporal_relevance'] > 0.8:
            explanation += "✅ (Recent information)\n"
        elif analysis_results['temporal_relevance'] < 0.4:
            explanation += "❌ (Potentially outdated)\n"
        else:
            explanation += "⚠️ (Moderately recent)\n"
    
    explanation += f"""
    
    **💡 VERIFICATION RECOMMENDATIONS:**
    1. Cross-reference with multiple reliable sources
    2. Check for recent updates on this topic
    3. Verify claims with official statements
    4. Look for peer-reviewed research when applicable
    5. Be cautious of emotionally charged language
    """
    
    return final_score, explanation, analysis_results


def enhanced_image_analysis(image_file):
    """Enhanced image analysis using multiple techniques."""
    try:
        # Read image from file-like object (Streamlit uploader)
        image_bytes = image_file.read()
        original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        analysis_results = {
            'ela_score': 0,
            'metadata_analysis': {},
            'face_detection': {},
            'compression_analysis': {},
            'overall_score': 0
        }

        # 1. ERROR LEVEL ANALYSIS (ELA)
        st.subheader("🔍 Error Level Analysis (ELA)")
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            original_image.save(temp_file, 'JPEG', quality=95)
            temp_filename = temp_file.name

        resaved_image = Image.open(temp_filename)
        os.remove(temp_filename)

        ela_image = ImageChops.difference(original_image, resaved_image)

        # Calculate the scale for visualization
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff

        ela_image = Image.eval(ela_image, lambda x: x * scale)

        # Calculate ELA score
        ela_array = np.array(ela_image)
        mean_ela = ela_array.mean()
        std_ela = ela_array.std()

        # Enhanced ELA scoring
        if mean_ela < 5:
            ela_score = 85  # Very uniform compression
        elif mean_ela < 15:
            ela_score = 70  # Moderately uniform
        elif mean_ela < 30:
            ela_score = 50  # Some variation
        else:
            ela_score = 20  # High variation, potential manipulation

        analysis_results['ela_score'] = ela_score
        analysis_results['compression_analysis'] = {
            'mean_ela': mean_ela,
            'std_ela': std_ela,
            'score': ela_score
        }

        # Display ELA results
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(ela_image, caption="ELA Analysis Result", use_column_width=True)

        st.metric("ELA Score", f"{ela_score:.1f}%", f"Mean ELA: {mean_ela:.2f}")

        # 2. METADATA ANALYSIS
        st.subheader("📊 Metadata Analysis")
        try:
            metadata = original_image.info
            exif_data = original_image.getexif()

            metadata_analysis = {
                'has_exif': len(exif_data) > 0,
                'exif_count': len(exif_data),
                'software_used': metadata.get('Software', 'Unknown'),
                'creation_time': metadata.get('Creation Time', 'Unknown'),
                'modification_time': metadata.get('Modification Time', 'Unknown')
            }

            analysis_results['metadata_analysis'] = metadata_analysis

            # Display metadata
            if metadata_analysis['has_exif']:
                st.success(f"✅ EXIF data found: {metadata_analysis['exif_count']} fields")
                if metadata_analysis['software_used'] != 'Unknown':
                    st.info(f"Software: {metadata_analysis['software_used']}")
            else:
                st.warning("⚠️ No EXIF metadata found - this may indicate the image has been processed")

        except Exception as e:
            st.warning(f"Metadata analysis failed: {e}")

        # 3. FACE DETECTION (OpenCV)
        st.subheader("👤 Face Detection Analysis")
        try:
            # Convert PIL image to OpenCV format
            img_array = np.array(original_image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Load face detection model
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            face_analysis = {
                'face_count': len(faces),
                'faces_detected': len(faces) > 0,
                'face_areas': []
            }

            if len(faces) > 0:
                st.success(f"✅ {len(faces)} face(s) detected")
                for (x, y, w, h) in faces:
                    face_area = w * h
                    face_analysis['face_areas'].append(face_area)
                    img_area = img_cv.shape[0] * img_cv.shape[1]
                    face_ratio = face_area / img_area
                    if face_ratio > 0.3:
                        st.warning("⚠️ Unusually large face detected - possible AI generation")
                    elif face_ratio < 0.01:
                        st.warning("⚠️ Unusually small face detected")
                # Draw faces on image
                img_with_faces = img_cv.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
                img_with_faces_rgb = cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB)
                st.image(img_with_faces_rgb, caption="Face Detection Result", use_column_width=True)
            else:
                st.info("ℹ️ No faces detected in the image")

            analysis_results['face_detection'] = face_analysis

        except Exception as e:
            st.warning(f"Face detection failed: {e}")

        # 4. COMPRESSION ARTIFACT ANALYSIS
        st.subheader("🔧 Compression Analysis")
        try:
            gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
            dct = cv2.dct(np.float32(gray))
            dct_abs = np.abs(dct)
            high_freq_energy = np.sum(dct_abs[8:, 8:])
            total_energy = np.sum(dct_abs)
            compression_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

            compression_analysis = {
                'compression_ratio': compression_ratio,
                'high_freq_energy': high_freq_energy,
                'total_energy': total_energy
            }

            analysis_results['compression_analysis'].update(compression_analysis)

            if compression_ratio < 0.1:
                st.success("✅ Low compression artifacts - image appears natural")
            elif compression_ratio < 0.3:
                st.info("ℹ️ Moderate compression artifacts")
            else:
                st.warning("⚠️ High compression artifacts detected")

        except Exception as e:
            st.warning(f"Compression analysis failed: {e}")

        # 5. CALCULATE OVERALL SCORE
        overall_score = ela_score * 0.4  # ELA is primary indicator

        # Metadata bonus/penalty
        if analysis_results['metadata_analysis'].get('has_exif', False):
            overall_score += 10
        else:
            overall_score -= 5

        # Face detection bonus
        if analysis_results['face_detection'].get('faces_detected', False):
            face_count = analysis_results['face_detection'].get('face_count', 0)
            if face_count == 1:
                overall_score += 5
            elif face_count > 5:
                overall_score -= 10

        # Compression artifact penalty
        compression_ratio = analysis_results['compression_analysis'].get('compression_ratio', 0)
        if compression_ratio > 0.3:
            overall_score -= 15

        overall_score = max(0, min(100, overall_score))
        analysis_results['overall_score'] = overall_score

        # 6. GENERATE EXPLANATION
        explanation = f"""
        **🔍 ENHANCED IMAGE ANALYSIS RESULTS:**

        **📊 Overall Credibility Score:** {overall_score:.1f}%

        **🔧 Technical Analysis:**
        - **Error Level Analysis (ELA):** {ela_score:.1f}% - {get_ela_description(ela_score)}
        - **Metadata Analysis:** {'✅ EXIF data present' if analysis_results['metadata_analysis'].get('has_exif') else '⚠️ No EXIF data'}
        - **Face Detection:** {analysis_results['face_detection'].get('face_count', 0)} face(s) detected
        - **Compression Analysis:** {get_compression_description(compression_ratio)}

        **💡 Key Findings:
        """

        if ela_score < 30:
            explanation += "• High ELA variation detected - potential manipulation\n"
        if not analysis_results['metadata_analysis'].get('has_exif'):
            explanation += "• No EXIF metadata - image may have been processed\n"
        if analysis_results['face_detection'].get('face_count', 0) > 5:
            explanation += "• Unusually high number of faces - verify authenticity\n"
        if compression_ratio > 0.3:
            explanation += "• High compression artifacts - possible multiple edits\n"

        explanation += f"""

        **🔍 VERIFICATION RECOMMENDATIONS:**
        1. Perform reverse image search on Google Images or TinEye
        2. Check the original source of the image
        3. Look for multiple versions of the same image
        4. Verify with the original photographer or source
        5. Check for recent news coverage of the event depicted

        **⚠️ DISCLAIMER:** This analysis provides technical indicators but is not definitive proof of manipulation. Always verify with multiple sources.
        """

        return overall_score, explanation, analysis_results

    except Exception as e:
        st.error(f"An error occurred during enhanced image analysis: {e}")
        return 0, f"Error during analysis: {e}", {}


def enhanced_video_analysis(video_path):
    """Enhanced video analysis with comprehensive audio and visual analysis."""
    # Check if required libraries are available
    if WHISPER_IMPORT_ERROR:
        st.error(f"❌ Whisper library not available: {WHISPER_IMPORT_ERROR}")
        return None
    
    if CV2_IMPORT_ERROR:
        st.error(f"❌ OpenCV library not available: {CV2_IMPORT_ERROR}")
        return None
    
    if NUMPY_IMPORT_ERROR:
        st.error(f"❌ NumPy library not available: {NUMPY_IMPORT_ERROR}")
        return None
    
    if YT_DLP_IMPORT_ERROR:
        st.error(f"❌ yt-dlp library not available: {YT_DLP_IMPORT_ERROR}")
        return None
    
    text_score, text_exp, text_analysis_results = 0, "No audio found or transcribed.", {}
    frame_scores = []
    video_analysis_results = {
        'audio_analysis': {},
        'frame_analysis': [],
        'overall_score': 0
    }
    
    with st.spinner("🔍 Performing comprehensive video analysis... This may take several minutes."):
        # 1. ENHANCED AUDIO ANALYSIS
        st.subheader("🎤 Audio Analysis")
        st.write("Step 1/3: Transcribing and analyzing audio content...")
        
        try:
            if not whisper_model:
                st.error("❌ Whisper model not available")
                return None
            
            result = whisper_model.transcribe(video_path, fp16=False)
            transcribed_text = result["text"]
            
            if transcribed_text.strip():
                st.success(f"✅ Audio transcribed successfully ({len(transcribed_text)} characters)")
                st.info(f"**Transcribed Text Preview:**\n\n> {transcribed_text[:300]}...")
                
                # Enhanced text analysis
                text_score, text_exp, text_analysis_results = enhanced_text_analysis(transcribed_text)
                video_analysis_results['audio_analysis'] = {
                    'transcribed_text': transcribed_text,
                    'text_score': text_score,
                    'analysis_results': text_analysis_results
                }
            else:
                text_exp = "The video contains no discernible speech."
                video_analysis_results['audio_analysis'] = {
                    'transcribed_text': "",
                    'text_score': 0,
                    'analysis_results': {}
                }
        except Exception as e:
            text_exp = f"Could not process audio: {e}"
            st.warning(text_exp)
            video_analysis_results['audio_analysis'] = {
                'transcribed_text': "",
                'text_score': 0,
                'analysis_results': {}
            }

        # 2. ENHANCED FRAME ANALYSIS
        st.subheader("🎬 Visual Frame Analysis")
        st.write("Step 2/3: Analyzing key video frames for manipulation...")
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = frame_count / fps if fps > 0 else 0
            
            st.info(f"Video Info: {frame_count} frames, {fps} FPS, {duration:.1f} seconds")
            
            # Analyze frames at regular intervals
            num_frames_to_analyze = min(10, frame_count)  # Analyze up to 10 frames
            frame_indices = np.linspace(0, frame_count - 1, num=num_frames_to_analyze, dtype=int)
            
            progress_bar = st.progress(0)
            frame_analysis_container = st.container()
            
            with frame_analysis_container:
                for i, frame_index in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Convert frame to PIL Image for analysis
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Save to a temporary buffer for enhanced image analysis
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                            pil_image.save(f, format='PNG')
                            frame_score, frame_exp, frame_results = enhanced_image_analysis(f.name)
                            frame_scores.append(frame_score)
                            
                            video_analysis_results['frame_analysis'].append({
                                'frame_index': frame_index,
                                'timestamp': frame_index / fps if fps > 0 else 0,
                                'score': frame_score,
                                'results': frame_results
                            })
                        os.remove(f.name)
                    
                    progress_bar.progress((i + 1) / len(frame_indices))
            
            cap.release()
            
            if frame_scores:
                avg_frame_score = sum(frame_scores) / len(frame_scores)
                st.success(f"✅ Frame analysis complete. Average score: {avg_frame_score:.1f}%")
            else:
                avg_frame_score = 50  # Neutral score if no frames analyzed
                st.warning("⚠️ No frames could be analyzed")
                
        except Exception as e:
            st.warning(f"Could not process video frames: {e}")
            avg_frame_score = 50

        # 3. METADATA AND TECHNICAL ANALYSIS
        st.subheader("🔧 Technical Video Analysis")
        st.write("Step 3/3: Analyzing video metadata and technical characteristics...")
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_name = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
            
            technical_analysis = {
                'resolution': f"{width}x{height}",
                'codec': codec_name,
                'frame_count': frame_count,
                'fps': fps,
                'duration': duration,
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
            
            video_analysis_results['technical_analysis'] = technical_analysis
            
            # Display technical info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resolution", technical_analysis['resolution'])
            with col2:
                st.metric("Duration", f"{duration:.1f}s")
            with col3:
                st.metric("File Size", f"{technical_analysis['file_size_mb']:.1f} MB")
            
            # Check for suspicious characteristics
            suspicious_indicators = []
            if width < 480 or height < 360:
                suspicious_indicators.append("Low resolution video")
            if fps < 15:
                suspicious_indicators.append("Unusually low frame rate")
            if duration < 1:
                suspicious_indicators.append("Very short video")
            
            if suspicious_indicators:
                st.warning(f"⚠️ Suspicious characteristics detected: {', '.join(suspicious_indicators)}")
            else:
                st.success("✅ Video characteristics appear normal")
            
            cap.release()
            
        except Exception as e:
            st.warning(f"Technical analysis failed: {e}")

    # 4. CALCULATE OVERALL SCORE
    # Weight the components: audio (60%), visual (30%), technical (10%)
    audio_weight = 0.6
    visual_weight = 0.3
    technical_weight = 0.1
    
    # Technical score based on video quality
    technical_score = 80  # Base score
    if 'technical_analysis' in video_analysis_results:
        tech_analysis = video_analysis_results['technical_analysis']
        if tech_analysis.get('width', 0) >= 1920:  # HD or better
            technical_score += 10
        if tech_analysis.get('fps', 0) >= 30:  # Good frame rate
            technical_score += 10
        if tech_analysis.get('duration', 0) > 5:  # Reasonable length
            technical_score += 10
    
    final_score = (
        text_score * audio_weight +
        avg_frame_score * visual_weight +
        technical_score * technical_weight
    )
    
    video_analysis_results['overall_score'] = final_score
    
    # 5. GENERATE COMPREHENSIVE EXPLANATION
    explanation = f"""
    **🎬 ENHANCED VIDEO ANALYSIS RESULTS:**
    
    **📊 Overall Credibility Score:** {final_score:.1f}%
    
    **🎤 Audio Analysis (Weight: 60%):**
    - **Spoken Content Score:** {text_score:.1f}%
    - **Transcription Length:** {len(video_analysis_results['audio_analysis'].get('transcribed_text', ''))} characters
    - **Analysis:** {text_exp[:200]}...
    
    **🎬 Visual Analysis (Weight: 30%):**
    - **Frame Analysis Score:** {avg_frame_score:.1f}%
    - **Frames Analyzed:** {len(frame_scores)} frames
    - **Frame Score Range:** {min(frame_scores):.1f}% - {max(frame_scores):.1f}% (if frames analyzed)
    
    **🔧 Technical Analysis (Weight: 10%):**
    - **Technical Score:** {technical_score:.1f}%
    - **Video Quality:** {video_analysis_results.get('technical_analysis', {}).get('resolution', 'Unknown')}
    - **Duration:** {video_analysis_results.get('technical_analysis', {}).get('duration', 0):.1f} seconds
    """
    
    if suspicious_indicators:
        explanation += f"\n**⚠️ Technical Warnings:** {', '.join(suspicious_indicators)}"
    
    explanation += f"""
    
    **💡 VERIFICATION RECOMMENDATIONS:**
    1. Verify the original source of the video
    2. Check for multiple versions of the same video
    3. Look for official statements about the events depicted
    4. Cross-reference with news coverage
    5. Check the credibility of the video uploader
    6. Look for timestamps and location data
    
    **⚠️ DISCLAIMER:** Video analysis is complex and may not detect all forms of manipulation. Always verify with multiple sources.
    """
    
    return final_score, explanation, video_analysis_results, transcribed_text if 'transcribed_text' in locals() else ""


def enhanced_link_handler(url):
    """Enhanced link handler with better content extraction and analysis."""
    try:
        parsed_url = urlparse(url)
        
        # --- YouTube / Instagram Handler ---
        if 'youtube.com' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc or 'instagram.com' in parsed_url.netloc:
            if YT_DLP_IMPORT_ERROR:
                st.error(f"❌ yt-dlp library not available: {YT_DLP_IMPORT_ERROR}")
                st.info("Video analysis will be skipped. Please install yt-dlp for video link analysis.")
                return 0, "Video analysis not available due to missing yt-dlp library.", {}, ""
            
            st.info("🎬 Detected a video link. Downloading and analyzing content...")
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'outtmpl': os.path.join(temp_dir, 'video.mp4'),
                    'noplaylist': True,
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                video_path = os.path.join(temp_dir, 'video.mp4')
                if os.path.exists(video_path):
                    return enhanced_video_analysis(video_path)
                else:
                    st.error("❌ Could not download video from the link.")
                    return 0, "Video download failed.", {}, ""

        # --- News Article Handler ---
        else:
            st.info("📰 Detected a news article link. Extracting and analyzing content...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Enhanced content extraction
                article_text = ""
                
                # Try multiple extraction strategies
                extraction_methods = [
                    # Method 1: Look for article content
                    lambda: ' '.join([p.get_text().strip() for p in soup.find_all(['p', 'article']) if len(p.get_text().strip()) > 50]),
                    # Method 2: Look for main content area
                    lambda: ' '.join([p.get_text().strip() for p in soup.select('main p, .content p, .article-content p, .post-content p') if len(p.get_text().strip()) > 30]),
                    # Method 3: Look for any paragraph with substantial text
                    lambda: ' '.join([p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 100]),
                    # Method 4: Extract title and description
                    lambda: f"{soup.find('title').get_text() if soup.find('title') else ''} {soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else ''}"
                ]
                
                for method in extraction_methods:
                    try:
                        extracted_text = method()
                        if len(extracted_text) > len(article_text):
                            article_text = extracted_text
                    except:
                        continue
                
                # Fallback to basic extraction
                if len(article_text) < 100:
                    paragraphs = soup.find_all('p')
                    article_text = ' '.join([p.get_text() for p in paragraphs])
                
                if len(article_text) < 50:
                    st.warning("⚠️ Could not extract significant text. The result might be inaccurate.")
                    article_text = f"Content from: {url}"
                
                # Get publication date if available
                publication_date = None
                date_selectors = [
                    'meta[property="article:published_time"]',
                    'meta[name="pubdate"]',
                    'meta[name="publish_date"]',
                    '.published-date',
                    '.article-date',
                    'time'
                ]
                
                for selector in date_selectors:
                    try:
                        date_element = soup.select_one(selector)
                        if date_element:
                            publication_date = date_element.get('content') or date_element.get('datetime') or date_element.get_text()
                            break
                    except:
                                                                                                                                                                                                                                                                                             continue
                
                # Enhanced text analysis with source URL and timestamp

                score, explanation, analysis_results = enhanced_text_analysis(article_text, url, publication_date)
                return score, explanation, analysis_results, article_text

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to fetch content from URL: {e}")
                return 0, f"Could not fetch content from the link. Error: {e}", {}, ""

    except Exception as e:
        st.error(f"❌ Failed to process URL: {e}")
        return 0, f"Could not process the link. Error: {e}", {}, ""



def fact_check(query):
    """Legacy function - redirects to enhanced_fact_check for backward compatibility."""
    return enhanced_fact_check(query)

# --- STREAMLIT UI ---

st.title("🛡️ Veritas: Advanced AI Misinformation Detector")
st.markdown("**Your comprehensive tool to analyze content and fight fake news using cutting-edge AI technology.**")
st.markdown("---")

# Enhanced sidebar with model status
with st.sidebar:
    st.header("🔧 System Status")
    if models:
        st.success("✅ AI Models Loaded")
        st.info(f"• Primary Model: {MODEL_CONFIG['primary_model'].split('/')[-1]}")
        st.info(f"• Zero-shot Model: {MODEL_CONFIG['secondary_models'][1].split('/')[-1]}")
    else:
        st.error("❌ AI Models Failed to Load")
    
    st.header("📊 API Status")
    api_status = {
        "Google Fact Check": API_CONFIG['google_fact_check'] is not None,
        "News API": API_CONFIG['news_api'] is not None,
        "Snopes": API_CONFIG['snopes_api'] is not None,
        "PolitiFact": API_CONFIG['politifact_api'] is not None
    }
    
    for api_name, status in api_status.items():
        if status:
            st.success(f"✅ {api_name}")
        else:
            st.warning(f"⚠️ {api_name} (Not configured)")
    
    st.header("💡 Quick Tips")
    st.info("""
    • **Text Analysis:** Best for news articles, social media posts, and claims
    • **Image Analysis:** Detects manipulation using multiple techniques
    • **Video Analysis:** Comprehensive audio and visual analysis
    • **Link Analysis:** Direct analysis of web content and videos
    """)
    
    st.header("🔍 Quick Fact-Check")
    quick_query = st.text_input("Enter a claim to fact-check:", placeholder="e.g., Narendra Modi is dead")
    if st.button("🔍 Quick Check", key="quick_fact_check"):
        if quick_query:
            with st.spinner("🔍 Performing quick fact-check..."):
                fact_check_results = enhanced_fact_check(quick_query)
                st.info(f"**Overall Score:** {fact_check_results.get('overall_score', 0):.1f}%")
                st.info(f"**Summary:** {fact_check_results.get('verification_summary', 'No summary')}")
                
                if fact_check_results.get('recommendations'):
                    st.write("**Recommendations:**")
                    for rec in fact_check_results['recommendations'][:3]:
                        st.write(f"• {rec}")
        else:
            st.warning("Please enter a claim to check.")

# Create tabs for different input types
tab1, tab2, tab3, tab4 = st.tabs(["**✍️ Text Analysis**", "**🖼️ Image Analysis**", "**🎬 Video Analysis**", "**🔗 Link Analysis**"])

with tab1:
    st.header("📝 Enhanced Text Analysis")
    st.markdown("Analyze news articles, social media posts, claims, and any text content for misinformation.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area(
            "Paste the text you want to analyze:", 
            height=200, 
            placeholder="Example: Scientists have discovered that chocolate cures all diseases..."
        )
    
    with col2:
        st.markdown("**📋 Analysis Options:**")
        source_url = st.text_input("Source URL (optional):", placeholder="https://example.com")
        timestamp = st.text_input("Publication Date (optional):", placeholder="2024-01-15")
    
    if st.button("🔍 Analyze Text", key="text_btn", type="primary"):
        if text_input:
            with st.spinner("🤖 Performing comprehensive text analysis..."):
                credibility_score, explanation, analysis_results = enhanced_text_analysis(text_input, source_url, timestamp)
                fact_check_results = enhanced_fact_check(text_input[:500])  # Use first 500 chars for query
                display_enhanced_results(credibility_score, explanation, analysis_results, fact_check_results)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.header("🖼️ Enhanced Image Analysis")
    st.markdown("Detect image manipulation using Error Level Analysis (ELA), metadata analysis, face detection, and compression analysis.")
    
    image_file = st.file_uploader(
        "Upload an image (JPG, PNG) to check for manipulation:", 
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG. Maximum file size: 10MB"
    )
    
    if st.button("🔍 Analyze Image", key="img_btn", type="primary"):
        if image_file:
            with st.spinner("🔍 Performing comprehensive image analysis..."):
                credibility_score, explanation, analysis_results = enhanced_image_analysis(image_file)
                # Add OpenAI fact-checking for image analysis
                fact_check_results = enhanced_fact_check("Image analysis completed. Please review the visual analysis results for any suspicious indicators or manipulation signs.")
                display_enhanced_results(credibility_score, explanation, analysis_results, fact_check_results)
        else:
            st.warning("⚠️ Please upload an image file.")

with tab3:
    st.header("🎬 Enhanced Video Analysis")
    st.markdown("Comprehensive video analysis including audio transcription, frame analysis, and technical metadata.")
    
    video_file = st.file_uploader(
        "Upload a video file (MP4):", 
        type=['mp4'],
        help="Supported format: MP4. Maximum file size: 100MB"
    )
    
    if st.button("🔍 Analyze Video", key="vid_btn", type="primary"):
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_file.read())
                video_path = tfile.name
            
            with st.spinner("🎬 Performing comprehensive video analysis..."):
                score, explanation, video_results, transcribed_text = enhanced_video_analysis(video_path)
                fact_check_results = enhanced_fact_check(transcribed_text[:500]) if transcribed_text else None
                display_enhanced_results(score, explanation, video_results, fact_check_results)
            
            os.remove(video_path)  # Clean up temp file
        else:
            st.warning("⚠️ Please upload a video file.")

with tab4:
    st.header("🔗 Enhanced Link Analysis")
    st.markdown("Direct analysis of web content, news articles, YouTube videos, and social media posts.")
    
    url_input = st.text_input(
        "Paste the URL here:", 
        placeholder="https://www.example.com/news-article or https://youtube.com/watch?v=..."
    )
    
    if st.button("🔍 Analyze Link", key="link_btn", type="primary"):
        if url_input:
            with st.spinner("🔗 Fetching and analyzing content..."):
                score, explanation, analysis_results, extracted_text = enhanced_link_handler(url_input)
                fact_check_results = enhanced_fact_check(extracted_text[:500]) if extracted_text else None
                display_enhanced_results(score, explanation, analysis_results, fact_check_results)
        else:
            st.warning("⚠️ Please enter a URL.")

# Enhanced test section
with st.expander("🧪 Advanced Test Suite", expanded=False):
    st.header("🧪 Test the Enhanced Analysis")
    st.markdown("Test the system with various types of content to see how it performs.")
    
    test_categories = {
        "🚨 Obvious Misinformation": [
            "Narendra Modi is dead",
            "Putin is president of America",
            "Trump is president of India",
            "China is part of USA"
        ],
        "✅ Likely Credible": [
            "Scientists discover new planet in solar system",
            "Study shows coffee reduces cancer risk",
            "NASA confirms successful Mars landing",
            "WHO releases new health guidelines"
        ],
        "🤔 Sensationalist": [
            "You won't believe what doctors found!",
            "Shocking discovery that will change everything!",
            "Big Pharma doesn't want you to know this!",
            "Breaking: Incredible revelation exposed!"
        ],
        "🔬 Scientific Claims": [
            "Research published in Nature shows climate change acceleration",
            "Peer-reviewed study confirms vaccine effectiveness",
            "University study finds correlation between diet and health",
            "Meta-analysis of 50 studies reveals new insights"
        ]
    }
    
    selected_category = st.selectbox("Choose test category:", list(test_categories.keys()))
    
    if selected_category:
        test_texts = test_categories[selected_category]
        
        for i, test_text in enumerate(test_texts):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(test_text)
            with col2:
                if st.button(f"Test {i+1}", key=f"test_{selected_category}_{i}"):
                    with st.spinner("Testing..."):
                        score, explanation, analysis_results = enhanced_text_analysis(test_text)
                        fact_check_results = enhanced_fact_check(test_text[:500])
                        display_enhanced_results(score, explanation, analysis_results, fact_check_results)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h3>🛡️ Veritas: Advanced AI Misinformation Detector</h3>
    <p>Built with ❤️ using Streamlit, Hugging Face, OpenAI Whisper, and cutting-edge AI technology</p>
    <p><strong>Features:</strong> Multi-model ensemble • Enhanced fact-checking • Advanced pattern detection • Source credibility analysis • User feedback system</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**⚠️ DISCLAIMER:** This tool uses advanced AI technology but is not infallible. Always cross-reference with multiple trusted sources. 
The analysis provides technical indicators and should be used as part of a comprehensive fact-checking process.
""")