#!/usr/bin/env python3
"""
Test script for enhanced fact-checking functionality with multiple APIs
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

def test_enhanced_fact_checking():
    """Test the enhanced fact-checking with multiple APIs"""
    print("🧪 Testing Enhanced Fact-Checking...")
    print("=" * 60)
    
    # Test cases that should trigger enhanced analysis
    test_cases = [
        {
            "name": "False Claim - Miracle Cure",
            "text": "Scientists discover miracle cure for all diseases that guarantees 100% success rate. This breakthrough treatment was developed by Dr. Smith and will be available for free to everyone."
        },
        {
            "name": "Uncertain Claim - Recent Event",
            "text": "Breaking news: Major breakthrough in quantum computing announced today. Scientists claim to have achieved quantum supremacy with a new algorithm."
        },
        {
            "name": "True Claim - Historical Fact",
            "text": "The Earth orbits around the Sun. This was first proposed by Nicolaus Copernicus in the 16th century."
        }
    ]
    
    url = "http://localhost:5000/api/verify-news"
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            response = requests.post(url, json={"news_text": test_case['text']})
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Analysis completed!")
                print(f"   Verdict: {result.get('verdict', 'N/A')}")
                print(f"   Confidence: {result.get('confidence_score', 'N/A')}")
                print(f"   Analysis Type: {result.get('analysis_type', 'N/A')}")
                
                # Check if enhanced analysis was performed
                if result.get('analysis_type') == 'enhanced_with_additional_apis':
                    enhanced = result.get('enhanced_analysis', {})
                    print(f"   🔍 Enhanced Analysis: {enhanced.get('summary', 'No summary')}")
                    
                    # Show source counts
                    wiki_count = len(enhanced.get('wikipedia', []))
                    news_count = len(enhanced.get('google_news', []))
                    search_count = len(enhanced.get('google_search', []))
                    
                    print(f"   📚 Wikipedia Sources: {wiki_count}")
                    print(f"   📰 Google News Sources: {news_count}")
                    print(f"   🔍 Google Search Sources: {search_count}")
                    
                    # Show first source from each category
                    if enhanced.get('wikipedia'):
                        first_wiki = enhanced['wikipedia'][0]
                        print(f"   📚 First Wiki: {first_wiki.get('title', 'N/A')}")
                    
                    if enhanced.get('google_news'):
                        first_news = enhanced['google_news'][0]
                        print(f"   📰 First News: {first_news.get('title', 'N/A')}")
                    
                    if enhanced.get('google_search'):
                        first_search = enhanced['google_search'][0]
                        print(f"   🔍 First Search: {first_search.get('title', 'N/A')}")
                else:
                    print(f"   ℹ️ Standard OpenAI analysis only")
                
            else:
                print(f"❌ Analysis failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Test error: {e}")

def test_api_configuration():
    """Test API configuration status"""
    print("\n🔧 Testing API Configuration...")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:5000/api/config")
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Configuration retrieved successfully!")
            print(f"   OpenAI API: {'✅ Configured' if config.get('api_key_configured') else '❌ Not configured'}")
            print(f"   Model: {config.get('model', 'N/A')}")
            print(f"   Supported formats: {len(config.get('supported_formats', []))} formats")
        else:
            print(f"❌ Configuration failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

def test_health():
    """Test application health"""
    print("\n🏥 Testing Application Health...")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:5000/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check successful!")
            print(f"   Status: {health.get('status', 'N/A')}")
            print(f"   Message: {health.get('message', 'N/A')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def main():
    """Run all enhanced fact-checking tests"""
    print("🚀 Starting Enhanced Fact-Checking Tests...")
    print("=" * 60)
    
    # Test health first
    test_health()
    
    # Test configuration
    test_api_configuration()
    
    # Test enhanced fact-checking
    test_enhanced_fact_checking()
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Fact-Checking Tests Completed!")
    print("\n💡 Enhanced Features Tested:")
    print("   - 🔍 Automatic enhanced analysis for FALSE/UNCERTAIN verdicts")
    print("   - 📚 Wikipedia integration for factual verification")
    print("   - 📰 Google News integration for recent events")
    print("   - 🔍 Google Search integration for web sources")
    print("   - ⚡ Smart query generation from claims")
    print("   - 📊 Cross-referencing across multiple sources")
    print("\n📋 Expected Behavior:")
    print("   - TRUE verdicts: OpenAI analysis only")
    print("   - FALSE/UNCERTAIN verdicts: OpenAI + Enhanced APIs")
    print("   - Enhanced results include source links and summaries")

if __name__ == "__main__":
    main()
