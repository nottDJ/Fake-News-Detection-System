#!/usr/bin/env python3
"""
Test script for recent events like the Ahmedabad aircraft crash
"""

import requests
import json
import time

def test_ahmedabad_aircraft_crash():
    """Test the system with the Ahmedabad aircraft crash event"""
    print("🧪 Testing Ahmedabad Aircraft Crash Event...")
    print("=" * 60)
    
    # Test cases for the Ahmedabad aircraft crash
    test_cases = [
        {
            "name": "Ahmedabad Aircraft Crash - Basic",
            "text": "Aircraft crash in Ahmedabad airport. Emergency landing incident reported."
        },
        {
            "name": "Ahmedabad Aircraft Crash - Detailed",
            "text": "Breaking news: Aircraft crash at Ahmedabad airport in Gujarat, India. Emergency landing incident reported with multiple casualties. The incident occurred during landing approach."
        },
        {
            "name": "Ahmedabad Aircraft Crash - Recent",
            "text": "Recent aircraft crash at Ahmedabad airport in 2024. Emergency landing incident with casualties reported. The plane crashed during landing approach at Sardar Vallabhbhai Patel International Airport."
        },
        {
            "name": "Ahmedabad Aircraft Crash - Aviation Terms",
            "text": "Aviation accident at Ahmedabad airport. Aircraft emergency landing incident with runway crash. Multiple casualties reported in the aviation incident."
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

def test_other_recent_events():
    """Test other recent events to verify the system works for various recent incidents"""
    print("\n🧪 Testing Other Recent Events...")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Recent Aviation Incident - Generic",
            "text": "Recent aircraft incident at major airport. Emergency landing reported with minor injuries."
        },
        {
            "name": "Recent Breaking News",
            "text": "Breaking news: Major incident reported today. Emergency services responding to the scene."
        },
        {
            "name": "Recent Event with Date",
            "text": "Incident occurred in 2024 at major facility. Multiple agencies responding to the emergency."
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
                
                # Check enhanced analysis
                if result.get('analysis_type') == 'enhanced_with_additional_apis':
                    enhanced = result.get('enhanced_analysis', {})
                    print(f"   🔍 Enhanced Analysis: {enhanced.get('summary', 'No summary')}")
                    
                    # Show source counts
                    wiki_count = len(enhanced.get('wikipedia', []))
                    news_count = len(enhanced.get('google_news', []))
                    search_count = len(enhanced.get('google_search', []))
                    
                    print(f"   📚 Wikipedia: {wiki_count} | 📰 News: {news_count} | 🔍 Search: {search_count}")
                else:
                    print(f"   ℹ️ Standard OpenAI analysis only")
                
            else:
                print(f"❌ Analysis failed: {response.status_code}")
                
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
            
            # Check if Google News API is configured
            if not config.get('api_key_configured'):
                print("   ⚠️ OpenAI API key not configured - enhanced features will be limited")
                print("   💡 Add your OpenAI API key to config.env to enable full functionality")
            
            print("\n💡 For better recent event detection:")
            print("   1. Get Google News API key from: https://newsapi.org/ (FREE)")
            print("   2. Add it to config.env: GOOGLE_NEWS_API_KEY=your_key")
            print("   3. Restart the application")
            
        else:
            print(f"❌ Configuration failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

def main():
    """Run all recent event tests"""
    print("🚀 Starting Recent Event Detection Tests...")
    print("=" * 60)
    
    # Test configuration first
    test_api_configuration()
    
    # Test Ahmedabad aircraft crash specifically
    test_ahmedabad_aircraft_crash()
    
    # Test other recent events
    test_other_recent_events()
    
    print("\n" + "=" * 60)
    print("🎉 Recent Event Detection Tests Completed!")
    print("\n💡 Enhanced Features for Recent Events:")
    print("   - 🔍 Enhanced Wikipedia search with aviation-specific terms")
    print("   - 📰 Google News integration for recent articles")
    print("   - 🔍 Google Search for web verification")
    print("   - 🎯 Location-specific query generation")
    print("   - 📅 Date-specific search variations")
    print("   - ✈️ Aviation incident detection")
    print("\n📋 Expected Behavior for Recent Events:")
    print("   - Should detect recent event indicators")
    print("   - Should generate location-specific queries")
    print("   - Should search for recent news articles")
    print("   - Should find relevant Wikipedia pages")
    print("   - Should provide comprehensive verification")

if __name__ == "__main__":
    main()
