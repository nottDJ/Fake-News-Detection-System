#!/usr/bin/env python3
"""
Basic functionality test for the enhanced fake news detector
"""

import requests
import json
import time

def test_app_startup():
    """Test if the app can start and respond to basic requests"""
    print("🧪 Testing Application Startup...")
    print("=" * 50)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check successful!")
            print(f"   Status: {health.get('status', 'N/A')}")
            print(f"   Message: {health.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Application not running. Please start the app with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_config_endpoint():
    """Test configuration endpoint"""
    print("\n🔧 Testing Configuration Endpoint...")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:5000/api/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Configuration retrieved successfully!")
            print(f"   Model: {config.get('model', 'N/A')}")
            print(f"   Model Weight: {config.get('model_weight', 'N/A')}")
            print(f"   API Key Configured: {config.get('api_key_configured', 'N/A')}")
            print(f"   Supported Formats: {len(config.get('supported_formats', []))} formats")
            
            # Check if API key is configured
            if not config.get('api_key_configured'):
                print("   ⚠️ OpenAI API key not configured - enhanced features will be limited")
                print("   💡 Add your OpenAI API key to config.env to enable full functionality")
            
            return True
        else:
            print(f"❌ Configuration failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_enhanced_structure():
    """Test that the enhanced analysis structure is in place"""
    print("\n🔍 Testing Enhanced Analysis Structure...")
    print("=" * 50)
    
    try:
        # Test with a simple text input
        test_data = {
            "news_text": "This is a test claim for verification."
        }
        
        response = requests.post(
            "http://localhost:5000/api/verify-news", 
            json=test_data, 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API response structure test successful!")
            print(f"   Verdict: {result.get('verdict', 'N/A')}")
            print(f"   Analysis Type: {result.get('analysis_type', 'N/A')}")
            print(f"   Has Enhanced Analysis: {'enhanced_analysis' in result}")
            
            # Check for enhanced analysis fields
            if 'enhanced_analysis' in result:
                enhanced = result['enhanced_analysis']
                print(f"   📚 Wikipedia Sources: {len(enhanced.get('wikipedia', []))}")
                print(f"   📰 Google News Sources: {len(enhanced.get('google_news', []))}")
                print(f"   🔍 Google Search Sources: {len(enhanced.get('google_search', []))}")
                print(f"   📋 Summary: {enhanced.get('summary', 'No summary')}")
            
            return True
        elif response.status_code == 400:
            error_data = response.json()
            if 'error' in error_data and 'API key' in error_data['error']:
                print("✅ API structure test passed (expected API key error)")
                print("   ℹ️ This is expected when OpenAI API key is not configured")
                print("   💡 Add your OpenAI API key to config.env to enable full functionality")
                return True
            else:
                print(f"❌ Unexpected error: {error_data}")
                return False
        else:
            print(f"❌ API test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced structure test error: {e}")
        return False

def main():
    """Run all basic functionality tests"""
    print("🚀 Starting Basic Functionality Tests...")
    print("=" * 50)
    
    # Test if app is running
    if not test_app_startup():
        print("\n💡 To start the application:")
        print("   1. Open a new terminal")
        print("   2. Navigate to the project directory")
        print("   3. Run: python app.py")
        print("   4. Wait for the app to start")
        print("   5. Run this test again")
        return
    
    # Test configuration
    test_config_endpoint()
    
    # Test enhanced structure
    test_enhanced_structure()
    
    print("\n" + "=" * 50)
    print("🎉 Basic Functionality Tests Completed!")
    print("\n📋 Next Steps:")
    print("   1. Get your OpenAI API key from: https://platform.openai.com/")
    print("   2. Add it to config.env: OPENAI_API_KEY=your_actual_key")
    print("   3. Restart the application")
    print("   4. Run: python test_enhanced_fact_checking.py")
    print("\n💡 Optional APIs (for enhanced fact-checking):")
    print("   - Google News API: https://newsapi.org/ (free tier)")
    print("   - Google SERP API: https://serpapi.com/ (free tier)")
    print("   - Wikipedia API: No key needed (free)")

if __name__ == "__main__":
    main()
