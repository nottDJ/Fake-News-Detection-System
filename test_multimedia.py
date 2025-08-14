#!/usr/bin/env python3
"""
Test script for the enhanced fake news detector with multimedia support
"""

import requests
import json
import os

def test_text_analysis():
    """Test text input analysis"""
    print("🧪 Testing text analysis...")
    
    url = "http://localhost:5000/api/verify-news"
    data = {
        "news_text": "Scientists discover that drinking 8 glasses of water daily can cure all diseases and make you live forever. This breakthrough research was conducted by Dr. Smith at Harvard University."
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Text analysis successful!")
            print(f"   Verdict: {result.get('verdict', 'N/A')}")
            print(f"   Input Type: {result.get('input_type', 'N/A')}")
            print(f"   Confidence: {result.get('confidence_score', 'N/A')}")
        else:
            print(f"❌ Text analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Text analysis error: {e}")

def test_config():
    """Test configuration endpoint"""
    print("\n🧪 Testing configuration endpoint...")
    
    url = "http://localhost:5000/api/config"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Configuration retrieved successfully!")
            print(f"   Model: {config.get('model', 'N/A')}")
            print(f"   Supported formats: {config.get('supported_formats', 'N/A')}")
            print(f"   API key configured: {config.get('api_key_configured', 'N/A')}")
        else:
            print(f"❌ Configuration failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

def test_health():
    """Test health endpoint"""
    print("\n🧪 Testing health endpoint...")
    
    url = "http://localhost:5000/api/health"
    
    try:
        response = requests.get(url)
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
    """Run all tests"""
    print("🚀 Starting multimedia fake news detector tests...")
    print("=" * 50)
    
    # Test health first
    test_health()
    
    # Test configuration
    test_config()
    
    # Test text analysis
    test_text_analysis()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n💡 To test file uploads:")
    print("   1. Open http://localhost:5000 in your browser")
    print("   2. Switch to the 'File Upload' tab")
    print("   3. Upload an audio, image, or document file")
    print("   4. Click 'Analyze Content'")

if __name__ == "__main__":
    main()
