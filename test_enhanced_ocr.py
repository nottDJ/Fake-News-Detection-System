#!/usr/bin/env python3
"""
Test script for enhanced image analysis functionality
"""

import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_news_image():
    """Create a test image that looks like a news article"""
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add news-like content
    draw.text((20, 20), "BREAKING NEWS", fill='red', font=font)
    draw.text((20, 50), "Scientists Discover Miracle Cure for All Diseases", fill='black', font=font)
    draw.text((20, 80), "New study shows revolutionary treatment that", fill='black', font=font)
    draw.text((20, 110), "guarantees 100% success rate!", fill='black', font=font)
    draw.text((20, 150), "Doctors are shocked by this breakthrough", fill='black', font=font)
    draw.text((20, 180), "research that big pharma doesn't want", fill='black', font=font)
    draw.text((20, 210), "you to know about!", fill='black', font=font)
    
    return img

def create_test_social_media_image():
    """Create a test image that looks like a social media post"""
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add social media-like content
    draw.text((20, 20), "Amazing discovery!", fill='black', font=font)
    draw.text((20, 50), "This one weird trick will", fill='black', font=font)
    draw.text((20, 80), "change your life forever!", fill='black', font=font)
    draw.text((20, 120), "Click here to learn more", fill='red', font=font)
    draw.text((20, 150), "Limited time offer!", fill='red', font=font)
    
    return img

def create_test_meme_image():
    """Create a test image that looks like a meme"""
    img = Image.new('RGB', (500, 350), color='yellow')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add meme-like content
    draw.text((20, 20), "WHEN SOMEONE SAYS", fill='black', font=font)
    draw.text((20, 50), "FAKE NEWS ISN'T REAL", fill='black', font=font)
    draw.text((20, 100), "Me: *shows them this image*", fill='black', font=font)
    draw.text((20, 150), "Facts don't care about", fill='black', font=font)
    draw.text((20, 180), "your feelings!", fill='black', font=font)
    
    return img

def test_enhanced_image_analysis():
    """Test the enhanced image analysis functionality"""
    print("🧪 Testing Enhanced Image Analysis...")
    print("=" * 60)
    
    test_images = [
        ("News Article", create_test_news_image()),
        ("Social Media Post", create_test_social_media_image()),
        ("Meme", create_test_meme_image())
    ]
    
    for image_name, test_img in test_images:
        print(f"\n📸 Testing {image_name}...")
        
        try:
            # Convert PIL image to file-like object
            img_buffer = io.BytesIO()
            test_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Test comprehensive analysis
            from app import analyze_image
            analysis_result = analyze_image(img_buffer)
            
            print(f"✅ {image_name} analysis successful!")
            print(f"   📝 Extracted Text: {analysis_result['extracted_text'][:100]}...")
            print(f"   🖼️ Image Context: {analysis_result['image_context']}")
            print(f"   🎯 Claims Found: {analysis_result['claims_found'][:100]}...")
            print(f"   ⚠️ Suspicious Elements: {analysis_result['suspicious_elements'][:100]}...")
            
            if 'visual_elements' in analysis_result:
                print(f"   🎨 Visual Elements: {analysis_result['visual_elements']}")
            if 'image_type' in analysis_result:
                print(f"   📋 Image Type: {analysis_result['image_type']}")
            if 'image_metadata' in analysis_result:
                print(f"   📊 Metadata: {analysis_result['image_metadata']}")
                
        except Exception as e:
            print(f"❌ {image_name} analysis failed: {e}")

def test_ocr_accuracy():
    """Test OCR accuracy with different image types"""
    print("\n🔍 Testing OCR Accuracy...")
    print("=" * 60)
    
    try:
        from app import extract_text_from_image
        
        # Test with news image
        news_img = create_test_news_image()
        img_buffer = io.BytesIO()
        news_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        extracted_text = extract_text_from_image(img_buffer)
        
        print(f"✅ OCR Test Results:")
        print(f"   📝 Text Length: {len(extracted_text)} characters")
        print(f"   📊 Text Preview: {extracted_text[:200]}...")
        
        # Check for key words
        text_lower = extracted_text.lower()
        key_words = ['scientists', 'discover', 'miracle', 'cure', 'diseases']
        found_words = [word for word in key_words if word in text_lower]
        
        print(f"   🎯 Key Words Found: {found_words}")
        print(f"   📈 Accuracy Score: {len(found_words)}/{len(key_words)} key words detected")
        
    except Exception as e:
        print(f"❌ OCR accuracy test failed: {e}")

def test_context_detection():
    """Test context detection capabilities"""
    print("\n🧠 Testing Context Detection...")
    print("=" * 60)
    
    try:
        from app import analyze_image
        
        # Test with different image types
        test_cases = [
            ("News", create_test_news_image()),
            ("Social Media", create_test_social_media_image()),
            ("Meme", create_test_meme_image())
        ]
        
        for case_name, test_img in test_cases:
            img_buffer = io.BytesIO()
            test_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            analysis = analyze_image(img_buffer)
            
            print(f"📋 {case_name} Context Analysis:")
            print(f"   🎯 Type: {analysis.get('image_type', 'Unknown')}")
            print(f"   📝 Context: {analysis.get('image_context', 'No context')}")
            print(f"   ⚠️ Suspicious: {analysis.get('suspicious_elements', 'None')[:100]}...")
            
    except Exception as e:
        print(f"❌ Context detection test failed: {e}")

def main():
    """Run all enhanced image analysis tests"""
    print("🚀 Starting Enhanced Image Analysis Tests...")
    print("=" * 60)
    
    # Test comprehensive analysis
    test_enhanced_image_analysis()
    
    # Test OCR accuracy
    test_ocr_accuracy()
    
    # Test context detection
    test_context_detection()
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Image Analysis Tests Completed!")
    print("\n💡 Enhanced Features:")
    print("   - 📝 Advanced OCR with multiple preprocessing techniques")
    print("   - 🧠 Intelligent context detection")
    print("   - 🎯 Claim identification and analysis")
    print("   - ⚠️ Suspicious element detection")
    print("   - 🎨 Visual element analysis")
    print("   - 📋 Image type classification")
    print("   - 📊 Comprehensive metadata extraction")

if __name__ == "__main__":
    main()
