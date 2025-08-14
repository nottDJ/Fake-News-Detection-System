#!/usr/bin/env python3
"""
Test script for OCR functionality in the fake news detector
"""

import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_image():
    """Create a test image with text for OCR testing"""
    # Create a simple image with text
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text to the image
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add test text
    test_text = "Scientists discover miracle cure for all diseases"
    draw.text((20, 50), test_text, fill='black', font=font)
    
    draw.text((20, 100), "This is a test image for OCR", fill='black', font=font)
    draw.text((20, 150), "Testing fake news detection", fill='black', font=font)
    
    return img

def test_ocr_functionality():
    """Test the OCR functionality"""
    print("🧪 Testing OCR functionality...")
    
    try:
        # Import the OCR function
        from app import extract_text_from_image
        
        # Create a test image
        test_img = create_test_image()
        
        # Convert PIL image to file-like object
        img_buffer = io.BytesIO()
        test_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Test OCR extraction
        extracted_text = extract_text_from_image(img_buffer)
        
        print(f"✅ OCR test successful!")
        print(f"   Extracted text: {extracted_text[:100]}...")
        
        if extracted_text and len(extracted_text.strip()) > 0:
            print(f"   Text length: {len(extracted_text)} characters")
            print(f"   OCR working: ✅")
        else:
            print(f"   OCR working: ⚠️ (No text extracted)")
            
    except ImportError as e:
        print(f"❌ OCR test failed - Import error: {e}")
    except Exception as e:
        print(f"❌ OCR test failed: {e}")

def test_image_analysis():
    """Test the complete image analysis pipeline"""
    print("\n🧪 Testing complete image analysis pipeline...")
    
    try:
        from app import analyze_image
        
        # Create a test image
        test_img = create_test_image()
        
        # Convert PIL image to file-like object
        img_buffer = io.BytesIO()
        test_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Test complete analysis
        analysis_result = analyze_image(img_buffer)
        
        print(f"✅ Image analysis test successful!")
        print(f"   Extracted text: {analysis_result['extracted_text'][:100]}...")
        print(f"   Image context: {analysis_result['image_context']}")
        print(f"   Claims found: {analysis_result['claims_found'][:100]}...")
        
    except Exception as e:
        print(f"❌ Image analysis test failed: {e}")

def main():
    """Run all OCR tests"""
    print("🚀 Starting OCR functionality tests...")
    print("=" * 50)
    
    # Test basic OCR
    test_ocr_functionality()
    
    # Test complete image analysis
    test_image_analysis()
    
    print("\n" + "=" * 50)
    print("🎉 OCR tests completed!")
    print("\n💡 OCR Features:")
    print("   - Local image processing (no OpenAI quota needed)")
    print("   - Advanced image preprocessing with OpenCV")
    print("   - Multiple OCR attempts for better accuracy")
    print("   - Fallback handling for failed extractions")

if __name__ == "__main__":
    main()
