#!/usr/bin/env python3
"""
Simple test script for the editing inference functionality.
"""

import os
import sys
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference_editing import ImageEditingInference, EditingConfig

def test_text_to_image():
    """Test text-to-image generation."""
    print("🧪 Testing text-to-image generation...")
    
    config = EditingConfig()
    inference = ImageEditingInference(config)
    
    instruction = "A beautiful sunset over mountains"
    
    try:
        result = inference.generate_image(instruction, input_images=None)
        if result:
            result.save("test_t2i_output.png")
            print("✅ Text-to-image test passed! Saved to test_t2i_output.png")
        else:
            print("❌ Text-to-image test failed - no image generated")
    except Exception as e:
        print(f"❌ Text-to-image test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_image_editing():
    """Test image editing with a sample input."""
    print("🧪 Testing image editing...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (256, 256), color='red')
    test_image.save("test_input.png")
    
    config = EditingConfig()
    inference = ImageEditingInference(config)
    
    instruction = "Change the color to blue"
    input_images = [test_image]
    
    try:
        result = inference.generate_image(instruction, input_images=input_images)
        if result:
            result.save("test_editing_output.png")
            print("✅ Image editing test passed! Saved to test_editing_output.png")
        else:
            print("❌ Image editing test failed - no image generated")
    except Exception as e:
        print(f"❌ Image editing test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Running editing inference tests...")
    
    # Test text-to-image
    test_text_to_image()
    
    print("-" * 50)
    
    # Test image editing
    test_image_editing()
    
    print("✅ All tests completed!")