import ollama
import base64
from pathlib import Path
import sys
from typing import List, Optional
import logging
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionTester:
    def __init__(self, model_name: str = "llama3.2-vision"):
        self.model_name = model_name
        
    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None

    def test_image(self, image_path: str, prompt: str = "What is in this image?") -> dict:
        """Test a single image with the vision model."""
        try:
            # Verify image exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Create message for model
            messages = [{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]

            # Get response from model
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )

            return {
                'status': 'success',
                'prompt': prompt,
                'response': response['message']['content'],
                'image_path': image_path
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'image_path': image_path
            }

    def batch_test(self, image_dir: str, prompts: List[str] = None) -> List[dict]:
        """Test multiple images in a directory."""
        if prompts is None:
            prompts = ["What is in this image?"]

        results = []
        image_paths = Path(image_dir).glob('*.[jp][pn][g]')  # Match .jpg, .jpeg, .png

        for image_path in image_paths:
            for prompt in prompts:
                result = self.test_image(str(image_path), prompt)
                results.append(result)

        return results

def main():
    # Initialize tester
    tester = VisionTester()

    # Example prompts
    test_prompts = [
        "What is happening in this image?",
        "Describe the fire pattern and spread direction in this image.",
        "What safety concerns are visible in this image?",
        "What type of terrain and vegetation is visible in this image?",
        "Are there any visible fire control measures or emergency vehicles in the image?"
    ]

    # Test single image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print("\n" + "="*80)
        print(f"🔍 Analyzing image: {image_path}")
        print("="*80)
        
        # Test with each prompt
        for i, prompt in enumerate(test_prompts, 1):
            result = tester.test_image(image_path, prompt)
            print(f"\n📝 Analysis #{i}")
            print(f"Question: {prompt}")
            print("-"*40)
            if result['status'] == 'success':
                print(f"Answer: {result['response']}")
            else:
                print(f"❌ Error: {result['error']}")
            print("="*80)

        print("\n✅ Analysis complete!\n")

    # Test batch of images
    elif len(sys.argv) > 2 and sys.argv[1] == "--batch":
        image_dir = sys.argv[2]
        print(f"\n📁 Processing all images in: {image_dir}")
        results = tester.batch_test(image_dir, test_prompts)
        
        print("\n📊 Batch Results:")
        for result in results:
            print("\n" + "="*80)
            if result['status'] == 'success':
                print(f"📸 Image: {result['image_path']}")
                print(f"❓ Question: {result['prompt']}")
                print(f"💡 Answer: {result['response']}")
            else:
                print(f"❌ Error processing {result['image_path']}: {result['error']}")
            print("="*80)

    else:
        print("\n📌 Usage:")
        print("Single image: python test_vision.py <image_path>")
        print("Batch processing: python test_vision.py --batch <image_directory>\n")

if __name__ == "__main__":
    main() 