import os
import shutil
from pathlib import Path
import requests
import base64
from dotenv import load_dotenv
import json
import time
import logging
from tqdm import tqdm
import sys
import backoff
import yaml
from typing import Optional
import re
import cv2
import numpy as np

# Set the correct base directory (relative to project root)
BASE_DIR = Path(__file__).parent.parent  # This gets us to the project root
DATASET_DIR = BASE_DIR / 'data' / 'binary_fire_dataset'
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
VISUALIZATION_COLOR = (0, 255, 0)  # Green in BGR
DEFAULT_CONFIG = {
    'model': "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    'max_retries': 3,
    'base_delay': 2,
    'temperature': 0.3,
    'dataset_root': DATASET_DIR
}

def cleanup_directories(dataset_root: Path) -> Path:
    """Clean up previous labels and visualizations"""
    # Clean up labels
    labels_dir = dataset_root / 'labels'
    if labels_dir.exists():
        for label_file in labels_dir.glob('**/*.txt'):
            label_file.unlink()
        print(f"Cleaned up previous labels in {labels_dir}")

    # Clean up visualizations
    viz_dir = dataset_root / 'visualizations'
    if viz_dir.exists():
        for viz_file in viz_dir.glob('**/*.jpg'):
            viz_file.unlink()
        for viz_file in viz_dir.glob('**/*.png'):
            viz_file.unlink()
        print(f"Cleaned up previous visualizations in {viz_dir}")

    # Create fresh directory structure
    for split in ['train', 'val', 'test']:
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        (viz_dir / split).mkdir(parents=True, exist_ok=True)
    print("Created fresh directory structure for labels and visualizations")

    # Create and return logs directory
    logs_dir = dataset_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

# Clean up and create fresh directories
log_dir = cleanup_directories(DATASET_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'labeling.log', mode='w')
    ]
)
logger = logging.getLogger('FireLabeler')

load_dotenv()

class LlamaLabeler:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Validate API key first
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError("ERROR: TOGETHER_API_KEY not found in .env file")
            
        # Initialize paths
        self.dataset_root = Path(self.config['dataset_root'])
        self.images_dir = self.dataset_root / 'images'
        self.labels_dir = self.dataset_root / 'labels'
        self.viz_dir = self.dataset_root / 'visualizations'
        
        # Create all required directories
        for split in ['train', 'val', 'test']:
            for dir_path in [self.images_dir, self.labels_dir, self.viz_dir]:
                (dir_path / split).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('FireLabeler.Vision')
        
        # Store other config values
        self.model = self.config['model']
        self.max_retries = self.config['max_retries']
        
        self.logger.info(f"Initialized LlamaLabeler with root: {self.dataset_root}")

    def encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"❌ Failed to encode image {image_path}: {e}")
            return None

    def call_api(self, encoded_image: str, prompt: str) -> dict:
        """Handle API calls with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
        def make_api_request():
            return requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a precise fire classification system that returns only valid JSON objects with exact fields and values as specified."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                            ]
                        }
                    ],
                    "temperature": self.config['temperature'],
                    "max_tokens": 1024
                },
                timeout=30
            )
        return make_api_request().json()

    def get_image_metadata(self, image_path):
        """Extract useful metadata from image"""
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = w / h
        
        # Get basic image stats
        mean_brightness = np.mean(img)
        
        metadata = {
            "dimensions": f"{w}x{h} pixels",
            "width": w,
            "height": h,
            "aspect_ratio": f"{aspect_ratio:.2f}",
            "mean_brightness": f"{mean_brightness:.1f}",
            "center_x": w/2,
            "center_y": h/2
        }
        return metadata, img

    def get_fire_location(self, image_path):
        """Get binary fire detection"""
        encoded_image = self.encode_image(image_path)
        if not encoded_image:
            return None

        prompt = """Analyze this image for fire detection. Your task is simple:
1. Determine if there is ANY type of fire in the image
2. If fire is present, provide its bounding box location

Return ONLY a JSON object in this format:
{
    "has_fire": true/false,
    "bbox": {
        "x": <center_x>,  // normalized 0-1
        "y": <center_y>,  // normalized 0-1
        "w": <width>,     // normalized 0-1
        "h": <height>     // normalized 0-1
    }
}

If no fire is detected, return:
{
    "has_fire": false,
    "bbox": null
}"""

        try:
            response = self.call_api(encoded_image, prompt)
            if not response:
                return None
                
            # Parse and validate response
            fire_data = self.parse_llm_response(response)
            if fire_data and fire_data.get('has_fire') and not fire_data.get('bbox'):
                self.logger.error("Fire detected but no bbox provided")
                return None
                
            return fire_data
            
        except Exception as e:
            self.logger.error(f"Failed to get fire location: {str(e)}")
            return None

    def parse_llm_response(self, response: str) -> dict:
        """Extract JSON from LLM response, handling various formats"""
        try:
            # Handle "No fire detected" responses first
            if "No fire detected" in response:
                self.logger.info("No fire detected in image")
                return None
                
            # Look for JSON block between triple backticks
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
                
            # Look for JSON block after "Answer:" or similar keywords
            json_match = re.search(r'(?:Answer:|Therefore,?|The final answer is:)\s*(\{.*\})', response, re.DOTALL)
            if json_match:
                # Remove any comments before parsing
                json_str = re.sub(r'\s*//.*$', '', json_match.group(1), flags=re.MULTILINE)
                return json.loads(json_str)
                
            # Try to find any JSON-like structure
            json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response)
            if json_match:
                json_str = re.sub(r'\s*//.*$', '', json_match.group(0), flags=re.MULTILINE)
                return json.loads(json_str)
                
            raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {str(e)}\nResponse: {response}")
            raise

    def save_yolo_label(self, image_path: Path, fire_data: dict) -> Path:
        """Convert binary fire detection to YOLO format and save label file"""
        if not fire_data['has_fire']:
            # No fire = no label file needed
            self.logger.info(f"No fire detected in {image_path.name}, skipping label file")
            return None
            
        # For fire detection, class index is always 0 (not 1)
        class_idx = 0
        bbox = fire_data['bbox']
        
        # Create label file path (same name as image but .txt extension)
        label_path = self.labels_dir / image_path.parent.name / f"{image_path.stem}.txt"
        
        # Create YOLO format line: <class> <x> <y> <width> <height>
        yolo_line = f"{class_idx} {bbox['x']} {bbox['y']} {bbox['w']} {bbox['h']}\n"
        
        # Save label file
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w') as f:
            f.write(yolo_line)
            
        self.logger.info(f"Saved fire detection label to: {label_path}")
        return label_path

    def save_visualization(self, img_path: Path, label_data: dict) -> Path:
        """Draw bounding box on image and save visualization."""
        if not label_data.get('has_fire', False):
            self.logger.info(f"No fire to visualize in {img_path.name}")
            return None
            
        viz_dir = self.viz_dir / img_path.parent.name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            self.logger.error(f"Failed to read image: {img_path}")
            return None
            
        h, w = img.shape[:2]
        
        # Get bbox coordinates
        bbox = label_data.get('bbox', {})
        x = bbox.get('x', 0.5)
        y = bbox.get('y', 0.5)
        width = bbox.get('w', 0.8)
        height = bbox.get('h', 0.8)
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((x - width/2) * w)
        y1 = int((y - height/2) * h)
        x2 = int((x + width/2) * w)
        y2 = int((y + height/2) * h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), VISUALIZATION_COLOR, 2)
        
        # Add label with fire type
        label = f"{label_data['fire_type']}"
        
        # Add background to text for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width, y1), VISUALIZATION_COLOR, -1)
        
        # Add text
        cv2.putText(img, label, (x1, y1-5), font, font_scale, (0, 0, 0), thickness)
        
        # Save visualization
        viz_path = viz_dir / f"viz_{img_path.name}"
        cv2.imwrite(str(viz_path), img)
        
        self.logger.info(f"Saved visualization to {viz_path}")
        return viz_path

    def process_image(self, image_path: Path) -> bool:
        """Process a single image and save its label"""
        try:
            # Get fire location data
            fire_data = self.get_fire_location(image_path)
            if not fire_data:
                return False
                
            # Save YOLO format label
            label_path = self.save_yolo_label(image_path, fire_data)
            
            # Save visualization
            viz_path = self.save_visualization(image_path, fire_data)
            if not viz_path:
                self.logger.warning(f"Failed to save visualization for {image_path.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path.name}: {str(e)}")
            return False

    def save_dataset_config(self):
        """Save YOLOv dataset configuration file"""
        config = {
            'path': str(DATASET_DIR),
            'train': str(DATASET_DIR / 'images' / 'train'),
            'val': str(DATASET_DIR / 'images' / 'val'), 
            'test': str(DATASET_DIR / 'images' / 'test'),
            'names': ['fire'],  # Single class for binary detection
            'nc': 1  # number of classes (just fire/no-fire)
        }
        
        yaml_path = DATASET_DIR / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        self.logger.info(f"\nSaved dataset configuration to: {yaml_path}")

def main():
    try:
        # Initialize labeler
        labeler = LlamaLabeler()
        
        # Clean up and prepare directories
        cleanup_directories(labeler.dataset_root)
        
        # Process the full dataset directly
        print("\nProcessing full dataset...")
        
        # Process each split
        splits = ['train', 'val', 'test']
        for split in splits:
            print(f"\nProcessing {split} split...")
            split_dir = labeler.images_dir / split
            
            # Get all images in split
            image_files = list(split_dir.glob('*.[jp][pn][g]'))  # matches .jpg, .jpeg, .png
            print(f"Found {len(image_files)} images in {split}")
            
            # Process images with progress bar
            success_count = 0
            for img_path in tqdm(image_files, desc=split):
                try:
                    if labeler.process_image(img_path):
                        success_count += 1
                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")
                    continue
                    
            print(f"Successfully processed {success_count}/{len(image_files)} images in {split}")
        
        # Save dataset configuration
        labeler.save_dataset_config()
        print("\nDataset preparation completed")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()