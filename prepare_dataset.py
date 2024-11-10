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
DATASET_DIR = BASE_DIR / 'data' / 'fire_dataset'

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
        self.config = config or {
            'model': "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            'max_retries': 3,
            'base_delay': 2,
            'temperature': 0.3,
            'dataset_root': DATASET_DIR
        }
        
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError("ERROR: TOGETHER_API_KEY not found in .env file")
            
        self.model = self.config['model']
        self.logger = logging.getLogger('FireLabeler.Vision')
        self.max_retries = self.config['max_retries']
        self.dataset_root = Path(self.config['dataset_root'])
        self.images_dir = self.dataset_root / 'images'
        self.labels_dir = self.dataset_root / 'labels'
        
        # Create directories if they don't exist
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Dataset root: {self.dataset_root}")
        self.logger.info(f"Images directory: {self.images_dir}")
        self.logger.info(f"Labels directory: {self.labels_dir}")

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
        """Get fire location with metadata context"""
        metadata, img = self.get_image_metadata(image_path)
        
        system_prompt = """You are a precise YOLOv8 fire detection labeling assistant. Your ONLY task is to output a single, valid JSON object.

CRITICAL RULES:
1. ONLY output valid JSON - no explanations, no markdown, no additional text
2. NEVER skip labeling if you see fire in the image
3. NEVER include smoke or glow in bounding box measurements
4. NEVER use ranges (like "0.4-0.5") for coordinates - use single precise values
5. NEVER return coordinates outside 0.1-0.9 range
6. NEVER add fields that aren't explicitly requested
7. ALWAYS ensure bbox is tight around visible flames only"""

        task_prompt = f"""LABELING INSTRUCTIONS:
1. Identify fire type:
   - building_fire: Fires in/on buildings
   - forest_fire: Fires in nature/vegetation
   - vehicle_fire: Fires involving vehicles

2. Create precise bounding box:
   - x,y: Exact center point of visible flames
   - w,h: Width/height of visible flames only
   - All values must be between 0.1 and 0.9
   - Image dimensions: {metadata['dimensions']}
   - Center reference: ({metadata['center_x']}, {metadata['center_y']})

EXPECTED FORMAT:
{{
    "fire_type": "building_fire|forest_fire|vehicle_fire",
    "bbox": {{
        "x": float,  // center-x of flames (0.1-0.9)
        "y": float,  // center-y of flames (0.1-0.9)
        "w": float,  // width of flames (0.1-0.9)
        "h": float   // height of flames (0.1-0.9)
    }}
}}

Return ONLY the JSON object. No explanations or additional text."""

        encoded_image = self.encode_image(image_path)
        if not encoded_image:
            raise ValueError(f"Failed to encode image: {image_path}")

        try:
            response = self.call_api(encoded_image, task_prompt)
            
            # Check if response is already a dict
            if isinstance(response, dict):
                content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                content = response
            
            # Parse and validate the response
            fire_data = self.parse_fire_json(content)
            if fire_data:
                self.validate_fire_data(fire_data)
                return fire_data
            return None
            
        except Exception as e:
            self.logger.error(f"Error in get_fire_location: {str(e)}")
            raise

    def parse_fire_json(self, response):
        """Parse JSON from LLM response"""
        try:
            # Handle if response is already a string
            if isinstance(response, str):
                response = response.strip()
            
            # Find the first { and last } to extract just the JSON object
            if isinstance(response, str):
                start = response.find('{')
                end = response.rfind('}') + 1
                
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    # Remove any comments
                    json_str = re.sub(r'\s*//.*$', '', json_str, flags=re.MULTILINE)
                    return json.loads(json_str)
            elif isinstance(response, dict):
                return response
            
            raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {str(e)}\nResponse: {response}")
            raise

    def validate_bbox(self, bbox: dict) -> None:
        """Validate bounding box coordinates"""
        # Add tolerance for floating point comparison
        eps = 1e-6
        
        required_fields = ['x', 'y', 'w', 'h']
        for field in required_fields:
            if field not in bbox:
                raise ValueError(f"Missing required bbox field: {field}")
            
            val = bbox[field]
            # Handle string values
            if isinstance(val, str):
                # Remove any comments and whitespace
                val = re.sub(r'\s*//.*$', '', val).strip()
                # Handle range format like "0.4-0.5"
                if '-' in val:
                    val = float(val.split('-')[0])
                else:
                    val = float(val)
            
            # Validate numeric range
            if not isinstance(val, (int, float)):
                raise ValueError(f"Invalid bbox {field} type: {type(val)}")
                
            if not (0.1 - eps <= float(val) <= 0.9 + eps):
                raise ValueError(f"Invalid bbox {field}: {val} not in range 0.1-0.9")
            
            # Store validated value
            bbox[field] = float(val)

    def validate_fire_data(self, data: dict) -> None:
        """Validate the fire detection data"""
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data)}")
            
        required_fields = ['fire_type', 'bbox']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate fire type
        valid_types = {'forest_fire', 'building_fire', 'vehicle_fire'}
        if data['fire_type'] not in valid_types:
            raise ValueError(f"Invalid fire_type: {data['fire_type']}")

        # Validate bbox structure and values
        bbox = data['bbox']
        if not isinstance(bbox, dict):
            raise ValueError(f"bbox must be a dictionary, got {type(bbox)}")
            
        # Use the new validate_bbox method
        self.validate_bbox(bbox)

    def visualize_prediction(self, img_path: Path, label_data: dict) -> Path:
        """Draw bounding box on image and save visualization."""
        # Create visualization directory if it doesn't exist
        viz_dir = self.dataset_root / 'visualizations' / img_path.parent.name
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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add labels with fire type and severity
        label = f"{label_data['fire_type']} ({label_data['severity']})"
        
        # Add background to text for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 0), -1)
        
        # Add text
        cv2.putText(img, label, (x1, y1-5), font, font_scale, (0, 0, 0), thickness)
        
        # Save visualization
        viz_path = viz_dir / f"viz_{img_path.name}"
        cv2.imwrite(str(viz_path), img)
        
        self.logger.info(f"Saved visualization to {viz_path}")
        return viz_path

    def save_label(self, img_path: Path, label_data: dict) -> bool:
        """Save label in YOLO format for training"""
        try:
            # Create label directory if it doesn't exist
            label_dir = self.labels_dir / img_path.parent.name
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to labels directory with same name but .txt extension
            label_path = label_dir / img_path.with_suffix('.txt').name
            
            self.logger.info(f"Saving label to {label_path}")
            
            # Map fire type to class index
            fire_types = {
                'forest_fire': 0,
                'building_fire': 1,
                'vehicle_fire': 2
            }
            
            if label_data['fire_type'] not in fire_types:
                self.logger.error(f"Invalid fire type: {label_data['fire_type']}")
                return False
                
            class_idx = fire_types[label_data['fire_type']]
            
            # Get bbox coordinates
            bbox = label_data.get('bbox', {})
            x = bbox.get('x', 0.5)
            y = bbox.get('y', 0.5)
            w = bbox.get('w', 0.8)
            h = bbox.get('h', 0.8)
            
            # Validate coordinates
            if not all(0 <= coord <= 1 for coord in [x, y, w, h]):
                self.logger.error(f"Invalid bbox coordinates: {bbox}")
                return False
            
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            yolo_format = f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
            
            with open(label_path, 'w') as f:
                f.write(yolo_format)
                
            self.logger.info(f"Successfully saved label for {img_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save label for {img_path}: {str(e)}")
            return False

    def validate_and_label_image(self, img_path: Path) -> Optional[dict]:
        """Process single image and get label data"""
        try:
            self.logger.info(f"Processing image: {img_path.name}")
            label_data = self.get_fire_location(img_path)
            
            if label_data:
                # Save visualization
                viz_path = self.visualize_prediction(img_path, label_data)
                if viz_path:
                    self.logger.info(f"Visualization saved to: {viz_path}")
                
            return label_data
        except Exception as e:
            self.logger.error(f"Failed to process {img_path}: {str(e)}")
            return None

    def process_test_batch(self, num_images: int = 5):
        """Process a small batch of images for testing"""
        # Get sample images from test set
        test_images = list((self.images_dir / 'test').glob('*.jpg'))[:num_images]
        
        results = []
        for img_path in test_images:
            label_data = self.validate_and_label_image(img_path)
            if label_data:
                results.append({
                    'image': str(img_path),
                    'visualization': str(self.dataset_root / 'visualizations' / 'test' / f"viz_{img_path.name}"),
                    'predictions': label_data
                })
        
        # Print results summary
        print("\nProcessed Test Batch Results:")
        for result in results:
            print(f"\nImage: {Path(result['image']).name}")
            print(f"Visualization: {Path(result['visualization']).name}")
            print("Predictions:", json.dumps(result['predictions'], indent=2))
        
        return results

    def prepare_dataset(self, source_dir: str):
        """Prepare the dataset with validation"""
        self.logger.info("Starting dataset preparation")
        
        # Use absolute path
        source_path = BASE_DIR / source_dir
        self.logger.info(f"Processing dataset at: {source_path}")
        
        # Track statistics
        stats = {split: {'total': 0, 'success': 0, 'failed': 0} 
                for split in ['train', 'val', 'test']}
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_dir = source_path / 'images' / split
            self.logger.info(f"\nProcessing {split} split")
            self.logger.info(f"Directory: {split_dir}")
            
            if not split_dir.exists():
                self.logger.error(f"Directory not found: {split_dir}")
                continue
            
            # Get only image files (excluding .npy files)
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(split_dir.glob(f'*{ext}'))
            
            # Sort for consistent processing
            image_files = sorted(image_files)
            
            if not image_files:
                self.logger.warning(f"No images found in {split_dir}")
                continue
            
            self.logger.info(f"Found {len(image_files)} images")
            stats[split]['total'] = len(image_files)
            
            # Process images with progress bar
            for img_path in tqdm(image_files, desc=split):
                try:
                    # Skip .npy files
                    if img_path.suffix == '.npy':
                        continue
                    
                    # Get label data
                    label_data = self.validate_and_label_image(img_path)
                    
                    if label_data:
                        # Save visualization and label
                        viz_path = self.visualize_prediction(img_path, label_data)
                        if viz_path:
                            stats[split]['success'] += 1
                            if stats[split]['success'] % 10 == 0:  # Log every 10 successful images
                                self.logger.info(f"\nProcessed {img_path.name}:")
                                self.logger.info(f"- Type: {label_data['fire_type']}")
                                self.logger.info(f"- Severity: {label_data['severity']}")
                                self.logger.info(f"- Description: {label_data['description']}")
                        else:
                            stats[split]['failed'] += 1
                    else:
                        stats[split]['failed'] += 1
                        
                except Exception as e:
                    stats[split]['failed'] += 1
                    self.logger.error(f"Error processing {img_path.name}: {str(e)}")
        
        # Generate data.yaml
        yaml_path = self.dataset_root / 'data.yaml'
        yaml_content = {
            'path': str(self.dataset_root.absolute()),
            'train': str(self.images_dir / 'train'),
            'val': str(self.images_dir / 'val'),
            'test': str(self.images_dir / 'test'),
            'nc': 3,  # number of classes
            'names': ['forest_fire', 'building_fire', 'vehicle_fire']
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        self.logger.info(f"\nSaved dataset configuration to: {yaml_path}")
        
        # Log final statistics
        self.logger.info("\nDataset preparation completed")
        total_processed = sum(s['total'] for s in stats.values())
        total_success = sum(s['success'] for s in stats.values())
        total_failed = sum(s['failed'] for s in stats.values())
        
        self.logger.info(f"Total processed: {total_processed}")
        self.logger.info(f"Total success: {total_success}")
        self.logger.info(f"Total failed: {total_failed}")
        self.logger.info(f"Overall success rate: {(total_success/total_processed)*100:.1f}%")
        
        # Log split-wise statistics
        for split, stat in stats.items():
            success_rate = (stat['success']/stat['total'])*100 if stat['total'] > 0 else 0
            self.logger.info(f"\n{split.capitalize()} split:")
            self.logger.info(f"  Images: {stat['success']}/{stat['total']}")
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
            self.logger.info(f"  Directory: {self.images_dir/split}")
            self.logger.info(f"  Labels: {self.labels_dir/split}")

    def get_object_location(self, image_path):
        """Detect all objects in image with metadata context"""
        metadata, img = self.get_image_metadata(image_path)
        
        # Build prompt parts separately
        context = f"""IMAGE CONTEXT:
- Dimensions: {metadata['dimensions']}
- Aspect Ratio: {metadata['aspect_ratio']}
- Center Point: ({metadata['center_x']}, {metadata['center_y']})
- Mean Brightness: {metadata['mean_brightness']}"""

        instructions = f"""INSTRUCTIONS:
1. Identify ALL distinct objects in the image
2. For EACH object provide:
   - "class": object type (e.g., "person", "car", "dog", "chair")
   - "confidence": how sure you are (0.0-1.0)
   - "bbox": normalized coordinates (0-1) of the object:
     - "x": center x coordinate (relative to width {metadata['width']})
     - "y": center y coordinate (relative to height {metadata['height']})
     - "w": width (as fraction of {metadata['width']})
     - "h": height (as fraction of {metadata['height']})"""

        coordinate_guide = f"""COORDINATE GUIDE:
- Center of image is at ({metadata['center_x']}, {metadata['center_y']})
- Normalize coordinates by dividing by image dimensions
- x values range from 0 to 1 (left to right)
- y values range from 0 to 1 (top to bottom)
- w,h should be proportional to image size"""

        example_response = """EXAMPLE RESPONSE:
{
    "objects": [
        {
            "class": "person",
            "confidence": 0.95,
            "bbox": {"x": 0.3, "y": 0.5, "w": 0.2, "h": 0.6}
        },
        {
            "class": "dog",
            "confidence": 0.88,
            "bbox": {"x": 0.7, "y": 0.6, "w": 0.15, "h": 0.2}
        }
    ],
    "description": "A person walking a dog in a park"
}"""

        final_note = f"The bbox coordinates should tightly bound each object relative to the {metadata['dimensions']} image."

        # Combine all parts
        prompt = f"""You are a precise object detection system. Analyze this image and return a JSON object with ALL visible objects.

{context}

{instructions}

{coordinate_guide}

{example_response}

{final_note}"""

        return self.process_image_with_prompt(image_path, prompt)

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
        """Convert fire detection data to YOLO format and save label file"""
        # Convert fire_type to class index
        fire_type_to_class = {
            'building_fire': 0,
            'forest_fire': 1,
            'vehicle_fire': 2
        }
        
        class_idx = fire_type_to_class[fire_data['fire_type']]
        bbox = fire_data['bbox']
        
        # Create label file path (same name as image but .txt extension)
        label_path = self.labels_dir / image_path.parent.name / f"{image_path.stem}.txt"
        
        # Create YOLO format line: <class> <x> <y> <width> <height>
        yolo_line = f"{class_idx} {bbox['x']} {bbox['y']} {bbox['w']} {bbox['h']}\n"
        
        # Save label file
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w') as f:
            f.write(yolo_line)
            
        self.logger.info(f"Saved YOLO label to: {label_path}")
        return label_path

    def save_visualization(self, img_path: Path, label_data: dict) -> Path:
        """Draw bounding box on image and save visualization."""
        # Create visualization directory if it doesn't exist
        viz_dir = self.dataset_root / 'visualizations' / img_path.parent.name
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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with fire type
        label = f"{label_data['fire_type']}"
        
        # Add background to text for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 0), -1)
        
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
            'names': {
                0: 'building_fire',
                1: 'forest_fire', 
                2: 'vehicle_fire'
            },
            'nc': 3  # number of classes
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