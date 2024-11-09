import os
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/fire_dataset/labeling.log', mode='w')
    ]
)
logger = logging.getLogger('FireLabeler')

load_dotenv()

class LlamaLabeler:
    def __init__(self):
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError("ERROR: TOGETHER_API_KEY not found in .env file")
        self.model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
        self.logger = logging.getLogger('FireLabeler.Vision')
        self.max_retries = 3
        self.dataset_root = Path('data/fire_dataset')
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

    @backoff.on_exception(
        backoff.expo,
        (json.JSONDecodeError, requests.exceptions.RequestException),
        max_tries=3,
        giveup=lambda e: isinstance(e, ValueError)
    )
    def get_fire_location(self, image_path):
        """Get fire location using Llama Vision"""
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                encoded_image = self.encode_image(image_path)
                if not encoded_image:
                    return None

                prompt = """You are a precise fire classification system. Your task is to analyze the image and return ONLY a JSON object.

IMPORTANT INSTRUCTIONS:
1. Return ONLY the JSON object, no explanations, no markdown, no additional text
2. Use EXACTLY these fields and allowed values:
   - "fire_type": MUST BE ONE OF ["forest_fire", "building_fire", "vehicle_fire"]
   - "severity": MUST BE ONE OF ["low", "medium", "high"]
   - "description": MUST BE under 100 characters

EXAMPLE CORRECT RESPONSE:
{
    "fire_type": "building_fire",
    "severity": "high",
    "description": "Two-story house engulfed in flames with heavy smoke"
}

DO NOT:
- Add any explanations or text before/after the JSON
- Use markdown formatting
- Use any fire types other than the three specified
- Use any severity levels other than the three specified
- Write descriptions longer than 100 characters

REMEMBER: Return ONLY the JSON object. Any additional text will cause errors.

Now analyze the image and provide the classification:"""

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                # Add retry logic for API calls
                @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
                def make_api_request():
                    response = requests.post(
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
                                        {
                                            "type": "text",
                                            "text": prompt
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{encoded_image}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            "temperature": 0.3,  # Lower temperature for more consistent outputs
                            "max_tokens": 1024
                        },
                        timeout=30
                    )
                    if "error" in response.json():
                        raise requests.exceptions.RequestException(response.json()["error"])
                    return response

                response = make_api_request()
                
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    try:
                        content = result['choices'][0]['message']['content']
                        self.logger.info(f"Raw content (Attempt {attempt + 1}/{max_retries}): {content}")
                        
                        # Find the JSON object in the content
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = content[start_idx:end_idx]
                            
                            try:
                                parsed_data = json.loads(json_str)
                                
                                # Validate required fields and values
                                if not all(k in parsed_data for k in ['fire_type', 'severity', 'description']):
                                    raise ValueError("Missing required fields")
                                    
                                if parsed_data['fire_type'] not in ['forest_fire', 'building_fire', 'vehicle_fire']:
                                    raise ValueError(f"Invalid fire_type: {parsed_data['fire_type']}")
                                    
                                if parsed_data['severity'] not in ['low', 'medium', 'high']:
                                    raise ValueError(f"Invalid severity: {parsed_data['severity']}")
                                
                                if len(parsed_data['description']) > 100:
                                    raise ValueError("Description too long")
                                    
                                self.logger.info(f"Successfully parsed data on attempt {attempt + 1}")
                                return parsed_data
                                
                            except (json.JSONDecodeError, ValueError) as e:
                                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                                if attempt < max_retries - 1:
                                    delay = base_delay * (attempt + 1)  # Exponential backoff
                                    self.logger.info(f"Retrying in {delay} seconds...")
                                    time.sleep(delay)
                                continue
                        else:
                            self.logger.warning(f"No JSON object found in response (Attempt {attempt + 1})")
                            if attempt < max_retries - 1:
                                delay = base_delay * (attempt + 1)
                                self.logger.info(f"Retrying in {delay} seconds...")
                                time.sleep(delay)
                            continue
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process content (Attempt {attempt + 1}): {str(e)}")
                        if attempt < max_retries - 1:
                            delay = base_delay * (attempt + 1)
                            self.logger.info(f"Retrying in {delay} seconds...")
                            time.sleep(delay)
                        continue
                
                self.logger.warning(f"Invalid API response format (Attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                continue
                    
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                continue
        
        self.logger.error(f"Failed to get valid response after {max_retries} attempts")
        return None

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
            
            # If fire type isn't one of our defined classes, fail
            if label_data['fire_type'] not in fire_types:
                self.logger.error(f"Invalid fire type: {label_data['fire_type']}")
                return False
                
            class_idx = fire_types[label_data['fire_type']]
            
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            yolo_format = f"{class_idx} 0.5 0.5 0.8 0.8\n"
            
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
            return self.get_fire_location(img_path)
        except Exception as e:
            self.logger.error(f"Failed to process {img_path}: {str(e)}")
            return None

    def prepare_dataset(self, source_dir: str):
        """Prepare the dataset with validation"""
        self.logger.info("Starting dataset preparation")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        
        # Track statistics
        stats = {split: {'total': 0, 'success': 0, 'failed': 0} 
                for split in ['train', 'val', 'test']}
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_dir = Path(source_dir) / 'images' / split
            self.logger.info(f"\nProcessing {split} split")
            self.logger.info(f"Directory: {split_dir}")
            
            image_files = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png'))
            self.logger.info(f"Found {len(image_files)} images")
            
            stats[split]['total'] = len(image_files)
            
            # Process images with progress bar
            for img_path in tqdm(image_files, desc=split):
                try:
                    # Get label data with validation
                    label_data = self.validate_and_label_image(img_path)
                    
                    if label_data is not None:
                        # Save valid data
                        if self.save_label(img_path, label_data):
                            stats[split]['success'] += 1
                        else:
                            stats[split]['failed'] += 1
                            self.logger.warning(f"Failed to save: {img_path.name}")
                    else:
                        stats[split]['failed'] += 1
                        self.logger.warning(f"Failed to process: {img_path.name}")
                        
                except Exception as e:
                    stats[split]['failed'] += 1
                    self.logger.error(f"Error processing {img_path}: {e}")
        
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

def main():
    labeler = LlamaLabeler()
    labeler.prepare_dataset('data/fire_dataset')

if __name__ == "__main__":
    main()