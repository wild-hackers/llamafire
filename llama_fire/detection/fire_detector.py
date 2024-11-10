from ultralytics import YOLO
import cv2
import numpy as np
import logging
from pathlib import Path
import torch
import platform

class FireDetector:
    def __init__(self, model_path=None):
        self.logger = logging.getLogger(__name__)
        self.root_dir = Path(__file__).parent.parent.parent  # Get project root
        
        # Determine device (M1/M2 Macs use MPS)
        self.device = 'mps' if platform.processor() == 'arm' else 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.logger.info(f"Using device: {self.device}")
        
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Check for trained model in custom directory
            custom_model = self.root_dir / 'models' / 'custom' / 'fire_detection.pt'
            if custom_model.exists():
                self.logger.info(f"Loading custom model: {custom_model}")
                self.model = YOLO(custom_model)
            else:
                # Train new model if none exists
                self.logger.info("No custom model found. Training new model...")
                base_model = self.root_dir / 'models' / 'pretrained' / 'yolov8n.pt'
                self.model = YOLO(base_model)
                self.train_model()
        
        # Move model to appropriate device
        self.model.to(self.device)
        
        # Fire detection thresholds
        self.confidence_threshold = 0.5
        self.fire_color_lower = np.array([0, 50, 50])
        self.fire_color_upper = np.array([25, 255, 255])
        
    def train_model(self):
        """Train the model on fire dataset"""
        data_yaml = self.root_dir / 'data' / 'fire_dataset' / 'data.yaml'
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
            
        self.logger.info("Starting model training...")
        results = self.model.train(
            data=str(data_yaml),
            epochs=50,
            imgsz=640,
            batch=16,
            device=self.device,
            patience=20,
            save=True,
            project=str(self.root_dir / 'models' / 'custom'),
            name='fire_detection'
        )
        
        # Save the trained model
        save_path = self.root_dir / 'models' / 'custom' / 'fire_detection.pt'
        self.model.save(str(save_path))
        self.logger.info(f"Model saved to {save_path}")
        
    def detect_fire(self, frame):
        """
        Multi-stage fire detection:
        1. Color-based detection for initial screening
        2. YOLO model for confirmation and classification
        """
        try:
            # First check for fire-like colors
            if not self.color_detection(frame):
                return False, None, 0.0, None
            
            # Move frame to appropriate device
            results = self.model(frame, 
                               conf=self.confidence_threshold,
                               device=self.device)
            
            for result in results:
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get class and confidence
                        cls = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        
                        if conf > self.confidence_threshold:
                            # Return class type along with detection
                            return True, box.xyxy[0].tolist(), conf, cls
                            
            return False, None, 0.0, None
            
        except Exception as e:
            self.logger.error(f"Fire detection error: {e}")
            return False, None, 0.0, None
            
    def color_detection(self, frame):
        """Detect fire-like colors in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.fire_color_lower, self.fire_color_upper)
        fire_pixel_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        return fire_pixel_ratio > 0.01