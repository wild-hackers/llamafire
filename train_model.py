from ultralytics import YOLO
import logging
import os
from pathlib import Path
import torch
import torch.multiprocessing as mp
import yaml
import shutil
import sys

# Set the correct base directory (relative to project root)
BASE_DIR = Path.cwd()  # Use current working directory
DATASET_DIR = BASE_DIR / 'data' / 'binary_fire_dataset'

# Create logs directory if it doesn't exist
logs_dir = DATASET_DIR / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

# Configure logging with absolute path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / 'training.log', mode='w')
    ]
)
logger = logging.getLogger('FireTrainer')

def setup_environment():
    """Configure training environment"""
    logger.info("Setting up training environment...")
    
    # Force multiprocessing settings
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['PYTORCH_WORKERS'] = '4'
    
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.backends.mps.enable_ddp = False
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    torch.set_num_threads(4)
    return device

def verify_dataset():
    """Verify dataset is ready for training"""
    logger.info("Verifying dataset...")
    
    # Check data.yaml exists
    yaml_path = DATASET_DIR / 'data.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    
    # Check images and labels exist
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_DIR / 'images' / split
        label_dir = DATASET_DIR / 'labels' / split
        
        images = list(img_dir.glob('*.[jp][pn][g]'))
        labels = list(label_dir.glob('*.txt'))
        
        logger.info(f"{split} split: {len(images)} images, {len(labels)} labels")
        
        if len(images) == 0:
            raise ValueError(f"No images found in {img_dir}")
        if len(labels) == 0:
            raise ValueError(f"No labels found in {label_dir}")
    
    return True

def train_fire_detector():
    """Train the fire detection model"""
    try:
        # Setup environment
        device = setup_environment()
        
        # Verify dataset is ready
        verify_dataset()
        
        # Setup paths
        pretrained_model_path = BASE_DIR / 'models' / 'pretrained' / 'yolov8n.pt'
        runs_path = BASE_DIR / 'runs/detect'
        
        # Log dataset info
        logger.info(f"Dataset path: {DATASET_DIR}")
        logger.info(f"Pretrained model: {pretrained_model_path}")
        
        # Load model
        logger.info("Loading YOLOv8n base model")
        model = YOLO(str(pretrained_model_path))
        
        # Create data.yaml with absolute paths
        yaml_content = {
            'path': str(DATASET_DIR.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # binary classification
            'names': ['fire']
        }
        
        yaml_path = DATASET_DIR / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        logger.info(f"Updated dataset config at: {yaml_path}")
        
        # Start training
        logger.info(f"Starting training on device: {device}")
        results = model.train(
            data=str(yaml_path),
            epochs=100,
            imgsz=640,
            patience=50,
            batch=8,
            mosaic=0.5,
            degrees=10.0,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            copy_paste=0.0,
            mixup=0.0,
            save=True,
            device='mps',
            workers=0,
            cache='disk',
            amp=False,
            optimizer='SGD',
            cos_lr=True,
            warmup_epochs=10,
            weight_decay=0.0005,
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            seed=42,
            deterministic=True,
            project='runs/detect',
            name='binary_fire_detection_v2',
            exist_ok=True,
            verbose=True,
            max_det=100,
            single_cls=True
        )
        
        # Copy best model to custom models directory
        best_model = runs_path / 'binary_fire_detection_v2' / 'weights' / 'best.pt'
        custom_models_dir = BASE_DIR / 'models' / 'custom'
        custom_models_dir.mkdir(exist_ok=True, parents=True)
        
        if best_model.exists():
            shutil.copy2(best_model, custom_models_dir / 'fire_detection_binary.pt')
            logger.info(f"Saved best model to: {custom_models_dir/'fire_detection_binary.pt'}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        
        # Run training
        results = train_fire_detector()
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)