from ultralytics import YOLO
import logging
import os
from pathlib import Path
import torch
import torch.multiprocessing as mp
import yaml
import shutil

# Set the correct base directory (relative to project root)
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / 'data' / 'fire_dataset'

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
    # Force multiprocessing settings
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['MKL_NUM_THREADS'] = '4'  # Reduced from 8
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['PYTORCH_WORKERS'] = '4'  # Explicitly set worker count
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.backends.mps.enable_ddp = False
    else:
        device = torch.device("cpu")
    
    torch.set_num_threads(4)  # Match with environment variables
    return device

def train_fire_detector():
    # Setup environment
    device = setup_environment()
    
    # Setup paths using absolute paths from BASE_DIR
    dataset_path = BASE_DIR / 'data' / 'fire_dataset'
    pretrained_model_path = BASE_DIR / 'models' / 'pretrained' / 'yolov8n.pt'
    runs_path = BASE_DIR / 'runs/detect'
    
    # Log dataset info
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Training images: {len(list((dataset_path/'images'/'train').glob('*.[jp][pn][g]')))}")
    logger.info(f"Validation images: {len(list((dataset_path/'images'/'val').glob('*.[jp][pn][g]')))}")
    logger.info(f"Test images: {len(list((dataset_path/'images'/'test').glob('*.[jp][pn][g]')))}")
    
    # Load model from local pretrained file
    logger.info(f"Loading YOLOv8n base model from {pretrained_model_path}")
    model = YOLO(str(pretrained_model_path))
    
    # Create data.yaml with absolute paths
    yaml_content = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 3,
        'names': ['building_fire', 'forest_fire', 'vehicle_fire']
    }
    
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    logger.info(f"Updated dataset config at: {yaml_path}")
    
    # Start training with optimized parameters for small dataset
    logger.info(f"Starting training on device: {device}")
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=100,
            imgsz=640,
            patience=20,
            batch=16,  # Smaller batch size
            mosaic=1.0,  # Increase mosaic probability
            degrees=20.0,  # More rotation
            translate=0.2,  # More translation
            scale=0.9,  # More scaling
            fliplr=0.5,
            flipud=0.3,
            hsv_h=0.015,  # Color augmentation
            hsv_s=0.7,
            hsv_v=0.4,
            copy_paste=0.3,  # Enable copy-paste augmentation
            mixup=0.3,      # Enable mixup augmentation
            save=True,
            device=device,
            workers=0,      
            cache=True,
            amp=True,
            plots=False,
            optimizer='AdamW',
            close_mosaic=10,
            nbs=32,
            cos_lr=True,
            warmup_epochs=5,
            weight_decay=0.001,
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            seed=42,
            deterministic=True,
            project='runs/detect',
            name='train_optimized_small',
            exist_ok=True,
            verbose=True,
            max_det=100
        )
        
        # Copy best model to custom models directory
        best_model = runs_path / 'train_optimized_small' / 'weights' / 'best.pt'
        custom_models_dir = BASE_DIR / 'models' / 'custom'
        custom_models_dir.mkdir(exist_ok=True, parents=True)
        
        if best_model.exists():
            shutil.copy2(best_model, custom_models_dir / 'fire_detection_small.pt')
            logger.info(f"Copied best model to: {custom_models_dir/'fire_detection_small.pt'}")
        
        return results
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        mp.freeze_support()
        
        # Run training
        train_fire_detector()
    except Exception as e:
        logger.error(f"Training failed: {e}") 