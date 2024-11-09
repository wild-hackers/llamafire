from ultralytics import YOLO
import logging
import os
from pathlib import Path
import torch
import torch.multiprocessing as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/fire_dataset/training.log', mode='w')
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
    
    # Setup paths using absolute paths
    base_path = Path.cwd()
    dataset_path = base_path / 'data/fire_dataset'
    model_path = base_path / 'models'
    runs_path = base_path / 'runs/detect'
    
    model_path.mkdir(exist_ok=True, parents=True)
    
    # Log dataset info
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Training images: {len(list((dataset_path/'images'/'train').glob('*')))}")
    logger.info(f"Validation images: {len(list((dataset_path/'images'/'val').glob('*')))}")
    logger.info(f"Test images: {len(list((dataset_path/'images'/'test').glob('*')))}")
    
    # Load model
    logger.info("Loading YOLOv8l base model")
    model = YOLO('yolov8l.pt')
    
    # Start training with optimized parameters
    logger.info(f"Starting training on device: {device}")
    results = model.train(
        data=str(dataset_path/'data.yaml'),
        epochs=100,
        imgsz=640,
        patience=20,
        batch=32,
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
        name='train_optimized_large',
        exist_ok=True,
        verbose=True,
        max_det=100
    )
    
    # Save and log results
    try:
        # Save final model
        final_model_path = model_path / 'fire_detection.pt'
        model.save(str(final_model_path))
        logger.info(f"Model exported to: {final_model_path}")
        
        # Log metrics
        metrics = results.results_dict
        logger.info("Final Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.3f}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

if __name__ == '__main__':
    try:
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        mp.freeze_support()
        
        # Run training
        train_fire_detector()
    except Exception as e:
        logger.error(f"Training failed: {e}") 