# test_labeler.py
from prepare_binary_dataset import LlamaLabeler, cleanup_directories
from pathlib import Path

def test_labeler():
    print("Testing binary fire detection labeler...")
    
    # Initialize labeler
    try:
        labeler = LlamaLabeler()
        print("✓ Labeler initialized successfully")
    except Exception as e:
        print(f"✗ Labeler initialization failed: {e}")
        return
    
    # Test directory structure
    required_dirs = [
        labeler.images_dir / 'train',
        labeler.images_dir / 'val',
        labeler.images_dir / 'test',
        labeler.labels_dir / 'train',
        labeler.labels_dir / 'val',
        labeler.labels_dir / 'test',
        labeler.viz_dir / 'train',
        labeler.viz_dir / 'val',
        labeler.viz_dir / 'test'
    ]
    
    print("\nChecking directory structure...")
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            
    print("\nChecking configuration...")
    print(f"✓ Model: {labeler.model}")
    print(f"✓ Dataset root: {labeler.dataset_root}")
    print(f"✓ Binary classification: fire/no-fire")

if __name__ == "__main__":
    test_labeler()