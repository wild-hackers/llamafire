# test_labeler.py
from prepare_binary_dataset import LlamaLabeler, cleanup_directories
from pathlib import Path

def test_labeler():
    print("Testing LlamaLabeler initialization...")
    
    # Initialize labeler
    try:
        labeler = LlamaLabeler()
        print("✓ Labeler initialized successfully")
    except Exception as e:
        print(f"✗ Labeler initialization failed: {e}")
        return
    
    # Test directory structure
    required_dirs = [
        labeler.images_dir,
        labeler.labels_dir,
        labeler.viz_dir
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")

if __name__ == "__main__":
    test_labeler()