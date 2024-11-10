# test_setup.py
from pathlib import Path
import cv2
import numpy as np

def test_environment():
    print("Testing environment setup...")
    
    # Test paths
    dataset_dir = Path("data/binary_fire_dataset")
    required_dirs = [
        dataset_dir / 'images' / split 
        for split in ['train', 'val', 'test']
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Test OpenCV
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (10, 10), (90, 90), (0, 255, 0), 2)
    test_path = dataset_dir / 'test_image.jpg'
    cv2.imwrite(str(test_path), test_img)
    
    if test_path.exists():
        print("OpenCV working correctly")
        test_path.unlink()  # Clean up test image
    
    print("Environment test complete!")

if __name__ == "__main__":
    test_environment()