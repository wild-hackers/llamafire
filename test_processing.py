# test_processing.py
from prepare_binary_dataset import LlamaLabeler
from pathlib import Path
import logging

def test_process_images():
    # Initialize labeler with debug logging
    logging.basicConfig(level=logging.DEBUG)  # Change to DEBUG level
    labeler = LlamaLabeler()
    
    # Get a few test images from train directory
    train_dir = Path("data/binary_fire_dataset/images/train")
    test_images = list(train_dir.glob("*.[jp][pn][g]"))[:3]
    
    print(f"\nFound {len(test_images)} test images")
    
    # Process each test image
    for img_path in test_images:
        print(f"\nProcessing: {img_path.name}")
        try:
            # Get raw response first
            response = labeler.get_llama_response(img_path)
            print(f"Raw API response: {response}")
            
            # Process the image
            result = labeler.process_image(img_path)
            print(f"Processing result: {'Success' if result else 'Failed'}")
            
            # Check if label file was created
            label_path = labeler.labels_dir / 'train' / f"{img_path.stem}.txt"
            if label_path.exists():
                print(f"✓ Label file created: {label_path.name}")
                with open(label_path) as f:
                    print(f"Label content: {f.read().strip()}")
                
            # Check if visualization was created
            viz_path = labeler.viz_dir / 'train' / f"viz_{img_path.name}"
            if viz_path.exists():
                print(f"✓ Visualization created: {viz_path.name}")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_process_images()