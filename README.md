# Fire Detection System

Real-time fire detection and analysis using YOLOv8 and Llama 3.2 Vision.

## Installation

1. Create virtual environment:
python3.11 -m venv venv
source venv/bin/activate

2. Install dependencies:
pip install -r requirements.txt

3. Set up environment variables:
echo "TOGETHER_API_KEY=your_key_here" > .env

## Dataset Preparation

1. Place fire images in:
data/fire_dataset/images/
├── train/  # 70% of images
├── val/    # 20% of images
└── test/   # 10% of images

2. Label dataset:
python scripts/prepare_dataset.py

3. Train model:
python scripts/train_model.py

## Running

Start detection system:
python run.py
