# Fire Detection System

Real-time fire detection and analysis using YOLOv8 and Llama 3.2 Vision.

## Installation

1. Create virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file with your Together AI API key
echo "TOGETHER_API_KEY=your_key_here" > .env
```

## Dataset Preparation & Training

1. Label Dataset with Llama Vision:
```bash
# This will:
# - Use Llama Vision to analyze images
# - Create YOLO format labels
# - Generate visualizations
python llama_annotate.py
```

2. Use Web Interface for Annotation:
```bash
# Create static directories for images
mkdir -p static/images/{train,val,test,viz/train,viz/val,viz/test}

# Start the annotation web interface
python app_annotate.py

# Visit http://127.0.0.1:5003 in your browser to:
# - View all images in the dataset
# - Process images with Llama Vision
# - See fire detection visualizations
# - Monitor annotation progress
```

3. Train YOLOv8 Model:
```bash
# This will:
# - Train on the labeled dataset
# - Save best model to models/custom/fire_detection_binary.pt
python train_model.py
```

## Running Fire Detection System

Start the real-time detection system:
```bash
python -m llama_fire.main
```

This will:
- Use your webcam as a mock drone feed
- Run real-time fire detection
- Show visualization with bounding boxes
- Trigger Llama analysis when fire is detected

### Controls
- 'q': Quit
- 't': Takeoff (mock)
- 'l': Landing (mock)

## Project Structure
```
.
├── data/
│   └── binary_fire_dataset/     # Dataset directory
│       ├── images/              # Original images
│       ├── labels/              # YOLO format labels
│       └── visualizations/      # Debug visualizations
├── models/
│   ├── custom/                  # Trained models
│   └── pretrained/             # Base YOLOv8 models
├── llama_fire/                  # Main package
│   ├── detection/              # Fire detection
│   ├── analysis/               # Llama Vision analysis
│   └── mock/                   # Mock drone interface
├── static/                     # Web interface assets
│   └── images/                 # Served images
└── templates/                  # HTML templates
```

## Requirements
- Python 3.11+
- Together AI API key (get from together.ai)
- Webcam for mock drone feed
- GPU recommended but not required (MPS on Apple Silicon works well)

## Notes
- The system uses Together AI's Llama 3.2 Vision for initial dataset labeling
- Web interface available for viewing and processing annotations
- YOLOv8 is used for real-time detection after training
- MPS (Metal Performance Shaders) is used on Apple Silicon Macs
- CUDA is used if available on NVIDIA GPUs
