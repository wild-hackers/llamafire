import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from src.main import FireMonitoringSystem
import asyncio

if __name__ == "__main__":
    # Initialize with large model
    monitor = FireMonitoringSystem(model_path='models/custom/fire_detection.pt')
    asyncio.run(monitor.run()) 