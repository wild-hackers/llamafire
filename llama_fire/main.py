import asyncio
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging to be more focused
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
# Suppress verbose logging from libraries
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger('FireMonitor')

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.mock.dji_mock import LlamaFireControlMock
from src.detection.fire_detector import FireDetector
from src.analysis.llama_analyzer import LlamaAnalyzer
import cv2
import time

class FireMonitoringSystem:
    def __init__(self, model_path=None):
        self.mock_dji = LlamaFireControlMock()
        self.detector = FireDetector(model_path=model_path)
        self.analyzer = LlamaAnalyzer()
        self.last_analysis_time = 0
        self.analysis_cooldown = 60
        self.frame_delay = 0.5
        self.detection_count = 0
        self.consecutive_detections = 0
        self.min_detections = 3  # Require multiple consecutive detections
        
        # Create necessary directories
        self.data_dir = Path("data")
        self.images_dir = self.data_dir / "detected_fires"
        self.analysis_dir = self.data_dir / "analysis"
        
        for directory in [self.images_dir, self.analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info("🚀 Fire Monitoring System Initialized")
        logger.info("Press 'q' to quit, 't' for takeoff, 'l' for landing")

    async def save_analysis(self, image_path, analysis, timestamp):
        """Save analysis results with metadata"""
        analysis_data = {
            "timestamp": timestamp,
            "image_path": str(image_path),
            "analysis": analysis,
            "telemetry": {
                "altitude": self.mock_dji.altitude,
                "gps": self.mock_dji.gps,
                "battery": self.mock_dji.battery
            }
        }
        
        analysis_file = self.analysis_dir / f"analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_data, f, indent=4)
        
        logger.info(f"📝 Analysis saved: {analysis_file.name}")
        
    async def run(self):
        """Main monitoring loop"""
        try:
            logger.info("▶️  Starting monitoring...")
            
            while True:
                frame = await self.mock_dji.get_frame()
                fire_detected, box, confidence, cls = self.detector.detect_fire(frame)
                
                # Add delay between frames
                await asyncio.sleep(self.frame_delay)
                
                if fire_detected:
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0
                    
                # Only trigger fire detection after multiple consecutive detections
                if self.consecutive_detections >= self.min_detections:
                    self.detection_count += 1
                    current_time = time.time()
                    
                    # Only log every 5th detection to reduce console spam
                    if self.detection_count % 5 == 0:
                        logger.info(f"🔥 Fire detected (confidence: {confidence:.2f})")
                    
                    if current_time - self.last_analysis_time > self.analysis_cooldown:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Status update
                        logger.info(f"📸 Capturing frame for analysis...")
                        image_path = self.images_dir / f"fire_{timestamp}.jpg"
                        cv2.imwrite(str(image_path), frame)
                        
                        # Analyze with Llama
                        logger.info("🤖 Analyzing with Llama Vision...")
                        analysis = self.analyzer.analyze_fire(str(image_path))
                        
                        if analysis['status'] == 'success':
                            await self.save_analysis(image_path, analysis['analysis'], timestamp)
                            
                            # Format the analysis output nicely
                            logger.info("\n🔍 Fire Analysis Report:")
                            logger.info("-" * 50)
                            for line in analysis['analysis'].split('\n'):
                                if line.strip():
                                    logger.info(f"  {line.strip()}")
                            logger.info("-" * 50)
                        
                        self.last_analysis_time = current_time
                
                # Display frame with detection box if fire detected
                if fire_detected and box:
                    cv2.rectangle(frame, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                (0, 0, 255), 2)
                    cv2.putText(frame, f"Fire: {confidence:.2f}", 
                              (int(box[0]), int(box[1]-10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                
                # Show frame
                cv2.imshow("Fire Monitoring", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("👋 Shutting down...")
                    break
                elif key == ord('t'):
                    logger.info("🛫 Takeoff initiated")
                elif key == ord('l'):
                    logger.info("🛬 Landing initiated")
                    
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.mock_dji.close()
            cv2.destroyAllWindows()
            logger.info("✅ System shutdown complete")

if __name__ == "__main__":
    monitor = FireMonitoringSystem()
    asyncio.run(monitor.run())