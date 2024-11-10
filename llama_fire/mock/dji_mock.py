import cv2
import asyncio
import numpy as np
from datetime import datetime
import time
import math
import logging

class LlamaFireControlMock:
    def __init__(self, port=8089):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger('LlamaFireMock')
        
        self.port = port
        self.video_window = None
        self.recording = False
        self.video_writer = None
        self.camera = cv2.VideoCapture(0)
        
        # Mock drone state
        self.is_flying = False
        self.battery = 100
        self.altitude = 0.0
        self.speed = 0.0
        self.gps = {"lat": 37.7749, "lon": -122.4194}  # Mock GPS coordinates
        self.gimbal_pitch = 0  # -90 to 0 degrees
        self.camera_mode = "NORMAL"  # NORMAL, WIDE, ZOOM
        self.zoom_level = 1.0  # 1.0x to 4.0x for Air 2S
        
        self.logger.info("LlamaFire Mock Controller initialized")

    def update_mock_telemetry(self):
        """Update mock telemetry data"""
        if self.is_flying:
            old_battery = self.battery
            old_alt = self.altitude
            
            # Simulate battery drain
            self.battery = max(0, self.battery - 0.01)
            # Simulate slight GPS drift
            self.gps["lat"] += np.random.normal(0, 0.0001)
            self.gps["lon"] += np.random.normal(0, 0.0001)
            # Simulate altitude variations
            self.altitude += np.random.normal(0, 0.1)
            self.altitude = max(0, min(500, self.altitude))  # Keep between 0-500m
            # Simulate speed variations
            self.speed = abs(np.random.normal(5, 1))  # Around 5 m/s with variation
            
            # Log significant changes
            if int(old_battery) != int(self.battery):
                self.logger.info(f"Battery: {int(self.battery)}%")
            if abs(old_alt - self.altitude) > 1.0:
                self.logger.info(f"Altitude change: {self.altitude:.1f}m")

    def draw_hud(self, frame):
        """Draw drone-like HUD overlay"""
        height, width = frame.shape[:2]
        
        # Add telemetry data
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Battery indicator (top-right)
        battery_text = f"Battery: {int(self.battery)}%"
        cv2.putText(frame, battery_text, (width-150, 30), font, 0.5, (0,255,0), 1)
        
        # Altitude and Speed (top-left)
        alt_text = f"ALT: {self.altitude:.1f}m"
        spd_text = f"SPD: {self.speed:.1f}m/s"
        cv2.putText(frame, alt_text, (20, 30), font, 0.5, (0,255,0), 1)
        cv2.putText(frame, spd_text, (20, 50), font, 0.5, (0,255,0), 1)
        
        # GPS coordinates (bottom-left)
        gps_text = f"GPS: {self.gps['lat']:.4f}, {self.gps['lon']:.4f}"
        cv2.putText(frame, gps_text, (20, height-20), font, 0.5, (0,255,0), 1)
        
        # Recording indicator (top-center)
        if self.recording:
            cv2.circle(frame, (width//2, 20), 10, (0,0,255), -1)
            
        # Flight mode and camera info (top-center)
        mode_text = f"Mode: {self.camera_mode} {self.zoom_level}x"
        cv2.putText(frame, mode_text, (width//2-50, 40), font, 0.5, (0,255,0), 1)
        
        # Artificial horizon (center)
        center_x, center_y = width//2, height//2
        cv2.line(frame, (center_x-40, center_y), (center_x+40, center_y), (0,255,0), 1)
        cv2.line(frame, (center_x, center_y-40), (center_x, center_y+40), (0,255,0), 1)
        
        return frame

    async def start(self):
        """Start the video display from MacBook camera"""
        self.video_window = cv2.namedWindow("LlamaFire Mock Feed", cv2.WINDOW_NORMAL)
        last_time = time.time()
        
        while True:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Update mock telemetry every 100ms
                current_time = time.time()
                if current_time - last_time > 0.1:
                    self.update_mock_telemetry()
                    last_time = current_time
                
                # Apply zoom if needed
                if self.zoom_level > 1.0:
                    height, width = frame.shape[:2]
                    crop_size = int(min(height, width) / self.zoom_level)
                    start_y = (height - crop_size) // 2
                    start_x = (width - crop_size) // 2
                    frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
                    frame = cv2.resize(frame, (width, height))
                
                # Add HUD overlay
                frame = self.draw_hud(frame)
                
                # Display frame
                cv2.imshow("LlamaFire Mock Feed", frame)
                
                # Record if enabled
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # Handle key commands
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.toggle_recording()
                elif key == ord('t'):
                    self.takeoff()
                elif key == ord('l'):
                    self.land()
                elif key == ord('w'):
                    self.gimbal_pitch = max(-90, self.gimbal_pitch - 5)
                elif key == ord('s'):
                    self.gimbal_pitch = min(0, self.gimbal_pitch + 5)
                elif key == ord('z'):
                    self.zoom_level = max(1.0, self.zoom_level - 0.2)
                elif key == ord('x'):
                    self.zoom_level = min(4.0, self.zoom_level + 0.2)
                elif key == ord('c'):
                    self.cycle_camera_mode()
                
            except Exception as e:
                print(f"Error: {e}")
                break

    def takeoff(self):
        """Simulate takeoff"""
        if not self.is_flying:
            print("Taking off...")
            self.is_flying = True
            self.altitude = 1.5  # Initial hover altitude

    def land(self):
        """Simulate landing"""
        if self.is_flying:
            print("Landing...")
            self.is_flying = False
            self.altitude = 0.0
            self.speed = 0.0

    def cycle_camera_mode(self):
        """Cycle through camera modes"""
        modes = ["NORMAL", "WIDE", "ZOOM"]
        current_index = modes.index(self.camera_mode)
        self.camera_mode = modes[(current_index + 1) % len(modes)]
        print(f"Camera mode: {self.camera_mode}")

    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            filename = f"llamafire_mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, 
                                              (frame_width, frame_height))
            self.recording = True
            print("Recording started")
        else:
            # Stop recording
            self.video_writer.release()
            self.recording = False
            print("Recording stopped")

    def close(self):
        """Cleanup resources"""
        if self.video_writer:
            self.video_writer.release()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

    async def get_frame(self):
        """Get current frame from camera"""
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to grab frame")
        
        # Only update telemetry every 30 frames
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 0
            
        if self.frame_count % 30 == 0:
            self.update_mock_telemetry()
            frame = self.draw_hud(frame)
            
        return frame

# Usage example
if __name__ == "__main__":
    controller = LlamaFireControlMock()
    try:
        asyncio.get_event_loop().run_until_complete(controller.start())
    except KeyboardInterrupt:
        print("\nShutting down LlamaFire Mock Control...")
    finally:
        controller.close() 