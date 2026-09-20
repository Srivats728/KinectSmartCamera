"""
Kinect v2 Manager for macOS using libfreenect2 Python bindings.
Handles color, depth, and IR stream capture.
"""

import logging
import threading
import time
from typing import Optional, Tuple, Callable, Dict, Any
import numpy as np

try:
    from freenect2 import Device, FrameType, Frame
    FREENECT2_AVAILABLE = True
except ImportError:
    FREENECT2_AVAILABLE = False
    Device = None
    FrameType = None
    Frame = None

logger = logging.getLogger(__name__)


class KinectManager:
    """Manages Kinect v2 device connection and frame capture."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device: Optional[Device] = None
        self.running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_callbacks: Dict[FrameType, list] = {
            FrameType.Color: [] if FREENECT2_AVAILABLE else [],
            FrameType.Depth: [] if FREENECT2_AVAILABLE else [],
            FrameType.IR: [] if FREENECT2_AVAILABLE else [],
        }
        self._latest_frames: Dict[FrameType, Optional[np.ndarray]] = {
            FrameType.Color: None,
            FrameType.Depth: None,
            FrameType.IR: None,
        }
        self._frame_lock = threading.Lock()
        self._fps_counter = {FrameType.Color: 0, FrameType.Depth: 0, FrameType.IR: 0}
        self._last_fps_time = time.time()
        
    def initialize(self) -> bool:
        """Initialize the Kinect device."""
        if not FREENECT2_AVAILABLE:
            logger.error("freenect2 not available. Install with: pip install freenect2")
            return False
            
        try:
            self.device = Device()
            logger.info("Kinect device initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kinect: {e}")
            return False
    
    def start(self) -> bool:
        """Start frame capture."""
        if not self.device:
            logger.error("Device not initialized")
            return False
            
        if self.running:
            logger.warning("Already running")
            return True
            
        try:
            self.device.start()
            self.running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            logger.info("Kinect capture started")
            return True
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            return False
    
    def stop(self):
        """Stop frame capture."""
        self.running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self.device:
            try:
                self.device.stop()
            except Exception as e:
                logger.error(f"Error stopping device: {e}")
        logger.info("Kinect capture stopped")
    
    def _capture_loop(self):
        """Main capture loop."""
        if not self.device:
            return
            
        for frame_type, frame in self.device:
            if not self.running:
                break
                
            try:
                # Convert frame to numpy array
                if frame_type == FrameType.Color:
                    img = frame.to_image()
                    arr = np.array(img)
                    # Convert RGB to BGR for OpenCV
                    arr = arr[:, :, ::-1]
                elif frame_type == FrameType.Depth:
                    arr = frame.to_array()
                elif frame_type == FrameType.IR:
                    arr = frame.to_array()
                else:
                    continue
                
                with self._frame_lock:
                    self._latest_frames[frame_type] = arr
                    self._fps_counter[frame_type] += 1
                
                # Call registered callbacks
                for callback in self._frame_callbacks.get(frame_type, []):
                    try:
                        callback(arr.copy())
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
    
    def register_callback(self, frame_type: FrameType, callback: Callable[[np.ndarray], None]):
        """Register a callback for a specific frame type."""
        if frame_type in self._frame_callbacks:
            self._frame_callbacks[frame_type].append(callback)
    
    def unregister_callback(self, frame_type: FrameType, callback: Callable[[np.ndarray], None]):
        """Unregister a callback."""
        if frame_type in self._frame_callbacks and callback in self._frame_callbacks[frame_type]:
            self._frame_callbacks[frame_type].remove(callback)
    
    def get_latest_frame(self, frame_type: FrameType) -> Optional[np.ndarray]:
        """Get the latest frame of a specific type."""
        with self._frame_lock:
            return self._latest_frames.get(frame_type)
    
    def get_fps(self) -> Dict[str, float]:
        """Get current FPS for each stream."""
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            with self._frame_lock:
                fps = {k.name: v / elapsed for k, v in self._fps_counter.items()}
                self._fps_counter = {k: 0 for k in self._fps_counter}
                self._last_fps_time = now
            return fps
        return {k.name: 0 for k in self._fps_counter}
    
    def is_running(self) -> bool:
        return self.running
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        if not self.device:
            return {}
        return {
            "serial": getattr(self.device, 'serial', 'unknown'),
            "firmware": getattr(self.device, 'firmware_version', 'unknown'),
        }


class MockKinectManager:
    """Mock Kinect manager for testing without hardware."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_callbacks: Dict[str, list] = {"color": [], "depth": [], "ir": []}
        self._latest_frames = {"color": None, "depth": None, "ir": None}
        self._frame_lock = threading.Lock()
        
    def initialize(self) -> bool:
        logger.info("Mock Kinect initialized")
        return True
    
    def start(self) -> bool:
        self.running = True
        self._capture_thread = threading.Thread(target=self._mock_capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("Mock Kinect capture started")
        return True
    
    def stop(self):
        self.running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        logger.info("Mock Kinect capture stopped")
    
    def _mock_capture_loop(self):
        """Generate mock frames for testing."""
        import cv2
        while self.running:
            # Mock color frame (640x480)
            color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(color, "MOCK KINECT", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Mock depth frame (512x424)
            depth = np.random.randint(500, 4000, (424, 512), dtype=np.uint16)
            # Add a "person" shape in the middle
            cv2.rectangle(depth, (200, 150), (312, 350), 1500, -1)
            
            # Mock IR frame
            ir = np.random.randint(0, 1000, (424, 512), dtype=np.uint16)
            
            with self._frame_lock:
                self._latest_frames["color"] = color
                self._latest_frames["depth"] = depth
                self._latest_frames["ir"] = ir
            
            for cb in self._frame_callbacks["color"]:
                cb(color.copy())
            for cb in self._frame_callbacks["depth"]:
                cb(depth.copy())
            for cb in self._frame_callbacks["ir"]:
                cb(ir.copy())
                
            time.sleep(1/30)  # 30 FPS
    
    def register_callback(self, frame_type: str, callback: Callable[[np.ndarray], None]):
        if frame_type in self._frame_callbacks:
            self._frame_callbacks[frame_type].append(callback)
    
    def get_latest_frame(self, frame_type: str) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frames.get(frame_type)
    
    def get_fps(self) -> Dict[str, float]:
        return {"Color": 30, "Depth": 30, "IR": 30}
    
    def is_running(self) -> bool:
        return self.running
    
    def get_device_info(self) -> Dict[str, Any]:
        return {"serial": "MOCK", "firmware": "MOCK"}