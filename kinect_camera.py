#!/usr/bin/env python3
"""
Kinect Smart Camera - Single-file executable
Run: python3 kinect_camera.py [--mock]
"""

# Standard library imports
import logging
import threading
import time
import signal
import sys
import json
import pickle
import yaml
import uuid
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple, Deque
from collections import deque
import numpy as np
import cv2

# Third-party imports (guarded)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    sd = None

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    from freenect2 import Device, FrameType, Frame
    FREENECT2_AVAILABLE = True
except ImportError:
    FREENECT2_AVAILABLE = False
    Device = FrameType = Frame = None

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import rumps
    RUMPS_AVAILABLE = True
except ImportError:
    RUMPS_AVAILABLE = False
    rumps = None

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===== kinect_manager.py =====
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

# ===== human_detector.py =====
"""
Human detection using MediaPipe Pose and depth data from Kinect.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
    # Check MediaPipe version for API compatibility
    mp_version = getattr(mp, '__version__', '0.0.0')
    major_version = int(mp_version.split('.')[0])
    if major_version >= 1:
        # MediaPipe 1.0+ uses tasks API
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_NEW_API = True
    else:
        # Legacy API
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_NEW_API = False
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_NEW_API = False
    mp = None

logger = logging.getLogger(__name__)


class HumanDetector:
    """Detects humans using MediaPipe Pose and Kinect depth data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_config = config.get('detection', {}).get('human_detection', {})
        self.enabled = self.detection_config.get('enabled', True)
        self.confidence_threshold = self.detection_config.get('confidence_threshold', 0.5)
        self.min_person_height = self.detection_config.get('min_person_height', 0.5)
        self.max_distance = self.detection_config.get('max_distance', 5.0)
        self.use_depth = self.detection_config.get('use_depth', True)
        self.use_mediapipe = self.detection_config.get('use_mediapipe', True)
        
        self.pose_detector = None
        self.mp_drawing = None
        self.mp_pose = None
        
        self._last_detections: List[Dict[str, Any]] = []
        self._detection_lock = __import__('threading').Lock()
        
    def initialize(self) -> bool:
        """Initialize the human detector."""
        if not self.enabled:
            logger.info("Human detection disabled")
            return True
            
        if self.use_mediapipe and MEDIAPIPE_AVAILABLE:
            try:
                if MEDIAPIPE_NEW_API:
                    # MediaPipe 1.0+ tasks API - requires model file
                    # For now, fall back to depth-only detection
                    logger.warning("MediaPipe 1.0+ tasks API requires model file, using depth-only detection")
                    return True
                else:
                    # Legacy API
                    self.mp_pose = mp.solutions.pose
                    self.mp_drawing = mp.solutions.drawing_utils
                    self.pose_detector = self.mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=self.confidence_threshold,
                        min_tracking_confidence=0.5
                    )
                    logger.info("MediaPipe Pose initialized (legacy API)")
                    return True
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe: {e}")
                return False
        else:
            logger.warning("MediaPipe not available, using depth-only detection")
            return True
    
    def detect(self, color_frame: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Detect humans in the frame."""
        if not self.enabled:
            return []
            
        detections = []
        
        # MediaPipe Pose detection
        if self.use_mediapipe and self.pose_detector and color_frame is not None:
            mp_detections = self._detect_mediapipe(color_frame, depth_frame)
            detections.extend(mp_detections)
        
        # Depth-based detection (fallback or supplement)
        if self.use_depth and depth_frame is not None:
            depth_detections = self._detect_depth(depth_frame)
            detections.extend(depth_detections)
        
        # Merge overlapping detections
        merged = self._merge_detections(detections)
        
        with self._detection_lock:
            self._last_detections = merged
            
        return merged
    
    def _detect_mediapipe(self, color_frame: np.ndarray, depth_frame: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect humans using MediaPipe Pose."""
        detections = []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.pose_detector.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = color_frame.shape[:2]
            
            # Get bounding box from landmarks
            x_coords = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
            y_coords = [lm.y * h for lm in landmarks if lm.visibility > 0.5]
            
            if x_coords and y_coords:
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                pad_x = int((x_max - x_min) * 0.1)
                pad_y = int((y_max - y_min) * 0.1)
                x_min = max(0, x_min - pad_x)
                y_min = max(0, y_min - pad_y)
                x_max = min(w, x_max + pad_x)
                y_max = min(h, y_max + pad_y)
                
                # Calculate depth/distance if depth frame available
                distance = None
                if depth_frame is not None:
                    distance = self._get_depth_distance(depth_frame, x_min, y_min, x_max, y_max)
                
                # Calculate confidence from landmark visibility
                visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
                confidence = len(visible_landmarks) / len(landmarks) if landmarks else 0
                
                detection = {
                    'bbox': (x_min, y_min, x_max, y_max),
                    'confidence': confidence,
                    'distance': distance,
                    'landmarks': landmarks,
                    'source': 'mediapipe',
                    'timestamp': time.time()
                }
                detections.append(detection)
        
        return detections
    
    def _detect_depth(self, depth_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect human-like shapes in depth frame."""
        detections = []
        
        # Normalize depth for processing
        depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Threshold to find objects at reasonable distances
        # Depth values: closer = smaller values (typically)
        _, thresh = cv2.threshold(depth_normalized, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = depth_frame.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum area
                continue
                
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (human-like)
            aspect_ratio = h_box / w_box if w_box > 0 else 0
            if aspect_ratio < 1.5 or aspect_ratio > 4.0:
                continue
            
            # Get average depth in bounding box
            roi = depth_frame[y:y+h_box, x:x+w_box]
            valid_depths = roi[roi > 0]
            if len(valid_depths) == 0:
                continue
                
            avg_depth = np.mean(valid_depths)
            distance = avg_depth / 1000.0  # Convert to meters (assuming mm)
            
            if distance > self.max_distance or distance < self.min_person_height:
                continue
            
            detection = {
                'bbox': (x, y, x + w_box, y + h_box),
                'confidence': min(area / 10000.0, 1.0),
                'distance': distance,
                'landmarks': None,
                'source': 'depth',
                'timestamp': time.time()
            }
            detections.append(detection)
        
        return detections
    
    def _get_depth_distance(self, depth_frame: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int) -> Optional[float]:
        """Get average distance from depth frame in bounding box."""
        # Scale coordinates if depth frame has different resolution
        dh, dw = depth_frame.shape[:2]
        ch, cw = depth_frame.shape[:2]  # Will be overridden by actual color frame size
        
        # Simple scaling - assumes depth is 512x424 and color is 1920x1080 or similar
        scale_x = dw / cw if cw > 0 else 1
        scale_y = dh / ch if ch > 0 else 1
        
        dx_min = int(x_min * scale_x)
        dy_min = int(y_min * scale_y)
        dx_max = int(x_max * scale_x)
        dy_max = int(y_max * scale_y)
        
        dx_min = max(0, min(dx_min, dw - 1))
        dy_min = max(0, min(dy_min, dh - 1))
        dx_max = max(0, min(dx_max, dw))
        dy_max = max(0, min(dy_max, dh))
        
        if dx_max <= dx_min or dy_max <= dy_min:
            return None
            
        roi = depth_frame[dy_min:dy_max, dx_min:dx_max]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) == 0:
            return None
            
        # Depth is typically in millimeters
        return float(np.mean(valid_depths)) / 1000.0
    
    def _merge_detections(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Merge overlapping detections."""
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        merged = []
        
        for det in detections:
            bbox = det['bbox']
            overlapped = False
            
            for existing in merged:
                if self._iou(bbox, existing['bbox']) > iou_threshold:
                    overlapped = True
                    # Keep the one with higher confidence
                    if det['confidence'] > existing['confidence']:
                        existing.update(det)
                    break
            
            if not overlapped:
                merged.append(det)
        
        return merged
    
    def _iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
            
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes and info on frame."""
        annotated = frame.copy()
        
        for i, det in enumerate(detections):
            x_min, y_min, x_max, y_max = det['bbox']
            confidence = det['confidence']
            distance = det.get('distance')
            source = det.get('source', 'unknown')
            
            # Color based on source
            color = (0, 255, 0) if source == 'mediapipe' else (255, 165, 0)
            
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
            
            label = f"Person {i+1}: {confidence:.2f}"
            if distance:
                label += f" ({distance:.2f}m)"
            label += f" [{source}]"
            
            cv2.putText(annotated, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks if available
            if det.get('landmarks') and self.mp_drawing and self.mp_pose:
                # This would require converting landmarks back to image coordinates
                pass
        
        return annotated
    
    def get_last_detections(self) -> List[Dict[str, Any]]:
        with self._detection_lock:
            return self._last_detections.copy()
    
    def cleanup(self):
        if self.pose_detector:
            self.pose_detector.close()
            self.pose_detector = None

# ===== clap_detector.py =====
"""
Real-time clap detection using audio input.
"""

import logging
import threading
import time
from typing import Optional, Callable, List, Deque, Dict, Any
from collections import deque
import numpy as np

# Try to import sounddevice, fall back to mock if PortAudio not available
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    sd = None
    logger = logging.getLogger(__name__)
    logger.warning(f"sounddevice not available: {e}. Clap detection will use mock mode.")

logger = logging.getLogger(__name__)


class ClapDetector:
    """Detects claps in real-time audio stream."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clap_config = config.get('detection', {}).get('clap_detection', {})
        self.enabled = self.clap_config.get('enabled', True)
        self.sample_rate = self.clap_config.get('sample_rate', 44100)
        self.chunk_size = self.clap_config.get('chunk_size', 1024)
        self.rms_threshold = self.clap_config.get('rms_threshold', 0.02)
        self.peak_threshold = self.clap_config.get('peak_threshold', 0.15)
        self.cooldown_ms = self.clap_config.get('cooldown_ms', 500)
        self.min_clap_interval_ms = self.clap_config.get('min_clap_interval_ms', 200)
        self.max_clap_interval_ms = self.clap_config.get('max_clap_interval_ms', 1000)
        
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._callback: Optional[Callable[[int], None]] = None
        self._clap_callback: Optional[Callable[[dict], None]] = None
        
        # Clap detection state
        self._recent_peaks: Deque[float] = deque(maxlen=10)
        self._last_clap_time = 0
        self._clap_count = 0
        self._background_rms = 0.0
        self._background_samples = 0
        self._calibrating = True
        self._calibration_duration = 3.0  # seconds
        self._calibration_start = 0
        
        # Audio buffer for analysis
        self._audio_buffer: Deque[np.ndarray] = deque(maxlen=int(self.sample_rate / self.chunk_size * 2))
        
    def initialize(self) -> bool:
        """Initialize audio input."""
        if not self.enabled:
            logger.info("Clap detection disabled")
            return True
            
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("PortAudio not available, falling back to mock clap detector")
            return False
            
        try:
            # List available devices
            devices = sd.query_devices()
            logger.info(f"Available audio devices: {len(devices)}")
            
            # Find default input device
            default_input = sd.default.device[0]
            logger.info(f"Using default input device: {default_input}")
            
            self._calibration_start = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            return False
    
    def start(self, clap_callback: Optional[Callable[[dict], None]] = None) -> bool:
        """Start audio capture and clap detection."""
        if not self.enabled:
            return True
            
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("PortAudio not available, cannot start real clap detection")
            return False
            
        if self._running:
            return True
            
        self._clap_callback = clap_callback
        self._running = True
        
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='float32',
                callback=self._audio_callback
            )
            self._stream.start()
            logger.info("Clap detection started")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self._running = False
            return False
    
    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
        self._stream = None
        logger.info("Clap detection stopped")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Audio stream callback."""
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Flatten to mono
        audio_data = indata.flatten()
        self._audio_buffer.append(audio_data.copy())
        
        # Analyze for claps
        self._analyze_audio(audio_data)
    
    def _analyze_audio(self, audio_chunk: np.ndarray):
        """Analyze audio chunk for clap detection."""
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Calculate peak
        peak = np.max(np.abs(audio_chunk))
        
        # Calibration phase - learn background noise level
        if self._calibrating:
            elapsed = time.time() - self._calibration_start
            if elapsed < self._calibration_duration:
                self._background_rms = (self._background_rms * self._background_samples + rms) / (self._background_samples + 1)
                self._background_samples += 1
                return
            else:
                self._calibrating = False
                logger.info(f"Calibration complete. Background RMS: {self._background_rms:.6f}")
        
        # Adaptive threshold based on background
        adaptive_rms_threshold = max(self.rms_threshold, self._background_rms * 5)
        adaptive_peak_threshold = max(self.peak_threshold, self._background_rms * 20)
        
        # Check for clap: sudden peak with sufficient RMS
        current_time = time.time() * 1000  # ms
        
        if peak > adaptive_peak_threshold and rms > adaptive_rms_threshold:
            # Check cooldown
            if current_time - self._last_clap_time > self.cooldown_ms:
                self._recent_peaks.append(current_time)
                self._last_clap_time = current_time
                
                # Check for double/triple clap pattern
                clap_pattern = self._analyze_clap_pattern()
                
                clap_event = {
                    'timestamp': current_time / 1000.0,
                    'rms': float(rms),
                    'peak': float(peak),
                    'pattern': clap_pattern,
                    'count': self._clap_count
                }
                
                logger.info(f"Clap detected! Pattern: {clap_pattern}, RMS: {rms:.4f}, Peak: {peak:.4f}")
                
                if self._clap_callback:
                    try:
                        self._clap_callback(clap_event)
                    except Exception as e:
                        logger.error(f"Clap callback error: {e}")
    
    def _analyze_clap_pattern(self) -> str:
        """Analyze recent claps for patterns (single, double, triple)."""
        current_time = time.time() * 1000
        
        # Filter peaks within the max interval
        recent = [t for t in self._recent_peaks if current_time - t <= self.max_clap_interval_ms]
        
        if len(recent) == 1:
            self._clap_count = 1
            return "single"
        elif len(recent) == 2:
            interval = recent[1] - recent[0]
            if self.min_clap_interval_ms <= interval <= self.max_clap_interval_ms:
                self._clap_count = 2
                return "double"
        elif len(recent) >= 3:
            intervals = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            if all(self.min_clap_interval_ms <= iv <= self.max_clap_interval_ms for iv in intervals):
                self._clap_count = len(recent)
                return f"multi_{len(recent)}"
        
        self._clap_count = 1
        return "single"
    
    def set_thresholds(self, rms_threshold: float = None, peak_threshold: float = None):
        """Update detection thresholds."""
        if rms_threshold is not None:
            self.rms_threshold = rms_threshold
        if peak_threshold is not None:
            self.peak_threshold = peak_threshold
        logger.info(f"Thresholds updated: RMS={self.rms_threshold}, Peak={self.peak_threshold}")
    
    def recalibrate(self):
        """Recalibrate background noise level."""
        self._calibrating = True
        self._background_rms = 0.0
        self._background_samples = 0
        self._calibration_start = time.time()
        logger.info("Recalibration started")
    
    def get_status(self) -> dict:
        """Get current detector status."""
        return {
            'running': self._running,
            'enabled': self.enabled,
            'calibrating': self._calibrating,
            'background_rms': self._background_rms,
            'rms_threshold': self.rms_threshold,
            'peak_threshold': self.peak_threshold,
            'last_clap': self._last_clap_time / 1000.0 if self._last_clap_time > 0 else None,
            'clap_count': self._clap_count
        }


class MockClapDetector:
    """Mock clap detector for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self._running = False
        self._clap_callback = None
        self._thread = None
        
    def initialize(self) -> bool:
        return True
    
    def start(self, clap_callback: Optional[Callable[[dict], None]] = None) -> bool:
        self._clap_callback = clap_callback
        self._running = True
        self._thread = threading.Thread(target=self._mock_loop, daemon=True)
        self._thread.start()
        return True
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _mock_loop(self):
        import random
        while self._running:
            time.sleep(random.uniform(5, 15))
            if self._clap_callback and random.random() < 0.3:
                self._clap_callback({
                    'timestamp': time.time(),
                    'rms': 0.1,
                    'peak': 0.5,
                    'pattern': 'single',
                    'count': 1
                })
    
    def get_status(self) -> dict:
        return {'running': self._running, 'enabled': self.enabled}

# ===== gesture_trainer.py =====
"""
Custom gesture and action training system.
Records pose sequences and trains a classifier for custom actions.
"""

import logging
import time
import pickle
import threading
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import deque
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class GestureTrainer:
    """Handles recording, training, and recognition of custom gestures/actions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.train_config = config.get('training', {}).get('gesture_training', {})
        self.action_config = config.get('training', {}).get('action_training', {})
        
        self.enabled = self.train_config.get('enabled', True)
        self.sample_duration = self.train_config.get('sample_duration_sec', 3)
        self.min_samples = self.train_config.get('min_samples_per_gesture', 10)
        self.feature_type = self.train_config.get('feature_type', 'pose_landmarks')
        self.model_path = Path(self.train_config.get('model_path', 'models/gesture_classifier.pkl'))
        self.scaler_path = Path(self.train_config.get('scaler_path', 'models/scaler.pkl'))
        self.sequence_length = self.action_config.get('sequence_length', 30)
        self.overlap = self.action_config.get('overlap', 0.5)
        
        self.model: Optional[Pipeline] = None
        self.scaler: Optional[StandardScaler] = None
        self.classes: List[str] = []
        
        # Training state
        self._recording = False
        self._current_gesture: Optional[str] = None
        self._samples: Dict[str, List[np.ndarray]] = {}
        self._sequence_buffer: deque = deque(maxlen=self.sequence_length)
        self._recording_start = 0
        self._lock = threading.Lock()
        
        # Ensure model directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
    def initialize(self) -> bool:
        """Initialize the trainer, load existing model if available."""
        if not self.enabled:
            return True
            
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.classes = list(self.model.classes_)
                logger.info(f"Loaded existing model with classes: {self.classes}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        
        logger.info("No existing model found, ready for training")
        return True
    
    def start_recording(self, gesture_name: str) -> bool:
        """Start recording samples for a gesture."""
        if not self.enabled:
            return False
            
        with self._lock:
            if self._recording:
                logger.warning("Already recording")
                return False
                
            self._current_gesture = gesture_name
            self._recording = True
            self._recording_start = time.time()
            self._sequence_buffer.clear()
            
            if gesture_name not in self._samples:
                self._samples[gesture_name] = []
                
            logger.info(f"Started recording gesture: {gesture_name}")
            return True
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and return the gesture name."""
        with self._lock:
            if not self._recording:
                return None
                
            gesture = self._current_gesture
            self._recording = False
            self._current_gesture = None
            
            logger.info(f"Stopped recording gesture: {gesture} ({len(self._samples.get(gesture, []))} samples)")
            return gesture
    
    def add_frame(self, landmarks: Any, frame_type: str = 'pose') -> bool:
        """Add a frame to the current recording."""
        if not self._recording or not self._current_gesture:
            return False
            
        features = self._extract_features(landmarks, frame_type)
        if features is not None:
            self._sequence_buffer.append(features)
            
            # If we have enough frames for a sample, save it
            if len(self._sequence_buffer) >= self.sequence_length:
                sample = np.array(list(self._sequence_buffer))
                self._samples[self._current_gesture].append(sample)
                
                # Slide window by overlap
                step = max(1, int(self.sequence_length * (1 - self.overlap)))
                for _ in range(step):
                    if self._sequence_buffer:
                        self._sequence_buffer.popleft()
                        
            return True
        return False
    
    def _extract_features(self, landmarks: Any, frame_type: str) -> Optional[np.ndarray]:
        """Extract feature vector from landmarks."""
        try:
            if frame_type == 'pose' and hasattr(landmarks, 'landmark'):
                # MediaPipe Pose: 33 landmarks * 4 (x, y, z, visibility) = 132 features
                features = []
                for lm in landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
                return np.array(features, dtype=np.float32)
            elif frame_type == 'hand' and hasattr(landmarks, 'landmark'):
                # MediaPipe Hands: 21 landmarks * 3 (x, y, z) = 63 features per hand
                features = []
                for lm in landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                return np.array(features, dtype=np.float32)
            elif isinstance(landmarks, np.ndarray):
                return landmarks.flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
        return None
    
    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status."""
        with self._lock:
            if not self._recording:
                return {'recording': False}
            
            elapsed = time.time() - self._recording_start
            samples_count = len(self._samples.get(self._current_gesture, []))
            
            return {
                'recording': True,
                'gesture': self._current_gesture,
                'elapsed': elapsed,
                'samples_collected': samples_count,
                'target_samples': self.min_samples,
                'progress': min(elapsed / self.sample_duration, 1.0)
            }
    
    def get_collected_gestures(self) -> Dict[str, int]:
        """Get count of samples per gesture."""
        with self._lock:
            return {name: len(samples) for name, samples in self._samples.items()}
    
    def clear_gesture(self, gesture_name: str) -> bool:
        """Clear all samples for a gesture."""
        with self._lock:
            if gesture_name in self._samples:
                del self._samples[gesture_name]
                logger.info(f"Cleared samples for gesture: {gesture_name}")
                return True
        return False
    
    def clear_all(self):
        """Clear all collected samples."""
        with self._lock:
            self._samples.clear()
            logger.info("Cleared all gesture samples")
    
    def train(self) -> Dict[str, Any]:
        """Train the classifier on collected samples."""
        with self._lock:
            if not self._samples:
                return {'success': False, 'error': 'No samples collected'}
            
            # Check minimum samples per class
            for name, samples in self._samples.items():
                if len(samples) < self.min_samples:
                    return {
                        'success': False, 
                        'error': f'Gesture "{name}" has only {len(samples)} samples, need at least {self.min_samples}'
                    }
            
            # Prepare training data
            X = []
            y = []
            
            for gesture_name, samples in self._samples.items():
                for sample in samples:
                    # Flatten sequence: (seq_len, features) -> (seq_len * features)
                    # Or use statistical features: mean, std, max, min across time
                    features = self._sequence_to_features(sample)
                    X.append(features)
                    y.append(gesture_name)
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Training on {len(X)} samples, {len(np.unique(y))} classes, feature dim: {X.shape[1]}")
            
            # Create pipeline
            self.scaler = StandardScaler()
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model = Pipeline([
                ('scaler', self.scaler),
                ('classifier', classifier)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(np.unique(y))), scoring='accuracy')
            
            # Train on all data
            self.model.fit(X, y)
            self.classes = list(self.model.classes_)
            
            # Save model
            try:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"Model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
            
            return {
                'success': True,
                'classes': self.classes,
                'samples_per_class': {name: len(samples) for name, samples in self._samples.items()},
                'cv_accuracy': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'feature_dim': X.shape[1]
            }
    
    def _sequence_to_features(self, sequence: np.ndarray) -> np.ndarray:
        """Convert temporal sequence to feature vector using statistics."""
        # sequence shape: (seq_len, features)
        features = []
        
        # Statistical features across time
        features.append(np.mean(sequence, axis=0))
        features.append(np.std(sequence, axis=0))
        features.append(np.max(sequence, axis=0))
        features.append(np.min(sequence, axis=0))
        features.append(np.median(sequence, axis=0))
        
        # Temporal derivatives (velocity)
        if len(sequence) > 1:
            velocity = np.diff(sequence, axis=0)
            features.append(np.mean(velocity, axis=0))
            features.append(np.std(velocity, axis=0))
        
        return np.concatenate(features)
    
    def predict(self, landmarks: Any, frame_type: str = 'pose') -> Optional[Dict[str, Any]]:
        """Predict gesture from current frame (uses sequence buffer)."""
        if self.model is None or self.scaler is None:
            return None
            
        features = self._extract_features(landmarks, frame_type)
        if features is None:
            return None
            
        # Add to sequence buffer
        self._sequence_buffer.append(features)
        
        # Need full sequence for prediction
        if len(self._sequence_buffer) < self.sequence_length:
            return None
            
        # Convert to features
        sequence = np.array(list(self._sequence_buffer))
        X = self._sequence_to_features(sequence).reshape(1, -1)
        
        try:
            # Predict
            proba = self.model.predict_proba(X)[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]
            predicted_class = self.classes[pred_idx]
            
            # Get all probabilities
            all_probas = {cls: float(proba[i]) for i, cls in enumerate(self.classes)}
            
            return {
                'gesture': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': all_probas,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def predict_single_frame(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Predict from pre-extracted features (for non-sequence models)."""
        if self.model is None:
            return None
            
        X = features.reshape(1, -1)
        try:
            proba = self.model.predict_proba(X)[0]
            pred_idx = np.argmax(proba)
            return {
                'gesture': self.classes[pred_idx],
                'confidence': float(proba[pred_idx]),
                'all_probabilities': {cls: float(proba[i]) for i, cls in enumerate(self.classes)}
            }
        except Exception as e:
            logger.error(f"Single frame prediction error: {e}")
            return None
    
    def export_training_data(self, path: str) -> bool:
        """Export collected training data for backup."""
        try:
            with self._lock:
                data = {
                    'samples': {k: [s.tolist() for s in v] for k, v in self._samples.items()},
                    'config': {
                        'sequence_length': self.sequence_length,
                        'feature_type': self.feature_type
                    }
                }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Training data exported to {path}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_training_data(self, path: str) -> bool:
        """Import training data from backup."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            with self._lock:
                self._samples = {k: [np.array(s) for s in v] for k, v in data['samples'].items()}
            
            logger.info(f"Training data imported from {path}")
            return True
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        pass


class ActionTrainer(GestureTrainer):
    """Extended trainer for complex multi-step actions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Action trainer uses the same base but with different feature extraction
        self.action_sequences: Dict[str, List[List[np.ndarray]]] = {}
    
    def record_action_step(self, action_name: str, step_name: str, landmarks: Any):
        """Record a step in a multi-step action."""
        features = self._extract_features(landmarks, 'pose')
        if features is not None:
            if action_name not in self.action_sequences:
                self.action_sequences[action_name] = []
            if not self.action_sequences[action_name] or len(self.action_sequences[action_name][-1]) >= self.sequence_length:
                self.action_sequences[action_name].append([])
            self.action_sequences[action_name][-1].append(features)
    
    def train_actions(self) -> Dict[str, Any]:
        """Train on multi-step actions."""
        # Flatten action sequences into samples
        for action_name, steps in self.action_sequences.items():
            for step_sequence in steps:
                if len(step_sequence) >= self.sequence_length:
                    sample = np.array(step_sequence)
                    if action_name not in self._samples:
                        self._samples[action_name] = []
                    self._samples[action_name].append(sample)
        
        return self.train()

# ===== home_assistant_client.py =====
"""
Home Assistant integration via MQTT and REST API.
Supports auto-discovery, state publishing, and automation triggers.
"""

import logging
import json
import threading
import time
import uuid
from typing import Dict, Any, Optional, Callable
from pathlib import Path

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Home Assistant integration client supporting MQTT and REST."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ha_config = config.get('home_assistant', {})
        self.enabled = self.ha_config.get('enabled', True)
        self.connection_type = self.ha_config.get('connection_type', 'mqtt')
        
        self.mqtt_config = self.ha_config.get('mqtt', {})
        self.rest_config = self.ha_config.get('rest', {})
        self.entities_config = self.ha_config.get('entities', {})
        
        self.client_id = self.mqtt_config.get('client_id', f'kinect_camera_{uuid.uuid4().hex[:8]}')
        self.base_topic = self.mqtt_config.get('base_topic', 'kinect/smart_camera')
        self.discovery_prefix = self.mqtt_config.get('discovery_prefix', 'homeassistant')
        
        self.mqtt_client: Optional[mqtt.Client] = None
        self.connected = False
        self._lock = threading.Lock()
        self._message_callback: Optional[Callable[[str, dict], None]] = None
        
        # Entity state cache
        self._entity_states: Dict[str, Any] = {}
        
    def initialize(self) -> bool:
        """Initialize connection to Home Assistant."""
        if not self.enabled:
            logger.info("Home Assistant integration disabled")
            return True
            
        if self.connection_type == 'mqtt':
            return self._init_mqtt()
        elif self.connection_type == 'rest':
            return self._init_rest()
        else:
            logger.error(f"Unknown connection type: {self.connection_type}")
            return False
    
    def _init_mqtt(self) -> bool:
        """Initialize MQTT connection."""
        if not MQTT_AVAILABLE:
            logger.error("paho-mqtt not installed. Install with: pip install paho-mqtt")
            return False
            
        try:
            self.mqtt_client = mqtt.Client(
                client_id=self.client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
            
            username = self.mqtt_config.get('username')
            password = self.mqtt_config.get('password')
            if username and password:
                self.mqtt_client.username_pw_set(username, password)
            
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.on_message = self._on_mqtt_message
            
            # Set will message
            if self.mqtt_config.get('will_message', True):
                will_topic = f"{self.base_topic}/status"
                self.mqtt_client.will_set(will_topic, payload='offline', qos=1, retain=True)
            
            host = self.mqtt_config.get('host', 'localhost')
            port = self.mqtt_config.get('port', 1883)
            
            self.mqtt_client.connect_async(host, port, keepalive=60)
            self.mqtt_client.loop_start()
            
            logger.info(f"MQTT client initialized, connecting to {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"MQTT initialization failed: {e}")
            return False
    
    def _init_rest(self) -> bool:
        """Initialize REST API connection."""
        if not REQUESTS_AVAILABLE:
            logger.error("requests not installed. Install with: pip install requests")
            return False
            
        url = self.rest_config.get('url', 'http://localhost:8123')
        token = self.rest_config.get('token')
        
        if not token:
            logger.error("Home Assistant REST API requires a long-lived access token")
            return False
            
        self.rest_headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        self.rest_url = url.rstrip('/')
        self.rest_verify = self.rest_config.get('verify_ssl', True)
        
        # Test connection
        try:
            resp = requests.get(f"{self.rest_url}/api/", headers=self.rest_headers, timeout=5, verify=self.rest_verify)
            if resp.status_code == 200:
                logger.info("REST API connection verified")
                return True
            else:
                logger.error(f"REST API test failed: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"REST API connection failed: {e}")
            return False
    
    def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT connect callback."""
        if reason_code == 0:
            self.connected = True
            logger.info("MQTT connected")
            
            # Publish birth message
            if self.mqtt_config.get('birth_message', True):
                self.publish(f"{self.base_topic}/status", 'online', retain=True)
            
            # Subscribe to command topics
            client.subscribe(f"{self.base_topic}/commands/+")
            client.subscribe(f"{self.base_topic}/config/+")
            
            # Publish discovery configs
            self._publish_discovery()
        else:
            logger.error(f"MQTT connection failed: {reason_code}")
            self.connected = False
    
    def _on_mqtt_disconnect(self, client, userdata, flags, reason_code, properties):
        """MQTT disconnect callback."""
        self.connected = False
        logger.warning(f"MQTT disconnected: {reason_code}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload) if payload else {}
            
            if self._message_callback:
                self._message_callback(topic, data)
        except Exception as e:
            logger.error(f"MQTT message error: {e}")
    
    def set_message_callback(self, callback: Callable[[str, dict], None]):
        """Set callback for incoming messages."""
        self._message_callback = callback
    
    def publish(self, topic: str, payload: Any, retain: bool = False, qos: int = 1) -> bool:
        """Publish message to MQTT or REST."""
        if not self.enabled:
            return False
            
        if self.connection_type == 'mqtt':
            return self._publish_mqtt(topic, payload, retain, qos)
        elif self.connection_type == 'rest':
            return self._publish_rest(topic, payload)
        return False
    
    def _publish_mqtt(self, topic: str, payload: Any, retain: bool, qos: int) -> bool:
        """Publish to MQTT."""
        if not self.mqtt_client or not self.connected:
            return False
            
        try:
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload)
            elif not isinstance(payload, str):
                payload = str(payload)
                
            result = self.mqtt_client.publish(topic, payload, qos=qos, retain=retain)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")
            return False
    
    def _publish_rest(self, topic: str, payload: Any) -> bool:
        """Publish to Home Assistant via REST (state update)."""
        # Map topic to entity_id
        entity_id = topic.replace(f"{self.base_topic}/", "").replace("/", ".")
        
        try:
            state = payload if isinstance(payload, str) else json.dumps(payload)
            url = f"{self.rest_url}/api/states/{entity_id}"
            data = {'state': state, 'attributes': {}}
            
            if isinstance(payload, dict):
                data['attributes'] = {k: v for k, v in payload.items() if k != 'state'}
                data['state'] = payload.get('state', 'unknown')
            
            resp = requests.post(url, headers=self.rest_headers, json=data, timeout=5, verify=self.rest_verify)
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.error(f"REST publish error: {e}")
            return False
    
    def _publish_discovery(self):
        """Publish Home Assistant MQTT discovery configs."""
        if self.connection_type != 'mqtt':
            return
            
        device_info = {
            'identifiers': [self.client_id],
            'name': 'Kinect Smart Camera',
            'model': 'Xbox One Kinect v2',
            'manufacturer': 'Microsoft',
            'sw_version': '1.0.0'
        }
        
        # Human Presence Sensor
        if self.entities_config.get('human_presence', {}).get('enabled', True):
            self._publish_sensor_discovery(
                'human_presence',
                'Human Presence',
                'presence',
                device_info,
                'binary_sensor'
            )
        
        # Human Count Sensor
        if self.entities_config.get('human_count', {}).get('enabled', True):
            self._publish_sensor_discovery(
                'human_count',
                'Human Count',
                'counter',
                device_info,
                'sensor'
            )
        
        # Clap Detection
        if self.entities_config.get('clap_detected', {}).get('enabled', True):
            self._publish_sensor_discovery(
                'clap_detected',
                'Clap Detected',
                'timestamp',
                device_info,
                'sensor'
            )
        
        # Gesture Detection
        if self.entities_config.get('gesture_detected', {}).get('enabled', True):
            self._publish_sensor_discovery(
                'gesture_detected',
                'Gesture Detected',
                'enum',
                device_info,
                'sensor'
            )
        
        # Custom Action Detection
        if self.entities_config.get('custom_action_detected', {}).get('enabled', True):
            self._publish_sensor_discovery(
                'custom_action_detected',
                'Custom Action Detected',
                'enum',
                device_info,
                'sensor'
            )
        
        # Camera entity for preview
        self._publish_camera_discovery(device_info)
    
    def _publish_sensor_discovery(self, entity_suffix: str, name: str, device_class: str, device_info: dict, component: str):
        """Publish discovery config for a sensor."""
        topic = f"{self.discovery_prefix}/{component}/{self.client_id}/{entity_suffix}/config"
        
        config = {
            'name': name,
            'unique_id': f"{self.client_id}_{entity_suffix}",
            'state_topic': f"{self.base_topic}/{entity_suffix}",
            'device_class': device_class,
            'device': device_info,
            'availability_topic': f"{self.base_topic}/status",
            'payload_available': 'online',
            'payload_not_available': 'offline',
            'json_attributes_topic': f"{self.base_topic}/{entity_suffix}_attrs"
        }
        
        if component == 'binary_sensor':
            config['payload_on'] = 'on'
            config['payload_off'] = 'off'
        
        self.publish(topic, config, retain=True)
    
    def _publish_camera_discovery(self, device_info: dict):
        """Publish camera discovery for preview stream."""
        topic = f"{self.discovery_prefix}/camera/{self.client_id}/preview/config"
        
        config = {
            'name': 'Kinect Camera Preview',
            'unique_id': f"{self.client_id}_preview",
            'topic': f"{self.base_topic}/preview",
            'device': device_info,
            'availability_topic': f"{self.base_topic}/status"
        }
        
        self.publish(topic, config, retain=True)
    
    # High-level entity update methods
    
    def update_human_presence(self, detected: bool, count: int = 0, details: dict = None):
        """Update human presence sensor."""
        state = 'on' if detected else 'off'
        self.publish(f"{self.base_topic}/human_presence", state)
        
        attrs = {'count': count}
        if details:
            attrs.update(details)
        self.publish(f"{self.base_topic}/human_presence_attrs", attrs)
        
        if self.entities_config.get('human_count', {}).get('enabled', True):
            self.publish(f"{self.base_topic}/human_count", count)
    
    def update_clap_detected(self, pattern: str, count: int = 1, details: dict = None):
        """Update clap detection sensor."""
        timestamp = time.time()
        self.publish(f"{self.base_topic}/clap_detected", timestamp)
        
        attrs = {'pattern': pattern, 'count': count}
        if details:
            attrs.update(details)
        self.publish(f"{self.base_topic}/clap_detected_attrs", attrs)
    
    def update_gesture_detected(self, gesture: str, confidence: float, details: dict = None):
        """Update gesture detection sensor."""
        self.publish(f"{self.base_topic}/gesture_detected", gesture)
        
        attrs = {'confidence': confidence}
        if details:
            attrs.update(details)
        self.publish(f"{self.base_topic}/gesture_detected_attrs", attrs)
    
    def update_custom_action_detected(self, action: str, confidence: float, details: dict = None):
        """Update custom action detection sensor."""
        self.publish(f"{self.base_topic}/custom_action_detected", action)
        
        attrs = {'confidence': confidence}
        if details:
            attrs.update(details)
        self.publish(f"{self.base_topic}/custom_action_detected_attrs", attrs)
    
    def publish_preview_frame(self, frame_data: bytes):
        """Publish camera preview frame (MQTT only)."""
        if self.connection_type == 'mqtt':
            self.publish(f"{self.base_topic}/preview", frame_data, retain=False, qos=0)
    
    def call_service(self, domain: str, service: str, service_data: dict) -> bool:
        """Call a Home Assistant service (REST only)."""
        if self.connection_type != 'rest':
            logger.warning("Service calls only supported via REST")
            return False
            
        try:
            url = f"{self.rest_url}/api/services/{domain}/{service}"
            resp = requests.post(url, headers=self.rest_headers, json=service_data, timeout=10, verify=self.rest_verify)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Service call failed: {e}")
            return False
    
    def trigger_automation(self, automation_id: str) -> bool:
        """Trigger a Home Assistant automation."""
        return self.call_service('automation', 'trigger', {'entity_id': automation_id})
    
    def fire_event(self, event_type: str, event_data: dict) -> bool:
        """Fire a custom event in Home Assistant."""
        return self.call_service('event', 'fire', {'event_type': event_type, 'event_data': event_data})
    
    def get_state(self, entity_id: str) -> Optional[dict]:
        """Get entity state (REST only)."""
        if self.connection_type != 'rest':
            return None
            
        try:
            url = f"{self.rest_url}/api/states/{entity_id}"
            resp = requests.get(url, headers=self.rest_headers, timeout=5, verify=self.rest_verify)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Get state failed: {e}")
        return None
    
    def shutdown(self):
        """Clean shutdown."""
        if self.connection_type == 'mqtt' and self.mqtt_client:
            if self.mqtt_config.get('birth_message', True):
                self.publish(f"{self.base_topic}/status", 'offline', retain=True)
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        logger.info("Home Assistant client shutdown")


class MockHomeAssistantClient:
    """Mock Home Assistant client for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.connected = True
        
    def initialize(self) -> bool:
        logger.info("Mock Home Assistant client initialized")
        return True
    
    def publish(self, topic: str, payload: Any, retain: bool = False, qos: int = 1) -> bool:
        logger.debug(f"Mock HA publish: {topic} = {payload}")
        return True
    
    def update_human_presence(self, detected: bool, count: int = 0, details: dict = None):
        logger.info(f"Mock HA: Human presence = {detected}, count = {count}")
    
    def update_clap_detected(self, pattern: str, count: int = 1, details: dict = None):
        logger.info(f"Mock HA: Clap detected = {pattern} (x{count})")
    
    def update_gesture_detected(self, gesture: str, confidence: float, details: dict = None):
        logger.info(f"Mock HA: Gesture = {gesture} ({confidence:.2f})")
    
    def update_custom_action_detected(self, action: str, confidence: float, details: dict = None):
        logger.info(f"Mock HA: Custom action = {action} ({confidence:.2f})")
    
    def trigger_automation(self, automation_id: str) -> bool:
        logger.info(f"Mock HA: Triggered automation {automation_id}")
        return True
    
    def fire_event(self, event_type: str, event_data: dict) -> bool:
        logger.info(f"Mock HA: Fired event {event_type}")
        return True
    
    def shutdown(self):
        pass

# ===== main_app.py =====
"""
Main application coordinator - ties together all components.
"""

import logging
import threading
import time
import signal
import sys
from typing import Dict, Any, Optional, Callable
from pathlib import Path


logger = logging.getLogger(__name__)


class KinectSmartCameraApp:
    """Main application class coordinating all components."""
    
    def __init__(self, config: Dict[str, Any], mock_mode: bool = False):
        self.config = config
        self.mock_mode = mock_mode
        self.running = False
        
        # Components
        self.kinect: Optional[KinectManager] = None
        self.human_detector: Optional[HumanDetector] = None
        self.clap_detector: Optional[ClapDetector] = None
        self.gesture_trainer: Optional[GestureTrainer] = None
        self.ha_client: Optional[HomeAssistantClient] = None
        
        # State
        self._human_present = False
        self._human_count = 0
        self._last_detections = []
        self._processing_thread: Optional[threading.Thread] = None
        self._preview_callback: Optional[Callable[[Any], None]] = None
        self._status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Training mode
        self._training_mode = False
        self._training_gesture = None
        
        # FPS tracking
        self._fps = {'color': 0, 'depth': 0, 'ir': 0}
        self._last_fps_update = 0
        
    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Kinect Smart Camera...")
        
        # Initialize Kinect
        if self.mock_mode:
            self.kinect = MockKinectManager(self.config)
        else:
            self.kinect = KinectManager(self.config)
        
        if not self.kinect.initialize():
            logger.error("Failed to initialize Kinect")
            return False
        
        # Register frame callbacks
        self.kinect.register_callback('color', self._on_color_frame)
        self.kinect.register_callback('depth', self._on_depth_frame)
        self.kinect.register_callback('ir', self._on_ir_frame)
        
        # Initialize human detector
        self.human_detector = HumanDetector(self.config)
        if not self.human_detector.initialize():
            logger.warning("Human detector initialization had issues")
        
        # Initialize clap detector
        if self.mock_mode:
            self.clap_detector = MockClapDetector(self.config)
        else:
            self.clap_detector = ClapDetector(self.config)
        
        if not self.clap_detector.initialize():
            logger.warning("Clap detector initialization failed, falling back to mock mode")
            self.clap_detector = MockClapDetector(self.config)
            self.clap_detector.initialize()
        
        # Initialize gesture trainer
        self.gesture_trainer = GestureTrainer(self.config)
        if not self.gesture_trainer.initialize():
            logger.warning("Gesture trainer initialization had issues")
        
        # Initialize Home Assistant client
        if self.mock_mode:
            self.ha_client = MockHomeAssistantClient(self.config)
        else:
            self.ha_client = HomeAssistantClient(self.config)
        
        if not self.ha_client.initialize():
            logger.warning("Home Assistant client initialization had issues")
        
        # Set up clap callback
        self.clap_detector.start(clap_callback=self._on_clap_detected)
        
        logger.info("All components initialized")
        return True
    
    def start(self) -> bool:
        """Start the application."""
        if self.running:
            return True
            
        if not self.kinect.start():
            logger.error("Failed to start Kinect")
            return False
        
        self.running = True
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        logger.info("Kinect Smart Camera started")
        return True
    
    def stop(self):
        """Stop the application."""
        self.running = False
        
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        
        if self.kinect:
            self.kinect.stop()
        
        if self.clap_detector:
            self.clap_detector.stop()
        
        if self.ha_client:
            self.ha_client.shutdown()
        
        if self.gesture_trainer:
            self.gesture_trainer.cleanup()
        
        if self.human_detector:
            self.human_detector.cleanup()
        
        logger.info("Kinect Smart Camera stopped")
    
    def _processing_loop(self):
        """Main processing loop for detection."""
        while self.running:
            try:
                color_frame = self.kinect.get_latest_frame('color')
                depth_frame = self.kinect.get_latest_frame('depth')
                
                if color_frame is not None:
                    # Human detection
                    detections = self.human_detector.detect(color_frame, depth_frame)
                    self._process_detections(detections, color_frame, depth_frame)
                    
                    # Gesture training/recognition
                    if self._training_mode and self._training_gesture:
                        # Get pose landmarks from detector
                        # This would need integration with the detector's internal state
                        pass
                    
                    # Gesture recognition (if model trained)
                    if self.gesture_trainer and self.gesture_trainer.model:
                        # Would need pose landmarks from detector
                        pass
                
                # Update FPS
                self._update_fps()
                
                # Call status callback
                if self._status_callback:
                    self._status_callback(self.get_status())
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
            
            time.sleep(1/30)  # 30 FPS max
    
    def _on_color_frame(self, frame):
        """Handle color frame."""
        if self._preview_callback:
            self._preview_callback(('color', frame))
    
    def _on_depth_frame(self, frame):
        """Handle depth frame."""
        if self._preview_callback:
            self._preview_callback(('depth', frame))
    
    def _on_ir_frame(self, frame):
        """Handle IR frame."""
        if self._preview_callback:
            self._preview_callback(('ir', frame))
    
    def _process_detections(self, detections, color_frame, depth_frame):
        """Process human detections and update HA."""
        human_present = len(detections) > 0
        human_count = len(detections)
        
        # Update HA if state changed
        if human_present != self._human_present or human_count != self._human_count:
            self._human_present = human_present
            self._human_count = human_count
            
            details = {
                'detections': [
                    {
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'distance': det.get('distance'),
                        'source': det.get('source')
                    }
                    for det in detections
                ]
            }
            
            self.ha_client.update_human_presence(human_present, human_count, details)
            
            # Fire event for automations
            if human_present and not self._human_present:
                self.ha_client.fire_event('kinect_human_entered', {'count': human_count})
            elif not human_present and self._human_present:
                self.ha_client.fire_event('kinect_human_left', {})
        
        self._last_detections = detections
    
    def _on_clap_detected(self, clap_event: dict):
        """Handle clap detection."""
        logger.info(f"Clap detected: {clap_event['pattern']}")
        
        self.ha_client.update_clap_detected(
            clap_event['pattern'],
            clap_event['count'],
            {'rms': clap_event['rms'], 'peak': clap_event['peak']}
        )
        
        # Fire event for automations
        self.ha_client.fire_event('kinect_clap_detected', clap_event)
        
        # Trigger automation if configured
        if clap_event['pattern'] == 'double':
            self.ha_client.trigger_automation('automation.kinect_double_clap')
        elif clap_event['pattern'] == 'triple':
            self.ha_client.trigger_automation('automation.kinect_triple_clap')
    
    def _update_fps(self):
        """Update FPS counters."""
        fps_data = self.kinect.get_fps()
        self._fps = {k.lower(): v for k, v in fps_data.items()}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current application status."""
        return {
            'running': self.running,
            'mock_mode': self.mock_mode,
            'human_present': self._human_present,
            'human_count': self._human_count,
            'fps': self._fps,
            'kinect_connected': self.kinect.is_running() if self.kinect else False,
            'clap_detector': self.clap_detector.get_status() if self.clap_detector else {},
            'ha_connected': self.ha_client.connected if self.ha_client else False,
            'training_mode': self._training_mode,
            'training_gesture': self._training_gesture,
            'trained_gestures': self.gesture_trainer.get_collected_gestures() if self.gesture_trainer else {},
            'model_loaded': self.gesture_trainer.model is not None if self.gesture_trainer else False
        }
    
    def set_preview_callback(self, callback: Callable[[Any], None]):
        """Set callback for preview frames."""
        self._preview_callback = callback
    
    def set_status_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for status updates."""
        self._status_callback = callback
    
    # Training methods
    
    def start_training(self, gesture_name: str) -> bool:
        """Start training a new gesture."""
        if self.gesture_trainer:
            self._training_mode = True
            self._training_gesture = gesture_name
            return self.gesture_trainer.start_recording(gesture_name)
        return False
    
    def stop_training(self) -> Optional[str]:
        """Stop training current gesture."""
        self._training_mode = False
        gesture = self._training_gesture
        self._training_gesture = None
        if self.gesture_trainer:
            return self.gesture_trainer.stop_recording()
        return gesture
    
    def train_model(self) -> Dict[str, Any]:
        """Train the gesture recognition model."""
        if self.gesture_trainer:
            return self.gesture_trainer.train()
        return {'success': False, 'error': 'Gesture trainer not available'}
    
    def clear_gesture(self, gesture_name: str) -> bool:
        """Clear training data for a gesture."""
        if self.gesture_trainer:
            return self.gesture_trainer.clear_gesture(gesture_name)
        return False
    
    def clear_all_training(self):
        """Clear all training data."""
        if self.gesture_trainer:
            self.gesture_trainer.clear_all()
    
    def recalibrate_clap(self):
        """Recalibrate clap detector."""
        if self.clap_detector:
            self.clap_detector.recalibrate()
    
    def set_clap_thresholds(self, rms: float = None, peak: float = None):
        """Set clap detection thresholds."""
        if self.clap_detector:
            self.clap_detector.set_thresholds(rms, peak)
    
    def export_training_data(self, path: str) -> bool:
        """Export training data."""
        if self.gesture_trainer:
            return self.gesture_trainer.export_training_data(path)
        return False
    
    def import_training_data(self, path: str) -> bool:
        """Import training data."""
        if self.gesture_trainer:
            return self.gesture_trainer.import_training_data(path)
        return False


def create_app(config_path: str = None, mock_mode: bool = False) -> KinectSmartCameraApp:
    """Factory function to create and configure the app."""
    import yaml
    
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Merge with defaults
    default_config_path = Path(__file__).parent.parent / 'config.yaml'
    if default_config_path.exists():
        with open(default_config_path, 'r') as f:
            default_config = yaml.safe_load(f) or {}
        # Deep merge
        def merge_dict(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    merge_dict(base[k], v)
                else:
                    base[k] = v
        merge_dict(default_config, config)
        config = default_config
    
    return KinectSmartCameraApp(config, mock_mode)


# ===== MODERN GUI (replaces menu_bar_app.py) =====
"""
Modern GUI for Kinect Smart Camera using rumps + OpenCV with modern styling.
Features:
- Native macOS menu bar integration
- Modern dark-themed preview window with overlays
- Settings dialog with modern styling
- Real-time status dashboard
- Gesture training wizard
"""

import threading
import time
import json
import subprocess
from typing import Optional, Dict, Any, Callable
import numpy as np
import cv2

try:
    import rumps
    RUMPS_AVAILABLE = True
except ImportError:
    RUMPS_AVAILABLE = False
    rumps = None

# Modern color scheme (dark theme)
COLORS = {
    'bg_dark': (25, 25, 30),
    'bg_card': (35, 35, 42),
    'accent': (0, 180, 255),
    'accent_hover': (0, 200, 255),
    'success': (0, 220, 130),
    'warning': (255, 180, 0),
    'danger': (255, 70, 85),
    'text_primary': (245, 245, 250),
    'text_secondary': (160, 160, 170),
    'border': (60, 60, 70),
}

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=12):
    """Draw rounded rectangle on image."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, (x2-x1)//2, (y2-y1)//2)
    # Draw main rectangles
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    # Draw corners
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

def put_text_modern(img, text, pos, font_scale=0.6, color=COLORS['text_primary'], thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw text with subtle shadow for readability."""
    x, y = pos
    # Shadow
    cv2.putText(img, text, (x+1, y+1), font, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
    # Text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

class ModernPreviewWindow:
    """Modern dark-themed preview window with live overlays."""
    
    def __init__(self, title="Kinect Smart Camera", width=960, height=540):
        self.title = title
        self.width = width
        self.height = height
        self.running = False
        self.thread = None
        self._frame_lock = threading.Lock()
        self._latest_data = {
            'color': None,
            'depth': None,
            'detections': [],
            'fps': {'color': 0, 'depth': 0, 'ir': 0},
            'human_present': False,
            'human_count': 0,
            'training_mode': False,
            'training_gesture': '',
            'training_progress': 0.0,
            'clap_status': {},
            'model_loaded': False,
            'trained_gestures': {},
        }
        
    def update(self, **kwargs):
        with self._frame_lock:
            self._latest_data.update(kwargs)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        cv2.destroyWindow(self.title)
    
    def _run_loop(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.title, self.width, self.height)
        # Set window to dark mode on macOS
        try:
            cv2.setWindowProperty(self.title, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except:
            pass
        
        last_fps_time = time.time()
        frame_count = 0
        
        while self.running:
            with self._frame_lock:
                data = self._latest_data.copy()
            
            frame = self._render_frame(data)
            
            cv2.imshow(self.title, frame)
            
            key = cv2.waitKey(16) & 0xFF  # ~60 FPS
            if key == 27:  # ESC
                break
            elif key == ord('f'):  # Toggle fullscreen
                cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, 
                    cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN) == 0 else cv2.WINDOW_NORMAL)
        
        self.running = False
    
    def _render_frame(self, data):
        """Render modern UI frame with all overlays."""
        color_frame = data.get('color')
        detections = data.get('detections', [])
        
        # Create base canvas
        if color_frame is not None:
            frame = cv2.resize(color_frame, (self.width, self.height))
        else:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = COLORS['bg_dark']
            put_text_modern(frame, "Waiting for Kinect...", (self.width//2 - 150, self.height//2), 
                          1.0, COLORS['text_secondary'], 2)
        
        # Draw semi-transparent overlay panel (top-left)
        panel_w, panel_h = 320, 220
        overlay = frame.copy()
        draw_rounded_rect(overlay, (10, 10), (10+panel_w, 10+panel_h), COLORS['bg_card'], -1, 12)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        
        # Draw border
        draw_rounded_rect(frame, (10, 10), (10+panel_w, 10+panel_h), COLORS['border'], 1, 12)
        
        # Status header
        put_text_modern(frame, "KINECT SMART CAMERA", (20, 36), 0.7, COLORS['accent'], 2)
        put_text_modern(frame, "v1.0.0  |  Intel macOS", (20, 58), 0.45, COLORS['text_secondary'], 1)
        
        # Divider
        cv2.line(frame, (20, 68), (20+panel_w-20, 68), COLORS['border'], 1)
        
        # Human detection status
        y = 85
        status_color = COLORS['success'] if data['human_present'] else COLORS['text_secondary']
        status_text = "HUMAN DETECTED" if data['human_present'] else "NO HUMAN"
        put_text_modern(frame, status_text, (20, y), 0.6, status_color, 2)
        put_text_modern(frame, f"Count: {data['human_count']}", (20, y+25), 0.5, COLORS['text_secondary'], 1)
        
        # Detection details
        for i, det in enumerate(detections[:3]):
            y_det = y + 50 + i * 22
            conf = det.get('confidence', 0)
            dist = det.get('distance')
            src = det.get('source', '?')
            put_text_modern(frame, f"Person {i+1}: {conf:.0%}  {f'{dist:.1f}m' if dist else 'N/A'}  [{src}]", 
                          (20, y_det), 0.42, COLORS['text_secondary'], 1)
        
        # FPS panel (top-right)
        fps_w, fps_h = 180, 90
        overlay = frame.copy()
        draw_rounded_rect(overlay, (self.width-10-fps_w, 10), (self.width-10, 10+fps_h), COLORS['bg_card'], -1, 10)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        draw_rounded_rect(frame, (self.width-10-fps_w, 10), (self.width-10, 10+fps_h), COLORS['border'], 1, 10)
        
        put_text_modern(frame, "PERFORMANCE", (self.width-10-fps_w+12, 32), 0.5, COLORS['accent'], 1)
        cv2.line(frame, (self.width-10-fps_w+12, 38), (self.width-22, 38), COLORS['border'], 1)
        
        fps = data.get('fps', {})
        put_text_modern(frame, f"Color:  {fps.get('color', 0):.1f} FPS", (self.width-10-fps_w+12, 58), 0.45, COLORS['text_primary'], 1)
        put_text_modern(frame, f"Depth:  {fps.get('depth', 0):.1f} FPS", (self.width-10-fps_w+12, 78), 0.45, COLORS['text_primary'], 1)
        put_text_modern(frame, f"IR:     {fps.get('ir', 0):.1f} FPS", (self.width-10-fps_w+12, 98), 0.45, COLORS['text_primary'], 1)
        
        # Bottom status bar
        bar_h = 60
        overlay = frame.copy()
        draw_rounded_rect(overlay, (10, self.height-10-bar_h), (self.width-10, self.height-10), COLORS['bg_card'], -1, 10)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        draw_rounded_rect(frame, (10, self.height-10-bar_h), (self.width-10, self.height-10), COLORS['border'], 1, 10)
        
        # Training status
        if data['training_mode']:
            put_text_modern(frame, f"TRAINING: {data['training_gesture']}", (20, self.height-10-bar_h+25), 
                          0.6, COLORS['warning'], 2)
            # Progress bar
            bar_x, bar_y = 20, self.height-10-bar_h+42
            bar_w = 300
            progress = data.get('training_progress', 0)
            draw_rounded_rect(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+10), COLORS['border'], -1, 5)
            draw_rounded_rect(frame, (bar_x, bar_y), (bar_x+int(bar_w*progress), bar_y+10), COLORS['warning'], -1, 5)
            put_text_modern(frame, f"{progress:.0%}", (bar_x+bar_w+10, bar_y+8), 0.4, COLORS['text_secondary'], 1)
        else:
            put_text_modern(frame, "READY  |  Press ESC to close  |  F for fullscreen", (20, self.height-10-bar_h+25), 
                          0.5, COLORS['text_secondary'], 1)
        
        # Model status
        if data['model_loaded']:
            put_text_modern(frame, "Model: LOADED", (self.width-200, self.height-10-bar_h+25), 0.5, COLORS['success'], 1)
            gestures = data.get('trained_gestures', {})
            if gestures:
                put_text_modern(frame, f"Gestures: {len(gestures)}", (self.width-200, self.height-10-bar_h+45), 0.4, COLORS['text_secondary'], 1)
        
        # Draw detection boxes on main frame
        if color_frame is not None and detections:
            scale_x = self.width / color_frame.shape[1]
            scale_y = self.height / color_frame.shape[0]
            
            for det in detections:
                bbox = det.get('bbox')
                if bbox:
                    x1, y1, x2, y2 = bbox
                    x1, x2 = int(x1*scale_x), int(x2*scale_x)
                    y1, y2 = int(y1*scale_y), int(y2*scale_y)
                    conf = det.get('confidence', 0)
                    color = COLORS['success'] if det.get('source') == 'mediapipe' else COLORS['warning']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Confidence badge
                    badge = f"{conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
                    put_text_modern(frame, badge, (x1+3, y1-4), 0.4, (0, 0, 0), 1)
        
        return frame

class ModernMenuBarApp:
    """Enhanced modern menu bar application."""
    
    def __init__(self, app_core):
        self.app_core = app_core
        self.preview_window = ModernPreviewWindow()
        self.preview_window.set_status_callback(self._on_status_update)
        self.preview_window.set_preview_callback(self._on_preview_frame)
        
        if not RUMPS_AVAILABLE:
            print("rumps not available, running headless")
            return
        
        self.menu_app = rumps.App("Kinect Camera", quit_button=None)
        self._setup_menu()
        
        # Start preview update timer
        self._start_update_timer()
    
    def _setup_menu(self):
        # Status items (read-only)
        self.status_item = rumps.MenuItem("● Status: Stopped")
        self.human_item = rumps.MenuItem("👤 Human: Not Detected")
        self.count_item = rumps.MenuItem("👥 Count: 0")
        self.fps_item = rumps.MenuItem("⚡ FPS: 0")
        
        # Control items
        self.start_item = rumps.MenuItem("▶ Start Camera", callback=self._toggle_camera)
        self.mock_item = rumps.MenuItem("🧪 Mock Mode", callback=self._toggle_mock)
        self.mock_item.state = self.app_core.mock_mode
        
        # Training submenu
        self.train_menu = rumps.MenuItem("🎓 Training")
        self.train_new_item = rumps.MenuItem("➕ Train New Gesture…", callback=self._train_new_gesture)
        self.train_stop_item = rumps.MenuItem("⏹ Stop Training", callback=self._stop_training)
        self.train_model_item = rumps.MenuItem("🧠 Train Model", callback=self._train_model)
        self.train_clear_item = rumps.MenuItem("🗑 Clear All Training", callback=self._clear_training)
        self.train_menu.add(self.train_new_item)
        self.train_menu.add(self.train_stop_item)
        self.train_menu.add(self.train_model_item)
        self.train_menu.add(self.train_clear_item)
        
        # Clap detection
        self.clap_menu = rumps.MenuItem("👏 Clap Detection")
        self.clap_recal_item = rumps.MenuItem("🔄 Recalibrate", callback=self._recalibrate_clap)
        self.clap_thresh_item = rumps.MenuItem("⚙ Set Thresholds…", callback=self._set_clap_thresholds)
        self.clap_menu.add(self.clap_recal_item)
        self.clap_menu.add(self.clap_thresh_item)
        
        # Home Assistant
        self.ha_menu = rumps.MenuItem("🏠 Home Assistant")
        self.ha_status_item = rumps.MenuItem("Status: Disconnected")
        self.ha_test_item = rumps.MenuItem("Test Connection", callback=self._test_ha)
        self.ha_menu.add(self.ha_status_item)
        self.ha_menu.add(self.ha_test_item)
        
        # Preview
        self.preview_item = rumps.MenuItem("🖥 Show Preview", callback=self._toggle_preview)
        
        # Settings
        self.config_item = rumps.MenuItem("⚙ Settings", callback=self._open_settings)
        self.reload_item = rumps.MenuItem("🔄 Reload Config", callback=self._reload_config)
        
        # Quit
        self.quit_item = rumps.MenuItem("❌ Quit", callback=self._quit)
        
        # Build menu
        self.menu_app.menu = [
            self.status_item, self.human_item, self.count_item, self.fps_item,
            None,
            self.start_item, self.mock_item,
            None,
            self.train_menu,
            self.clap_menu,
            self.ha_menu,
            None,
            self.preview_item,
            None,
            self.config_item, self.reload_item,
            None,
            self.quit_item
        ]
    
    def _start_update_timer(self):
        @rumps.timer(1)
        def update(_):
            if hasattr(self, 'app_core') and self.app_core:
                self._update_from_status(self.app_core.get_status())
    
    def _update_from_status(self, status):
        if not RUMPS_AVAILABLE:
            return
        running = status.get('running', False)
        self.status_item.title = f"{'●' if running else '○'} Status: {'Running' if running else 'Stopped'}"
        self.start_item.title = "⏹ Stop Camera" if running else "▶ Start Camera"
        
        hp = status.get('human_present', False)
        hc = status.get('human_count', 0)
        self.human_item.title = f"👤 Human: {'Detected' if hp else 'Not Detected'}"
        self.count_item.title = f"👥 Count: {hc}"
        
        fps = status.get('fps', {})
        self.fps_item.title = f"⚡ FPS: {fps.get('color', 0):.1f}"
        
        ha = status.get('ha_connected', False)
        self.ha_status_item.title = f"Status: {'Connected' if ha else 'Disconnected'}"
        
        tm = status.get('training_mode', False)
        tg = status.get('training_gesture', '')
        self.train_new_item.title = f"⏺ Training: {tg}…" if tm else "➕ Train New Gesture…"
        self.train_model_item.title = "🔄 Retrain Model" if status.get('model_loaded') else "🧠 Train Model"
    
    def _toggle_camera(self, _):
        if self.app_core.running:
            self.app_core.stop()
            rumps.notification("Kinect Smart Camera", "Stopped", "Camera stopped")
        else:
            if self.app_core.start():
                rumps.notification("Kinect Smart Camera", "Started", "Camera running")
            else:
                rumps.alert("Failed to start camera")
    
    def _toggle_mock(self, sender):
        sender.state = not sender.state
        self.app_core.mock_mode = sender.state
        rumps.alert("Restart app for mock mode change to take effect")
    
    def _toggle_preview(self, _):
        if self.preview_window.running:
            self.preview_window.stop()
            self.preview_item.title = "🖥 Show Preview"
        else:
            self.preview_window.start()
            self.preview_item.title = "🖥 Hide Preview"
    
    def _train_new_gesture(self, _):
        win = rumps.Window("Train New Gesture", "Enter gesture name:", 
                          ok="Start", cancel="Cancel", dimensions=(300, 100))
        resp = win.run()
        if resp.clicked and resp.text.strip():
            if self.app_core.start_training(resp.text.strip()):
                rumps.notification("Training Started", "", f"Recording: {resp.text.strip()}")
            else:
                rumps.alert("Failed to start training")
    
    def _stop_training(self, _):
        g = self.app_core.stop_training()
        if g:
            rumps.notification("Training Stopped", "", f"Finished: {g}")
        else:
            rumps.alert("Not currently training")
    
    def _train_model(self, _):
        rumps.notification("Training Model", "", "Training in background…")
        def do_train():
            result = self.app_core.train_model()
            rumps.notification("Training Complete" if result.get('success') else "Training Failed", 
                             "", str(result))
        threading.Thread(target=do_train, daemon=True).start()
    
    def _clear_training(self, _):
        if rumps.alert("Clear all training data?", "This cannot be undone.", ok="Yes", cancel="No"):
            self.app_core.clear_all_training()
            rumps.notification("Training Cleared", "", "All gesture data removed")
    
    def _recalibrate_clap(self, _):
        self.app_core.recalibrate_clap()
        rumps.notification("Clap Detection", "", "Recalibration started")
    
    def _set_clap_thresholds(self, _):
        win = rumps.Window("Clap Thresholds", "RMS Threshold:\nPeak Threshold:", 
                          ok="Set", cancel="Cancel", dimensions=(300, 150))
        win.default_text = "0.02\n0.15"
        resp = win.run()
        if resp.clicked:
            try:
                rms, peak = map(float, resp.text.strip().split())
                self.app_core.set_clap_thresholds(rms, peak)
                rumps.notification("Thresholds Updated", "", f"RMS: {rms}, Peak: {peak}")
            except:
                rumps.alert("Invalid format. Use: RMS Peak (e.g., 0.02 0.15)")
    
    def _test_ha(self, _):
        if self.app_core.ha_client:
            if self.app_core.ha_client.connection_type == 'mqtt':
                ok = self.app_core.ha_client.publish(f"{self.app_core.ha_client.base_topic}/test", "test")
                msg = "MQTT test sent" if ok else "MQTT failed"
            else:
                ok = self.app_core.ha_client.get_state("sensor.test") is not None
                msg = "REST OK" if ok else "REST failed"
            rumps.notification("HA Test", "", msg)
    
    def _open_settings(self, _):
        subprocess.run(['open', str(Path(__file__).parent / 'config.yaml')])
    
    def _reload_config(self, _):
        rumps.alert("Restart app to reload config")
    
    def _quit(self, _):
        if self.app_core.running:
            self.app_core.stop()
        if self.preview_window.running:
            self.preview_window.stop()
        rumps.quit_application()
    
    def _on_status_update(self, status):
        self.preview_window.update(**status)
    
    def _on_preview_frame(self, frame_data):
        ftype, frame = frame_data
        if ftype == 'color':
            self.preview_window.update(color=frame)
    
    def run(self):
        if RUMPS_AVAILABLE:
            self.menu_app.run()
        else:
            # Headless mode
            print("Running headless (no rumps)")
            self.app_core.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.app_core.stop()

# ===== END MODERN GUI =====


# ===== MAIN ENTRY POINT =====
def main():
    """Entry point for Kinect Smart Camera."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kinect Smart Camera - Modern GUI")
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode (no hardware)')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level')
    parser.add_argument('--headless', action='store_true', help='Run without GUI (menu bar only)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create core app
    app_core = create_app(config_path=args.config, mock_mode=args.mock)
    
    if args.headless or not RUMPS_AVAILABLE:
        # Headless mode
        print("Starting in headless mode...")
        if app_core.initialize():
            app_core.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                app_core.stop()
        else:
            print("Failed to initialize")
            sys.exit(1)
    else:
        # GUI mode
        menu_app = ModernMenuBarApp(app_core)
        menu_app.run()

if __name__ == '__main__':
    main()
