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