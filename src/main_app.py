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

from kinect_manager import KinectManager, MockKinectManager
from human_detector import HumanDetector
from clap_detector import ClapDetector, MockClapDetector
from gesture_trainer import GestureTrainer, ActionTrainer
from home_assistant_client import HomeAssistantClient, MockHomeAssistantClient

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