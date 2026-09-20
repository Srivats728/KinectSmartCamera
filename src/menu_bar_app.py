"""
macOS Menu Bar Application for Kinect Smart Camera.
"""

import rumps
import logging
import threading
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image

from main_app import create_app, KinectSmartCameraApp

logger = logging.getLogger(__name__)


class KinectMenuBarApp(rumps.App):
    """Menu bar application for Kinect Smart Camera."""
    
    def __init__(self, config_path: str = None, mock_mode: bool = False):
        super().__init__("Kinect Camera", quit_button=None)
        
        self.config_path = config_path
        self.mock_mode = mock_mode
        self.app: Optional[KinectSmartCameraApp] = None
        self.app_thread: Optional[threading.Thread] = None
        
        # Menu items
        self.status_item = rumps.MenuItem("Status: Stopped")
        self.human_item = rumps.MenuItem("Human: Not Detected")
        self.count_item = rumps.MenuItem("Count: 0")
        self.fps_item = rumps.MenuItem("FPS: 0")
        
        self.start_item = rumps.MenuItem("Start Camera", callback=self.toggle_camera)
        self.mock_item = rumps.MenuItem("Mock Mode", callback=self.toggle_mock)
        self.mock_item.state = mock_mode
        
        # Training submenu
        self.training_menu = rumps.MenuItem("Training")
        self.train_gesture_item = rumps.MenuItem("Train New Gesture...", callback=self.start_training_dialog)
        self.stop_training_item = rumps.MenuItem("Stop Training", callback=self.stop_training)
        self.stop_training_item.set_callback(self.stop_training)
        self.train_model_item = rumps.MenuItem("Train Model", callback=self.train_model)
        self.clear_training_item = rumps.MenuItem("Clear All Training", callback=self.clear_all_training)
        self.training_menu.add(self.train_gesture_item)
        self.training_menu.add(self.stop_training_item)
        self.training_menu.add(self.train_model_item)
        self.training_menu.add(self.clear_training_item)
        
        # Clap detection
        self.clap_menu = rumps.MenuItem("Clap Detection")
        self.recalibrate_clap_item = rumps.MenuItem("Recalibrate", callback=self.recalibrate_clap)
        self.clap_thresholds_item = rumps.MenuItem("Set Thresholds...", callback=self.set_clap_thresholds)
        self.clap_menu.add(self.recalibrate_clap_item)
        self.clap_menu.add(self.clap_thresholds_item)
        
        # Home Assistant
        self.ha_menu = rumps.MenuItem("Home Assistant")
        self.ha_status_item = rumps.MenuItem("HA: Disconnected")
        self.test_ha_item = rumps.MenuItem("Test Connection", callback=self.test_ha)
        self.ha_menu.add(self.ha_status_item)
        self.ha_menu.add(self.test_ha_item)
        
        # Preview
        self.preview_item = rumps.MenuItem("Show Preview", callback=self.toggle_preview)
        self.preview_window = None
        
        # Config
        self.config_item = rumps.MenuItem("Open Config", callback=self.open_config)
        self.reload_config_item = rumps.MenuItem("Reload Config", callback=self.reload_config)
        
        # Quit
        self.quit_item = rumps.MenuItem("Quit", callback=self.quit_app)
        
        # Build menu
        self.menu = [
            self.status_item,
            self.human_item,
            self.count_item,
            self.fps_item,
            None,
            self.start_item,
            self.mock_item,
            None,
            self.training_menu,
            self.clap_menu,
            self.ha_menu,
            None,
            self.preview_item,
            None,
            self.config_item,
            self.reload_config_item,
            None,
            self.quit_item
        ]
        
        # Preview update timer
        self._preview_timer = None
        self._last_preview_frame = None
        
    def run_app(self):
        """Initialize and start the background app."""
        self.app = create_app(self.config_path, self.mock_mode)
        self.app.set_status_callback(self._on_status_update)
        self.app.set_preview_callback(self._on_preview_frame)
        
        if self.app.initialize():
            self._update_menu_status()
        else:
            rumps.alert("Failed to initialize Kinect Smart Camera")
    
    def toggle_camera(self, sender):
        """Start or stop the camera."""
        if not self.app:
            return
            
        if self.app.running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start the camera and processing."""
        if self.app and self.app.start():
            self.start_item.title = "Stop Camera"
            self.status_item.title = "Status: Running"
            rumps.notification("Kinect Smart Camera", "Started", "Camera is now running")
        else:
            rumps.alert("Failed to start camera")
    
    def stop_camera(self):
        """Stop the camera and processing."""
        if self.app:
            self.app.stop()
            self.start_item.title = "Start Camera"
            self.status_item.title = "Status: Stopped"
            self.human_item.title = "Human: Not Detected"
            self.count_item.title = "Count: 0"
            rumps.notification("Kinect Smart Camera", "Stopped", "Camera has been stopped")
    
    def toggle_mock(self, sender):
        """Toggle mock mode (requires restart)."""
        sender.state = not sender.state
        self.mock_mode = sender.state
        rumps.alert("Mock mode changed. Please restart the app for changes to take effect.")
    
    def _on_status_update(self, status: Dict[str, Any]):
        """Handle status updates from the main app."""
        # Update menu items on main thread
        rumps.call_later(0, lambda: self._update_menu_from_status(status))
    
    def _update_menu_from_status(self, status: Dict[str, Any]):
        """Update menu items from status dict."""
        running = status.get('running', False)
        self.status_item.title = f"Status: {'Running' if running else 'Stopped'}"
        self.start_item.title = "Stop Camera" if running else "Start Camera"
        
        human_present = status.get('human_present', False)
        human_count = status.get('human_count', 0)
        self.human_item.title = f"Human: {'Detected' if human_present else 'Not Detected'}"
        self.count_item.title = f"Count: {human_count}"
        
        fps = status.get('fps', {})
        color_fps = fps.get('color', 0)
        self.fps_item.title = f"FPS: {color_fps:.1f}"
        
        ha_connected = status.get('ha_connected', False)
        self.ha_status_item.title = f"HA: {'Connected' if ha_connected else 'Disconnected'}"
        
        training_mode = status.get('training_mode', False)
        training_gesture = status.get('training_gesture', '')
        if training_mode:
            self.train_gesture_item.title = f"Training: {training_gesture}..."
            self.stop_training_item.set_callback(self.stop_training)
        else:
            self.train_gesture_item.title = "Train New Gesture..."
        
        model_loaded = status.get('model_loaded', False)
        self.train_model_item.title = "Retrain Model" if model_loaded else "Train Model"
    
    def _update_menu_status(self):
        """Initial menu status update."""
        if self.app:
            self._update_menu_from_status(self.app.get_status())
    
    def _on_preview_frame(self, frame_data):
        """Handle preview frame from main app."""
        frame_type, frame = frame_data
        if frame_type == 'color' and self.preview_window:
            self._last_preview_frame = frame.copy()
    
    def toggle_preview(self, sender):
        """Toggle preview window."""
        if self.preview_window:
            self.close_preview()
        else:
            self.open_preview()
    
    def open_preview(self):
        """Open preview window."""
        import cv2
        cv2.namedWindow("Kinect Smart Camera Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Kinect Smart Camera Preview", 640, 480)
        self.preview_window = True
        self.preview_item.title = "Hide Preview"
        
        # Start preview update loop
        def update_preview():
            while self.preview_window and self.app and self.app.running:
                if self._last_preview_frame is not None:
                    frame = self._last_preview_frame.copy()
                    
                    # Draw detections
                    detections = self.app._last_detections if hasattr(self.app, '_last_detections') else []
                    if detections and hasattr(self.app, 'human_detector'):
                        frame = self.app.human_detector.draw_detections(frame, detections)
                    
                    # Add status overlay
                    status_text = f"Human: {'Yes' if self.app._human_present else 'No'} | Count: {self.app._human_count}"
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    fps_text = f"FPS: {self.app._fps.get('color', 0):.1f}"
                    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Kinect Smart Camera Preview", frame)
                
                key = cv2.waitKey(33) & 0xFF
                if key == 27:  # ESC
                    break
            
            self.close_preview()
        
        self._preview_thread = threading.Thread(target=update_preview, daemon=True)
        self._preview_thread.start()
    
    def close_preview(self):
        """Close preview window."""
        import cv2
        self.preview_window = False
        cv2.destroyWindow("Kinect Smart Camera Preview")
        self.preview_item.title = "Show Preview"
    
    # Training methods
    
    def start_training_dialog(self, sender):
        """Show dialog to start training a new gesture."""
        if not self.app:
            return
            
        response = rumps.Window(
            title="Train New Gesture",
            message="Enter a name for the new gesture:",
            default_text="",
            ok="Start Training",
            cancel="Cancel",
            dimensions=(300, 100)
        ).run()
        
        if response.clicked and response.text.strip():
            gesture_name = response.text.strip()
            if self.app.start_training(gesture_name):
                rumps.notification("Kinect Smart Camera", "Training Started", f"Recording gesture: {gesture_name}")
            else:
                rumps.alert("Failed to start training")
    
    def stop_training(self, sender):
        """Stop training current gesture."""
        if self.app:
            gesture = self.app.stop_training()
            if gesture:
                rumps.notification("Kinect Smart Camera", "Training Stopped", f"Finished recording: {gesture}")
            else:
                rumps.alert("Not currently training")
    
    def train_model(self, sender):
        """Train the gesture recognition model."""
        if not self.app:
            return
            
        # Show progress
        rumps.notification("Kinect Smart Camera", "Training Model", "Training in progress...")
        
        def do_train():
            result = self.app.train_model()
            rumps.call_later(0, lambda: self._on_training_complete(result))
        
        threading.Thread(target=do_train, daemon=True).start()
    
    def _on_training_complete(self, result: Dict[str, Any]):
        """Handle training completion."""
        if result.get('success'):
            classes = result.get('classes', [])
            accuracy = result.get('cv_accuracy', 0)
            msg = f"Model trained successfully!\nClasses: {', '.join(classes)}\nCV Accuracy: {accuracy:.2%}"
            rumps.notification("Kinect Smart Camera", "Training Complete", msg)
        else:
            error = result.get('error', 'Unknown error')
            rumps.alert(f"Training failed: {error}")
    
    def clear_all_training(self, sender):
        """Clear all training data."""
        if rumps.alert("Clear all training data?", "This cannot be undone.", ok="Yes", cancel="No"):
            if self.app:
                self.app.clear_all_training()
                rumps.notification("Kinect Smart Camera", "Training Cleared", "All gesture data has been cleared")
    
    # Clap detection methods
    
    def recalibrate_clap(self, sender):
        """Recalibrate clap detector."""
        if self.app:
            self.app.recalibrate_clap()
            rumps.notification("Kinect Smart Camera", "Clap Recalibration", "Recalibration started")
    
    def set_clap_thresholds(self, sender):
        """Set clap detection thresholds."""
        if not self.app:
            return
            
        window = rumps.Window(
            title="Clap Detection Thresholds",
            message="RMS Threshold (0.01-0.1):\nPeak Threshold (0.05-0.5):",
            default_text="0.02\n0.15",
            ok="Set",
            cancel="Cancel",
            dimensions=(300, 150)
        )
        response = window.run()
        
        if response.clicked:
            try:
                lines = response.text.strip().split('\n')
                rms = float(lines[0].strip())
                peak = float(lines[1].strip())
                self.app.set_clap_thresholds(rms, peak)
                rumps.notification("Kinect Smart Camera", "Thresholds Updated", f"RMS: {rms}, Peak: {peak}")
            except Exception as e:
                rumps.alert(f"Invalid thresholds: {e}")
    
    # Home Assistant methods
    
    def test_ha(self, sender):
        """Test Home Assistant connection."""
        if self.app and self.app.ha_client:
            if self.app.ha_client.connection_type == 'mqtt':
                result = self.app.ha_client.publish(f"{self.app.ha_client.base_topic}/test", "test")
                msg = "MQTT test message sent" if result else "MQTT test failed"
            else:
                result = self.app.ha_client.get_state("sensor.test")
                msg = "REST API connection OK" if result is not None else "REST API test failed"
            rumps.notification("Kinect Smart Camera", "HA Test", msg)
    
    # Config methods
    
    def open_config(self, sender):
        """Open config file in default editor."""
        import subprocess
        config_file = self.config_path or (Path(__file__).parent.parent / 'config.yaml')
        subprocess.run(['open', str(config_file)])
    
    def reload_config(self, sender):
        """Reload configuration."""
        rumps.alert("Config reload requires app restart. Please quit and restart the app.")
    
    def quit_app(self, sender):
        """Quit the application."""
        if self.app and self.app.running:
            self.stop_camera()
        
        if self.preview_window:
            self.close_preview()
        
        rumps.quit_application()
    
    @rumps.timer(1)
    def update_timer(self, sender):
        """Periodic status update."""
        if self.app:
            self._update_menu_from_status(self.app.get_status())


def main():
    """Entry point for the menu bar app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kinect Smart Camera Menu Bar App")
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode (no hardware)')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run app
    app = KinectMenuBarApp(config_path=args.config, mock_mode=args.mock)
    app.run_app()
    app.run()


if __name__ == '__main__':
    main()