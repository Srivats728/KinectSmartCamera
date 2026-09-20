"""
Kinect Smart Camera - Main package.
"""

__version__ = "1.0.0"
__author__ = "Kinect Smart Camera"

from .main_app import create_app, KinectSmartCameraApp
from .kinect_manager import KinectManager, MockKinectManager
from .human_detector import HumanDetector
from .clap_detector import ClapDetector, MockClapDetector
from .gesture_trainer import GestureTrainer, ActionTrainer
from .home_assistant_client import HomeAssistantClient, MockHomeAssistantClient

__all__ = [
    'create_app',
    'KinectSmartCameraApp',
    'KinectManager',
    'MockKinectManager',
    'HumanDetector',
    'ClapDetector',
    'MockClapDetector',
    'GestureTrainer',
    'ActionTrainer',
    'HomeAssistantClient',
    'MockHomeAssistantClient',
]