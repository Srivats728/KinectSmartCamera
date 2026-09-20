"""
Setup script for building Kinect Smart Camera as a macOS .app bundle using py2app.
"""

from setuptools import setup
import sys
from pathlib import Path

# Read version
VERSION = "1.0.0"

APP = ['src/menu_bar_app.py']
DATA_FILES = [
    ('config', ['config.yaml']),
    ('models', []),  # Will be populated at runtime
]

OPTIONS = {
    'argv_emulation': False,
    'strip': True,
    'optimize': 2,
    'includes': [
        'rumps',
        'cv2',
        'numpy',
        'mediapipe',
        'freenect2',
        'sounddevice',
        'scipy',
        'paho.mqtt.client',
        'yaml',
        'PIL',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.preprocessing',
        'sklearn.pipeline',
        'sklearn.model_selection',
    ],
    'excludes': [
        'tkinter',
        'matplotlib',
        'pytest',
        'setuptools',
        'pip',
        'wheel',
    ],
    'packages': [
        'src',
    ],
    'resources': [
        'config.yaml',
    ],
    'plist': {
        'CFBundleName': 'Kinect Smart Camera',
        'CFBundleDisplayName': 'Kinect Smart Camera',
        'CFBundleIdentifier': 'com.kinect.smartcamera',
        'CFBundleVersion': VERSION,
        'CFBundleShortVersionString': VERSION,
        'CFBundleIconFile': 'app_icon.icns',
        'LSUIElement': True,  # Menu bar app (no dock icon)
        'NSHighResolutionCapable': True,
        'NSCameraUsageDescription': 'Kinect Smart Camera needs camera access for Kinect color stream',
        'NSMicrophoneUsageDescription': 'Kinect Smart Camera needs microphone access for clap detection',
        'NSHumanReadableCopyright': 'Copyright © 2024',
    },
    'iconfile': 'assets/app_icon.icns' if Path('assets/app_icon.icns').exists() else None,
}

# Filter out None values
OPTIONS = {k: v for k, v in OPTIONS.items() if v is not None}

setup(
    name='Kinect Smart Camera',
    version=VERSION,
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    author='Kinect Smart Camera',
    description='Smart camera app for Xbox One Kinect v2 with human detection, clap recognition, and Home Assistant integration',
    python_requires='>=3.8',
)