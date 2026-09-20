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