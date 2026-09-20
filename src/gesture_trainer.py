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