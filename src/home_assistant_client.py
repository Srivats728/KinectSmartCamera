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