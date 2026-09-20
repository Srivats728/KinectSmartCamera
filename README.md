# Kinect Smart Camera for macOS

A macOS menu bar application that turns your Xbox One Kinect v2 into a smart camera with:
- **Human presence detection** using MediaPipe Pose + Kinect depth data
- **Clap detection** for gesture-free control
- **Custom gesture/action training** - teach it your own physical actions
- **Home Assistant integration** via MQTT or REST API for automations

## Requirements

- macOS (Intel or Apple Silicon)
- Xbox One Kinect v2 with 12V power mod (USB 3.0 connection)
- Python 3.8+
- Home Assistant (optional, for automations)

### Kinect Hardware Setup

The Xbox One Kinect v2 requires a 12V power supply (the original wall wart) and a USB 3.0 connection. The "solder mod" you mentioned typically involves:
1. Providing 12V to the Kinect's power input
2. Connecting the USB 3.0 data lines to your Mac

**Important**: The Kinect v2 draws significant power (~1.5A at 12V). Use a quality 12V 2A+ power supply.

## Installation

### Quick Start (Development)

```bash
# Clone and enter directory
cd KinectSmartCamera

# Install dependencies
pip3 install -r requirements.txt

# Install libfreenect2 (required for Kinect v2)
# On macOS with Homebrew:
brew install libusb pkg-config cmake
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(sysctl -n hw.ncpu)
sudo make install
cd ../..

# Run in mock mode (no hardware needed)
python3 -m src.menu_bar_app --mock

# Or run with real hardware
python3 -m src.menu_bar_app
```

### Build as .app Bundle

```bash
# Make build script executable
chmod +x build.sh

# Run build (installs deps, compiles, creates .app)
./build.sh
```

The built app will be at `dist/Kinect Smart Camera.app`. You can:
- Run directly: `open "dist/Kinect Smart Camera.app"`
- Copy to Applications: `cp -R "dist/Kinect Smart Camera.app" /Applications/`

## Configuration

Copy `config.yaml` and customize:

```yaml
kinect:
  device_index: 0
  enable_color: true
  enable_depth: true
  enable_ir: true

detection:
  human_detection:
    enabled: true
    confidence_threshold: 0.5
    min_person_height: 0.5
    max_distance: 5.0
    use_depth: true
    use_mediapipe: true
  
  clap_detection:
    enabled: true
    rms_threshold: 0.02
    peak_threshold: 0.15
    cooldown_ms: 500

home_assistant:
  enabled: true
  connection_type: "mqtt"  # or "rest"
  mqtt:
    host: "localhost"
    port: 1883
    username: ""
    password: ""
    base_topic: "kinect/smart_camera"
```

### Home Assistant Setup

#### MQTT (Recommended)

Add to your `configuration.yaml`:

```yaml
mqtt:
  binary_sensor:
    - name: "Kinect Human Presence"
      state_topic: "kinect/smart_camera/human_presence"
      payload_on: "on"
      payload_off: "off"
      device_class: "presence"
  
  sensor:
    - name: "Kinect Human Count"
      state_topic: "kinect/smart_camera/human_count"
      device_class: "counter"
    
    - name: "Kinect Clap Detected"
      state_topic: "kinect/smart_camera/clap_detected"
      device_class: "timestamp"
    
    - name: "Kinect Gesture"
      state_topic: "kinect/smart_camera/gesture_detected"
```

Auto-discovery is enabled by default - entities will appear automatically!

#### REST API

Generate a Long-Lived Access Token in Home Assistant (Profile → Security) and add to config:

```yaml
home_assistant:
  connection_type: "rest"
  rest:
    url: "http://homeassistant.local:8123"
    token: "YOUR_LONG_LIVED_TOKEN"
```

## Usage

### Menu Bar App

The app runs in your menu bar (top-right). Click the icon to access:

- **Start/Stop Camera** - Toggle Kinect capture
- **Mock Mode** - Test without hardware
- **Training** - Record and train custom gestures
- **Clap Detection** - Recalibrate or adjust thresholds
- **Home Assistant** - Test connection
- **Show Preview** - Open OpenCV preview window
- **Open Config** - Edit config.yaml

### Human Detection

The app detects humans using:
1. **MediaPipe Pose** - 33 body landmarks from color camera
2. **Depth silhouette** - Body-shaped blobs from depth data

Detection events sent to Home Assistant:
- `binary_sensor.kinect_human_presence` - on/off
- `sensor.kinect_human_count` - number of people
- Events: `kinect_human_entered`, `kinect_human_left`

### Clap Detection

Clap your hands to trigger automations:
- **Single clap** - Basic trigger
- **Double clap** - Triggers `automation.kinect_double_clap`
- **Triple clap** - Triggers `automation.kinect_triple_clap`

Adjust sensitivity in the menu or config.

### Custom Gesture Training

1. Click **Train New Gesture...** in the menu
2. Enter a name (e.g., "wave", "thumbs_up", "sit_down")
3. Perform the gesture repeatedly for ~3 seconds
4. Click **Stop Training**
5. Repeat for more gestures (minimum 10 samples each)
6. Click **Train Model** to create the classifier

Trained gestures are recognized in real-time and published to:
- `sensor.kinect_gesture_detected` - gesture name
- `sensor.kinect_custom_action_detected` - for complex actions

### Example Home Assistant Automations

```yaml
automation:
  - alias: "Kinect - Lights on when human enters"
    trigger:
      - platform: event
        event_type: kinect_human_entered
    action:
      - service: light.turn_on
        target:
          entity_id: light.living_room

  - alias: "Kinect - Double clap toggles TV"
    trigger:
      - platform: event
        event_type: kinect_clap_detected
        event_data:
          pattern: "double"
    action:
      - service: media_player.toggle
        target:
          entity_id: media_player.living_room_tv

  - alias: "Kinect - Wave gesture opens blinds"
    trigger:
      - platform: state
        entity_id: sensor.kinect_gesture_detected
        to: "wave"
    action:
      - service: cover.open_cover
        target:
          entity_id: cover.living_room_blinds
```

## Troubleshooting

### Kinect not detected
- Check USB 3.0 connection (Kinect v2 requires USB 3.0)
- Verify 12V power supply is connected and working
- Run `system_profiler SPUSBDataType` to see if device appears
- Try: `sudo kextunload -b com.apple.driver.usb.AppleUSBXHCI` then reload

### libfreenect2 issues
```bash
# Rebuild libfreenect2 for your architecture
cd libfreenect2
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
make -j$(sysctl -n hw.ncpu)
sudo make install
```

### Permission errors
- Grant Camera permission in System Settings → Privacy & Security → Camera
- Grant Microphone permission for clap detection

### Low FPS
- Reduce color resolution in config.yaml
- Disable IR stream if not needed
- Close other camera-using apps

## Architecture

```
src/
├── kinect_manager.py    # libfreenect2 wrapper (color/depth/IR streams)
├── human_detector.py    # MediaPipe Pose + depth-based detection
├── clap_detector.py     # Real-time audio clap detection
├── gesture_trainer.py   # Custom gesture recording & classification
├── home_assistant_client.py  # MQTT/REST HA integration
├── main_app.py          # Main coordinator
└── menu_bar_app.py      # rumps menu bar interface
```

## License

MIT License - Feel free to modify and distribute.

## Credits

- [libfreenect2](https://github.com/OpenKinect/libfreenect2) - Kinect v2 drivers
- [MediaPipe](https://mediapipe.dev/) - Pose/hand detection
- [rumps](https://github.com/jaredks/rumps) - Menu bar app framework
- [py2app](https://py2app.readthedocs.io/) - macOS app bundling