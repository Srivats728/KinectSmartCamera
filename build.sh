#!/bin/bash
# Build script for Kinect Smart Camera macOS .app

set -e

echo "=== Kinect Smart Camera Build Script ==="
echo

# Find a compatible Python version (3.11 or 3.12 preferred for mediapipe wheels)
PYTHON_CMD=""
for py in python3.11 python3.12 python3.13 python3; do
    if command -v "$py" &> /dev/null; then
        version=$($py --version 2>&1 | cut -d' ' -f2)
        major_minor=$(echo $version | cut -d. -f1,2)
        if [[ "$major_minor" == "3.11" ]] || [[ "$major_minor" == "3.12" ]] || [[ "$major_minor" == "3.13" ]]; then
            PYTHON_CMD="$py"
            echo "Using Python $version at $(command -v $py)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: No compatible Python found (need 3.11, 3.12, or 3.13)."
    echo "Install with: brew install python@3.11"
    exit 1
fi

python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Check if on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Warning: This script is designed for macOS. Building .app on other platforms may not work correctly."
fi

# Install dependencies
echo
echo "=== Installing Python dependencies ==="
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel --break-system-packages
$PYTHON_CMD -m pip install -r requirements.txt --break-system-packages

# Install py2app
echo
echo "=== Installing py2app ==="
$PYTHON_CMD -m pip install py2app --break-system-packages

# Create icon if not exists
if [ ! -f assets/app_icon.icns ]; then
    echo
    echo "=== Creating app icon ==="
    $PYTHON_CMD scripts/create_icon.py
fi

# Build the app
echo
echo "=== Building .app bundle ==="
$PYTHON_CMD setup.py py2app

# Check result
app_path="dist/Kinect Smart Camera.app"
if [ -d "$app_path" ]; then
    echo
    echo "=== Build successful! ==="
    echo "App location: $app_path"
    echo
    echo "To run the app:"
    echo "  open \"$app_path\""
    echo
    echo "To install to Applications:"
    echo "  cp -R \"$app_path\" /Applications/"
else
    echo
    echo "=== Build failed! ==="
    echo "Check the output above for errors."
    exit 1
fi

# Optional: Create DMG
read -p "Create DMG installer? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "=== Creating DMG ==="
    dmg_name="KinectSmartCamera-${VERSION:-1.0.0}.dmg"
    hdiutil create -volname "Kinect Smart Camera" -srcfolder "$app_path" -ov -format UDZO "$dmg_name"
    echo "DMG created: $dmg_name"
fi

echo
echo "Done!"