#!/bin/bash
# Build script for Kinect Smart Camera macOS .app

set -e

echo "=== Kinect Smart Camera Build Script ==="
echo

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Check if on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Warning: This script is designed for macOS. Building .app on other platforms may not work correctly."
fi

# Install dependencies
echo
echo "=== Installing Python dependencies ==="
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements.txt

# Install py2app
echo
echo "=== Installing py2app ==="
pip3 install py2app

# Create icon if not exists
if [ ! -f assets/app_icon.icns ]; then
    echo
    echo "=== Creating app icon ==="
    python3 scripts/create_icon.py
fi

# Build the app
echo
echo "=== Building .app bundle ==="
python3 setup.py py2app

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