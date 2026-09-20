#!/usr/bin/env python3
"""
Create a simple app icon for the Kinect Smart Camera.
Run this to generate app_icon.icns from a generated icon.
"""

from PIL import Image, ImageDraw
import os

def create_icon():
    """Create a simple camera/Kinect icon."""
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    images = []
    
    for size in sizes:
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Background circle
        margin = size // 8
        draw.ellipse([margin, margin, size - margin, size - margin], 
                    fill=(30, 130, 200, 255), outline=(20, 100, 180, 255), width=max(1, size//64))
        
        # Camera body
        body_margin = size // 3
        body_top = size // 3
        body_bottom = size * 2 // 3
        draw.rounded_rectangle(
            [body_margin, body_top, size - body_margin, body_bottom],
            radius=size // 16,
            fill=(50, 50, 60, 255)
        )
        
        # Lens
        lens_center = (size // 2, size // 2)
        lens_radius = size // 6
        draw.ellipse([
            lens_center[0] - lens_radius, lens_center[1] - lens_radius,
            lens_center[0] + lens_radius, lens_center[1] + lens_radius
        ], fill=(20, 20, 30, 255), outline=(100, 180, 255, 255), width=max(1, size//128))
        
        # Inner lens
        inner_radius = lens_radius // 2
        draw.ellipse([
            lens_center[0] - inner_radius, lens_center[1] - inner_radius,
            lens_center[0] + inner_radius, lens_center[1] + inner_radius
        ], fill=(30, 130, 200, 200))
        
        # Kinect-style indicator dots (IR projectors)
        dot_size = size // 20
        dot_y = body_top + size // 12
        for i in range(3):
            dot_x = body_margin + (size - 2*body_margin) * (i + 1) // 4
            draw.ellipse([
                dot_x - dot_size, dot_y - dot_size,
                dot_x + dot_size, dot_y + dot_size
            ], fill=(255, 100, 50, 255))
        
        # Human silhouette in lens (small)
        human_size = lens_radius // 3
        human_x = lens_center[0]
        human_y = lens_center[1]
        # Head
        draw.ellipse([
            human_x - human_size//2, human_y - human_size,
            human_x + human_size//2, human_y
        ], fill=(255, 255, 255, 180))
        # Body
        draw.rectangle([
            human_x - human_size//3, human_y,
            human_x + human_size//3, human_y + human_size
        ], fill=(255, 255, 255, 180))
        
        images.append(img)
    
    # Save as ICNS
    os.makedirs('assets', exist_ok=True)
    images[0].save(
        'assets/app_icon.icns',
        format='ICNS',
        sizes=[(img.width, img.height) for img in images],
        append_images=images[1:]
    )
    print("Icon created at assets/app_icon.icns")

if __name__ == '__main__':
    create_icon()