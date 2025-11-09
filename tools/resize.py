import os
from PIL import Image
from pathlib import Path

# Source and destination directories
source_dir = r"C:\ai8x-framework\ai8x-training\data\pizza_not_pizza"
output_dir = r"C:\ai8x-framework\ai8x-training\data\pizza_not_pizza64"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Maintain the same subdirectory structure
for root, dirs, files in os.walk(source_dir):
    # Get relative path from source directory
    rel_path = os.path.relpath(root, source_dir)
    
    # Create corresponding subdirectory in output directory
    if rel_path != '.':
        os.makedirs(os.path.join(output_dir, rel_path), exist_ok=True)
    
    # Process each image file
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                # Full path to source image
                img_path = os.path.join(root, file)
                
                # Open image
                img = Image.open(img_path)
                
                # Resize to 64x64 using high-quality resampling
                resized_img = img.resize((64, 64), Image.LANCZOS)
                
                # Save path (preserving directory structure)
                if rel_path == '.':
                    save_path = os.path.join(output_dir, file)
                else:
                    save_path = os.path.join(output_dir, rel_path, file)
                
                # Save with same format and quality
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension == '.png':
                    resized_img.save(save_path, "PNG")
                elif file_extension in ['.jpg', '.jpeg']:
                    resized_img.save(save_path, "JPEG", quality=95)
                elif file_extension == '.bmp':
                    resized_img.save(save_path, "BMP")
                
                print(f"Processed: {img_path} → {save_path}")
                
            except Exception as e:
                print(f"Error processing {file}: {e}")

print("Conversion complete!")