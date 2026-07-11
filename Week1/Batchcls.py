import cv2
import os
import numpy as np

# SETUP INPUT AND OUTPUT FOLDERS
input_folder = "input_images"
output_folder = "output_images"

# CREATE OUTPUT FOLDER SAFELY
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"Starting batch process on folder: {input_folder}...\n")

# THE BATCH LOOP
# This loop looks at every single file inside input folder
for filename in os.listdir(input_folder):
    
    # We only want to process actual image files, ignoring random hidden system files
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        
        # Build the exact path: "input_images/my_photo.jpg"
        img_path = os.path.join(input_folder, filename)
        
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {filename}. Skipping.")
            continue
            
        print(f"Processing: {filename}")
        
        
        # Resize to 800x800 square
        img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
        
        # Add the semi-transparent watermark bar at the bottom
        overlay = img.copy()
        height, width = img.shape[:2]
        cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
        
        alpha = 0.7 
        beta = 0.3 
        blended = cv2.addWeighted(img, alpha, overlay, beta, 0)
        
        # Add the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blended, "PIYUSH'S IMAGES", (20, height - 30), font, 1.2, (255, 255, 255), 2)
        
        # Build the exact save path: "output_images/my_photo.jpg"
        output_path = os.path.join(output_folder, filename)
        
        # Write the file to the hard drive
        cv2.imwrite(output_path, blended)

print("\nSuccess! Check the 'output_images' folder for your final images.")