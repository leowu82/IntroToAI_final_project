import cv2
import os

# Input directory containing images
input_dir = 'input'
# Output directory to store resized images
output_dir = 'resized_images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Target size
target_size = (178, 218)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    # Construct input and output file paths
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Read image
    img = cv2.imread(input_path)

    # Resize image
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Save resized image
    cv2.imwrite(output_path, resized_img)

    print(f"Resized {filename} to {target_size}")
