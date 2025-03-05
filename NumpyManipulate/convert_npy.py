import os
import numpy as np
from PIL import Image

# Set the directory containing the numpy files
input_directory = '/home/lixion/rgbd/data_process/DenseMap/lidar/'
output_directory = '/home/lixion/rgbd/data_process/lidar/'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.png'):
        # Load the PNG file
        file_path = os.path.join(input_directory, filename)
        image = Image.open(file_path)
        
        # Convert the image to a numpy array
        data = np.array(image)
        print(f"Loaded {filename} with shape: {data.shape}")

        # Save the numpy array to a file with the same name but .npy extension
        output_filename = filename.replace('.png', '.npy')
        output_path = os.path.join(output_directory, output_filename)
        
        # Save the numpy array
        np.save(output_path, data)
        print(f"Saved {output_path}")
