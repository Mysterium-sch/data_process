import os
import numpy as np
from PIL import Image

# Set the directory containing the numpy files
input_directory = '/home/lixion/rgbd/data_process/d/'
output_directory = '/home/lixion/rgbd/data_process/i/'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.npy'):
        # Load the numpy file
        file_path = os.path.join(input_directory, filename)
        data = np.load(file_path)
        print(data.shape)

        # Check if the data has more than two dimensions
        if len(data.shape) > 2:
            # Handle 3D or 4D arrays
            if data.ndim == 3:  # If the data is 3D (height, width, channels)
                second_channel = data[:, :, 0]*3  # Extract the second channel (index 1)
                #print(np.max(data[:, :, 1]))
            elif data.ndim == 4:  # If the data is 4D (batch, height, width, channels)
                second_channel = data[0, :, :, 1]  # Extract the second channel from the first batch

            # Convert the second channel to an image (ensure the dtype is uint8 for proper saving)
            image = Image.fromarray(second_channel.astype(np.uint8))


            # Save the extracted channel as a new PNG file
            output_filename = filename.replace('.npy', '.png')
            output_path = os.path.join(output_directory, filename)
            #image.save(output_path)
            np.save(output_path, second_channel)
            print(f"Saved {output_path}")
        else:
            print(f"{filename} does not have multiple channels.")
