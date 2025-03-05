import os
import numpy as np

# Set the directory containing the numpy files
input_directory = '/home/lixion/rgbd/data_process/d/'
output_directory = '/home/lixion/rgbd/data_process/blown/'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.npy'):
        # Load the numpy file
        file_path = os.path.join(input_directory, filename)
        data = np.load(file_path)

        # Check if the data has more than one channel (assuming 3D or 4D array)
        if len(data.shape) > 2:
            # Extract the second channel (index 1)
            second_channel = data[:, :, 1] if data.ndim == 3 else data[:, :, :, 0]*3

            # Save the extracted channel as a new numpy file
            output_path = os.path.join(output_directory, f"{filename}")
            np.save(output_path, second_channel)
            print(f"Saved {output_path}")
        else:
            print(f"{filename} does not have multiple channels.")
