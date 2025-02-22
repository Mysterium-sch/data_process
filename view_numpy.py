import numpy as np
import matplotlib.pyplot as plt

# Replace this with the path to your .npy file
npy_file_path = './processed/lidar/00000100.npy'

# Load the NumPy file
try:
    data = np.load(npy_file_path)

    plt.imshow(data[:, :, 0])  # 'gray' for grayscale image
    plt.title("2D Array Visualization")
    plt.show()
    
    
except Exception as e:
    print(f"Error loading .npy file: {e}")
