import numpy as np
import matplotlib.pyplot as plt

# Replace this with the path to your .npy file
npy_file_path = '/home/lixion/rgbd/data_process/d_r/00000100.npy'

# Load the NumPy file
try:
    data = np.load(npy_file_path)

    print(data.shape)

    plt.imshow(data)  # 'gray' for grayscale image
    plt.title("2D Array Visualization")
    plt.show()
    
    
except Exception as e:
    print(f"Error loading .npy file: {e}")
