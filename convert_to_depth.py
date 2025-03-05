import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from depth_map import dense_map
import time

# Class for the calibration matrices for KITTI data
class Calibration:
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3,4])

        self.L2C = calibs['Tr_velo_to_cam']
        self.L2C = np.reshape(self.L2C, [3,4])

        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

    @staticmethod
    def read_calib_file(filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    # From LiDAR coordinate system to Camera Coordinate system
    def lidar2cam(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        pts_3d_cam_rec = np.transpose(np.dot(self.R0, np.transpose(np.dot(np.hstack((pts_3d_lidar, np.ones((n,1)))), np.transpose(self.L2C)))))
        return pts_3d_cam_rec
    
    # From Camera Coordinate system to Image frame
    def rect2Img(self, rect_pts, img_width, img_height):
        n = rect_pts.shape[0]
        points_2d = np.dot(np.hstack((rect_pts, np.ones((n,1)))), np.transpose(self.P)) # nx3
        points_2d[:,0] /= points_2d[:,2]
        points_2d[:,1] /= points_2d[:,2]
        
        mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= img_width) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= img_height)
        mask = mask & (rect_pts[:,2] > 2)
        return points_2d[mask,0:2], mask

if __name__ == "__main__":
    root = "/home/lixion/rgbd/data/"
    image_dir = os.path.join(root, "data_object_image_2/training/image_2")
    velodyne_dir = os.path.join(root, "data_object_velodyne/training/velodyne")
    calib_dir = os.path.join(root, "data_object_calib/training/calib")

    total_time = 0
    total_images = 0

    # Loop through all files in the image directory
    for image_file in os.listdir(image_dir):
        # Check if the file is an image file with a .png extension
        if image_file.endswith('.png'):
            # Extract the ID from the filename (assuming it follows the format 'xxxxxx.png')
            cur_id = int(image_file.split('.')[0])
            
            # Loading the image
            img = cv2.imread(os.path.join(image_dir, image_file))
            
            # Loading the LiDAR data
            lidar = np.fromfile(os.path.join(velodyne_dir, "%06d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
            
            # Loading Calibration
            calib = Calibration(os.path.join(calib_dir, "%06d.txt" % cur_id))
            
            # From LiDAR coordinate system to Camera Coordinate system
            start_time = time.time()
            lidar_rect = calib.lidar2cam(lidar[:, 0:3])
            
            # From Camera Coordinate system to Image frame
            lidarOnImage, mask = calib.rect2Img(lidar_rect, img.shape[1], img.shape[0])
            
            # Concatenate LiDAR position with the intensity (3), with (2) we would have the depth
            lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask, 2].reshape(-1, 1)), 1)
            
            # Generate the depth map

            #print(img.shape[1], "" img.shape[0],)
            
            out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
            end_time = time.time()
            
            elapsed_time = (end_time - start_time) * 1000
            total_time += elapsed_time
            total_images += 1

            # Print the time for this image
            print(f"Processing time for {image_file}: {elapsed_time:.2f} ms")
            
            # Save the output
            cv2.imwrite(f"lidar/{cur_id:06d}.png", out)
    
    # Calculate and print the average processing time
    if total_images > 0:
        avg_time = total_time / total_images
        print(f"\nAverage processing time for {total_images} images: {avg_time:.2f} ms")
    else:
        print("No images processed.")