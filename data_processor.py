import numpy as np
import cv2
import os
from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#import pykitti
from kitti_fondation_tracks import Kitti, Kitti_util
from src import parseTrackletXML as pt_XML
import torch
from PIL import Image


class data_proc:
    
    def __init__(self, images, vel_cal, cam_cal, lidar, labels, out, num):
        self.image_path = images
        self.vel_cal = vel_cal
        self.cam_cal = cam_cal
        self.lidar_path = lidar
        self.labels_path = labels
        self.output = out

        #self.lidar_data = self.bin_file_to_float_matrix(self.lidar_path)
        #self.image_data = self.load_images(self.image_path)
        #self.calibration = self.loadCalib(self.cali_path)

        self.processPoints(self.image_path, self.lidar_path, self.vel_cal, self.cam_cal, self.output, num)
        self.processTracks(self.image_path, self.lidar_path, self.vel_cal, self.cam_cal, self.labels_path, self.output, num)

        # save one frame about projecting velodyne points into camera image
        '''
        image_type = 'rgb'  # 'gray' or 'color' image
        mode = '00' if image_type == 'gray' else '02'  # image_00 = 'graye image' , image_02 = 'color image'

        self.image_path += 'image_' + mode + '/data'
        velo_path = self.lidar_path

        v_fov, h_fov = (-24.9, 2.0), (-90, 90)

        v2c_filepath = vel_cal
        c2c_filepath = cam_cal

        res = Kitti_util(frame=89, camera_path=self.image_path, velo_path=velo_path, \
                        v2c_path=v2c_filepath, c2c_path=c2c_filepath)

        img, pnt, xyz, c_ = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)
        zc = xyz[::][2]

        tracks = pt_XML.parseXML(labels)
        

        result = self.print_projection_plt(pnt, c_, img)
        '''


        
    def print_projection_cv2(self, points, color, image):
        """ project converted velodyne points into camera image """
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for i in range(points.shape[1]):
            cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def print_projection_plt(self, points, color, image):
        """ project converted velodyne points into camera image """
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for i in range(points.shape[1]):
            x = np.int32(points[0][i])
            y = np.int32(points[1][i])
            cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    def bin_file_to_float_matrix(self, file_path):

        pc = []
        for file in os.listdir(file_path):
            scan = np.fromfile(os.path.join(file_path, file), dtype=np.float32)
            scan = scan.reshape((-1, 4))
            pc.append(scan)
        
        return pc
    
    def load_images(self, img_file):
        x = []
        i = 0
        for dir in os.listdir(img_file):
            print(dir)
            if 'image' in dir:
                for file in os.listdir(os.path.join(img_file, dir, 'data')):
                    if file.endswith('.png'):
                        imgpath = os.path.join(img_file, dir, 'data', file)
                        img = cv2.imread(imgpath, 0)
                        if img is not None:
                            x.append(img)
                i +=1
        return x

    def loadCalib(self, path):
        calib_data = {
        'calib_time': None,
        'R': [],
        'T': [],
        'delta_f': [],
        'delta_c': []
        }

        with open(path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('calib_time'):
                calib_data['calib_time'] = line.split(":")[1].strip()
            elif line.startswith('R:'):
                calib_data['R'] = list(map(float, line.split(":")[1].strip().split()))
            elif line.startswith('T:'):
                calib_data['T'] = list(map(float, line.split(":")[1].strip().split()))
            elif line.startswith('delta_f:'):
                calib_data['delta_f'] = list(map(float, line.split(":")[1].strip().split()))
            elif line.startswith('delta_c:'):
                calib_data['delta_c'] = list(map(float, line.split(":")[1].strip().split()))

        return calib_data

    def printTracklet(self, tracks):
        print(tracks.objectType)
        print(tracks.size)
        print(tracks.firstFrame)
        print(tracks.trans)
        print(tracks.rots)    # n x 3 float array (x,y,z)
        print(tracks.states)  # len-n uint8 array of states
        print(tracks.occs)    # n x 2 uint8 array  (occlusion, occlusion_kf)
        print(tracks.truncs)  # len-n uint8 array of truncation
        print(tracks.amtOccs)    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
        print(tracks.amtBorders)    # None (n x 3) float array  (amt_border_l / _r / _kf)
        print(tracks.nFrames)

    def processPoints(self, images, lidar, vel_cal, cam_cal, out, num):
        lidar_path = os.path.join(out, 'processed', 'lidar')
        os.makedirs(lidar_path, exist_ok=True)
        images_path = os.path.join(out, 'processed', 'images')
        os.makedirs(images_path, exist_ok=True)
        print("Starting processing of: ", images_path)

        images_dir = Path(images)
        
        main_dir = -1
        for subdir in images_dir.iterdir():
            main_dir += 1
            if 'image_02' in subdir.name:
                data_dir = subdir / 'data'
                if data_dir.exists() and data_dir.is_dir():
                    count = 0
                    for im in data_dir.iterdir():
                        try:
                            v_fov, h_fov = (-24.9, 2.0), (-90, 90)
                            velo_path = Path(lidar, im.name)

                            v2c_filepath = vel_cal
                            c2c_filepath = cam_cal

                            res = Kitti_util(frame=count, camera_path=str(data_dir), velo_path=str(lidar), \
                                                v2c_path=v2c_filepath, c2c_path=c2c_filepath)

                            img, pnt, xyz, c_ = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)
                            zc = xyz[::][2]

                            np_im = np.array(img)
                            image = Image.fromarray(img)

                            np_lidar = np.zeros((np_im.shape[0], np_im.shape[1]))

                            name = f"{num}00{main_dir}00{count}"

                            index = 0
                            for z in zc:
                                x = np.int32(pnt[0, index])
                                y = np.int32(pnt[1, index])
                                if (x < np_lidar.shape[1] and y < np_lidar.shape[0] and x > 0 and y > 0 ):
                                    np_lidar[y, x] = z
                                index += 1
                            np.save(Path(lidar_path, (name + ".npy")), np_lidar)    
                            image.save(Path(images_path, (name + ".png")))

                        except Exception as e:
                            print(f"Error processing data at count {count}: {e}")
                        count += 1
                        
    def processTracks(self, images, lidar, vel_cal, cam_cal, tracks_path, out, num):
        labels_path = os.path.join(out, 'processed', 'labels')
        os.makedirs(labels_path, exist_ok=True)
        #images_path = os.path.join('.', 'processed', 'images')

        print("Starting processing of: ", labels_path)

        images_dir = Path(images)
        
        main_dir = -1
        for subdir in images_dir.iterdir():
            main_dir += 1
            if 'image_02' in subdir.name:
                data_dir = subdir / 'data'
                if data_dir.exists() and data_dir.is_dir():
                    count = 0
                    for im in data_dir.iterdir():

                        v_fov, h_fov = (-24.9, 2.0), (-90, 90)
                        velo_path = Path(lidar, im.name)

                        v2c_filepath = vel_cal
                        c2c_filepath = cam_cal

                        res = Kitti_util(frame=count, camera_path=str(data_dir), velo_path=str(lidar), \
                                            xml_path=tracks_path, v2c_path=v2c_filepath, c2c_path=c2c_filepath)
                        
                        points = res.velo_file
                        tracklet_, type_, descrption = res.tracklet_info
                        image = res.camera_file

                        tracklet2d = []
                        words = []
                        try:
                            for i, j, w in zip(tracklet_[count], type_[count], descrption[count]):
                                try:
                                    # Process the 3D track data
                                    point = i.T
                                    x = round(np.mean(i[0]), 2)
                                    y = round(np.mean(i[1]), 2)
                                    z = round(np.mean(i[2]), 2)
                                    realworld = str(x) + " " + str(y) + " " + str(z) + " "

                                    # Calculate direction vector and angle
                                    a = np.array([i[0][0], i[1][0], i[2][0]])
                                    b = np.array([i[0][-1], i[1][-1], i[2][-1]])
                                    direction_vector = b - a
                                    reference_vector = np.array([1, 0, 0])
                                    dot_product = np.dot(direction_vector, reference_vector)
                                    magnitude_direction = np.linalg.norm(direction_vector)
                                    magnitude_reference = np.linalg.norm(reference_vector)
                                    cos_angle = dot_product / (magnitude_direction * magnitude_reference)
                                    angle_rad = round(np.arccos(cos_angle), 2)

                                    # Project tracks and compute bounding box
                                    ans, xyz_c, c_ = res.project_tracks(point)
                                    x_min = round(np.min(ans[0]), 2)
                                    x_max = round(np.max(ans[0]), 2)
                                    y_min = round(np.min(ans[1]), 2)
                                    y_max = round(np.max(ans[1]), 2)
                                    text = str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) + " "

                                    # Process the label description
                                    word = w.split()
                                    words.append(word[0] + " " + word[1] + " " + word[2] + " " + word[6] + " " + text + word[3] + " " + word[4] + " " + word[5] + " " + realworld + str(angle_rad))
                                    tracklet2d.append(ans)

                                except Exception as e:
                                    print(f"Error processing tracklet at count {count}: {e}")
                                    continue  # Continue with the next iteration if there's an error

                            # Type color mappings
                            type_c = {
                                'Car': (0, 0, 255), 'Van': (0, 255, 0), 'Truck': (255, 0, 0), 'Pedestrian': (0,255,255),
                                'Person (sitting)': (255, 0, 255), 'Cyclist': (255, 255, 0), 'Tram': (0, 0, 0), 'Misc': (255, 255, 255)
                            }

                            line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])

                            # Define output path for the file
                            name = os.path.join(labels_path, f"{num}00{main_dir}00{count}.txt")

                            # Write results to the file
                            with open(name, "w") as file:
                                for i, j, w in zip(tracklet2d, type_[count], words):
                                    try:
                                        file.write(w + "\n")
                                    except Exception as e:
                                        print(f"Error writing to file {name}: {e}")

                        except Exception as e:
                            print(f"Error processing data at count {count}: {e}")



                        count += 1

                        
      
class kittiDataset:
    def __init__(self, images, lidar, tracklets):
        self.imgs = images
        self.lidar = lidar
        self.tracks = tracklets


directory_path = Path("/home/lixion/rgbd/2011_09_26")
count = 0



for item in directory_path.iterdir():
    vel_to_cam = os.path.join(directory_path, 'calib_velo_to_cam.txt')
    cam_to_cam = os.path.join(directory_path, 'calib_cam_to_cam.txt')
    lidar = os.path.join(item, 'velodyne_points/data/')
    out = item
    num = count
    labels = os.path.join(item, item.name, 'tracklet_labels.xml')
        #images = '/home/lixion/stuff/kitti/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/'
    try:
        kitti_data = data_proc(item, vel_to_cam, cam_to_cam, lidar, labels, out, num)
    except Exception as e:
        print(f"Error processing data at count {count}: {e}")
    count += 1