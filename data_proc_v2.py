
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import shutil
import os
from pathlib import Path
import cv2
import time
import csv

def project_points(img, binary, cali):

    sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
    with open(cali,'r') as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
    points = scan[:, 0:3] # lidar xyz (front, left, up)
    ref = scan[:, 3]
    # TODO: use fov filter? 
    velo = np.insert(points,3,1,axis=1).T
    v_ref = ref.T

    delete_indices = np.where(velo[0, :] < 0)[0]  # [0] to get the indices as a 1D array
    velo = np.delete(velo, delete_indices, axis=1)
    v_ref = np.delete(v_ref, delete_indices)
    #velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))

    del_cam = np.where(cam[2,:]<0)
    cam = np.delete(cam,del_cam,axis=1)
    c_ref = np.delete(v_ref, del_cam)
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    #plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
    png = mpimg.imread(img)
    IMG_H,IMG_W,_ = png.shape
    # restrict canvas in range
    #plt.axis([0,IMG_W,IMG_H,0])
    #plt.imshow(png)
    # filter point out of canvas
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    # generate color map from depth
    u,v,z = cam

    np_lidar = np.zeros((png.shape[0], png.shape[1], 2))

    for count in range(len(z)):
        np_lidar[int(v[count]), int(u[count]), 0] = z[count]
        np_lidar[int(v[count]), int(u[count]), 1] = c_ref[count]

    print(np_lidar.shape)

    #plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
    #plt.title(name)
    #plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
    #plt.show()
    return np_lidar

labels = '/home/lixion/rgbd/data/data_object_label_2/training/label_2'
images = '/home/lixion/rgbd/data/data_object_image_2/training/image_2'
lidar = '/home/lixion/rgbd/data/data_object_velodyne/training/velodyne'
cali = '/home/lixion/rgbd/data/data_object_calib/training/calib'
out = '.'

lidard_path = os.path.join(out, 'processed', 'lidar', 'd_r')
lidardr_path = os.path.join(out, 'processed', 'lidar', 'd')
images_path = os.path.join(out, 'processed', 'images')
labels_path = os.path.join(out, 'processed', 'labels')

os.makedirs(lidard_path, exist_ok=True)
os.makedirs(lidardr_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)

images_dir = Path(images)
lidar_dir = Path(lidar)
labels_dir = Path(labels)
cali_dir = Path(cali)



main_dir = -1
count = 0
elapsed_time = []
for im in images_dir.iterdir():

    name = im.stem
    li = Path(lidar_dir, (name + ".bin"))
    ca = Path(cali_dir, (name + ".txt"))
    la = Path(labels_dir, (name + ".txt"))

    image = cv2.imread(str(im))

    print(im, " ", li, " ", ca, " ", la)

    start_time = time.time()
    lidar = project_points(im, li, ca)
    end_time = time.time()

    elapsed_time.append((end_time - start_time)*1000)

    name = "00000" + str(count)
    np.save(Path(lidard_path, (name + ".npy")), lidar[:, :, 0])
    np.save(Path(lidardr_path, (name + ".npy")), lidar)        
    image_out = Path(images_path, (name + ".png"))
    shutil.copy(im, image_out)

    label_out = Path(labels_path, (name + ".txt"))
    shutil.copy(la, label_out)
    count += 1

with open('processing_times.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header if the file is empty
    if file.tell() == 0:
        writer.writerow(["Processing Time (ms)"])

    # Write the command and its processing time
    for t in elapsed_time:
        writer.writerow([t])

