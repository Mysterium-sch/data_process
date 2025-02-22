'''
Pulled from https://github.com/azureology/kitti-velo2cam/blob/master/proj_velo2cam.py
'''


import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
name = '%06d'%sn # 6 digit zeropadding
img = f'/home/lixion/rgbd/data/data_object_image_2/testing/image_2/{name}.png'
binary = f'/home/lixion/rgbd/data/data_object_velodyne/testing/velodyne/{name}.bin'
with open(f'/home/lixion/rgbd/data/data_object_calib/testing/calib/{name}.txt','r') as f:
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
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(png)
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

plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
plt.title(name)
#plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
plt.show()