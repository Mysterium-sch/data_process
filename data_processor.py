import numpy as np
import cv2
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#from kitti_fondation import Kitti, Kitti_util
#from src import parseTrackletXML as pt_XML
from PIL import Image


class data_proc:
    
    def __init__(self, images, cali, lidar, labels, out):
        self.image_path = images
        self.cali = cali
        self.lidar_path = lidar
        self.labels_path = labels

        #self.lidar_data = self.bin_file_to_float_matrix(self.lidar_path)
        #self.image_data = self.load_images(self.image_path)
        #self.calibration = self.loadCalib(self.cali_path)

        self.processPoints(self.image_path, self.lidar_path, self.cali, self.labels_path, out)
        #self.processTracks(self.image_path, self.lidar_path, self.cali, self.labels_path, out)

        
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

    def processPoints(self, images, lidar, cali, labels, out):
        lidar_path = os.path.join(out, 'processed', 'lidar')
        images_path = os.path.join(out, 'processed', 'images')
        labels_path = os.path.join(out, 'processed', 'labels')

        os.makedirs(lidar_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

        images_dir = Path(images)
        lidar_dir = Path(lidar)
        labels_dir = Path(labels)
        cali_dir = Path(cali)


        
        main_dir = -1
        count = 0
        for im in images_dir.iterdir():

            name = im.stem
            li = Path(lidar_dir, (name + ".bin"))
            ca = Path(cali_dir, (name + ".txt"))
            la = Path(labels_dir, (name + ".txt"))

            print(im, " ", li, " ", ca, " ", la)

            v_fov, h_fov = (-24.9, 2.0), (-90, 90)


            res = kittiDataset(im, li, ca)

            img, pnt, xyz, c_ = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)
            cv2.imshow("image", pnt)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

            zc = xyz[::][2]

            np_im = np.array(img)
            image = Image.fromarray(img)

            np_lidar = np.zeros((np_im.shape[0], np_im.shape[1]))

            name = "00000" + str(count)

            index = 0
            for z in zc:
                x = np.int32(pnt[0, index])
                y = np.int32(pnt[1, index])
                if (x < np_lidar.shape[1] and y < np_lidar.shape[0] and x > 0 and y > 0 ):
                    np_lidar[y, x] = z
                    index += 1
            np.save(Path(lidar_path, (name + ".npy")), np_lidar)    
            image.save(Path(images_path, (name + ".png")))
            count += 1
                        
    def processTracks(self, images, lidar, vel_cal, cam_cal, tracks_path, out):
        labels_path = os.path.join(out, 'processed', 'labels')
        #images_path = os.path.join('.', 'processed', 'images')

        images_dir = Path(images)
        
        main_dir = -1
        for subdir in images_dir.iterdir():
            main_dir += 1
            if 'image' in subdir.name:
                data_dir = subdir / 'data'
                if data_dir.exists() and data_dir.is_dir():
                    count = 0
                    for im in data_dir.iterdir():

                        v_fov, h_fov = (-24.9, 2.0), (-90, 90)
                        velo_path = Path(lidar, im.name)
                        print(count)

                        v2c_filepath = vel_cal
                        c2c_filepath = cam_cal

                        res = kittiDataset(im, lidar_path, cali)
                        
                        points = res.velo_file
                        tracklet_, type_, descrption = res.tracklet_info
                        image = res.camera_file

                        tracklet2d = []
                        words = []
                        for i, j, w in zip(tracklet_[count], type_[count], descrption[count]):
                            point = i.T
                            x = round(np.mean(i[0]), 2)
                            y = round(np.mean(i[1]), 2)
                            z = round(np.mean(i[2]), 2)
                            realworld = str(x) + " " + str(y) + " " + str(z) + " "

                            a = np.array([i[0][0], i[1][0], i[2][0]])
                            b = np.array([i[0][-1], i[1][-1], i[2][-1]])
                            direction_vector = b - a
                            reference_vector = np.array([1, 0, 0])
                            dot_product = np.dot(direction_vector, reference_vector)
                            magnitude_direction = np.linalg.norm(direction_vector)
                            magnitude_reference = np.linalg.norm(reference_vector)
                            cos_angle = dot_product / (magnitude_direction * magnitude_reference)
                            angle_rad = round(np.arccos(cos_angle), 2)


                            ans, xyz_c, c_ = res.project_tracks(point)
                            x_min = round(np.min(ans[0]), 2)
                            x_max = round(np.max(ans[0]), 2)
                            y_min = round(np.min(ans[1]), 2)
                            y_max = round(np.max(ans[1]), 2)
                            text = str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) + " "

                            word = w.split()
                            words.append(word[0] + " " + word[1] + " " + word[2] + " " + word[6] + " " + text + word[3] + " " + word[4] + " " + word[5] + " " + realworld + str(angle_rad))
                            tracklet2d.append(ans)
                            

                        type_c = { 'Car': (0, 0, 255), 'Van': (0, 255, 0), 'Truck': (255, 0, 0), 'Pedestrian': (0,255,255), \
                            'Person (sitting)': (255, 0, 255), 'Cyclist': (255, 255, 0), 'Tram': (0, 0, 0), 'Misc': (255, 255, 255)}

                        line_order = ([0, 1], [1, 2],[2, 3],[3, 0], [4, 5], [5, 6], \
                                [6 ,7], [7, 4], [4, 0], [5, 1], [6 ,2], [7, 3])
                        

                        name = os.path.join(labels_path, (str(main_dir) + "00000" + str(count) + ".txt"))
                        
                        with open(name, "w") as file:
                            for i, j, w in zip(tracklet2d, type_[count], words):

                                file.write(w + "\n")

                                for k in line_order:    
                                    cv2.line(image, (int(i[0][k[0]]), int(i[1][k[0]])), (int(i[0][k[1]]), int(i[1][k[1]])), type_c[j], 2)

                        plt.subplots(1,1, figsize = (12,4))
                        plt.title("3D Tracklet display on image")
                        plt.axis('off')
                        plt.imshow(image)
                        plt.waitforbuttonpress()

                        count += 1

                        
      
class kittiDataset:
    
    def __init__(self, image_path, lidar_path, cali):
        self.__h_min, self.__h_max = -180, 180
        self.__v_min, self.__v_max = -24.9, 2.0
        self.__v_res, self.__h_res = 0.42, 0.35
        self.__img_size = None
        self.image = self.__get_camera_frame(image_path)
        self.lidar = self.__get_velo_frame(lidar_path)
        self.cali = self.__load_velo2cam(cali)

    def __get_camera_frame(self, files):
        """ Return image for one frame """
        frame = cv2.imread(str(files))
        self.__img_size = frame.shape
        return frame
    
    def __get_velo_frame(self, file):
        """ Convert bin to numpy array for one frame """
        points = np.fromfile(str(file), dtype=np.float32).reshape(-1, 4)
        return points[:, :3]
    
    def __load_velo2cam(self, cali):
        """ load Velodyne to Camera calibration info file """
        with open(str(cali), "r") as f:
            file = f.readlines()
            return file
        
    def __calib_velo2cam(self, cali_file):
        """
        get Rotation(R : 3x3), Translation(T : 3x1) matrix info
        using R,T matrix, we can convert velodyne coordinates to camera coordinates
        """

        for line in cali_file:
            (key, val) = line.split(':', 1)
            if key == 'Tr_velo_to_cam':
                TR = np.fromstring(val, sep=' ')
                TR = TR.reshape(3, 4)
                R = TR[:3, :3]
                T = TR[:3, 3]
                return TR

    def __calib_cam2cam(self, cali_file):
        """
        If your image is 'rectified image' :
            get only Projection(P : 3x4) matrix is enough
        but if your image is 'distorted image'(not rectified image) :
            you need undistortion step using distortion coefficients(5 : D)

        In this code, only P matrix info is used for rectified image
        """

        for line in cali_file:
            (key, val) = line.split(':', 1)
            if key == 'P2':
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
                return P_

    def __velo_2_img_projection(self, points):
        """ convert velodyne coordinates to camera image coordinates """

        # rough velodyne azimuth range corresponding to camera horizontal fov
        if self.__h_fov is None:
            self.__h_fov = (-50, 50)
        if self.__h_fov[0] < -50:
            self.__h_fov = (-50,) + self.__h_fov[1:]
        if self.__h_fov[1] > 50:
            self.__h_fov = self.__h_fov[:1] + (50,)

        # R_vc = Rotation matrix ( velodyne -> camera )
        # T_vc = Translation matrix ( velodyne -> camera )
        RT_ = self.__calib_velo2cam(self.cali)
        

        # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
        P_ = self.__calib_cam2cam(self.cali)
        print(P_)

        """
        xyz_v - 3D velodyne points corresponding to h, v FOV limit in the velodyne coordinates
        c_    - color value(HSV's Hue vaule) corresponding to distance(m)

                 [x_1 , x_2 , .. ]
        xyz_v =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
                 [ 1  ,  1  , .. ]
        """
        xyz_v, c_ = self.__point_matrix(points)

        """
        RT_ - rotation matrix & translation matrix
            ( velodyne coordinates -> camera coordinates )

                [r_11 , r_12 , r_13 , t_x ]
        RT_  =  [r_21 , r_22 , r_23 , t_y ]
                [r_31 , r_32 , r_33 , t_z ]
        """
        #RT_ = np.concatenate((R_vc, T_vc), axis=1)

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

        """
        xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
                 [x_1 , x_2 , .. ]
        xyz_c =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
        """

        xyz_c = np.delete(xyz_v, 3, axis=0)


        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

        plt.scatter(xyz_c[0], xyz_c[1], cmap='viridis')  # Use xyz_c[0] as x, xyz_c[1] as y, and color by z
        plt.title("2D Scatter of 3D Velodyne Points")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Z')  # Optional: color bar for the Z values
        plt.show()     

        """
        xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
        ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
                 [s_1*x_1 , s_2*x_2 , .. ]
        xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]
                 [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
        """
        xy_i = xyz_c[::] / xyz_c[::][2]
        ans = np.delete(xy_i, 2, axis=0)

        return ans, xyz_c, c_

    def project_tracks(self, points):
        """ convert velodyne coordinates to camera image coordinates """

        # rough velodyne azimuth range corresponding to camera horizontal fov
        if self.__h_fov is None:
            self.__h_fov = (-50, 50)
        if self.__h_fov[0] < -50:
            self.__h_fov = (-50,) + self.__h_fov[1:]
        if self.__h_fov[1] > 50:
            self.__h_fov = self.__h_fov[:1] + (50,)

        # R_vc = Rotation matrix ( velodyne -> camera )
        # T_vc = Translation matrix ( velodyne -> camera )
        R_vc, T_vc = self.__calib_velo2cam()

        # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
        P_ = self.__calib_cam2cam(self.cali)

        """
        xyz_v - 3D velodyne points corresponding to h, v FOV limit in the velodyne coordinates
        c_    - color value(HSV's Hue vaule) corresponding to distance(m)

                 [x_1 , x_2 , .. ]
        xyz_v =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
                 [ 1  ,  1  , .. ]
        """
        xyz_v, c_ = self.__point_matrix(points)

        """
        RT_ - rotation matrix & translation matrix
            ( velodyne coordinates -> camera coordinates )

                [r_11 , r_12 , r_13 , t_x ]
        RT_  =  [r_21 , r_22 , r_23 , t_y ]
                [r_31 , r_32 , r_33 , t_z ]
        """
        RT_ = np.concatenate((R_vc, T_vc), axis=1)

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

        """
        xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
                 [x_1 , x_2 , .. ]
        xyz_c =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
        """
        xyz_c = np.delete(xyz_v, 3, axis=0)

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

        """
        xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
        ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
                 [s_1*x_1 , s_2*x_2 , .. ]
        xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]
                 [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
        """
        xy_i = xyz_c[::] / xyz_c[::][2]
        ans = np.delete(xy_i, 2, axis=0)

        return ans, xyz_c, c_ 
    
    def velo_projection_frame(self, h_fov=None, v_fov=None, x_range=None, y_range=None, z_range=None):
        """ print velodyne 3D points corresponding to camera 2D image """

        self.__v_fov, self.__h_fov = v_fov, h_fov
        self.__x_range, self.__y_range, self.__z_range = x_range, y_range, z_range
        velo_gen, cam_gen = self.lidar, self.image

        if velo_gen is None:
            raise ValueError("Velo data is not included in this class")
        if cam_gen is None:
            raise ValueError("Cam data is not included in this class")
        res, xyz, c_ = self.__velo_2_img_projection(velo_gen)
        return cam_gen, res, xyz, c_

    def __point_matrix(self, points):
        """ extract points corresponding to FOV setting """

        # filter in range points based on fov, x,y,z range setting
        self.__points_filter(points)

        # Stack arrays in sequence horizontally
        xyz_ = np.hstack((self.__x[:, None], self.__y[:, None], self.__z[:, None]))
        xyz_ = xyz_.T

        # stack (1,n) arrays filled with the number 1
        one_mat = np.full((1, xyz_.shape[1]), 1)
        xyz_ = np.concatenate((xyz_, one_mat), axis=0)

        # need dist info for points color
        color = self.__normalize_data(self.__d, min=1, max=70, scale=120, clip=True)

        return xyz_, color

    def __points_filter(self, points):
        """
        filter points based on h,v FOV and x,y,z distance range.
        x,y,z direction is based on velodyne coordinates
        1. azimuth & elevation angle limit check
        2. x,y,z distance limit
        """

        # upload current points
        self.__upload_points(points)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if self.__h_fov is not None and self.__v_fov is not None:
            if self.__h_fov[1] == self.__h_max and self.__h_fov[0] == self.__h_min and \
                            self.__v_fov[1] == self.__v_max and self.__v_fov[0] == self.__v_min:
                pass
            elif self.__h_fov[1] == self.__h_max and self.__h_fov[0] == self.__h_min:
                con = self.__hv_in_range(d, z, self.__v_fov, fov_type='v')
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
            elif self.__v_fov[1] == self.__v_max and self.__v_fov[0] == self.__v_min:
                con = self.__hv_in_range(x, y, self.__h_fov, fov_type='h')
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
            else:
                h_points = self.__hv_in_range(x, y, self.__h_fov, fov_type='h')
                v_points = self.__hv_in_range(d, z, self.__v_fov, fov_type='v')
                con = np.logical_and(h_points, v_points)
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
        else:
            pass

        if self.__x_range is None and self.__y_range is None and self.__z_range is None:
            pass
        elif self.__x_range is not None and self.__y_range is not None and self.__z_range is not None:
            # extract in-range points
            temp_x, temp_y = self.__3d_in_range(self.__x), self.__3d_in_range(self.__y)
            temp_z, temp_d = self.__3d_in_range(self.__z), self.__3d_in_range(self.__d)
            self.__x, self.__y, self.__z, self.__d = temp_x, temp_y, temp_z, temp_d
        else:
            raise ValueError("Please input x,y,z's min, max range(m) based on velodyne coordinates. ")

    def __upload_points(self, points):
        self.__x = points[:, 0]
        self.__y = points[:, 1]
        self.__z = points[:, 2]
        self.__d = np.sqrt(self.__x ** 2 + self.__y ** 2 + self.__z ** 2)

    def __3d_in_range(self, points):
        """ extract filtered in-range velodyne coordinates based on x,y,z limit """
        return points[np.logical_and.reduce((self.__x > self.__x_range[0], self.__x < self.__x_range[1], \
                                             self.__y > self.__y_range[0], self.__y < self.__y_range[1], \
                                             self.__z > self.__z_range[0], self.__z < self.__z_range[1]))]

    def __hv_in_range(self, m, n, fov, fov_type='h'):
        """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
            horizontal limit = azimuth angle limit
            vertical limit = elevation angle limit
        """

        if fov_type == 'h':
            return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                  np.arctan2(n, m) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                  np.arctan2(n, m) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def __normalize_data(self, val, min, max, scale, depth=False, clip=False):
        """ Return normalized data """
        if clip:
            # limit the values in an array
            np.clip(val, min, max, out=val)
        if depth:
            """
            print 'normalized depth value'
            normalize values to (0 - scale) & close distance value has high value. (similar to stereo vision's disparity map)
            """
            return (((max - val) / (max - min)) * scale).astype(np.uint8)
        else:
            """
            print 'normalized value'
            normalize values to (0 - scale) & close distance value has low value.
            """
            return (((val - min) / (max - min)) * scale).astype(np.uint8)

        
labels = '/home/lixion/rgbd/data/data_object_label_2/training/label_2'
images = '/home/lixion/rgbd/data/data_object_image_2/training/image_2'
lidar = '/home/lixion/rgbd/data/data_object_velodyne/training/velodyne'
cali = '/home/lixion/rgbd/data/data_object_calib/training/calib'

out = "."

# lidar data is in X, Y, Z, reflectivity 

kitti_data = data_proc(images, cali, lidar, labels, out)
