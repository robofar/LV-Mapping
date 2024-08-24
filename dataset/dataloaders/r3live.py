# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# 2024 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to mse, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE msE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import os

import cv2
import numpy as np

import yaml

from datetime import datetime

# R3Live dataset in kitti-like format
# https://github.com/ziv-lin/r3live_dataset

# camera parameters
# r3live_vio:
#    image_width: 1280
#    image_height: 1024
#    camera_intrinsic:
#       [863.4241, 0.0, 640.6808,
#       0.0,  863.4171, 518.3392,
#       0.0, 0.0, 1.0 ] 
#    camera_dist_coeffs: [-0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222]  #k1, k2, p1, p2, k3
#    # Fine extrinsic value. form camera-LiDAR calibration.
#    camera_ext_R:
#          [-0.00113207, -0.0158688, 0.999873,
#             -0.9999999,  -0.000486594, -0.00113994,
#             0.000504622,  -0.999874,  -0.0158682]
#    # camera_ext_t: [0.050166, 0.0474116, -0.0312415] 
#    camera_ext_t: [0,0,0] 


class R3LiveDataset:
    def __init__(self, data_dir, *_, **__):
        
        self.use_only_colorized_points = False

        self.livox_dir = os.path.join(data_dir, "livox_points", "data/")
        self.scan_files = sorted(glob.glob(self.livox_dir + "*.bin"))
        scan_count = len(self.scan_files)
        self.scan_ts = self.read_timestamps(os.path.join(data_dir, "livox_points", "timestamps.txt"))

        # camera
        self.img_dir = os.path.join(data_dir, "camera_image_color_compressed", "data/")
        self.img_files = sorted(glob.glob(self.img_dir + "*.png"))
        self.img_ts = self.read_timestamps(os.path.join(data_dir, "camera_image_color_compressed", "timestamps.txt"))

        # synchronize lidar and camera
        self.img_ts_sync, img_idx_sync = self.associate_img_to_lidar(self.scan_ts, self.img_ts)
        self.img_files = [self.img_files[i] for i in img_idx_sync] # get the synchronized files
        img_count = len(self.img_files)

        self.main_cam_name = "cam"

        H, W = 1024, 1280


        self.K_mats = {}
        self.T_c_l_mats = {}
       
        self.fx = 863.4241
        self.fy = 863.4171
        self.cx = 640.6808
        self.cy = 518.3392
        K_mat = np.eye(3)
        K_mat[0,0] = self.fx
        K_mat[0,2] = self.cx
        K_mat[1,1] = self.fy
        K_mat[1,2] = self.cy

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(height=H,
                                      width=W,
                                      fx=self.fx,
                                      fy=self.fy,
                                      cx=self.cx,
                                      cy=self.cy)


        self.K_mats[self.main_cam_name] = K_mat

        T_c_l = np.eye(4)
        T_c_l[:3,:3] = np.array([[-0.00113207, -0.0158688, 0.999873],
                                [-0.9999999,  -0.000486594, -0.00113994],
                                [0.000504622,  -0.999874,  -0.0158682]])

        self.T_c_l_mats[self.main_cam_name] = T_c_l

        # no gt pose available
        # gt_poses_file = os.path.join(data_dir, "gt.txt")
        # if os.path.exists(gt_poses_file):
        #     self.gt_poses, self.scan_timestamps = self.load_tum_format_gt_poses(gt_poses_file) 
        #     self.gt_poses = np.array(self.gt_poses)

    def __getitem__(self, idx):
        
        points = self.scans(idx)

        img = self.read_img(self.img_files[idx]) # just for vis here

        points_color = np.ones_like(points)

        # project to the image plane to get the corresponding color
        points_color, depth_map = self.project_points_to_cam(points, points_color, img, 
            self.T_c_l_mats[self.main_cam_name], self.K_mats[self.main_cam_name])

        if self.use_only_colorized_points:
            with_rgb_mask = (points_rgb[:, 3] == 0)
            points = points[with_rgb_mask]
            points_rgb = points_rgb[with_rgb_mask]

        # we skip the intensity here for now (and also the color mask)
        points = np.hstack((points[:,:3], points_color[:,:3]))

        img = np.concatenate((img, np.expand_dims(depth_map, axis=-1)), axis=-1) # 4 channels
        img_dict = {self.main_cam_name: img}

        frame_data = {"points": points, "point_ts": None, "img": img_dict}

        return frame_data

    def __len__(self):
        return len(self.scan_files)
    
    def read_timestamps(self, file_path):
        timestamps = []
        # pip install datetime
        with open(file_path, 'r') as file:
            for line in file:
                time_part = line.split("T")[1]
                # print(time_part)
                # Parse the time string into a datetime object
                time_obj = datetime.strptime(time_part[:-4], "%H:%M:%S.%f") # we ignore the nanosecond digits and count for the microsecond part 
                # Calculate the total number of seconds since 00:00:00
                time_seconds = (time_obj.hour * 3600) + (time_obj.minute * 60) + time_obj.second + (time_obj.microsecond / 1_000_000)
                timestamps.append(time_seconds)
                # print(time_seconds)
        timestamps = np.array(timestamps)   
        return timestamps
    
    def associate_img_to_lidar(self, lidar_ts, img_ts):
        # for each lidar ts, find the closest img_ts
        img_ts_associated = []
        img_associated_idx = []
        for i in range(lidar_ts.shape[0]):
            cur_lidar_ts = lidar_ts[i]
            j = np.argmin(np.abs(img_ts - cur_lidar_ts))
            img_ts_associated.append(img_ts[j])
            img_associated_idx.append(j)
        img_ts_associated = np.array(img_ts_associated)
        img_associated_idx = np.array(img_associated_idx, dtype=np.int32)
        # print(img_associated_idx)
        return img_ts_associated, img_associated_idx        

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)
        return points # N, 4
    
    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        # print(img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img) # as np array
        
        return img

    def load_tum_format_gt_poses(self, filename: str):
        """
        read pose file (with the tum format), support txt file
        # timestamp tx ty tz qx qy qz qw
        returns -> list, transformation before calibration transformation
        """
        from pyquaternion import Quaternion

        poses = []
        timestamps = []
        with open(filename, 'r') as file:
            first_line = file.readline().strip()
            
            # check if the first line contains any numeric characters
            # if contain, then skip the first line # timestamp tx ty tz qx qy qz qw
            if any(char.isdigit() for char in first_line):
                file.seek(0)
            
            for line in file: # read each line in the file 
                values = line.strip().split()
                if len(values) != 8 and len(values) != 9: 
                    print('Not a tum format pose file')
                    return None, None
                # some tum format pose file also contain the idx before timestamp
                idx_col =  len(values) - 8 # 0 or 1
                values = [float(value) for value in values]
                timestamps.append(values[idx_col])
                trans = np.array(values[1+idx_col:4+idx_col])
                quat = Quaternion(np.array([values[7+idx_col], values[4+idx_col], values[5+idx_col], values[6+idx_col]])) # w, i, j, k
                rot = quat.rotation_matrix
                # Build numpy transform matrix
                odom_tf = np.eye(4)
                odom_tf[0:3, 3] = trans
                odom_tf[0:3, 0:3] = rot
                poses.append(odom_tf)
        
        return poses, timestamps

    def read_calib_file(self, yaml_file_path: str) -> dict:

        # TODO: add for other sensors
        calib_dict = {}
        with open(yaml_file_path, 'r') as file:
            calib_dict = yaml.safe_load(file)

            # for the left camera
            cam_left_calib_dict = calib_dict["cam_l"]
            cam_left_intrinsic = cam_left_calib_dict["intrinsics"]

            # intrinsic
            self.fx = cam_left_intrinsic[0]
            self.fy = cam_left_intrinsic[1]
            self.cx = cam_left_intrinsic[2]
            self.cy = cam_left_intrinsic[3]

            self.K_mat = np.eye(3)
            self.K_mat[0,0]=self.fx
            self.K_mat[1,1]=self.fy
            self.K_mat[0,2]=self.cx
            self.K_mat[1,2]=self.cy

            # extrinsic
            self.T_l_c = np.array(cam_left_calib_dict['T_b'], dtype=np.float64) # T_l_c
            self.T_c_l = np.linalg.inv(self.T_l_c)

            self.T_c_l_mats[self.left_cam_name] = self.T_c_l
            self.K_mats[self.left_cam_name] = self.K_mat

        return calib_dict
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate

        # transfrom velodyne points to camera coordinate
        points_cam = np.matmul(T_c_l, points.T).T # N, 4
        points_cam = points_cam[:,:3] # N, 3

        # project to image space
        u, v, depth= self.persepective_cam2image(points_cam.T, K_mat) 
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        img_height, img_width, _ = np.shape(img)

        # prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width))
        depth_img = np.zeros((img_height, img_width, 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        
        # visualize points within 30 meters
        min_depth = 1.0
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid] = depth[mask]

        # print(np.shape(points_rgb))

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        return points_rgb, depth_map
    
    def persepective_cam2image(self, points, K_mat):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(K_mat[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int32)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int32)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth
