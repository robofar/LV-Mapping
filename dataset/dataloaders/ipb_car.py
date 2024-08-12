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

# IPB car dataset in kitti-like format

class IPBCarDataset:
    def __init__(self, data_dir, *_, **__):
        
        self.use_only_lidar_h = True

        self.lidar_h_topic_name = "os_h_points" # "lidar_horizontal_points"
        self.lidar_v_topic_name = "os_v_points" # "lidar_vertical_points"
        self.cam_left_topic_name = "cam_left"   # "camera_left"
        self.cam_right_topic_name = "cam_right" # "camera_right"
        self.cam_front_topic_name = "cam_front" # "camera_front"
        self.cam_rear_topic_name = "cam_rear"   # "camera_rear"

        self.K_mats = {}
        self.T_c_l_mats = {}

        # horizontal lidar
        self.lidar_horizontal_dir = os.path.join(data_dir, self.lidar_h_topic_name, "data/")
        self.lidar_horizontal_files = sorted(glob.glob(self.lidar_horizontal_dir + "*.bin"))
        self.lidar_horizontal_ts = self.read_timestamps(os.path.join(data_dir, self.lidar_h_topic_name, "timestamps.txt"))

        # vertical lidar
        self.lidar_vertical_dir = os.path.join(data_dir, self.lidar_v_topic_name, "data/")
        self.lidar_vertical_files = sorted(glob.glob(self.lidar_vertical_dir + "*.bin"))
        self.lidar_vertical_ts = self.read_timestamps(os.path.join(data_dir, self.lidar_v_topic_name, "timestamps.txt"))

        # camera left
        self.img_left_dir = os.path.join(data_dir, self.cam_left_topic_name, "data/")
        self.img_left_files = sorted(glob.glob(self.img_left_dir + "*.png"))
        self.img_left_ts = self.read_timestamps(os.path.join(data_dir, self.cam_left_topic_name, "timestamps.txt"))

        # camera right
        self.img_right_dir = os.path.join(data_dir, self.cam_right_topic_name, "data/")
        self.img_right_files = sorted(glob.glob(self.img_right_dir + "*.png"))
        self.img_right_ts = self.read_timestamps(os.path.join(data_dir, self.cam_right_topic_name, "timestamps.txt"))

        # camera front
        self.img_front_dir = os.path.join(data_dir, self.cam_front_topic_name, "data/")
        self.img_front_files = sorted(glob.glob(self.img_front_dir + "*.png"))
        self.img_front_ts = self.read_timestamps(os.path.join(data_dir, self.cam_front_topic_name, "timestamps.txt"))

        # camera rear
        self.img_rear_dir = os.path.join(data_dir, self.cam_rear_topic_name, "data/")
        self.img_rear_files = sorted(glob.glob(self.img_rear_dir + "*.png"))
        self.img_rear_ts = self.read_timestamps(os.path.join(data_dir, self.cam_rear_topic_name, "timestamps.txt"))

        # synchronize lidar and camera (reference: lidar_horizontal_ts)
        self.img_left_ts_sync, img_left_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.img_left_ts)
        self.img_left_files = [self.img_left_files[i] for i in img_left_idx_sync] # get the synchronized files

        self.img_right_ts_sync, img_right_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.img_right_ts)
        self.img_right_files = [self.img_right_files[i] for i in img_right_idx_sync] # get the synchronized files

        self.img_front_ts_sync, img_front_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.img_front_ts)
        self.img_front_files = [self.img_front_files[i] for i in img_front_idx_sync] # get the synchronized files

        self.img_rear_ts_sync, img_rear_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.img_rear_ts)
        self.img_rear_files = [self.img_rear_files[i] for i in img_rear_idx_sync] # get the synchronized files

        self.lidar_vertical_ts_sync, lidar_vertical_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.lidar_vertical_ts)
        self.lidar_vertical_files = [self.lidar_vertical_files[i] for i in lidar_vertical_idx_sync] # get the synchronized files

        self.calibration_dict = self.read_calib_file(os.path.join(data_dir, "calib.yaml"))

        # no gt pose yet (TODO)

    def __getitem__(self, idx):
        
        points = self.read_point_cloud(self.lidar_horizontal_files[idx]) # lidar_h_points
        points_ts = self.get_timestamps()
        
        if not self.use_only_lidar_h:
            lidar_v_points = self.read_point_cloud(self.lidar_vertical_files[idx])

            lidar_v_points_homo = np.hstack((lidar_v_points[:,:3], np.ones((np.shape(lidar_v_points)[0], 1))))

            lidar_v_points_h_frame = lidar_v_points_homo @ self.T_lv_lh.T
            lidar_v_points[:,:3] = lidar_v_points_h_frame[:,:3]

            points = np.concatenate((points, lidar_v_points), axis=0) # 2N, 4

            points_ts = np.tile(points_ts, (2, 1)) # 2N, 1

        img_left = self.read_img(self.img_left_files[idx])
        img_right = self.read_img(self.img_right_files[idx])
        img_front = self.read_img(self.img_front_files[idx])
        img_rear = self.read_img(self.img_rear_files[idx])

        img_dict = {self.cam_front_topic_name: img_front, self.cam_left_topic_name: img_left, 
                    self.cam_rear_topic_name: img_rear, self.cam_right_topic_name: img_right}

        points_rgb = np.ones_like(points) # N,4, last channel for the mask
        
        for cam_name in list(img_dict.keys()):
            # calib seems to be somehow wrong, figure it out. TODO
            points_rgb = self.project_points_to_cam(points, points_rgb, img_dict[cam_name], self.T_c_l_mats[cam_name], self.K_mats[cam_name])

        # # we skip the intensity here for now (and also the color mask)
        points = np.hstack((points[:,:3], points_rgb[:,:3]))

        frame_data = {"points": points, "point_ts": points_ts, "img": img_dict}

        return frame_data

    def __len__(self):
        return len(self.lidar_horizontal_files)
    
    # ouster-128 lidar (point-wise timestamp)
    @staticmethod
    def get_timestamps():
        timestamps = np.expand_dims(np.floor(np.arange(128 * 2048) / 128) / 2048, axis=1)
        return timestamps
    
    # frame timestamp
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

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)
        return points # N, 4
    
    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        # print(img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img) # as np array
        
        return img

    def read_calib_file(self, yaml_file_path: str) -> dict:

        # TODO: add for other sensors
        calib_dict = {}
        with open(yaml_file_path, 'r') as file:
            calib_dict = yaml.safe_load(file)

            self.T_bacs2opencv = np.asarray(
                [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            ) # we need to convert to the opencv camera frame

            lidar_h_calib = calib_dict["os_hpoints"]

            lidar_v_calib = calib_dict["os_vpoints"]
            self.T_lv_lh = np.array(lidar_v_calib["os_hpoints"])

            camera_front_calib = calib_dict[self.cam_front_topic_name]
            self.K_mats[self.cam_front_topic_name] = np.array(camera_front_calib["K"])
            self.T_c_l_mats[self.cam_front_topic_name] = self.T_bacs2opencv @ np.array(lidar_h_calib[self.cam_front_topic_name])

            # print(self.K_mats[self.cam_front_topic_name])
            # print(self.T_c_l_mats[self.cam_front_topic_name])

            camera_rear_calib = calib_dict[self.cam_rear_topic_name]
            self.K_mats[self.cam_rear_topic_name] = np.array(camera_rear_calib["K"])
            self.T_c_l_mats[self.cam_rear_topic_name] = self.T_bacs2opencv @ np.array(lidar_h_calib[self.cam_rear_topic_name])

            # print(self.K_mats[self.cam_rear_topic_name])
            # print(self.T_c_l_mats[self.cam_rear_topic_name])

            camera_left_calib = calib_dict[self.cam_left_topic_name]
            self.K_mats[self.cam_left_topic_name] = np.array(camera_left_calib["K"])
            self.T_c_l_mats[self.cam_left_topic_name] = self.T_bacs2opencv @ np.array(lidar_h_calib[self.cam_left_topic_name])

            camera_right_calib = calib_dict[self.cam_right_topic_name]
            self.K_mats[self.cam_right_topic_name] = np.array(camera_right_calib["K"])
            self.T_c_l_mats[self.cam_right_topic_name] = self.T_bacs2opencv @ np.array(lidar_h_calib[self.cam_right_topic_name])

        return calib_dict
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate # TODO, this would override the intensity part

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
        # overwrite the points_rgb
        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        return points_rgb
    
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
