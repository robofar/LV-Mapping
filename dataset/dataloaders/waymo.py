import glob
import os
from pathlib import Path
from typing import List

import cv2
import json
import numpy as np
import open3d as o3d

# TODO: edit
class WaymoDataset:
    def __init__(self, data_dir, *_, **__):
        
        self.use_only_lidar_top = True 

        self.use_only_colorized_points = True

        self.lidar_top_topic_name = "lidar_TOP" # "lidar_horizontal_points"
        # self.lidar_v_topic_name = "os_v_points" # "lidar_vertical_points"
        

        self.cam_front_topic_name = "FRONT"  
        self.cam_front_left_topic_name = "FRONT_LEFT" 
        self.cam_front_right_topic_name = "FRONT_RIGHT"
        self.cam_side_left_topic_name = "SIDE_LEFT"
        self.cam_side_right_topic_name = "SIDE_RIGHT"
        self.cam_names = [self.cam_front_topic_name, self.cam_front_left_topic_name, self.cam_front_right_topic_name, self.cam_side_left_topic_name, self.cam_side_right_topic_name]

        self.K_mats = {}
        self.T_c_l_mats = {}


        self.lidar_dir = os.path.join(data_dir, "lidars")
        # top center lidar
        self.lidar_top_dir = os.path.join(self.lidar_dir, self.lidar_top_topic_name)
        self.lidar_top_files = sorted(glob.glob(self.lidar_top_dir + "/*.pcd"))

        self.img_dir = os.path.join(data_dir, "images") 
        # self.img_dir = os.path.join(data_dir, "images_ud") # already undistorted?

        # camera front
        self.img_front_dir = os.path.join(self.img_dir, self.cam_front_topic_name)
        self.img_front_files = sorted(glob.glob(self.img_front_dir + "/*.jpg"))
        # camera front left
        self.img_front_left_dir = os.path.join(self.img_dir, self.cam_front_left_topic_name)
        self.img_front_left_files = sorted(glob.glob(self.img_front_left_dir + "/*.jpg"))
        # camera front right
        self.img_front_right_dir = os.path.join(self.img_dir, self.cam_front_right_topic_name)
        self.img_front_right_files = sorted(glob.glob(self.img_front_right_dir + "/*.jpg"))
        # camera side left
        self.img_side_left_dir = os.path.join(self.img_dir, self.cam_side_left_topic_name)
        self.img_side_left_files = sorted(glob.glob(self.img_side_left_dir + "/*.jpg"))
        # camera side right
        self.img_side_right_dir = os.path.join(self.img_dir, self.cam_side_right_topic_name)
        self.img_side_right_files = sorted(glob.glob(self.img_side_right_dir + "/*.jpg"))

        self.transforms = self.read_transform(os.path.join(data_dir, "transform.json"))

        # print(self.lidar_top_extrinsic) # T_b_l

        # # vertical lidar
        # self.lidar_vertical_dir = os.path.join(data_dir, self.lidar_v_topic_name, "data/")
        # self.lidar_vertical_files = sorted(glob.glob(self.lidar_vertical_dir + "*.bin"))
        # self.lidar_vertical_ts = self.read_timestamps(os.path.join(data_dir, self.lidar_v_topic_name, "timestamps.txt"))

        # # camera left
        # self.img_left_dir = os.path.join(data_dir, self.cam_left_topic_name, "data/")
        # self.img_left_files = sorted(glob.glob(self.img_left_dir + "*.png"))
        # self.img_left_ts = self.read_timestamps(os.path.join(data_dir, self.cam_left_topic_name, "timestamps.txt"))

        # # camera right
        # self.img_right_dir = os.path.join(data_dir, self.cam_right_topic_name, "data/")
        # self.img_right_files = sorted(glob.glob(self.img_right_dir + "*.png"))
        # self.img_right_ts = self.read_timestamps(os.path.join(data_dir, self.cam_right_topic_name, "timestamps.txt"))



        # # camera rear
        # self.img_rear_dir = os.path.join(data_dir, self.cam_rear_topic_name, "data/")
        # self.img_rear_files = sorted(glob.glob(self.img_rear_dir + "*.png"))
        # self.img_rear_ts = self.read_timestamps(os.path.join(data_dir, self.cam_rear_topic_name, "timestamps.txt"))

        # synchronize lidar and camera (reference: lidar_horizontal_ts)
        # self.img_left_ts_sync, img_left_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.img_left_ts)
        # self.img_left_files = [self.img_left_files[i] for i in img_left_idx_sync] # get the synchronized files

        # self.img_right_ts_sync, img_right_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.img_right_ts)
        # self.img_right_files = [self.img_right_files[i] for i in img_right_idx_sync] # get the synchronized files

        # self.img_front_ts_sync, img_front_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.img_front_ts)
        # self.img_front_files = [self.img_front_files[i] for i in img_front_idx_sync] # get the synchronized files

        # self.img_rear_ts_sync, img_rear_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.img_rear_ts)
        # self.img_rear_files = [self.img_rear_files[i] for i in img_rear_idx_sync] # get the synchronized files

        # self.lidar_vertical_ts_sync, lidar_vertical_idx_sync = self.associate_img_to_lidar(self.lidar_horizontal_ts, self.lidar_vertical_ts)
        # self.lidar_vertical_files = [self.lidar_vertical_files[i] for i in lidar_vertical_idx_sync] # get the synchronized files

        # self.calibration_dict = self.read_calib_file(os.path.join(data_dir, "calib.yaml"))

        # no gt pose yet (TODO)

    def __getitem__(self, idx):
        
        points = self.read_point_cloud(self.lidar_top_files[idx])

        points_homo = np.hstack((points[:,:3], np.ones((np.shape(points)[0], 1))))

        points_homo_lidar_frame = points_homo @ np.linalg.inv(self.lidar_top_extrinsic).T # T_l_b.T

        points_homo[:,:3] = points_homo_lidar_frame[:,:3]

        points = points_homo

        # points_ts = self.get_timestamps()

        img_front = self.read_img(self.img_front_files[idx])
        img_front_left = self.read_img(self.img_front_left_files[idx])
        img_front_right = self.read_img(self.img_front_right_files[idx])
        img_side_left = self.read_img(self.img_side_left_files[idx])
        img_side_right = self.read_img(self.img_side_right_files[idx])

        img_dict = {self.cam_side_left_topic_name: img_side_left,
                    self.cam_front_left_topic_name: img_front_left, 
                    self.cam_front_topic_name: img_front,
                    self.cam_front_right_topic_name: img_front_right,
                    self.cam_side_right_topic_name: img_side_right}

        points_rgb = np.ones_like(points) # N,4, last channel for the mask
        
        for cam_name in list(img_dict.keys()):
            points_rgb = self.project_points_to_cam(points, points_rgb, img_dict[cam_name], 
                                                    self.T_c_l_mats[cam_name], self.K_mats[cam_name])

        if self.use_only_colorized_points:
            with_rgb_mask = (points_rgb[:, 3] == 0)
            points = points[with_rgb_mask]
            points_rgb = points_rgb[with_rgb_mask]

        # # # we skip the intensity here for now (and also the color mask)
        points = np.hstack((points[:,:3], points_rgb[:,:3]))

        frame_data = {"points": points, "point_ts": None, "img": img_dict}

        return frame_data

    def __len__(self):
        return len(self.lidar_top_files)
    
    # ouster-128 lidar (point-wise timestamp)
    @staticmethod
    def get_timestamps():
        timestamps = np.expand_dims(np.floor(np.arange(128 * 2048) / 128) / 2048, axis=1)
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
        points = np.asarray(o3d.io.read_point_cloud(scan_file).points, dtype=np.float64)
        return points.astype(np.float64)
    
    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        # print(img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img) # as np array
        
        return img

    def read_transform(self, json_file_path: str) -> dict:
        transforms_dict = {}

        with open(json_file_path, 'r') as infile: # load intrinsic json file
            transforms_dict = json.load(infile)

            sensor_params = transforms_dict["sensor_params"]
            lidar_top_params = sensor_params["lidar_TOP"]
            self.lidar_top_extrinsic = np.array(lidar_top_params["extrinsic"]) # T_b_l

            for cam_name in self.cam_names:
                cam_params = sensor_params[cam_name]
                self.K_mats[cam_name] = np.array(cam_params["camera_intrinsic"])
                cam_extrinsic = np.array(cam_params["extrinsic"]) # T_b_c
                self.T_c_l_mats[cam_name] = np.linalg.inv(cam_extrinsic) @ self.lidar_top_extrinsic # T_c_l

        return transforms_dict
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate # TODO, this would override the intensity part

        # transfrom velodyne points to camera coordinate
        points_cam = np.matmul(T_c_l, points.T).T # N, 4
        points_cam = points_cam[:,:3] # N, 3

        # project to image space
        u, v, depth= self.persepective_cam2image(points_cam.T, K_mat) 
        u = u.astype(int)
        v = v.astype(int)

        img_height, img_width, _ = np.shape(img)

        # prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width))
        # depth_img = np.zeros((img_height, img_width, 3))
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
        depth[depth==0] = -1e-5

        u = np.round(points_proj[:,0,:]/np.abs(depth))
        v = np.round(points_proj[:,1,:]/np.abs(depth))

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth
