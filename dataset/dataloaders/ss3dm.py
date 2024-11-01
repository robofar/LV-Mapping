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

import open3d as o3d
import cv2
import numpy as np
import pickle

import yaml

from datetime import datetime

from utils.tools import get_time

# SS3DM Dataset: Benchmarking Street-View Surface Reconstruction with a Synthetic 3D Mesh Dataset (NeurIPS 2024) 
# https://ss3dm.top/
# https://arxiv.org/pdf/2410.21739

class SS3DMDataset:
    def __init__(self, data_dir, cam_name: str, *_, **__):
        
        self.load_img = False # default

        self.use_only_colorized_points = False
        
        self.use_only_lidar_h = False # use lidar_h or both (lidar_h + lidar_v)

        self.min_lidar_radius_m = 1.0

        self.lidar_list_all = ["TOP", "FRONT", "LEFT", "REAR", "RIGHT"]
        # NOTE that the LiDAR point cloud simulated from this dataset has a noise of 0.1m std

        self.cam_list_all = ["FRONT", "FRONT_LEFT", "BACK_LEFT", "BACK", "BACK_RIGHT", "FRONT_RIGHT"]

        self.main_lidar_name = "TOP"

        if cam_name in self.cam_list_all: 
            self.main_cam_only = True
            self.main_cam_name = cam_name
            self.cam_list = [self.main_cam_name]
            print("Use {} camera only".format(cam_name))
        else: 
            # use all the cameras
            self.main_cam_only = False
            self.main_cam_name = self.cam_list_all[0]
            self.cam_list = self.cam_list_all
            print("Use all the cameras")

        self.lidar_list = self.lidar_list_all

        scenario_file_path = os.path.join(data_dir, "scenario.pt")
        # print(scenario_file_path)

        with open(scenario_file_path, 'rb') as file:
            scenario_data_all = pickle.load(file) # ok done
        # print(list(scenario_data_all.keys()))

        scenario_data = scenario_data_all["observers"]

        self.img_files = {}
        self.lidar_files = {}
        self.K_mats = {}
        self.T_c_l_mats = {}

        self.cam_widths = {}
        self.cam_heights = {}

        # img_size: 2064x1024
        H, W = 1080, 1920 
        fx = fy = 672.19923668
        cx, cy = 960, 540
        K_mat = np.eye(3)
        K_mat[0,0] = fx
        K_mat[0,2] = cx
        K_mat[1,1] = fy 
        K_mat[1,2] = cy

        # c2w needs to be in OpenCV coordinate system, so this matrix is required Opencv/gl conversion
        self.T_f_v = np.array([[0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])

        self.T_l_v = np.array([[0, 0, -1, 0],
                                [1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])

        ego_car_data = (scenario_data["ego_car"])["data"]
        self.T_w_v = ego_car_data["v2w"] # N,4,4
        self.gt_poses = np.matmul(self.T_w_v, np.linalg.inv(self.T_l_v)) # T_w_l

        lidar_name_full = "lidar_{}".format(self.main_lidar_name)
        lidar_meta_data = scenario_data[lidar_name_full]
        self.T_v_wl = (lidar_meta_data["data"])["l2v"] # N,4,4 # l here is another world frame (lidar world frame, which still has some difference from w)

        for cam_name in self.cam_list:
            cur_cam_dir = os.path.join(data_dir, "images", "camera_{}/".format(cam_name))
            cur_img_files = sorted(glob.glob(cur_cam_dir + "*.jpg"))
        
            self.img_files[cam_name] = cur_img_files

            self.cam_widths[cam_name] = W
            self.cam_heights[cam_name] = H
            self.K_mats[cam_name] = K_mat

            cam_name_full = "camera_{}".format(cam_name)
            cam_meta_data = scenario_data[cam_name_full]

            cur_cam_T_v_c = ((cam_meta_data["data"])["c2v"])[0]

            self.T_c_l_mats[cam_name] = np.linalg.inv(self.T_l_v @ cur_cam_T_v_c)  # still wrong

            # print(self.T_c_l_mats[cam_name])

        for lidar_name in self.lidar_list:
            cur_lidar_dir = os.path.join(data_dir, "lidars", "lidar_{}/".format(lidar_name))
            cur_lidar_files = sorted(glob.glob(cur_lidar_dir + "*.npz"))
        
            self.lidar_files[lidar_name] = cur_lidar_files

            # all the lidar l2v are the same, lidar point cloud already under a world frame

        # self.gt_poses = self.lidar_poses[self.main_lidar_name]

        # read from the scenario_file_path
        
        # # main cam parameters
        # self.intrinsic = o3d.camera.PinholeCameraIntrinsic()

        # # NOTE for the rear camera, from about h=920 is the ego car's tail

        # self.intrinsic.set_intrinsics(
        #                             height=H,
        #                             width=W,
        #                             fx=self.K_mats[self.main_cam_name][0,0],
        #                             fy=self.K_mats[self.main_cam_name][1,1],
        #                             cx=self.K_mats[self.main_cam_name][0,2],
        #                             cy=self.K_mats[self.main_cam_name][1,2])

        # self.extrinsic = self.T_c_l_mats[self.main_cam_name] # T_c_l

        self.mono_depth_for_high_z: bool = True # complete the low Z part 


    def __getitem__(self, idx): 

        points = np.empty((0,4))

        for lidar_name in self.lidar_list:
            
            cur_lidar_file = self.lidar_files[lidar_name][idx]

            lidar_data = np.load(cur_lidar_file)

            # here already under world frame
            cur_lidar_xyz = lidar_data['rays_o'] + lidar_data['rays_d'] * lidar_data['ranges'].reshape((-1, 1)) # xyz # already in world frame?

            cur_lidar_xyz_homo = np.hstack((cur_lidar_xyz, np.ones((np.shape(cur_lidar_xyz)[0], 1))))

            # if lidar_name == self.main_lidar_name:
            #     print((lidar_data['rays_o'])[0])

            # cur_lidar_T_w_l = (self.lidar_poses[lidar_name])[idx] # 4,4 # F**K, this T_v_l has nothing to do with the lidar frame, I felt really confused
            # cur_lidar_T_l_w = np.linalg.inv(cur_lidar_T_w_l) # 4,4

            cur_T_l_wl = self.T_l_v @ self.T_v_wl[idx] # this is back to the unified lidar frame

            # print(cur_T_l_wl[:3,3])

            # convert to vehicle frame
            cur_lidar_xyz_l_frame = (cur_lidar_xyz_homo @ cur_T_l_wl.T) # v frame actually as front camera # N, 4

            points = np.concatenate((points, cur_lidar_xyz_l_frame), axis=0) # N, 4

            # valid_mask = ~np.all(np.abs(points[:,:3]) < self.min_lidar_radius_m, axis=1) 
            # points = points[valid_mask]

        if self.load_img:
            
            img_dict = {}
            depth_img_dict = {}

            points_rgb = np.ones_like(points) # N,4, last channel for the mask

            for cam_name in self.cam_list:
                
                # tic_0 = get_time()
                # slow, but would be hard to speed up

                # print("{} ts: {}".format(cam_name, self.img_ts[cam_name][idx]))

                cur_img_file = self.img_files[cam_name][idx]

                # print(cur_img_file)

                img_cam = self.read_img(cur_img_file) 
                
                # toc_1 = get_time()
                
                # TODO: a bit slow, try to speed it up
                # we do not to do this here anymore
                # FIXME: not used now
                # points_rgb, depth_map = self.project_points_to_cam(points, points_rgb, img_cam, self.T_c_l_mats[cam_name], self.K_mats[cam_name])

                # toc_1 = get_time()

                # depth_img_dict[cam_name] = depth_map # H, W, 1
                
                img_dict[cam_name] = img_cam # H, W, 3

            # we skip the intensity here for now (and also the color mask)
            points = np.hstack((points[:,:3], points_rgb[:,:3]))

            frame_data = {"points": points, "img": img_dict}
            
        else:
            frame_data = {"points": points}

        return frame_data

        # # tic_read_pc = get_time()

        # h_lidar_ref_ts = self.lidar_horizontal_ts[idx] # unit: s
        # # print("H Lidar ts: {}".format(h_lidar_ref_ts))

        # # TODO: read ply is a bot too slow, try to use *.bin (done), but for *.bin, there some problem of the timestamp loading
        # # read bin is very fast
        # points, point_ts = self.read_point_cloud_ply(self.lidar_horizontal_files[idx]) # lidar_h_points

        # # toc_read_pc = get_time()
        # # print("pc reading time (ms):" , (toc_read_pc -tic_read_pc)*1e3)

        # valid_mask = ~np.all(np.abs(points[:,:3]) < self.min_lidar_radius_m, axis=1) 
        # points = points[valid_mask]
        # point_ts = point_ts[valid_mask] # unit: s

        # # print(point_ts)
        
        # point_lidar_idx = np.zeros_like(point_ts) # all zero

        # lidar_h_point_count = np.shape(points)[0]
        # # from 0 to lidar_h_point_count-1: lidar_h
        # # from lidar_h_point_count to end: lidar_v

        # # TODO: it's not correct to firstly combine the two point clouds are then apply undistortion
        # # FIXME: you need to apply a T_lv_lh to the points from the vertical liadr during the undistortion
        # # For multiple-LiDAR cases, you still have something to deal with
        # if not self.use_only_lidar_h:
            
        #     v_lidar_ref_ts = self.lidar_vertical_ts[idx]
        #     # print("V Lidar ts: {}".format(v_lidar_ref_ts))

        #     lidar_v_points, lidar_v_points_ts = self.read_point_cloud_ply(self.lidar_vertical_files[idx])
        #     # print(np.shape(lidar_v_points)[0])

        #     lidar_v_points_homo = np.hstack((lidar_v_points[:,:3], np.ones((np.shape(lidar_v_points)[0], 1))))

        #     lidar_v_points_h_frame = lidar_v_points_homo @ self.T_lv_lh.T
        #     lidar_v_points[:,:3] = lidar_v_points_h_frame[:,:3]

        #     valid_mask = ~np.all(np.abs(lidar_v_points[:,:3]) < self.min_lidar_radius_m, axis=1) 
        #     lidar_v_points = lidar_v_points[valid_mask]
        #     lidar_v_points_ts = lidar_v_points_ts[valid_mask]

        #     # this is actually not a very big number, consider it later FIXME
        #     # lidar_v_points_ts += (v_lidar_ref_ts - h_lidar_ref_ts) # convert to the same reference time of the horizontal lidar

        #     added_ref_ts_ratio = (h_lidar_ref_ts - v_lidar_ref_ts) / 0.1
        #     self.ts_ref_ratio_diffs[0] = added_ref_ts_ratio
        #     # print(added_ref_ts_ratio)

        #     lidar_v_point_lidar_idx = np.ones_like(lidar_v_points_ts)

        #     points = np.concatenate((points, lidar_v_points), axis=0) # K, 4
        #     point_ts = np.concatenate((point_ts, lidar_v_points_ts), axis=0)
        #     point_lidar_idx = np.concatenate((point_lidar_idx, lidar_v_point_lidar_idx), axis=0)

        # if self.load_img:
        #     img_dict = {}
        #     depth_img_dict = {}

        #     points_rgb = np.ones_like(points) # N,4, last channel for the mask

        #     for cam_name in self.cam_list:
                
        #         # tic_0 = get_time()
        #         # slow, but would be hard to speed up

        #         # print("{} ts: {}".format(cam_name, self.img_ts[cam_name][idx]))

        #         cur_img_file = self.img_files[cam_name][idx]

        #         img_file_split = cur_img_file.split("/")
        #         img_file_split[-2] = "data_undistorted" # data --> data_undistorted
        #         cur_img_file_distorted = "/".join(img_file_split)

        #         if os.path.exists(cur_img_file_distorted):
        #             undistort_on = False # if already exists the undistorted file, then use it
        #             cur_img_file = cur_img_file_distorted
        #         else:
        #             undistort_on = True # otherwise, do the distortion and save the file

        #         # print(cur_img_file)

        #         img_cam = self.read_img(cur_img_file, undistort_on, self.K_mats[cam_name], self.dist_coeffs[cam_name]) 
                
        #         # toc_1 = get_time()
                
        #         # TODO: a bit slow, try to speed it up
        #         # we do not to do this here anymore
        #         # FIXME: not used now
        #         # points_rgb, depth_map = self.project_points_to_cam(points, points_rgb, img_cam, self.T_c_l_mats[cam_name], self.K_mats[cam_name])

        #         # toc_1 = get_time()

        #         # depth_img_dict[cam_name] = depth_map # H, W, 1
                
        #         img_dict[cam_name] = img_cam # H, W, 3
                
        #         # print("img reading time (ms):" , (toc_0 - tic_0)*1e3)
        #         # print("pc colorize time (ms):" , (toc_1 - toc_0)*1e3)

        #     # FIXME
        #     # if self.use_only_colorized_points:
        #     #     with_rgb_mask = (points_rgb[:, 3] == 0)
        #     #     points = points[with_rgb_mask]
        #     #     points_rgb = points_rgb[with_rgb_mask]
        #     #     point_ts = point_ts[with_rgb_mask]
        #     #     point_lidar_idx = point_lidar_idx[with_rgb_mask]

        #     # we skip the intensity here for now (and also the color mask)
        #     points = np.hstack((points[:,:3], points_rgb[:,:3]))

        #     # print(point_ts) # correct

        #     frame_data = {"points": points, "point_ts": point_ts, "point_lidar_idx": point_lidar_idx, "img": img_dict}
        #     # frame_data = {"points": points, "point_ts": point_ts, "point_lidar_idx": point_lidar_idx, "img": img_dict, "depth": depth_img_dict}
        # else:
        #     frame_data = {"points": points, "point_ts": point_ts, "point_lidar_idx": point_lidar_idx}

        # return frame_data

    def __len__(self):
        return len(self.lidar_files[self.lidar_list[0]])
    
    # ouster-128 lidar (point-wise timestamp)
    # this does not work well
    # @staticmethod
    # def get_timestamps(v_beam=128, h_beam=2048):
    #     timestamps = np.expand_dims(np.floor(np.arange(v_beam * h_beam) / v_beam) / h_beam, axis=1)
    #     return timestamps
    
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

    # # read bin format
    # def read_point_cloud(self, scan_file: str):
    #     points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4)).astype(np.float64)
    #     return points[:, :4] # N, 4
    
    # read ply format
    def read_point_cloud_ply(self, scan_file: str):

        # tic_read_pc = get_time()

        pc_load = o3d.t.io.read_point_cloud(scan_file)
        pc_load = {k: v.numpy() for k, v in pc_load.point.items()}

        keys = list(pc_load.keys())
        # print("available attributes:", keys)
        
        points = pc_load["positions"]

        assert "t" in keys, "The point cloud ply file must have the t field for point wise timestamp"

        ts = pc_load["t"] # already in seconds

        point_count = np.shape(points)[0]
        points = np.concatenate((points, np.ones((point_count, 1))), axis=1) # N, 4
        
        # toc_read_pc = get_time()
        # print("pc reading time ply (ms): {:.2f}".format((toc_read_pc -tic_read_pc)*1e3))

        return points, ts

    def read_img(self, img_file: str, undistort_on: bool = False, K_mat = None, dist_coeffs = None):
        
        tic_0 = get_time()
        img = cv2.imread(img_file)
        toc_0 = get_time()

        # print("img reading time (ms):" , (toc_0 -tic_0)*1e3)

        # apply undistortion:
        # tic_undistort = get_time()
        if undistort_on and K_mat is not None and dist_coeffs is not None:
            img = cv2.undistort(img, K_mat, dist_coeffs)
            img_file_split = img_file.split("/")
            img_file_split[-2] = "data_undistorted" # data --> data_undistorted # ugly fix here
            out_img_file = "/".join(img_file_split)
            # print(out_img_file)

            cv2.imwrite(out_img_file, img)

        # toc_undistort = get_time() 

        # print("Image undistortion time (ms):", (toc_undistort-tic_undistort)*1e3) # 12 ms / each 
        # for 4 imgs, takes about 50ms, better to do this offline
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def read_calib_file(self, yaml_file_path: str) -> dict:

        calib_dict = {}
        with open(yaml_file_path, 'r') as file:
            calib_dict = yaml.safe_load(file)

            lidar_h_calib = calib_dict["lidarhorizontalpoints"]
            lidar_v_calib = calib_dict["lidarverticalpoints"]
            T_cf_lh = np.array(lidar_h_calib["extrinsics"])
            T_cf_lv = np.array(lidar_v_calib["extrinsics"])
            self.T_lv_lh = np.linalg.inv(T_cf_lv) @ T_cf_lh
            self.T_l_lm_mats.append(self.T_lv_lh)

            for cam_name in self.cam_list:
                cur_cam_calib_name = "camera{}image_raw".format(cam_name)
                cur_camera_calib = calib_dict[cur_cam_calib_name]
                self.K_mats[cam_name] = np.array(cur_camera_calib["K"])
                self.dist_coeffs[cam_name] = np.array(cur_camera_calib["distortion_coeff"])
                self.T_c_l_mats[cam_name] = np.linalg.inv(np.array(cur_camera_calib["extrinsics"])) @ T_cf_lh 

        return calib_dict
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        tic_0 = get_time()

        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate

        # transfrom velodyne points to camera coordinate
        points_cam = np.matmul(T_c_l, points.T).T # N, 4
        points_cam = points_cam[:,:3] # N, 3

        toc_0 = get_time() # fast

        # project to image space
        u, v, depth = self.persepective_cam2image(points_cam.T, K_mat) 
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        img_height, img_width, _ = np.shape(img)

        toc_1 = get_time() # relatively fast

        # prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width, 1)) # H, W, 1
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        
        # visualize points within 30 meters
        min_depth = 1.0
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid, 0] = depth[mask]

        # print(np.shape(points_rgb))

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        toc_2 = get_time() # slow

        # print("colorize time 1 (ms):" , (toc_0 -tic_0)*1e3)
        # print("colorize time 2 (ms):" , (toc_1 -toc_0)*1e3)
        # print("colorize time 3 (ms):" , (toc_2 -toc_1)*1e3)

        return points_rgb, depth_map
    
    def persepective_cam2image(self, points, K_mat):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(K_mat[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth


    def read_kitti_format_poses(self, filename: str):
        """
        read pose file (with the kitti format)
        returns -> np.array, transformation before calibration transformation
        if the format is incorrect, return None
        """
        poses = []
        with open(filename, 'r') as file:            
            for line in file:
                values = line.strip().split()
                if len(values) < 12: # FIXME: > 12 means maybe it's a 4x4 matrix
                    print('Not a kitti format pose file')
                    return None

                values = [float(value) for value in values]
                pose = np.zeros((4, 4))
                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0
                poses.append(pose)
        
        return np.array(poses)

