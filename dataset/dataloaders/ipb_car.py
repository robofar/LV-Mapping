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

import yaml

from datetime import datetime

from utils.tools import get_time

# for the new test data

class IPBCarDataset:
    def __init__(self, data_dir, cam_name: str, *_, **__):
        
        # for cam_name, select from "left", "right", "front", "rear" all "all"
        self.use_only_colorized_points = True
        
        self.use_only_lidar_h = True # use lidar_h or both (lidar_h + lidar_v)

        self.lidar_h_topic_name = "os_h" 
        self.lidar_v_topic_name = "os_v" 
        
        self.cam_left_topic_name = "left"  
        self.cam_right_topic_name = "right" 
        self.cam_front_topic_name = "front" 
        self.cam_rear_topic_name = "rear"

        cam_list_all = [self.cam_left_topic_name, self.cam_right_topic_name, self.cam_front_topic_name, self.cam_rear_topic_name]

        if cam_name in cam_list_all: 
            self.main_cam_only = True
            self.main_cam_name = cam_name
            self.cam_list = [self.main_cam_name]
            print("Use {} camera only".format(cam_name))
        else: 
            # use all the cameras
            self.main_cam_only = False
            self.main_cam_name = self.cam_front_topic_name
            self.cam_list = cam_list_all
            print("Use all the cameras")

        self.img_files = {}
        self.K_mats = {}
        self.dist_coeffs = {}
        self.T_c_l_mats = {}

        # horizontal lidar
        self.lidar_horizontal_dir = os.path.join(data_dir, "lidar", self.lidar_h_topic_name, "points/")
        self.lidar_horizontal_files = sorted(glob.glob(self.lidar_horizontal_dir + "*.ply"))

        # img_size: 2064x1024

        for cam_name in self.cam_list:
            cur_cam_dir = os.path.join(data_dir, "camera", cam_name, "image_undistorted/")
            if os.path.exists(cur_cam_dir):
                self.img_already_undistorted = True
            else:
                cur_cam_dir = os.path.join(data_dir, "camera", cam_name, "image_raw/")
                self.img_already_undistorted = False
            cur_img_files = sorted(glob.glob(cur_cam_dir + "*.png"))
            self.img_files[cam_name] = cur_img_files

        # read calib
        self.calibration_dict = self.read_calib_file(os.path.join(data_dir, "calibration", "results.yaml"))

        # read reference poses (by Louis)
        self.gt_poses = np.load(os.path.join(data_dir, "poses", "latest.npy")) # is this the pose in LiDAR frame? (ask louis)
        # print(self.gt_poses)
        
        # main cam parameters
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        H, W = 1024, 2064
        self.intrinsic.set_intrinsics(
                                    height=H,
                                    width=W,
                                    fx=self.K_mats[self.main_cam_name][0,0],
                                    fy=self.K_mats[self.main_cam_name][1,1],
                                    cx=self.K_mats[self.main_cam_name][0,2],
                                    cy=self.K_mats[self.main_cam_name][1,2])

        self.extrinsic = self.T_c_l_mats[self.main_cam_name] # T_c_l

        self.cam_widths = {self.main_cam_name: W}
        self.cam_heights = {self.main_cam_name: H}
        
        self.mono_depth_for_high_z: bool = False # complete the low Z part 


    def __getitem__(self, idx):
        # TODO: read ply is a bot too slow, try to use *.bin
        points = self.read_point_cloud(self.lidar_horizontal_files[idx]) # lidar_h_points
        point_ts = self.get_timestamps()
        
        # if not self.use_only_lidar_h:
        #     lidar_v_points = self.read_point_cloud(self.lidar_vertical_files[idx])

        #     lidar_v_points_homo = np.hstack((lidar_v_points[:,:3], np.ones((np.shape(lidar_v_points)[0], 1))))

        #     lidar_v_points_h_frame = lidar_v_points_homo @ self.T_lv_lh.T
        #     lidar_v_points[:,:3] = lidar_v_points_h_frame[:,:3]

        #     points = np.concatenate((points, lidar_v_points), axis=0) # 2N, 4

        #     point_ts = np.tile(point_ts, (2, 1)) # 2N, 1

        valid_mask = ~np.all(points[:,:3] == 0, axis=1) 
        points = points[valid_mask]
        point_ts = point_ts[valid_mask]

        points_rgb = np.ones_like(points) # N,4, last channel for the mask
        undistort_on = not self.img_already_undistorted
        img_dict = {}
        for cam_name in self.cam_list:
            img_cam = self.read_img(self.img_files[cam_name][idx], undistort_on, self.K_mats[cam_name], self.dist_coeffs[cam_name]) 
            # TODO: to slow, try to speed it up
            points_rgb, depth_map = self.project_points_to_cam(points, points_rgb, img_cam, self.T_c_l_mats[cam_name], self.K_mats[cam_name])
            img_cam = np.concatenate((img_cam, np.expand_dims(depth_map, axis=-1)), axis=-1) # 4 channels
            img_dict[cam_name] = img_cam

        if self.use_only_colorized_points:
            with_rgb_mask = (points_rgb[:, 3] == 0)
            points = points[with_rgb_mask]
            points_rgb = points_rgb[with_rgb_mask]
            point_ts = point_ts[with_rgb_mask]

        # we skip the intensity here for now (and also the color mask)
        points = np.hstack((points[:,:3], points_rgb[:,:3]))

        frame_data = {"points": points, "point_ts": point_ts, "img": img_dict}

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

    # def read_point_cloud(self, scan_file: str):
    #     points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)
    #     return points # N, 4
    
    def read_point_cloud(self, scan_file: str):
        pcd = o3d.io.read_point_cloud(scan_file)
        points = np.array(pcd.points, dtype=np.float64) # N, 3
        point_count = np.shape(points)[0]
        points = np.concatenate((points, np.ones((point_count, 1))), axis=1) # N, 4
        return points

    def read_img(self, img_file: str, undistort_on: bool = False, K_mat = None, dist_coeffs = None):
        img = cv2.imread(img_file)

        # apply undistortion:
        # tic_undistort = get_time()
        if undistort_on and K_mat is not None and dist_coeffs is not None:
            img = cv2.undistort(img, K_mat, dist_coeffs)
            cv2.imwrite(img_file.replace("image_raw", "image_undistorted"), img)

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
            T_cf_l = np.array(lidar_h_calib["extrinsics"])

            for cam_name in self.cam_list:
                cur_cam_calib_name = "camera{}image_raw".format(cam_name)
                cur_camera_calib = calib_dict[cur_cam_calib_name]
                self.K_mats[cam_name] = np.array(cur_camera_calib["K"])
                self.dist_coeffs[cam_name] = np.array(cur_camera_calib["distortion_coeff"])
                self.T_c_l_mats[cam_name] = np.linalg.inv(np.array(cur_camera_calib["extrinsics"])) @ T_cf_l 

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
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth

# image distortion
PX_OFFSET_128 = 32 * [48, 32, 16, 0]  # Check os*.json
PX_OFFSET_32 = 32 * [16]
color_only_visible = True

def getLUT(points, px_offset=PX_OFFSET_128):
    H = points.shape[0]
    W = points.shape[1]
    # row, col = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    row, col = np.ogrid[:H, :W]
    col = col - np.array(px_offset)[:, None]
    return row, col

def tccApplyLut(coords, lut):
    # Apply lut
    coords_distorted = np.asarray(coords, dtype="float32")
    coords_int = np.asarray(np.round(coords), dtype="int")
    coords_int[coords_int[:, 0] < 0, 0] = 0
    coords_int[coords_int[:, 1] < 0, 1] = 0
    coords_int[coords_int[:, 0] >= lut.shape[0], 0] = lut.shape[0] - 1
    coords_int[coords_int[:, 1] >= lut.shape[1], 1] = lut.shape[1] - 1

    coords_distorted[:, 0] += lut[coords_int[:, 0], coords_int[:, 1], 1]
    coords_distorted[:, 1] += lut[coords_int[:, 0], coords_int[:, 1], 0]
    return coords_distorted


def tccReadLut(lutfilename: str):
    """Read lookup table create by calibration software tcc.

    Args:
      lutfilename: Complete filename of lookup table, usual ending is .lut
                   and .ilut

    Returns:
      the lookup table as a numpy array of size (no_of_image_rows, no_of_image_column, 2)
      [:,:,0] is the offset in x (or column) direction, [:,:,1] the offset in y (or row) direction
    """
    with open(lutfilename, "rt") as fstream:
        # Read Header with the identifier "distortiontable"
        line = fstream.readline()
        while line[0] == "#":
            line = fstream.readline()

        if line[0:15] != "distortiontable":
            print(
                "ERROR: given filename %s seems not to be a tcc lookup table, wrong header !"
                % lutfilename
            )
            # return

        # Line with basex basey dimx dimy
        line = fstream.readline()
        while line[0] == "#":
            line = fstream.readline()

        basex, basey, dimx, dimy = np.asarray(line.split(), dtype="int")

        # Now the actual lut values
        lut = np.loadtxt(fstream, comments="#", dtype="float32")

        lut = np.reshape(lut, (dimy, dimx, 2))

        return lut

def project_lidar(K, coors3d, lut, T_laser2cam, T_bacs2opencv):
    T_laser2cam_opencv = T_bacs2opencv @ T_laser2cam
    P = K @ T_laser2cam_opencv[0:3, :]
    Xh = np.hstack((coors3d, np.ones_like(coors3d[:, :1]))).T
    xh = P @ Xh
    pos_depth_idx = xh[2, :] > 0

    coors_cam = ((T_bacs2opencv @ T_laser2cam @ Xh)[:3, :]).T
    depth = np.linalg.norm(coors_cam, axis=-1)
    xh = xh / np.repeat(xh[2:, :], 3, axis=0)
    lprojected = xh[[1, 0], :].T  # lproject = [row, column] (nx2)

    lprojected = tccApplyLut(lprojected, lut)
    return lprojected, depth, pos_depth_idx