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
from pathlib import Path

from pyquaternion import Quaternion

from utils.tools import get_time


class OxfordDataset:
    def __init__(self, data_dir, cam_name: str, *_, **__):
        
        self.load_img = False # default

        self.use_only_colorized_points = False

        self.min_lidar_radius_m = 0.5

        # poses_file = os.path.join(data_dir, "processed", "trajectory", "vilens-slam-tum.txt")
        poses_file = os.path.join(data_dir, "processed", "trajectory", "gt-tum.txt")

        gt_poses, pose_ts = load_tum_format_poses(poses_file)

        self.poses_count = len(gt_poses)

        self.gt_poses = np.array(gt_poses)
        pose_ts = np.array(pose_ts)

        self.lidar_files = [None] * self.poses_count
        self.cam0_files = [None] * self.poses_count
        self.cam1_files = [None] * self.poses_count
        self.cam2_files = [None] * self.poses_count

        # note these lidar point clouds are in base frame
        lidar_dir = os.path.join(data_dir, "processed", "vilens-slam", "undist-clouds/")
        lidar_files = sorted(glob.glob(lidar_dir + "*.pcd"))

        lidar_ts = extract_time_from_lidar_filenames(lidar_files)

        img_dir_base = os.path.join(data_dir, "processed", "colmap", "images_rectified")

        cam0_dir = os.path.join(img_dir_base, "alphasense_driver_ros_cam0_debayered_image_compressed/")
        cam1_dir = os.path.join(img_dir_base, "alphasense_driver_ros_cam1_debayered_image_compressed/")
        cam2_dir = os.path.join(img_dir_base, "alphasense_driver_ros_cam2_debayered_image_compressed/")

        cam0_files = sorted(glob.glob(cam0_dir + "*.jpg"))
        cam0_ts = extract_time_from_cam_filenames(cam0_files)

        cam1_files = sorted(glob.glob(cam1_dir + "*.jpg"))
        cam1_ts = extract_time_from_cam_filenames(cam1_files)

        cam2_files = sorted(glob.glob(cam2_dir + "*.jpg"))
        cam2_ts = extract_time_from_cam_filenames(cam2_files)

        # lidar - pose association
        lidar_pose_associated_idx, lidar_associated_idx = associate_sensor_to_pose(lidar_ts, pose_ts)

        lidar_associated_count = np.shape(lidar_pose_associated_idx)[0]

        for i in range(lidar_associated_count):
            self.lidar_files[lidar_pose_associated_idx[i]] = lidar_files[lidar_associated_idx[i]]

        # camera 0 - pose association
        cam0_pose_associated_idx, cam0_associated_idx = associate_sensor_to_pose(cam0_ts, pose_ts)

        cam0_associated_count = np.shape(cam0_pose_associated_idx)[0]

        for i in range(cam0_associated_count):
            self.cam0_files[cam0_pose_associated_idx[i]] = cam0_files[cam0_associated_idx[i]]

        # camera 1 - pose association
        cam1_pose_associated_idx, cam1_associated_idx = associate_sensor_to_pose(cam1_ts, pose_ts)

        cam1_associated_count = np.shape(cam1_pose_associated_idx)[0]

        for i in range(cam1_associated_count):
            self.cam1_files[cam1_pose_associated_idx[i]] = cam1_files[cam1_associated_idx[i]]

        # camera 2 - pose association
        cam2_pose_associated_idx, cam2_associated_idx = associate_sensor_to_pose(cam2_ts, pose_ts)

        cam2_associated_count = np.shape(cam2_pose_associated_idx)[0]

        for i in range(cam2_associated_count):
            self.cam2_files[cam2_pose_associated_idx[i]] = cam2_files[cam2_associated_idx[i]]
        
        dataset_parent_path = Path(data_dir).parent

        self.K_mats = {}
        self.T_c_l_mats = {}
        self.cam_widths = {}
        self.cam_heights = {}

        self.cam_list = ["cam0", "cam1", "cam2"] # front, left, right
        self.main_cam_name = "cam0"
        
        calib_file = os.path.join(dataset_parent_path, "calibration", "cam-lidar-imu.yaml")
        self.read_calib_file(calib_file)


        # gt_poses_associated = gt_poses[pose_associated_idx]

        # self.gt_poses = gt_poses_associated

        # self.lidar_files = [self.lidar_files[i] for i in lidar_associated_idx]


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

        # self.mono_depth_for_high_z: bool = False # complete the low Z part 


    def __getitem__(self, idx):

        frame_data = {}
        cur_lidar_file = self.lidar_files[idx]
        if cur_lidar_file is not None:
            points = self.read_point_cloud(cur_lidar_file)

            # transform from base frame to lidar frame
            points_homo = np.hstack((points[:,:3], np.ones((np.shape(points)[0], 1))))

            points = (points_homo @ self.T_l_b_mat.T)[:,:3]

            points_rgb = np.ones_like(points) # only for further processing
            points = np.hstack((points[:,:3], points_rgb[:,:3]))

            frame_data["points"] = points

        if self.load_img:
            
            img_dict = {}
            cur_cam0_file = self.cam0_files[idx]
            if cur_cam0_file is not None:
                img_cam0 = self.read_img(cur_cam0_file)                 
                img_dict["cam0"] = img_cam0

            cur_cam1_file = self.cam1_files[idx]
            if cur_cam1_file is not None:
                img_cam1 = self.read_img(cur_cam1_file)                 
                img_dict["cam1"] = img_cam1
            
            cur_cam2_file = self.cam2_files[idx]
            if cur_cam2_file is not None:
                img_cam2 = self.read_img(cur_cam2_file)                 
                img_dict["cam2"] = img_cam2
    
            if len(img_dict.keys())>0: # at least one img got associated to this timestamp
                
                # print("Img loaded: ", len(img_dict.keys()))
                
                frame_data["img"] = img_dict
        
        return frame_data
    
    def __len__(self):
        return self.poses_count
    
    # read pcd format
    def read_point_cloud(self, scan_file: str):
        pcd_o3d = o3d.io.read_point_cloud(scan_file)
        out_points = np.asarray(pcd_o3d.points, dtype=np.float64)
        
        return out_points # N, 3

    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def read_calib_file(self, yaml_file_path: str):

        calib_dict = {}
        with open(yaml_file_path, 'r') as file:
            calib_dict = yaml.safe_load(file)

            for cam_name in self.cam_list:
                cur_camera_calib = calib_dict[cam_name]
                self.K_mats[cam_name] = np.array(cur_camera_calib["K_rect"])

                # load extrinsics version 1
                # T_c_l_mat = np.array(cur_camera_calib["T_cam_lidar"])

                # load extrinsics version 2
                T_c_l_t_q = np.array(cur_camera_calib["T_cam_lidar_t_xyz_q_xyzw_overwrite"])
                t_c_l = T_c_l_t_q[:3]
                quat_rot_c_l = np.array([T_c_l_t_q[6], T_c_l_t_q[3], T_c_l_t_q[4], T_c_l_t_q[5]]) 
                T_c_l_mat = tran_quat_to_mat(t_c_l, quat_rot_c_l)

                # rotate around z axis by 180 degree
                # flip_mat = np.eye(4)
                # flip_mat[0,0] = flip_mat[1,1] = -1
                # T_c_l_mat = T_c_l_mat @ flip_mat

                self.T_c_l_mats[cam_name] = T_c_l_mat

                self.cam_widths[cam_name] = int(cur_camera_calib["width"])
                self.cam_heights[cam_name] = int(cur_camera_calib["height"])

            T_b_l_t_q = np.array(calib_dict["T_base_lidar_t_xyz_q_xyzw"])
            t_b_l = T_b_l_t_q[:3]
            quat_rot_b_l = np.array([T_b_l_t_q[6], T_b_l_t_q[3], T_b_l_t_q[4], T_b_l_t_q[5]]) 
            self.T_b_l_mat = tran_quat_to_mat(t_b_l, quat_rot_b_l)
            self.T_l_b_mat = np.linalg.inv(self.T_b_l_mat)


def extract_time_from_lidar_filenames(filenames):

    ts_list = []
    for filename in filenames:

        filename = filename.replace(".pcd", "")
        
        # Split the filename by underscores and take the last two parts
        _, secs, nsecs = filename.split("_")
        
        # Convert to integer
        secs = int(secs)
        nsecs = int(nsecs)
        
        # Convert ROS timestamp to seconds
        total_time_seconds = secs + nsecs * 1e-9
        ts_list.append(total_time_seconds)
    
    ts_np = np.array(ts_list)

    return ts_np

def extract_time_from_cam_filenames(filenames):

    ts_list = []
    for filename in filenames:
        
        filename = filename.split("/")[-1]
        secs, nsecs, _ = filename.split(".")
        
        # Convert to integer
        secs = int(secs)
        nsecs = int(nsecs)
        
        # Convert ROS timestamp to seconds
        total_time_seconds = secs + nsecs * 1e-9
        ts_list.append(total_time_seconds)

    ts_np = np.array(ts_list)

    return ts_np

def load_tum_format_poses(filename: str):
    """
    read pose file (with the tum format), support txt file
    # count sec nsec tx ty tz qx qy qz qw
    returns -> list, transformation before calibration transformation
    """

    poses = []
    timestamps = []
    # timestamps_sec = []
    # timestamps_nsec = []
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        
        # check if the first line contains any numeric characters
        # if contain, then skip the first line # timestamp tx ty tz qx qy qz qw
        if any(char.isdigit() for char in first_line):
            file.seek(0)
        
        for line in file: # read each line in the file 
            values = line.strip().split() # split with space
            # some tum format pose file also contain the idx before timestamp
            values = [float(value) for value in values]
            timestamps.append(values[0])
            # timestamps_sec.append(values[1])
            # timestamps_nsec.append(values[2])
            trans = np.array(values[1:4])
            quat_rot = np.array([values[7], values[4], values[5], values[6]]) # w, i, j, k
            
            odom_tf = tran_quat_to_mat(trans, quat_rot)
            poses.append(odom_tf)
    
    return poses, timestamps

def tran_quat_to_mat(trans, quat_rot):

    tran_mat = np.eye(4)
    quats = Quaternion(quat_rot)
    tran_mat[0:3, 3] = trans
    tran_mat[0:3, 0:3] = quats.rotation_matrix

    return tran_mat

def associate_sensor_to_pose(sensor_ts, pose_ts, max_dt=0.025):
    # for each lidar ts, find the closest pose
    # pose_ts_associated = []

    pose_associated_idx = []
    sensor_associated_idx = []
    for i in range(sensor_ts.shape[0]):
        cur_sensor_ts = sensor_ts[i]
        j = np.argmin(np.abs(pose_ts - cur_sensor_ts))
        # pose_ts_associated.append(pose_ts[j])
        if np.abs(pose_ts[j] - cur_sensor_ts) < max_dt:
            pose_associated_idx.append(j)
            sensor_associated_idx.append(i)

    pose_associated_idx = np.array(pose_associated_idx, dtype=np.int32)
    sensor_associated_idx = np.array(sensor_associated_idx, dtype=np.int32)

    return pose_associated_idx, sensor_associated_idx        



    # # read ply format
    # def read_point_cloud_ply(self, scan_file: str):

    #     # tic_read_pc = get_time()

    #     pc_load = o3d.t.io.read_point_cloud(scan_file)
    #     pc_load = {k: v.numpy() for k, v in pc_load.point.items()}

    #     keys = list(pc_load.keys())
    #     # print("available attributes:", keys)
        
    #     points = pc_load["positions"]

    #     assert "t" in keys, "The point cloud ply file must have the t field for point wise timestamp"

    #     ts = pc_load["t"] # already in seconds

    #     point_count = np.shape(points)[0]
    #     points = np.concatenate((points, np.ones((point_count, 1))), axis=1) # N, 4
        
    #     # toc_read_pc = get_time()
    #     # print("pc reading time ply (ms): {:.2f}".format((toc_read_pc -tic_read_pc)*1e3))

    #     return points, ts

    # def read_img(self, img_file: str, undistort_on: bool = False, K_mat = None, dist_coeffs = None):
        
    #     tic_0 = get_time()
    #     img = cv2.imread(img_file)
    #     toc_0 = get_time()

    #     # print("img reading time (ms):" , (toc_0 -tic_0)*1e3)

    #     # apply undistortion:
    #     # tic_undistort = get_time()
    #     if undistort_on and K_mat is not None and dist_coeffs is not None:
    #         img = cv2.undistort(img, K_mat, dist_coeffs)
    #         img_file_split = img_file.split("/")
    #         img_file_split[-2] = "data_undistorted" # data --> data_undistorted # ugly fix here
    #         out_img_file = "/".join(img_file_split)
    #         # print(out_img_file)

    #         cv2.imwrite(out_img_file, img)

    #     # toc_undistort = get_time() 

    #     # print("Image undistortion time (ms):", (toc_undistort-tic_undistort)*1e3) # 12 ms / each 
    #     # for 4 imgs, takes about 50ms, better to do this offline
        
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     return img

    # def read_calib_file(self, yaml_file_path: str) -> dict:

    #     calib_dict = {}
    #     with open(yaml_file_path, 'r') as file:
    #         calib_dict = yaml.safe_load(file)

    #         lidar_h_calib = calib_dict["lidarhorizontalpoints"]
    #         lidar_v_calib = calib_dict["lidarverticalpoints"]
    #         T_cf_lh = np.array(lidar_h_calib["extrinsics"])
    #         T_cf_lv = np.array(lidar_v_calib["extrinsics"])
    #         self.T_lv_lh = np.linalg.inv(T_cf_lv) @ T_cf_lh
    #         self.T_l_lm_mats.append(self.T_lv_lh)

    #         for cam_name in self.cam_list:
    #             cur_cam_calib_name = "camera{}image_raw".format(cam_name)
    #             cur_camera_calib = calib_dict[cur_cam_calib_name]
    #             self.K_mats[cam_name] = np.array(cur_camera_calib["K"])
    #             self.dist_coeffs[cam_name] = np.array(cur_camera_calib["distortion_coeff"])
    #             self.T_c_l_mats[cam_name] = np.linalg.inv(np.array(cur_camera_calib["extrinsics"])) @ T_cf_lh 

    #     return calib_dict
    
    # def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
    #     tic_0 = get_time()

    #     # points as np.numpy (N,4)
    #     points[:,3] = 1 # homo coordinate

    #     # transfrom velodyne points to camera coordinate
    #     points_cam = np.matmul(T_c_l, points.T).T # N, 4
    #     points_cam = points_cam[:,:3] # N, 3

    #     toc_0 = get_time() # fast

    #     # project to image space
    #     u, v, depth = self.persepective_cam2image(points_cam.T, K_mat) 
    #     u = u.astype(np.int32)
    #     v = v.astype(np.int32)

    #     img_height, img_width, _ = np.shape(img)

    #     toc_1 = get_time() # relatively fast

    #     # prepare depth map for visualization
    #     depth_map = np.zeros((img_height, img_width, 1)) # H, W, 1
    #     mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        
    #     # visualize points within 30 meters
    #     min_depth = 1.0
    #     max_depth = 100.0
    #     mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
    #     v_valid = v[mask]
    #     u_valid = u[mask]

    #     depth_map[v_valid,u_valid, 0] = depth[mask]

    #     # print(np.shape(points_rgb))

    #     points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
    #     points_rgb[mask, 3] = 0 # has color

    #     toc_2 = get_time() # slow

    #     # print("colorize time 1 (ms):" , (toc_0 -tic_0)*1e3)
    #     # print("colorize time 2 (ms):" , (toc_1 -toc_0)*1e3)
    #     # print("colorize time 3 (ms):" , (toc_2 -toc_1)*1e3)

    #     return points_rgb, depth_map
    
    # def persepective_cam2image(self, points, K_mat):
    #     ndim = points.ndim
    #     if ndim == 2:
    #         points = np.expand_dims(points, 0)
    #     points_proj = np.matmul(K_mat[:3,:3].reshape([1,3,3]), points)
    #     depth = points_proj[:,2,:]
    #     depth[depth==0] = -1e-6
    #     u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
    #     v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

    #     if ndim==2:
    #         u = u[0]; v=v[0]; depth=depth[0]
    #     return u, v, depth


    # def read_kitti_format_poses(self, filename: str):
    #     """
    #     read pose file (with the kitti format)
    #     returns -> np.array, transformation before calibration transformation
    #     if the format is incorrect, return None
    #     """
    #     poses = []
    #     with open(filename, 'r') as file:            
    #         for line in file:
    #             values = line.strip().split()
    #             if len(values) < 12: # FIXME: > 12 means maybe it's a 4x4 matrix
    #                 print('Not a kitti format pose file')
    #                 return None

    #             values = [float(value) for value in values]
    #             pose = np.zeros((4, 4))
    #             pose[0, 0:4] = values[0:4]
    #             pose[1, 0:4] = values[4:8]
    #             pose[2, 0:4] = values[8:12]
    #             pose[3, 3] = 1.0
    #             poses.append(pose)
        
    #     return np.array(poses)