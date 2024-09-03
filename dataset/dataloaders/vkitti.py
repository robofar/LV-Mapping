# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
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
import importlib
import os
from collections import namedtuple

import cv2
import numpy as np
import open3d as o3d

# https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
# actually as a RGB-D like dataset

class VirtualKITTIDataset:
    def __init__(self, data_dir, sequence: str, *_, **__):
        
        self.sequence_id = "Scene" + str(sequence).zfill(2)
        # include the data dir as kitti_mot/training/
        # self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)

        self.scenario_name = "clone" 
        self.used_cam_name = "Camera_0"
        
        self.depth_img_dir = os.path.join(data_dir, "depth", self.sequence_id, self.scenario_name, "frames", "depth", self.used_cam_name) 
        self.rgb_img_dir = os.path.join(data_dir, "rgb", self.sequence_id, self.scenario_name, "frames", "rgb", self.used_cam_name) 
        
        self.depth_frames = sorted(glob.glob(self.depth_img_dir + "/*.png"))
        self.rgb_frames =  sorted(glob.glob(self.rgb_img_dir + "/*.jpg"))

        frame_count = len(self.depth_frames)

        # same for two cameras, fixed for this synthetic dataset
        H, W = 375, 1242

        self.fx = 725.0087
        self.fy = 725.0087 
        self.cx = 620.5 
        self.cy = 187.0

        self.K_mat = np.eye(3)
        self.K_mat[0,0]=self.fx
        self.K_mat[1,1]=self.fy
        self.K_mat[0,2]=self.cx
        self.K_mat[1,2]=self.cy

        self.meta_data_dir = os.path.join(data_dir, "textgt", self.sequence_id, self.scenario_name) 
        self.extrinsics_file = os.path.join(self.meta_data_dir, "extrinsic.txt")

        self.depth_scale = 100.0 # 1 correspondong to 1cm
        self.max_depth_m = 100.0

        self.T_l_c = np.eye(4)
        self.T_c_l = np.linalg.inv(self.T_l_c)

        self.main_cam_name = self.used_cam_name 

        self.T_c_l_mats = {self.main_cam_name: self.T_c_l} 
        self.K_mats = {self.main_cam_name: self.K_mat}
        self.cam_widths = {self.main_cam_name: W}
        self.cam_heights = {self.main_cam_name: H}

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(height=H,
                                      width=W,
                                      fx=self.fx,
                                      fy=self.fy,
                                      cx=self.cx,
                                      cy=self.cy)

        self.extrinsic = self.T_c_l

        gt_poses_cam0, gt_poses_cam1 = self.load_poses(self.extrinsics_file)
        self.gt_poses = gt_poses_cam0


    def __getitem__(self, idx):

        rgb_image = o3d.io.read_image(self.rgb_frames[idx])
        depth_image = o3d.io.read_image(self.depth_frames[idx])
        # depth_image = cv2.imread(self.depth_frames[idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, 
                                                                        depth_image, 
                                                                        depth_scale=self.depth_scale, 
                                                                        depth_trunc=self.max_depth_m, 
                                                                        convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.intrinsic)

        points_xyz = np.array(pcd.points, dtype=np.float64)
        points_rgb = np.array(pcd.colors, dtype=np.float64)
        points_xyzrgb = np.hstack((points_xyz, points_rgb))

        rgb_image = np.array(rgb_image)

        depth_image = np.array(depth_image)/self.depth_scale
        rgbd_image = np.concatenate((rgb_image, np.expand_dims(depth_image, axis=-1)), axis=-1) # 4 channels

        image_dict = {self.main_cam_name: rgbd_image}

        frame_data = {"points": points_xyzrgb, "img": image_dict}
        
        return frame_data

    def __len__(self):
        return len(self.depth_frames)

    def load_poses(self, extrinsic_file):
        data = np.loadtxt(extrinsic_file, delimiter=" ", skiprows=1)
        poses = data[:, 2:]
        gt_poses_cam0 = poses[::2].reshape((-1, 4, 4))
        gt_poses_cam1 = poses[1::2].reshape((-1, 4, 4))

        # inverse
        gt_poses_cam0 = np.array([np.linalg.inv(matrix) for matrix in gt_poses_cam0])
        gt_poses_cam1 = np.array([np.linalg.inv(matrix) for matrix in gt_poses_cam1])

        return gt_poses_cam0, gt_poses_cam1