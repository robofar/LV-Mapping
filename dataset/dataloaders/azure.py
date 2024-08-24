# MIT License
#
# Copyright (c) 2024 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import importlib
import glob
import os
import json
from pathlib import Path

import numpy as np
import open3d as o3d

# Kinect Azure RGBD dataset

class AzureDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):

        self.rgb_dir = os.path.join(data_dir, "color/")
        self.depth_dir = os.path.join(data_dir, "depth/")

        # 1280 x 720
        self.rgb_frames = sorted(glob.glob(self.rgb_dir + '*.jpg'))
        self.depth_frames = sorted(glob.glob(self.depth_dir + '*.png'))

        assert len(self.rgb_frames) == len(self.depth_frames), "RGB frame and depth frame count are not identical"

        # for the shape completion challenge dataset

        self.intrinsic_file = os.path.join(data_dir, "intrinsic", "intrinsic_color.txt")
        self.extrinsic_file = os.path.join(data_dir, "intrinsic", "extrinsic_color.txt")
        self.depth_scale = 1000.0

        intrinsic_mat = np.loadtxt(self.intrinsic_file)
        self.extrinsic = np.loadtxt(self.extrinsic_file)

        self.fx = intrinsic_mat[0,0]
        self.fy = intrinsic_mat[1,1]
        self.cx = intrinsic_mat[0,2]
        self.cy = intrinsic_mat[1,2]

        self.K_mat = intrinsic_mat[:3,:3]
        self.K_mats = {"cam": self.K_mat}

        self.T_l_c = np.eye(4)
        self.T_c_l = np.linalg.inv(self.T_l_c)

        self.T_c_l_mats = {"cam": self.T_c_l}
        self.main_cam_name = "cam"
                    
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(height=720,
                                    width=1280,
                                    fx=self.fx,
                                    fy=self.fy,
                                    cx=self.cx,
                                    cy=self.cy)

        # load reference poses
        pose_dir = os.path.join(data_dir, "pose/")

        self.pose_frames = sorted(glob.glob(pose_dir + '*.txt'))
        gt_poses_list = []
        for pose_frame in self.pose_frames:
            cur_pose = np.loadtxt(pose_frame)
            gt_poses_list.append(cur_pose)
        self.gt_poses = np.array(gt_poses_list) # N,4,4
        
        self.max_depth_m = 6.0
        self.down_sample_on = False
        self.rand_down_rate = 0.1

        self.filter_depth = False

    def __len__(self):
        return len(self.depth_frames)

    def __getitem__(self, idx):
        rgb_image = o3d.io.read_image(self.rgb_frames[idx])

        depth_image = o3d.io.read_image(self.depth_frames[idx])

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, 
                                                                        depth_image,
                                                                        depth_scale=self.depth_scale, 
                                                                        depth_trunc=self.max_depth_m, 
                                                                        convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.intrinsic, self.extrinsic)

        # print(pcd.points)

        if self.down_sample_on:
            pcd = pcd.random_down_sample(sampling_ratio=self.rand_down_rate)
        
        points_xyz = np.array(pcd.points, dtype=np.float64)
        points_rgb = np.array(pcd.colors, dtype=np.float64)
        points_xyzrgb = np.hstack((points_xyz, points_rgb))

        rgb_image = np.array(rgb_image)

        depth_image = np.array(depth_image)/self.depth_scale
        rgbd_image = np.concatenate((rgb_image, np.expand_dims(depth_image, axis=-1)), axis=-1) # 4 channels

        rgbd_image_dict = {"cam": rgbd_image}

        frame_data = {"points": points_xyzrgb, "img": rgbd_image_dict}

        return frame_data 