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

# CKA RGBD dataset in the greenhouse
# or general RGBD data collected by realsense

class CKADataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as err:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        self.rgb_dir = os.path.join(data_dir, "color/")
        self.depth_dir = os.path.join(data_dir, "depth/")

        self.rgb_frames = sorted(glob.glob(self.rgb_dir + '*.png'))
        self.depth_frames = sorted(glob.glob(self.depth_dir + '*.npy'))
        
        # self.poses_fn = os.path.join(sequence_dir, "traj.txt")
        # self.gt_poses = self.load_poses(self.poses_fn)

        self.intrinsic_file = os.path.join(data_dir, "intrinsic.json")

        with open(self.intrinsic_file, 'r') as infile: # load intrinsic json file
            intrinsic_data = json.load(infile)
            intrinsic_mat = intrinsic_data["intrinsic_matrix"]
            width = intrinsic_data["width"]
            height = intrinsic_data["height"]
            self.depth_scale = intrinsic_data["depth_scale"]
                    
            self.intrinsic = self.o3d.camera.PinholeCameraIntrinsic()
            self.intrinsic.set_intrinsics(height=height,
                                        width=width,
                                        fx=intrinsic_mat[0],
                                        fy=intrinsic_mat[4],
                                        cx=intrinsic_mat[6],
                                        cy=intrinsic_mat[7])
        
        self.max_depth_m = 1.5
        self.down_sample_on = False
        self.rand_down_rate = 0.1

        self.filter_depth = False

    def __len__(self):
        return len(self.depth_frames)


    def __getitem__(self, idx):
        rgb_image = self.o3d.io.read_image(self.rgb_frames[idx])

        depth = np.load(self.depth_frames[idx])

        rgbd_image = self.o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, 
                                                                            self.o3d.geometry.Image(depth),
                                                                            depth_scale=self.depth_scale, 
                                                                            depth_trunc=self.max_depth_m, 
                                                                            convert_rgb_to_intensity=False)

        pcd = self.o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.intrinsic)
        if self.down_sample_on:
            pcd = pcd.random_down_sample(sampling_ratio=self.rand_down_rate)
        
        points_xyz = np.array(pcd.points, dtype=np.float64)
        points_rgb = np.array(pcd.colors, dtype=np.float64)
        points_xyzrgb = np.hstack((points_xyz, points_rgb))

        frame_data = {"points": points_xyzrgb}

        return frame_data 