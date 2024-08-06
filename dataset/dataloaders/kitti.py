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
import os

import cv2
import numpy as np

# https://www.cvlibs.net/datasets/kitti/setup.php

class KITTIOdometryDataset:
    def __init__(self, data_dir, sequence: str, *_, **__):
        self.sequence_id = str(sequence).zfill(2)
        self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")
        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        scan_count = len(self.scan_files)

        # cam 2 (color)
        self.img2_dir = os.path.join(self.kitti_sequence_dir, "image_2/")
        self.img2_files = sorted(glob.glob(self.img2_dir + "*.png"))
        img2_count = len(self.img2_files)
        if img2_count == scan_count:
            self.image_available = True
        else:
            self.image_available = False

        # cam 3 (color)
        self.img3_dir = os.path.join(self.kitti_sequence_dir, "image_3/")
        self.img3_files = sorted(glob.glob(self.img3_dir + "*.png"))

        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))

        # if self.image_available: # now we use cam2
        #     # check this (FIXME)
        #     K_mat = self.calibration["P2"]
        #     self.fx = K_mat[0,0]
        #     self.fy = K_mat[1,1]
        #     self.cx = K_mat[0,2]
        #     self.cy = K_mat[1,2]

        # Load GT Poses (if available)
        if int(sequence) < 11:
            self.poses_fn = os.path.join(data_dir, f"poses/{self.sequence_id}.txt")
            self.gt_poses = self.load_poses(self.poses_fn)

    def __getitem__(self, idx):
        
        points = self.scans(idx)
        point_ts = self.get_timestamps(points)

        frame_data = {"points": points, "point_ts": point_ts}

        if self.image_available:
            img = self.read_img(self.img2_files[idx]) # just for vis here
            frame_data["img"] = img

        return frame_data

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        """Converts from Velodyne to Camera Frame"""
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
        return Tr @ poses @ np.linalg.inv(Tr)

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)
        return points
    
    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        # print(img.shape)
        
        cv2.imshow('cam0', img)
        cv2.waitKey(1) # 1ms

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img) # as np array
        
        return img
    
    # velodyne lidar
    @staticmethod
    def get_timestamps(points):
        x = points[:, 0]
        y = points[:, 1]
        yaw = -np.arctan2(y, x)
        timestamps = 0.5 * (yaw / np.pi + 1.0)
        return timestamps

    def load_poses(self, poses_file):
        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right

        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return _lidar_pose_gt(poses)

    def get_frames_timestamps(self) -> np.ndarray:
        timestamps = np.loadtxt(os.path.join(self.kitti_sequence_dir, "times.txt")).reshape(-1, 1)
        return timestamps

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)

                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict
