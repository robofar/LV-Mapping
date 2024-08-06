# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
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
import os
import sys
from pathlib import Path
from typing import List

import numpy as np


# https://github.com/fudan-zvg/S-NeRF/blob/main/scripts/nuscenes_preprocess.py
# TODO: add imgs
# search and check nuscenes img loaders
# what is the v1.01-mini split

class NuScenesDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):
        try:
            importlib.import_module("nuscenes")
        except ModuleNotFoundError:
            print("nuscenes-devkit is not installed on your system")
            print('run "pip install nuscenes-devkit"')
            sys.exit(1)

        # TODO: If someone needs more splits from nuScenes expose this 2 parameters
        #  nusc_version: str = "v1.0-trainval"
        #  split: str = "train"
        nusc_version: str = "v1.0-mini"
        split: str = "mini_train"
        self.lidar_name: str = "LIDAR_TOP"
        self.cam_name: str = "CAM_FRONT" # we actually need all these images

        # Lazy loading
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_logs

        self.sequence_id = str(sequence).zfill(4)

        self.nusc = NuScenes(dataroot=str(data_dir), version=nusc_version)
        self.scene_name = f"scene-{self.sequence_id}"
        if self.scene_name not in [s["name"] for s in self.nusc.scene]:
            print(f'[ERROR] Sequence "{self.sequence_id}" not available scenes')
            print("\nAvailable scenes:")
            self.nusc.list_scenes()
            sys.exit(1)

        # Load nuScenes read from file inside dataloader module
        self.load_point_cloud = importlib.import_module(
            "nuscenes.utils.data_classes"
        ).LidarPointCloud.from_file

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(split, self.nusc)

        # print(split_logs)

        # print(len(self.nusc.sample)) # 404 ?
        # print(self.nusc.sample)

        # you should use sweep instead of samples
        # sample data are keyframes sampled per 0.5s (2Hz)
        my_sample = self.nusc.sample[0] # this is list the frame

        # print(my_sample)
        # self.nusc.list_sample(my_sample['token'])
        # self.nusc.render_pointcloud_in_image(my_sample['token'], dot_size = 8, pointsensor_channel='LIDAR_TOP', camera_channel='CAM_BACK')

        # Use only the samples from the current split.
        scene_token = self._get_scene_token(split_logs) # get all the samples in a scene, each sample contains different sensor data 
        self.lidar_tokens = self._get_lidar_tokens(scene_token) # 382
        self.cam_front_tokens = self._get_img_tokens(scene_token) # 224 # why so few

        # print(len(self.lidar_tokens))
        # print(len(self.cam_front_tokens))

        self.gt_poses = self._load_poses()

    def __len__(self):
        return len(self.lidar_tokens)

    def __getitem__(self, idx):
        # self.read_img(self.cam_front_tokens[idx])
        return self.read_point_cloud(self.lidar_tokens[idx])

    def read_point_cloud(self, token: str):
        filename = self.nusc.get("sample_data", token)["filename"]
        # print("LiDAR name:", filename)
        pcl = self.load_point_cloud(os.path.join(self.nusc.dataroot, filename))
        points = pcl.points.T[:, :4] # include intensity
        return points.astype(np.float64)
    
    def read_img(self, token: str):
        filename = self.nusc.get("sample_data", token)["filename"]
        print("Img name:", filename)

    def _load_poses(self) -> np.ndarray:
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        poses = np.empty((len(self), 4, 4), dtype=np.float32)
        for i, lidar_token in enumerate(self.lidar_tokens):
            sd_record_lid = self.nusc.get("sample_data", lidar_token)
            cs_record_lid = self.nusc.get(
                "calibrated_sensor", sd_record_lid["calibrated_sensor_token"]
            )
            ep_record_lid = self.nusc.get("ego_pose", sd_record_lid["ego_pose_token"])

            car_to_velo = transform_matrix(
                cs_record_lid["translation"],
                Quaternion(cs_record_lid["rotation"]),
            ) # car body frame to lidar frame
            pose_car = transform_matrix(
                ep_record_lid["translation"],
                Quaternion(ep_record_lid["rotation"]),
            ) # in car body frame

            poses[i:, :] = pose_car @ car_to_velo # poses in LiDAR frame

        # print(poses) # global coordinate

        # Convert from global coordinate poses to local poses
        first_pose = poses[0, :, :]
        poses = np.linalg.inv(first_pose) @ poses
        return poses

    def _get_scene_token(self, split_logs: List[str]) -> str:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        scene_tokens = [s["token"] for s in self.nusc.scene if s["name"] == self.scene_name][0]
        scene = self.nusc.get("scene", scene_tokens)
        log = self.nusc.get("log", scene["log_token"])
        return scene["token"] if log["logfile"] in split_logs else ""

    def _get_lidar_tokens(self, scene_token: str) -> List[str]:
        # Get records from DB.
        scene_rec = self.nusc.get("scene", scene_token)
        start_sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.nusc.get("sample_data", start_sample_rec["data"][self.lidar_name])

        # Make list of frames
        cur_sd_rec = sd_rec
        sd_tokens = [cur_sd_rec["token"]]
        while cur_sd_rec["next"] != "":
            cur_sd_rec = self.nusc.get("sample_data", cur_sd_rec["next"])
            sd_tokens.append(cur_sd_rec["token"])
        return sd_tokens
    
    # TODO: img count is smaller, we should only use the keyframes (which are synchron)
    def _get_img_tokens(self, scene_token: str) -> List[str]: 
        # Get records from DB.
        scene_rec = self.nusc.get("scene", scene_token)
        start_sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.nusc.get("sample_data", start_sample_rec["data"][self.cam_name])

        # Make list of frames
        cur_sd_rec = sd_rec
        sd_tokens = [cur_sd_rec["token"]]
        while cur_sd_rec["next"] != "":
            cur_sd_rec = self.nusc.get("sample_data", cur_sd_rec["next"])
            sd_tokens.append(cur_sd_rec["token"])
        return sd_tokens
