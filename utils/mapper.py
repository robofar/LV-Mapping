#!/usr/bin/env python3
# @file      mapper.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import math
import sys
import csv
import os

import cv2 # TODO
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import random
import torch
import torch.nn.functional as F
import wandb
from rich import print
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dataset.slam_dataset import SLAMDataset
from model.decoder import Decoder
from model.neural_gaussians import NeuralPoints
from utils.config import Config
from utils.data_sampler import DataSampler
from utils.loss import color_diff_loss, sdf_bce_loss, sdf_diff_loss, sdf_zhong_loss
from utils.tools import (
    colorize_depth_maps,
    get_gradient,
    get_time,
    setup_optimizer,
    transform_batch_torch,
    transform_torch,
    voxel_down_sample_torch
)

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, sky_bce_loss, sky_mask_loss
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.general_utils import rotation2normal
from gaussian_splatting.utils.sh_utils import RGB2SH, SH2RGB
from gaussian_splatting.scene.cameras import CamImage

class Mapper:
    def __init__(
        self,
        config: Config,
        dataset: SLAMDataset,
        neural_points: NeuralPoints,
        decoders
    ):

        self.config = config
        self.silence = config.silence
        self.dataset = dataset
        self.neural_points = neural_points
        self.sdf_mlp = decoders["sdf"]
        self.sem_mlp = decoders["semantic"]
        self.color_mlp = decoders["color"]

        self.gaussian_xyz_mlp = decoders["gauss_xyz"] 
        self.gaussian_scale_mlp = decoders["gauss_scale"] 
        self.gaussian_rot_mlp = decoders["gauss_rot"] 
        self.gaussian_alpha_mlp = decoders["gauss_alpha"] 
        self.gaussian_color_mlp = decoders["gauss_color"] 

        self.device = config.device
        self.dtype = config.dtype
        self.used_poses = None
        self.require_gradient = False
        if (
            config.ekional_loss_on
            or config.proj_correction_on
            or config.consistency_loss_on
        ):
            self.require_gradient = True
        if (
            config.numerical_grad
            and not config.proj_correction_on
            and not config.consistency_loss_on
        ):
            self.require_gradient = False
        self.total_iter: int = 0
        self.sdf_scale = config.logistic_gaussian_ratio * config.sigma_sigmoid_m

        # initialize the data sampler
        self.sampler = DataSampler(config)
        self.ray_sample_count = (
            1 + config.surface_sample_n + config.free_behind_n + config.free_front_n
        )

        self.new_obs_ratio = 0.0

        self.new_idx = None
        self.ba_done_flag = False
        self.adaptive_iter_offset = 0

        # data pool
        self.coord_pool = torch.empty(
            (0, 3), device=self.device, dtype=self.dtype
        )  # coordinate in each frame's coordinate frame
        self.global_coord_pool = torch.empty(
            (0, 3), device=self.device, dtype=self.dtype
        )  # coordinate in global frame
        self.sdf_label_pool = torch.empty((0), device=self.device, dtype=self.dtype)
        self.color_pool = torch.empty(
            (0, self.config.color_channel), device=self.device, dtype=self.dtype
        )
        self.sem_label_pool = torch.empty((0), device=self.device, dtype=torch.int)
        self.normal_label_pool = torch.empty(
            (0, 3), device=self.device, dtype=self.dtype
        )
        self.weight_pool = torch.empty((0), device=self.device, dtype=self.dtype)
        self.time_pool = torch.empty((0), device=self.device, dtype=torch.int)

        # for GS
        self.cam_img_train_pool = []
        self.cam_img_train_ids = []

        self.cam_img_test_pool = []
        self.cam_img_test_ids = []

        self.gs_total_iter = 0
        self.gs_iter_window = config.gs_bs * config.gs_iters * config.img_pool_size

        self.T_w_c_cur_view = None # validate render view camera pose
        
        # evaluation
        self.rendered_pcd_o3d = None # rerendered point cloud

        self.val_psnr_list = []
        self.val_ssim_list = []
        self.val_lpips_list = []
        self.val_depthl1_list = []
        self.val_depth_rmse_list = []
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device) 

    def dynamic_filter(self, points_torch, type_2_on: bool = True):

        if type_2_on:
            points_torch.requires_grad_(True)

        geo_feature, _, weight_knn, _, certainty = self.neural_points.query_feature(
            points_torch, training_mode=False
        )

        sdf_pred = self.sdf_mlp.sdf(
            geo_feature
        )  # predict the scaled sdf with the feature # [N, K, 1]
        if not self.config.weighted_first:
            sdf_pred = torch.sum(sdf_pred * weight_knn, dim=1).squeeze(1)  # N

        # print(sdf_pred[sdf_pred>2.0])

        if type_2_on:
            sdf_grad = get_gradient(
                points_torch, sdf_pred
            ).detach()  # use analytical gradient here
            grad_norm = sdf_grad.norm(dim=-1, keepdim=True).squeeze()

        # Strategy 1 [used]
        # measurements at the certain freespace would be filtered
        # dynamic objects are those have the measurement in the certain freespace
        static_mask = (certainty < self.config.dynamic_certainty_thre) | (
            sdf_pred < self.config.dynamic_sdf_ratio_thre * self.config.voxel_size_m
        )

        # Strategy 2 [not used]
        # dynamic objects's sdf are often underestimated or unstable (already used for source point cloud)
        if type_2_on:
            min_grad_norm = self.config.dynamic_min_grad_norm_thre
            certainty_thre = self.config.dynamic_certainty_thre
            static_mask_2 = (grad_norm > min_grad_norm) | (certainty < certainty_thre)
            static_mask = static_mask & static_mask_2

        return static_mask

    def dynamic_filter_neural_points(self):

        geo_feature, _, weight_knn, _, certainty = self.neural_points.query_feature(
            self.neural_points.local_neural_points, training_mode=False
        )

        sdf_pred = self.sdf_mlp.sdf(geo_feature)    
        # predict the scaled sdf with the feature # [N, K, 1]
        if not self.config.weighted_first:
            sdf_pred = torch.sum(sdf_pred * weight_knn, dim=1).squeeze(1)  # N

        # print(sdf_pred[sdf_pred>2.0])

        static_mask = (certainty < self.config.dynamic_certainty_thre) | (
            sdf_pred < self.config.dynamic_sdf_ratio_thre * self.config.voxel_size_m
        )

        return static_mask

    def determine_used_pose(self):
        
        cur_frame = self.dataset.processed_frame
        if self.config.pgo_on:
            self.used_poses = torch.tensor(
                self.dataset.pgo_poses[:cur_frame+1],
                device=self.device,
                dtype=torch.float64,
            )
        elif self.config.track_on:
            self.used_poses = torch.tensor(
                self.dataset.odom_poses[:cur_frame+1],
                device=self.device,
                dtype=torch.float64,
            )
        elif self.dataset.gt_pose_provided:  # for pure reconstruction with known pose
            self.used_poses = torch.tensor(
                self.dataset.gt_poses[:cur_frame+1],
                device=self.device, 
                dtype=torch.float64
            )

    # begin mapping
    def process_frame(
        self,
        point_cloud_torch: torch.tensor,
        frame_label_torch: torch.tensor,
        frame_normal_torch: torch.tensor,
        cur_pose_torch: torch.tensor,
        frame_id: int,
        filter_dynamic: bool = False,
        mono_depth_point_cloud_torch: torch.tensor = None,
        mono_depth_point_normals_torch: torch.tensor = None,
    ):

        # points_torch contains both the coordinate and the color (intensity)
        # frame_id is the actually used frame id starting from 0 with no skip, 0, 1, 2, ......

        T0 = get_time()

        frame_origin_torch = cur_pose_torch[:3, 3]
        frame_orientation_torch = cur_pose_torch[:3, :3]

        cur_pose_rot = torch.eye(4).to(cur_pose_torch)
        cur_pose_rot[:3,:3] = frame_orientation_torch

        # point in local sensor frame
        frame_point_torch = point_cloud_torch[:, :3]

        # dynamic filtering
        self.static_mask = torch.ones(
            frame_point_torch.shape[0], dtype=torch.bool, device=self.config.device
        )

        if filter_dynamic:
            # reset local map (consider the frame description for loop with latency) 
            self.neural_points.reset_local_map(frame_origin_torch, frame_orientation_torch, frame_id)

            # transformed to the global frame
            frame_point_torch_global = transform_torch(
                frame_point_torch, cur_pose_torch
            )
            
            self.static_mask = self.dynamic_filter(frame_point_torch_global)
            dynamic_count = (self.static_mask == 0).sum().item()
            if not self.silence:
                print("# Dynamic points filtered: ", dynamic_count)
            frame_point_torch = frame_point_torch[self.static_mask]

        frame_color_torch = None
        if self.config.color_channel > 0:
            frame_color_torch = point_cloud_torch[:, 3:]
            if filter_dynamic:
                frame_color_torch = frame_color_torch[self.static_mask]

        if frame_label_torch is not None:
            if filter_dynamic:
                frame_label_torch = frame_label_torch[self.static_mask]

        if frame_normal_torch is not None: # not used yet
            frame_normal_torch = frame_normal_torch[self.static_mask]  

        # TODO

        self.dataset.static_mask = self.static_mask

        T1 = get_time()

        # sampling data for training
        (
            coord,
            sdf_label,
            normal_label,
            sem_label,
            color_label,
            weight,
        ) = self.sampler.sample(
            frame_point_torch, frame_normal_torch, frame_label_torch, frame_color_torch
        )
        # coord is in sensor local frame

        time_repeat = torch.tensor(
            frame_id, dtype=torch.int, device=self.device
        ).repeat(coord.shape[0])

        self.cur_sample_count = sdf_label.shape[0]  # before filtering
        self.pool_sample_count = self.sdf_label_pool.shape[0]

        T2 = get_time()

        update_colors = None
        update_normals = None 

        # update the neural point map
        if self.config.from_sample_points:
            if self.config.from_all_samples:
                update_points = coord
                if frame_color_torch is not None:
                    update_colors = color_label
                if frame_normal_torch is not None:
                    update_normals = normal_label
            else:
                sample_mask = torch.abs(sdf_label) < self.config.surface_sample_range_m * self.config.map_surface_ratio
                update_points = coord[sample_mask, :]
                if frame_color_torch is not None:
                    update_colors = color_label[sample_mask, :]
                if frame_normal_torch is not None:
                    update_normals = normal_label[sample_mask, :]
        else:
            update_points = frame_point_torch 
            update_colors = frame_color_torch
            update_normals = frame_normal_torch
        
        update_points = transform_torch(update_points, cur_pose_torch)

        if update_normals is not None:
            update_normals = transform_torch(update_normals, cur_pose_rot)
            
        # prune map and recreate hash
        if self.config.prune_map_on and ((frame_id + 1) % self.config.prune_freq_frame == 0):
            if self.neural_points.prune_map(self.config.max_prune_certainty):
                self.neural_points.recreate_hash(None, None, True, True, frame_id)
        # TODO: we can prune those free gaussians that has a very small opacity? # TODO: there's some floating gaussians in the sky due to wrong mono depth initialization
        

        # update neural point map
        self.neural_points.update(
            update_points, update_colors, update_normals, frame_origin_torch, frame_orientation_torch, frame_id
        )

        # update gaussians using mono depth predictions # TODO
        if mono_depth_point_cloud_torch is not None: 
            # use the mono depth estimation results to do the initialization
            mono_depth_point_cloud_torch[:, :3] = transform_torch(mono_depth_point_cloud_torch[:, :3], cur_pose_torch)

            # also need to transform the normal
            
            # we currently use a easy fix for ground robot to use only the points with large height value
            update_points_z_quantile = torch.quantile(update_points[:, 2], 0.98) # TODO # height ?
            mono_depth_point_used_mask = mono_depth_point_cloud_torch[:, 2] > update_points_z_quantile
            mono_depth_point_cloud_torch = mono_depth_point_cloud_torch[mono_depth_point_used_mask]

            # voxel downsampling (make it sparse) # TODO: but how sparse
            down_voxel_size = self.config.monodepth_gaussian_res # add to config # TODO
            idx = voxel_down_sample_torch(mono_depth_point_cloud_torch[:, :3], down_voxel_size)
            mono_depth_point_cloud_torch = mono_depth_point_cloud_torch[idx]

            if mono_depth_point_normals_torch is not None:
                cur_rot_torch = torch.eye(4)
                cur_rot_torch[:3,:3] = cur_pose_torch[:3,:3] # rotation part
                mono_depth_point_normals_torch = transform_torch(mono_depth_point_normals_torch, cur_pose_torch)
                mono_depth_point_normals_torch = mono_depth_point_normals_torch[mono_depth_point_used_mask]
                mono_depth_point_normals_torch = mono_depth_point_normals_torch[idx]

            if mono_depth_point_cloud_torch.shape[0] > 0:
                self.neural_points.update(
                    mono_depth_point_cloud_torch[:,:3], mono_depth_point_cloud_torch[:, 3:],
                    mono_depth_point_normals_torch, frame_origin_torch, 
                    frame_orientation_torch, frame_id, is_reliable = False
                )

        # TODO
        # update again with the mono_depth predicted point cloud, set another mask for these neural points


        # local map is also updated here

        if not self.silence:
            self.neural_points.print_memory()

        T3 = get_time()

        # concat with current observations
        self.coord_pool = torch.cat((self.coord_pool, coord), 0)
        self.weight_pool = torch.cat((self.weight_pool, weight), 0)
        self.sdf_label_pool = torch.cat((self.sdf_label_pool, sdf_label), 0)
        self.time_pool = torch.cat((self.time_pool, time_repeat), 0)

        if sem_label is not None:
            self.sem_label_pool = torch.cat((self.sem_label_pool, sem_label), 0)
        else:
            self.sem_label_pool = None
        if color_label is not None:
            self.color_pool = torch.cat((self.color_pool, color_label), 0)
        else:
            self.color_pool = None
        if normal_label is not None:
            self.normal_label_pool = torch.cat(
                (self.normal_label_pool, normal_label), 0
            )
        else:
            self.normal_label_pool = None

        # update the data pool
        # get the data pool ready for training

        # determine used poses # speed fixed
        self.determine_used_pose()

        if self.ba_done_flag:  # bundle adjustment is not done
            self.global_coord_pool = transform_batch_torch(
                self.coord_pool, self.used_poses[self.time_pool]
            )  # very slow here [if ba is not done, then you don't need to transform the whole data pool]
            self.ba_done_flag = False

        else:  # used when ba is not enabled
            global_coord = transform_torch(coord, cur_pose_torch)
            self.global_coord_pool = torch.cat(
                (self.global_coord_pool, global_coord), 0
            )
            # why so slow

        T3_1 = get_time()

        if (frame_id + 1) % self.config.pool_filter_freq == 0:
            pool_relatve = self.global_coord_pool - frame_origin_torch
            # print(pool_relatve.shape)
            pool_relative_dist = torch.sum(pool_relatve**2, dim=-1)
            dist_mask = pool_relative_dist < self.config.window_radius**2

            filter_mask = dist_mask

            true_indices = torch.nonzero(filter_mask).squeeze()

            pool_sample_count = true_indices.shape[0]

            if pool_sample_count > self.config.pool_capacity:
                discard_count = pool_sample_count - self.config.pool_capacity
                # randomly discard some of the data samples if it already exceed the maximum number allowed in the data pool
                discarded_index = torch.randint(
                    0, pool_sample_count, (discard_count,), device=self.device
                )
                filter_mask[
                    true_indices[discarded_index]
                ] = False  # Set the elements corresponding to the discard indices to False

            # filter the data pool
            self.coord_pool = self.coord_pool[filter_mask]
            self.global_coord_pool = self.global_coord_pool[
                filter_mask
            ]  # make global here
            self.sdf_label_pool = self.sdf_label_pool[filter_mask]
            self.weight_pool = self.weight_pool[filter_mask]
            self.time_pool = self.time_pool[filter_mask]

            if normal_label is not None:
                self.normal_label_pool = self.normal_label_pool[filter_mask]
            if sem_label is not None:
                self.sem_label_pool = self.sem_label_pool[filter_mask]
            if color_label is not None:
                self.color_pool = self.color_pool[filter_mask]

            cur_sample_filter_mask = filter_mask[
                -self.cur_sample_count :
            ]  # typically all true
            self.cur_sample_count = (
                cur_sample_filter_mask.sum().item()
            )  # number of current samples
            self.pool_sample_count = filter_mask.sum().item()
        else:
            self.cur_sample_count = coord.shape[0]
            self.pool_sample_count = self.coord_pool.shape[0]

        if not self.silence:
            print("# Total sample in pool: ", self.pool_sample_count)
            print("# Current sample      : ", self.cur_sample_count)

        T3_2 = get_time()

        if (
            self.config.bs_new_sample > 0
        ):  # learn more in the region that is newly observed

            cur_sample_filtered = self.global_coord_pool[
                -self.cur_sample_count :
            ]  # newly added samples
            cur_sample_filtered_count = cur_sample_filtered.shape[0]
            bs = self.config.infer_bs
            iter_n = math.ceil(cur_sample_filtered_count / bs)
            cur_sample_certainty = torch.zeros(
                cur_sample_filtered_count, device=self.device
            )
            cur_label_filtered = self.sdf_label_pool[-self.cur_sample_count :]

            self.neural_points.set_search_neighborhood(
                num_nei_cells=1, search_alpha=0.0
            )
            for n in range(iter_n):
                head = n * bs
                tail = min((n + 1) * bs, cur_sample_filtered_count)
                batch_coord = cur_sample_filtered[head:tail, :]
                batch_certainty = self.neural_points.query_certainty(batch_coord)
                cur_sample_certainty[head:tail] = batch_certainty

            # dirty fix
            self.neural_points.set_search_neighborhood(
                num_nei_cells=self.config.num_nei_cells,
                search_alpha=self.config.search_alpha,
            )

            # cur_sample_certainty = self.neural_points.query_certainty()
            # self.new_idx = torch.where(cur_sample_certainty < self.config.new_certainty_thre)[0] # both the surface and freespace new samples

            # use only the close-to-surface new samples
            self.new_idx = torch.where(
                (cur_sample_certainty < self.config.new_certainty_thre)
                & (
                    torch.abs(cur_label_filtered)
                    < self.config.surface_sample_range_m * 3.0
                )
            )[0]

            self.new_idx += (
                self.pool_sample_count - self.cur_sample_count
            )  # new idx in the data pool

            new_sample_count = self.new_idx.shape[0]
            # if not self.silence:
            #     print("# New sample          : ", new_sample_count)

            # for determine adaptive mapping iteration
            self.adaptive_iter_offset = 0
            self.new_obs_ratio = new_sample_count / self.cur_sample_count
            if self.config.adaptive_iters:
                if self.new_obs_ratio < self.config.new_sample_ratio_less:
                    # print('Train less:', self.new_obs_ratio)
                    self.adaptive_iter_offset = -5
                elif self.new_obs_ratio > self.config.new_sample_ratio_more:
                    # print('Train more:', self.new_obs_ratio)
                    self.adaptive_iter_offset = 5
                    if (
                        frame_id > self.config.freeze_after_frame
                        and self.new_obs_ratio > self.config.new_sample_ratio_restart
                    ):
                        self.adaptive_iter_offset = 10
            
            # use self.new_obs_ratio to determine keyframe


        T3_3 = get_time()

        T4 = get_time()

        # img related # TODO: use a better keyframe selection strategy
        # TODO: add keyframe selection here
        if self.dataset.cur_cam_img is not None:
            
            # set camera poses
            for cam_name in self.dataset.cam_names:
                cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                cur_view_cam.train_view = True

                T_w_l = self.used_poses[cur_view_cam.frame_id] # already in torch tensor, lidar pose
                T_c_l = torch.tensor(self.dataset.T_c_l_mats[cur_view_cam.cam_id], device=self.device) 
                T_w_c = T_w_l @ torch.linalg.inv(T_c_l) # need to convert to cam frame # Here there could be different cameras, support this
                cur_view_cam.set_pose(T_w_c) # set camera pose
            
            # training views
            if not self.dataset.stop_status and frame_id % self.config.gs_keyframe_interval==0:
                # better to use the newly added gaussians ratio (FIXME)
                if len(self.cam_img_train_pool) > self.config.img_pool_size: # TODO, change maximum pool size
                    self.cam_img_train_pool.pop(0) # pop the oldest cam

                for cam_name in self.dataset.cam_names:
                    cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                    cur_view_cam.train_view = True

                    self.cam_img_train_pool.append(cur_view_cam)
                    self.cam_img_train_ids.append(cur_view_cam.uid)
            
            # also add some testing views
            else:
                if len(self.cam_img_test_pool) > self.config.img_test_pool_size:
                    self.cam_img_test_pool.pop(0) # pop the oldest cam
                    
                cam_name = self.dataset.loader.main_cam_name
                cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                cur_view_cam.train_view = False

                self.cam_img_test_pool.append(cur_view_cam)
                self.cam_img_test_ids.append(cur_view_cam.uid)

                # print(self.cam_img_train_pool_id)

        # print("time for dynamic filtering     (ms):", (T1-T0)*1e3)
        # print("time for sampling              (ms):", (T2-T1)*1e3)
        # print("time for map updating          (ms):", (T3-T2)*1e3)
        # print("time for pool updating         (ms):", (T4-T3)*1e3) # mainly spent here
        # print("time for pool transforming     (ms):", (T3_1-T3_0)*1e3) # mainly spent here
        # print("time for filtering             (ms):", (T3_2-T3_1)*1e3)

    # get a batch of training samples and labels for map optimization
    def get_batch(self, global_coord=False):

        if (
            self.config.bs_new_sample > 0
            and self.new_idx is not None
            and not self.dataset.lose_track
            and not self.dataset.stop_status
        ):
            # partial, partial for the history and current samples
            new_idx_count = self.new_idx.shape[0]
            if new_idx_count > 0:
                bs_new = min(new_idx_count, self.config.bs_new_sample)
                bs_history = self.config.bs - bs_new
                index_history = torch.randint(
                    0, self.pool_sample_count, (bs_history,), device=self.device
                )
                index_new_batch = torch.randint(
                    0, new_idx_count, (bs_new,), device=self.device
                )
                index_new = self.new_idx[index_new_batch]
                index = torch.cat((index_history, index_new), dim=0)
            else:  # uniformly sample the pool
                index = torch.randint(
                    0, self.pool_sample_count, (self.config.bs,), device=self.device
                )
        else:  # uniformly sample the pool
            index = torch.randint(
                0, self.pool_sample_count, (self.config.bs,), device=self.device
            )

        if global_coord:
            coord = self.global_coord_pool[index, :]
        else:
            coord = self.coord_pool[index, :]
        sdf_label = self.sdf_label_pool[index]
        ts = self.time_pool[index]  # frame number as the timestamp
        weight = self.weight_pool[index]

        if self.sem_label_pool is not None:
            sem_label = self.sem_label_pool[index]
        else:
            sem_label = None
        if self.color_pool is not None:
            color_label = self.color_pool[index]
        else:
            color_label = None
        if self.normal_label_pool is not None:
            normal_label = self.normal_label_pool[index, :]
        else:
            normal_label = None

        return coord, sdf_label, ts, normal_label, sem_label, color_label, weight

    # get a batch of training samples (only those measured end points) and labels for local bundle adjustment
    def get_ba_samples(self, subsample_count):

        surface_sample_idx = torch.where(self.sdf_label_pool == 0)[0]
        surface_sample_count = surface_sample_idx.shape[0]

        coord_pool_surface = self.coord_pool[surface_sample_idx]
        time_pool_surface = self.time_pool[surface_sample_idx]
        weight_pool_surface = self.weight_pool[surface_sample_idx]

        # uniformly sample the pool
        index = torch.randint(
            0, surface_sample_count, (subsample_count,), device=self.device
        )

        local_coord = coord_pool_surface[index, :]
        weight = weight_pool_surface[index]
        ts = time_pool_surface[index]  # frame number as the timestamp

        return local_coord, weight, ts

    # transform the data pool after pgo pose correction
    def transform_data_pool(self, pose_diff_torch: torch.tensor):
        # pose_diff_torch [N,4,4]
        self.global_coord_pool = transform_batch_torch(
            self.global_coord_pool, pose_diff_torch[self.time_pool]
        )

    # for visualization
    def get_data_pool_o3d(self, down_rate=1, only_cur_data=False):

        if only_cur_data:
            pool_coord_np = (
                self.global_coord_pool[-self.cur_sample_count :: 3]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )
        else:
            pool_coord_np = (
                self.global_coord_pool[::down_rate]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )

        data_pool_pc_o3d = o3d.geometry.PointCloud()
        data_pool_pc_o3d.points = o3d.utility.Vector3dVector(pool_coord_np)

        if self.sdf_label_pool is None:
            return data_pool_pc_o3d
            
        if only_cur_data:
            pool_label_np = (
                self.sdf_label_pool[-self.cur_sample_count :: 3]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )
        else:
            pool_label_np = (
                self.sdf_label_pool[::down_rate]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )

        min_sdf = self.config.free_sample_end_dist_m * -2.0
        max_sdf = -min_sdf
        pool_label_np = np.clip(
            (pool_label_np - min_sdf) / (max_sdf - min_sdf), 0.0, 1.0
        )

        color_map = cm.get_cmap("seismic")
        colors = color_map(1.0 - pool_label_np)[:, :3].astype(np.float64) # change to blue (+) ---> red (-)

        data_pool_pc_o3d.colors = o3d.utility.Vector3dVector(colors)

        return data_pool_pc_o3d

    def free_pool(self):
        self.coord_pool = None
        self.weight_pool = None
        self.sdf_label_pool = None
        self.time_pool = None
        self.sem_label_pool = None
        self.color_pool = None
        self.normal_label_pool = None

    # PIN map online training (mapping) given the fixed pose
    # the main training function
    def mapping(self, iter_count):

        iter_count = max(1, iter_count + self.adaptive_iter_offset)

        neural_point_feat = [self.neural_points.local_geo_features, self.neural_points.local_color_features]

        sdf_mlp_param = list(self.sdf_mlp.parameters())
        if self.config.semantic_on:
            sem_mlp_param = list(self.sem_mlp.parameters())
        else:
            sem_mlp_param = None
        if self.config.color_on:
            color_mlp_param = list(self.color_mlp.parameters())
        else:
            color_mlp_param = None

        opt = setup_optimizer(
            self.config,
            neural_point_feat,
            sdf_mlp_param,
            sem_mlp_param,
            color_mlp_param,
        )

        for iter in tqdm(range(iter_count), disable=self.silence):
            # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)

            T00 = get_time()
            # we do not use the ray rendering loss here for the incremental mapping
            coord, sdf_label, ts, _, sem_label, color_label, weight = self.get_batch(
                global_coord=not self.ba_done_flag
            )  # coord here is in global frame if no ba pose update

            T01 = get_time()

            poses = self.used_poses[ts]
            origins = poses[:, :3, 3]

            if self.ba_done_flag:
                coord = transform_batch_torch(
                    coord, poses
                )  # transformed to global frame

            if self.require_gradient:
                coord.requires_grad_(True)

            (
                geo_feature,
                color_feature,
                weight_knn,
                _,
                certainty,
            ) = self.neural_points.query_feature(
                coord, ts, query_color_feature=self.config.color_on
            )

            T02 = get_time()
            
            # predict the scaled sdf with the feature
            sdf_pred = self.sdf_mlp.sdf(geo_feature) # [N, K, 1]  

            if not self.config.weighted_first:
                sdf_pred = torch.sum(sdf_pred * weight_knn, dim=1).squeeze(1)  # N

            if self.config.semantic_on:
                sem_pred = self.sem_mlp.sem_label_prob(geo_feature)
                if not self.config.weighted_first:
                    sem_pred = torch.sum(sem_pred * weight_knn, dim=1)  # N, S
            if self.config.color_on:
                color_pred = self.color_mlp.regress_color(color_feature)  # [N, K, C]
                if not self.config.weighted_first:
                    color_pred = torch.sum(color_pred * weight_knn, dim=1)  # N, C

            surface_mask = (
                torch.abs(sdf_label) < self.config.surface_sample_range_m
            )  # weight > 0

            if self.require_gradient:
                g = get_gradient(coord, sdf_pred)  # to unit m
            elif (
                self.config.numerical_grad
            ):  # do not use this for the tracking, still analytical grad for tracking
                g = self.get_numerical_gradient(
                    coord[:: self.config.gradient_decimation],
                    sdf_pred[:: self.config.gradient_decimation],
                    self.config.voxel_size_m * self.config.num_grad_step_ratio,
                )  #

            T03 = get_time()

            if self.config.proj_correction_on:  # [not used]
                cos = torch.abs(F.cosine_similarity(g, coord - origins))
                sdf_label = sdf_label * cos

            if self.config.consistency_loss_on:  # [not used]
                near_index = torch.randint(
                    0,
                    coord.shape[0],
                    (min(self.config.consistency_count, coord.shape[0]),),
                    device=self.device,
                )
                random_shift = (
                    torch.rand_like(coord) * 2 * self.config.consistency_range
                    - self.config.consistency_range
                )  # 10 cm
                coord_near = coord + random_shift
                coord_near = coord_near[
                    near_index, :
                ]  # only use a part of these coord to speed up
                coord_near.requires_grad_(True)
                (
                    geo_feature_near,
                    _,
                    weight_knn,
                    _,
                    _,
                ) = self.neural_points.query_feature(coord_near)
                pred_near = self.sdf_mlp.sdf(geo_feature_near)
                if not self.config.weighted_first:
                    pred_near = torch.sum(pred_near * weight_knn, dim=1).squeeze(1)  # N
                g_near = get_gradient(coord_near, pred_near)

            # calculate the loss
            cur_loss = 0.0
            # weight's sign indicate the sample is around the surface or in the free space
            weight = torch.abs(weight).detach() 

            if self.config.main_loss_type == "bce":  # [used]
                sdf_loss = sdf_bce_loss(
                    sdf_pred,
                    sdf_label,
                    self.sdf_scale,
                    weight,
                    self.config.loss_weight_on,
                )
            elif self.config.main_loss_type == "zhong":  # [not used]
                sdf_loss = sdf_zhong_loss(
                    sdf_pred, sdf_label, None, weight, self.config.loss_weight_on
                )
            elif self.config.main_loss_type == "sdf_l1":  # [not used]
                sdf_loss = sdf_diff_loss(sdf_pred, sdf_label, weight, l2_loss=False)
            elif self.config.main_loss_type == "sdf_l2":  # [not used]
                sdf_loss = sdf_diff_loss(sdf_pred, sdf_label, weight, l2_loss=True)
            else:
                sys.exit("Please choose a valid loss type")
            cur_loss += sdf_loss

            # optional consistency regularization loss
            consistency_loss = 0.0
            if self.config.consistency_loss_on:  # [not used]
                consistency_loss = (
                    1.0 - F.cosine_similarity(g[near_index, :], g_near)
                ).mean()
                cur_loss += self.config.weight_c * consistency_loss

            # ekional loss
            eikonal_loss = 0.0
            if (
                self.config.ekional_loss_on and self.config.weight_e > 0
            ):  # MSE with regards to 1
                surface_mask_decimated = surface_mask[
                    :: self.config.gradient_decimation
                ]
                # weight_used = (weight.clone())[::self.config.gradient_decimation] # point-wise weight not used
                if self.config.ekional_add_to == "freespace":
                    g_used = g[~surface_mask_decimated]
                    # weight_used = weight_used[~surface_mask_decimated]
                elif self.config.ekional_add_to == "surface":
                    g_used = g[surface_mask_decimated]
                    # weight_used = weight_used[surface_mask_decimated]
                else:  # "all"  # both the surface and the freespace, used here # [used]
                    g_used = g
                eikonal_loss = (
                    (g_used.norm(2, dim=-1) - 1.0) ** 2
                ).mean()  # both the surface and the freespace
                cur_loss += self.config.weight_e * eikonal_loss

            # optional semantic loss
            sem_loss = 0.0
            if self.config.semantic_on and self.config.weight_s > 0:
                loss_nll = torch.nn.NLLLoss(reduction="mean")
                if self.config.freespace_label_on:
                    label_mask = (
                        sem_label >= 0
                    )  # only use the points with labels (-1, unlabled would not be used)
                else:
                    label_mask = (
                        sem_label > 0
                    )  # only use the points with labels (even those with free space labels would not be used)
                sem_pred = sem_pred[label_mask]
                sem_label = sem_label[label_mask].long()
                sem_loss = loss_nll(
                    sem_pred[:: self.config.sem_label_decimation, :],
                    sem_label[:: self.config.sem_label_decimation],
                )
                cur_loss += self.config.weight_s * sem_loss

            # optional color (intensity) loss
            color_loss = 0.0
            if self.config.color_on and self.config.weight_i > 0:
                color_loss = color_diff_loss(
                    color_pred[surface_mask],
                    color_label[surface_mask],
                    weight[surface_mask],
                    self.config.loss_weight_on,
                    l2_loss=False,
                )
                cur_loss += self.config.weight_i * color_loss

            T04 = get_time()

            # print(cur_loss)

            opt.zero_grad(set_to_none=True)
            cur_loss.backward(retain_graph=False)
            opt.step()

            T05 = get_time()

            self.total_iter += 1

            # in ms
            # print("time for get data        :", (T01-T00) * 1e3) # \
            # print("time for feature querying:", (T02-T01) * 1e3) # \\\\\\\
            # print("time for sdf prediction  :", (T03-T02) * 1e3) # \\\\\\
            # print("time for loss calculation:", (T04-T03) * 1e3) # \\
            # print("time for back propogation:", (T05-T04) * 1e3) # \\\\\\

            if self.config.wandb_vis_on:
                wandb_log_content = {
                    "iter": self.total_iter,
                    "loss/total_loss": cur_loss,
                    "loss/sdf_loss": sdf_loss,
                    "loss/eikonal_loss": eikonal_loss,
                    "loss/consistency_loss": consistency_loss,
                    "loss/sem_loss": sem_loss,
                    "loss/color_loss": color_loss,
                }
                wandb.log(wandb_log_content)

        # update the global map
        self.neural_points.assign_local_to_global()


    def spawn_gaussians2(self, view_direction, distance):

        # TODO: only spawn points from the neural points inside the frustum
        # using the cuda function "in_frustum"
        # currently just use all the points in the local map
        
        xyz_displacement = self.config.voxel_size_m * torch.tanh(self.gaussian_xyz_mlp.mlp(self.neural_points.local_geo_features)[:-1]) # N, 3K # [-1,1]        
        # print(xyz_displacement)

        # this also need to be regularized (TODO)
        
        local_point_count = xyz_displacement.shape[0]
        gaussian_count_per_point = self.gaussian_xyz_mlp.out_k
        local_gaussian_count = local_point_count * gaussian_count_per_point

        gaussian_xyz = self.neural_points.local_neural_points.repeat(1, gaussian_count_per_point) + xyz_displacement # N, 3K
        
        gaussian_xyz = gaussian_xyz.view(local_gaussian_count, -1) # NK, 3

        gaussian_scale = 0.5 * self.config.voxel_size_m * torch.exp(self.gaussian_scale_mlp.mlp(self.neural_points.local_geo_features)[:-1]) # N, 2K
        gaussian_scale = gaussian_scale.view(local_gaussian_count, -1) # NK, 2 # positive (after activation)
        
        # print("mean scale:", gaussian_scale.mean().item())
        
        thin_dim_scale = torch.full((local_gaussian_count, 1), 1e-7).to(gaussian_scale) # already after activation, last dim, very thin
        gaussian_scale = torch.cat((gaussian_scale, thin_dim_scale), dim=1) # NK, 3

        gaussian_rot = self.gaussian_rot_mlp.mlp(self.neural_points.local_geo_features)[:-1] # N, 4K
        gaussian_rot = gaussian_rot.view(local_gaussian_count, -1) # NK , 4
        gaussian_rot = torch.nn.functional.normalize(gaussian_rot) # normalize (after activation)
        gaussian_rot = torch.nan_to_num(gaussian_rot, 0, 0)


        gaussian_alpha = torch.sigmoid(self.gaussian_alpha_mlp.mlp(self.neural_points.local_geo_features)[:-1]) 
        # gaussian_alpha = 0.9 + 0.1 * torch.sigmoid(self.gaussian_alpha_mlp.mlp(self.neural_points.local_geo_features)[:-1]) 
        # gaussian_alpha = 0.5-0.5*torch.tanh(self.gaussian_alpha_mlp.mlp(self.neural_points.local_geo_features)[:-1]) # N, K  #[-1,1] --> [0,1]
        
        # this is like RTG-SLAM
        # print(gaussian_alpha)

        gaussian_alpha = gaussian_alpha.view(local_gaussian_count, -1) # NK # [0-1] (after activation)

        # print("mean opacity:", gaussian_alpha.mean().item()) # the opacity is too low, may have some problem, better to have either 0 or 1 opacity

        # try to now use only one single feature vector
        # learn residual now
        gaussian_rgb_residual = self.gaussian_color_mlp.mlp(self.neural_points.local_color_features)[:-1] # N, 3K
        # print(gaussian_rgb_residual)
        # print(torch.abs(gaussian_rgb_residual).mean().item())
        
        gaussian_color = self.neural_points.local_point_colors.repeat(1, gaussian_count_per_point) + gaussian_rgb_residual # N, 3K
        gaussian_color = torch.clamp(gaussian_color, 0.0, 1.0)

        # gaussian_rgb_base = self.neural_points.local_point_colors.repeat(1, gaussian_count_per_point)
        gaussian_color = gaussian_color.view(local_gaussian_count, 1, -1) # NK, 1, 3
        # gaussian_sh = RGB2SH(gaussian_rgb_base)

        return gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_sh


    def spawn_gaussians(self, alpha_filter_on: bool = True):

        # TODO: only spawn points from the neural points inside the frustum
        # using the cuda function "in_frustum"
        # currently just use all the points in the local map
        
        xyz_displacement = self.config.voxel_size_m * torch.tanh(self.gaussian_xyz_mlp.mlp(self.neural_points.local_geo_features)[:-1]) # N, 3K # [-1,1]        
        # print(xyz_displacement)

        # this also need to be regularized (TODO)
        
        local_point_count = xyz_displacement.shape[0]
        gaussian_count_per_point = self.gaussian_xyz_mlp.out_k
        local_gaussian_count = local_point_count * gaussian_count_per_point

        gaussian_xyz = self.neural_points.local_neural_points.repeat(1, gaussian_count_per_point) + xyz_displacement # N, 3K
        
        gaussian_xyz = gaussian_xyz.view(local_gaussian_count, -1) # NK, 3

        # gaussian_scale = 0.5 * self.config.voxel_size_m * torch.exp(self.gaussian_scale_mlp.mlp(self.neural_points.local_geo_features)[:-1]) # N, 2K
        # FIXME
        # what should be the maximum size here?
        gaussian_scale = 2.0 * self.config.voxel_size_m * torch.sigmoid(self.gaussian_scale_mlp.mlp(self.neural_points.local_geo_features)[:-1]) # N, 2K
        
        gaussian_scale = gaussian_scale.view(local_gaussian_count, -1) # NK, 2 # positive (after activation)
        
        # print("mean scale:", gaussian_scale.mean().item())
        
        thin_dim_scale = torch.full((local_gaussian_count, 1), 1e-7).to(gaussian_scale) # already after activation, last dim, very thin
        gaussian_scale = torch.cat((gaussian_scale, thin_dim_scale), dim=1) # NK, 3

        gaussian_rot = self.gaussian_rot_mlp.mlp(self.neural_points.local_geo_features)[:-1] # N, 4K
        gaussian_rot = gaussian_rot.view(local_gaussian_count, -1) # NK , 4
        gaussian_rot = torch.nn.functional.normalize(gaussian_rot) # normalize (after activation)
        gaussian_rot = torch.nan_to_num(gaussian_rot, 0, 0)


        # gaussian_alpha = torch.sigmoid(self.gaussian_alpha_mlp.mlp(self.neural_points.local_geo_features)[:-1]) 
        gaussian_alpha = torch.tanh(self.gaussian_alpha_mlp.mlp(self.neural_points.local_geo_features)[:-1]) 
        # gaussian_alpha = 0.9 + 0.1 * torch.sigmoid(self.gaussian_alpha_mlp.mlp(self.neural_points.local_geo_features)[:-1]) 
        # gaussian_alpha = 0.5-0.5*torch.tanh(self.gaussian_alpha_mlp.mlp(self.neural_points.local_geo_features)[:-1]) # N, K  #[-1,1] --> [0,1]
        
        # this is like RTG-SLAM
        # print(gaussian_alpha)

        gaussian_alpha = gaussian_alpha.view(local_gaussian_count, -1) # NK, 1 # [0-1] (after activation)
        
        # print("mean opacity:", gaussian_alpha.mean().item()) # the opacity is too low, may have some problem, better to have either 0 or 1 opacity

        # try to now use only one single feature vector
        # learn residual now
        gaussian_rgb_residual = self.gaussian_color_mlp.mlp(self.neural_points.local_color_features)[:-1] # N, 3K
        # print(gaussian_rgb_residual)
        # print(torch.abs(gaussian_rgb_residual).mean().item())
        
        gaussian_color = self.neural_points.local_point_colors.repeat(1, gaussian_count_per_point) + gaussian_rgb_residual # N, 3K
        gaussian_color = torch.clamp(gaussian_color, 0.0, 1.0)

        gaussian_color = gaussian_color.view(local_gaussian_count, -1) # NK, 3 # not SH anymore

        # gaussian_rgb_base = self.neural_points.local_point_colors.repeat(1, gaussian_count_per_point)
        # gaussian_color = gaussian_color.view(local_gaussian_count, 1, -1) # NK, 1, 3
        # gaussian_sh = RGB2SH(gaussian_rgb_base)

        mean_alpha_all = gaussian_alpha.mean()

        # alpha threshold # but this cannot let the gradients to backpropagate (FIXME) # what's the better way to set an self-adpative mask
        if alpha_filter_on:
            before_size = gaussian_alpha.shape[0]

            alpha_thre = 0.0 # tanh [-1,1]
            alpha_mask_idx = torch.nonzero(gaussian_alpha.squeeze(-1) > alpha_thre).view(-1)

            gaussian_xyz = gaussian_xyz[alpha_mask_idx]
            gaussian_scale = gaussian_scale[alpha_mask_idx]
            gaussian_rot = gaussian_rot[alpha_mask_idx]
            gaussian_alpha = gaussian_alpha[alpha_mask_idx]
            gaussian_color = gaussian_color[alpha_mask_idx]

            after_shape = gaussian_alpha.shape[0]

            print("Gaussian count:", before_size, "-->", after_shape) # it's downsampled a bit too much, shall we have some inductive bias

        # also consider the entropy loss, let the opacity to be either 0 or 1

        return gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, mean_alpha_all


    # jointly optimize the neural point features and gaussian parameters
    def joint_gsdf_mapping(self, iter_count: int, sdf_loss_on = True,
         eval_on = False, lpips_eval_on = False, render_pcd = False):
        
        neural_point_feat = [self.neural_points.local_geo_features, self.neural_points.local_color_features]

        opt = setup_optimizer(
            self.config,
            neural_point_feat,
            mlp_sdf_param=list(self.sdf_mlp.parameters()),
            mlp_gs_xyz_param=list(self.gaussian_xyz_mlp.parameters()),
            mlp_gs_scale_param=list(self.gaussian_scale_mlp.parameters()),
            mlp_gs_rot_param=list(self.gaussian_rot_mlp.parameters()),
            mlp_gs_alpha_param=list(self.gaussian_alpha_mlp.parameters()),
            mlp_gs_color_param=list(self.gaussian_color_mlp.parameters())
        )
        
        background = torch.tensor(self.config.bg_color, dtype=self.dtype, device=self.device)
        bg_3d = background.view(3, 1, 1)
        
        if iter_count > 0:

            # print("GS fitting on ")
        
            # TODO 
            # fastest speed: 20 ms / iter (bs=1) including the gaussian parameter loss
            # fastest speed: 10 ms / iter (bs=1) excluding the gaussian parameter loss (but we need to constriant these gaussians)

            # self.neural_points.training_setup_gs(with_pin_feature=self.config.pin_gs_opt_on)

            renderd_image = None

            # still too slow, figure it out how to make the process faster

            down_rate = self.config.gs_down_rate # TODO: add to config. img downsample rate 2**down_rate, if down_rate=0, then use original img

            eval_depth_max = self.config.max_range
            eval_depth_min = self.config.min_range

            for iter in tqdm(range(iter_count), disable=self.silence):    

                gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, mean_alpha_all = self.spawn_gaussians()

                local_gaussian_count = gaussian_xyz.shape[0]
                gaussian_count_per_point = self.gaussian_xyz_mlp.out_k

                # TODO: predict the up-to-date GS here
                # find the neural points in the field of view of the batch
                # now we firstly test all the neural points in the local map

                # TO THINK
                # could be diffcult to set the learning rate
                # could be hard to directly integrate with the visualizer
                # could be better to set the batch size to 1 and operate per image
                # and then maybe you can write the gaussian prediction in the render function, then it can be induced by the visualizer
                # PINGS

                cur_img_pool_size = len(self.cam_img_train_pool)

                # rendering losses
                rgb_loss_batch = 0
                depth_loss_batch = 0

                # regularization losses
                normal_loss_batch = 0
                distort_loss_batch = 0
                mono_normal_loss_batch = 0

                sky_loss_batch = 0

                opacity_loss = 0
                if mean_alpha_all is not None: # let the opacity to be ideally larger
                    opacity_loss = 1.0 - mean_alpha_all # [0, 2]
                    print("Opacity loss:", opacity_loss.item())
                
                gs_bs = min(self.config.gs_bs, cur_img_pool_size)

                batch_visbility_mask = torch.zeros(local_gaussian_count, dtype=torch.bool, device=self.device)

                local_free_gs_mask = self.neural_points.local_free_gs_mask.repeat(1, gaussian_count_per_point).view(-1)

                for rand_idx in torch.randperm(cur_img_pool_size)[:gs_bs]:

                    # T1 = get_time()

                    viewpoint_cam: CamImage = self.cam_img_train_pool[rand_idx]

                    # print("Used cam id:", viewpoint_cam.uid)
                    # camera poses already set

                    gt_image = viewpoint_cam.original_image_list[down_rate]
                    
                    if gt_image.device != self.device: # this is one very time consuming part
                        gt_image.to(self.device)
                    
                    if viewpoint_cam.depth_on:
                        gt_rgb_image = gt_image[:3]
                        gt_depth_image = gt_image[3].unsqueeze(0)
                    else:
                        gt_rgb_image = gt_image
                        gt_depth_image = None

                    # T2 = get_time()

                    render_pkg = render(viewpoint_cam, None, gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, background, down_rate=down_rate) # render gaussians 

                    # T3 = get_time()

                    # rendered results
                    renderd_rgb_image, viewspace_point_tensor, visibility_filter = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"]

                    rend_normal = render_pkg['rend_normal'] # 3, H, W # what is this actually?
                    surf_depth = render_pkg["surf_depth"] # 1, H, W # rendered depth
                    depth_normal = render_pkg['surf_normal'] # 3, H, W # calculated from the depth map (depth --> normal)
                    rend_alpha = render_pkg["rend_alpha"] # 1, H, W accumulated opacity 
                    dist_distortion = render_pkg["rend_dist"] # depth distortion # 1, H, W 

                    # T4 = get_time()
                    
                    batch_visbility_mask = batch_visbility_mask | visibility_filter # update the gaussian visbility mask for this batch

                    # normalize the normals to norm == 1
                    if rend_normal is not None:
                        rend_normal = torch.nn.functional.normalize(rend_normal, dim=0) 
                    if depth_normal is not None:
                        depth_normal = torch.nn.functional.normalize(depth_normal, dim=0) 
                    
                    if viewpoint_cam.sky_mask_on: 
                        cur_sky_mask = viewpoint_cam.sky_mask_list[down_rate]
                        non_sky_mask = ~cur_sky_mask
                        if self.config.lambda_sky > 0 and rend_alpha is not None:
                            cur_sky_loss = sky_mask_loss(cur_sky_mask, rend_alpha) # sky part has 0 alpha
                            # cur_sky_loss = sky_bce_loss(cur_sky_mask, rend_alpha) # let the sky part has small opacity, the others have a large opacity
                            sky_loss_batch += cur_sky_loss
                        
                        if rend_normal is not None:
                            rend_normal = rend_normal * non_sky_mask
                        if depth_normal is not None:
                            depth_normal = depth_normal * non_sky_mask
                        if dist_distortion is not None:
                            dist_distortion = dist_distortion * non_sky_mask
                        
                        # masked the sky part as the background color
                        mask_broadcasted = cur_sky_mask.repeat(3,1,1)
                        gt_rgb_image[mask_broadcasted] = bg_3d.expand_as(gt_rgb_image)[mask_broadcasted]


                    # RGB rendering loss
                    loss_rgb_l1 = l1_loss(renderd_rgb_image, gt_rgb_image)
                    rgb_loss = (1.0 - self.config.lambda_dssim) * loss_rgb_l1 + self.config.lambda_dssim * (1.0 - ssim(renderd_rgb_image, gt_rgb_image))

                    rgb_loss_batch += rgb_loss
                    # ----------------

                    # Depth rendering loss
                    valid_depth_mask = None
                    if surf_depth is not None and gt_depth_image is not None and self.config.lambda_depth > 0:
                        valid_depth_mask = (gt_depth_image > eval_depth_min) & (surf_depth > eval_depth_min) & (gt_depth_image < eval_depth_max) & (surf_depth < eval_depth_max)
                        gt_depth_image = gt_depth_image[valid_depth_mask]
                        # print(gt_depth_image)
                        rend_depth_valid = surf_depth[valid_depth_mask]
                        if self.config.inverse_depth_loss:
                            depth_loss = l1_loss(1.0/gt_depth_image, 1.0/rend_depth_valid) # use inverse depth (then we will care more about the close range part)
                        else:
                            depth_loss = l1_loss(gt_depth_image, rend_depth_valid)

                        # print(" Depth loss:", depth_loss.item()) 
                        depth_loss_batch += depth_loss
                    # ----------------

                    # Regularization losses
                    # this normal consistency regularization loss seems to have some problem, figure it out (FIXME)
                    # if valid_depth_mask is not None:
                    #     rend_normal = rend_normal[:, valid_depth_mask]
                    #     depth_normal = depth_normal[:, valid_depth_mask]
                    #     dist_distortion = dist_distortion[:, valid_depth_mask]    

                    # Normal-Depth consistency regularization loss
                    if rend_normal is not None and depth_normal is not None:
                        rend_normal_norm = rend_normal.norm(2, dim=0) 
                        depth_normal_norm = depth_normal.norm(2, dim=0) 
                        normal_valid_mask = (rend_normal_norm > 0) & (depth_normal_norm > 0)

                        dot_product = (rend_normal * depth_normal).sum(dim=0) # H, W
                        normal_error = 1.0 - dot_product # dot product 
                        # normal_error = 1.0 - torch.abs(dot_product) 
                        normal_error_valid = torch.masked_select(normal_error, normal_valid_mask)

                        normal_loss = normal_error_valid.mean()
                        normal_loss_batch += normal_loss
                    # ----------------

                    # Mono normal regularization loss
                    if rend_normal is not None and viewpoint_cam.mono_normal_on and self.config.lambda_mono_normal > 0:
                        mono_normal = viewpoint_cam.normal_img_list[down_rate]
                        dot_product = (rend_normal * mono_normal).sum(dim=0) # H, W
                        mono_normal_error = 1.0 - dot_product # dot product 
                        mono_normal_error = torch.masked_select(mono_normal_error, normal_valid_mask)
                        mono_normal_loss = mono_normal_error.mean()
                        mono_normal_loss_batch += mono_normal_loss
                    # ----------------

                    # Normal along ray interval distance regularization loss
                    if dist_distortion is not None:
                        distort_loss = dist_distortion.mean()
                        distort_loss_batch += distort_loss
                    # ----------------

                    # T5 = get_time()
                    
                    # if not self.silence:
                    #     print("Render prepare time (ms):", (T2-T1)*1e3) # super fast here
                    #     print("Render time         (ms):", (T3-T2)*1e3) # the forward rendering is fast (about 300Hz)
                    #     print("Render process time (ms):", (T4-T3)*1e3) # sometimes slow here
                    #     print("Render loss time    (ms):", (T5-T4)*1e3) 

                # TODO:
                if not self.silence:
                    if gt_depth_image is not None and depth_loss_batch > 0.0 and self.config.lambda_depth > 0:
                        if self.config.inverse_depth_loss:
                            print(" Inverse depth rendering loss:", depth_loss_batch.item() / gs_bs)
                        else:
                            print(" Depth rendering loss (m):", depth_loss_batch.item() / gs_bs)
                    if normal_loss_batch > 0.0:
                        print(" Normal reg loss:", normal_loss_batch.item() / gs_bs)
                    # print(" Sky loss:", sky_loss_batch.item() / gs_bs)
                    if viewpoint_cam.mono_normal_on and mono_normal_loss_batch > 0.0 and self.config.lambda_mono_normal > 0:
                        print(" Mono normal loss:", mono_normal_loss_batch.item() / gs_bs)

                depth_loss_batch *= self.config.lambda_depth
                
                lambda_normal_linear_ratio = min(self.gs_total_iter / self.gs_iter_window, 1.0)
                lambda_normal = self.config.lambda_normal * lambda_normal_linear_ratio
                normal_loss_batch *= lambda_normal # should increase from 0 to config.lambda_normal (ref: gaussian surfel)
                
                # mono normal loss
                mono_normal_loss_batch *= self.config.lambda_mono_normal

                # depth distortion loss
                distort_loss_batch *= self.config.lambda_distort

                # sky mask
                # print(sky_loss_batch)
                sky_loss_batch *= self.config.lambda_sky

                # # actually we only need to use the points in the field of view (but this might already been handeled in CUDA)
                # # only these gaussians would be optimized
                # # also with the visible gaussians
                # # batch_visbility_mask is only for the valid gs
                # local_valid_gs_mask = self.neural_points.local_valid_gs_mask

                # batch_visbility_mask_in_all_local_gs = local_valid_gs_mask.clone()
                # batch_visbility_mask_in_all_local_gs[batch_visbility_mask_in_all_local_gs>0] = batch_visbility_mask

                # if self.neural_points.gs_dim_count == 2:
                #     constraint_mask = (~self.neural_points.local_free_gs_mask) & batch_visbility_mask_in_all_local_gs # & self.neural_points.local_valid_color_mask 
                # # for gaussian surfels, gaussains with opposite directional normal will not be regarded as visible, thus not optimized (this can be set with the new config[4] to disable the back face culling)
                # else:
                #     constraint_mask = (~self.neural_points.local_free_gs_mask) & local_valid_gs_mask

                # # constraint_mask = (~self.neural_points.local_free_gs_mask) & batch_visbility_mask_in_all_local_gs
                
                # constraint_mask = (~local_free_gs_mask) & batch_visbility_mask

                constraint_mask = batch_visbility_mask # also use the free ones

                true_count = torch.sum(constraint_mask).item()
                true_indices = torch.nonzero(constraint_mask, as_tuple=True)[0]
                gaussian_bs = int(self.config.bs * self.config.gaussian_bs_ratio) # TODO
                # gaussian_bs = self.config.gaussian_bs
                sample_bs = min(true_count, gaussian_bs)  # Number of indices to sample # infer_bs is a bit too large here, TODO: add to config
                # print("Sampled neural point count: " , sample_bs)
                # don't use all the points here (random sample some of them as a batch)
                sampled_indices = true_indices[torch.randperm(true_count)[:sample_bs]] # this is already the idx in all the local gaussians

                # how to find those gaussians only in the training field of views?

                # Gaussian scale istropic loss
                isotropic_loss = 0.0
                if self.config.lambda_isotropic > 0:
                    # scaling = self.neural_points.get_local_scaling[sampled_indices] # after activation, the real scale
                    
                    scaling = gaussian_scale[sampled_indices]

                    scaling = scaling[:,:2] # do not use the last one for Gaussian surfels
                    isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean()
                    # if not self.silence:
                    #     print(" Gaussian isotropic loss:", isotropic_loss.item())
                    isotropic_loss *= self.config.lambda_isotropic
                # ----------------

                # Area regularization loss
                area_loss = 0.0
                if self.config.lambda_area > 0:
                    scaling = gaussian_scale[sampled_indices]
                    area_loss = (scaling[:,0] * scaling[:,1]).mean() # but this is already mean value
                    area_loss *= self.config.lambda_area
                # ----------------
                
                # Opacity regularization loss (prefer large value, prefer positive value)
                opacity_loss *= self.config.lambda_opacity

                # Gaussian SDF consistency loss
                sdf_consistency_loss = 0.0
                sdf_normal_consistency_loss = 0.0
                if self.config.lambda_sdf_normal_cons > 0 or self.config.lambda_sdf_cons > 0:
                    # sampled_guassians_xyz = self.neural_points.get_local_xyz[sampled_indices]
                    # sampled_guassians_normals = rotation2normal(self.neural_points.get_local_rotation[sampled_indices]) # N, 3 # this is definitely normalized

                    sampled_guassians_xyz = gaussian_xyz[sampled_indices]
                    sampled_guassians_normals = rotation2normal(gaussian_rot[sampled_indices]) # N, 3 # this is definitely normalized

                    sampled_guassians_xyz.requires_grad_(True)

                    sampled_guassians_sdf = self.sdf(sampled_guassians_xyz)[0] # sdf, sdf_std
                    sampled_guassians_sdf_grad = get_gradient(sampled_guassians_xyz, sampled_guassians_sdf) # N, 3 # analytical one
                    grad_norm = sampled_guassians_sdf_grad.norm(dim=-1, keepdim=True).squeeze()  # unit: m # normalize 
                    # maybe relax this a bit
                    valid_grad_mask = (grad_norm < self.config.reg_max_grad_norm) & (grad_norm > self.config.reg_min_grad_norm)
                    valid_grad_mask = valid_grad_mask.detach()
                    valid_grad_count = torch.sum(valid_grad_mask).item()
                    # if not self.silence:
                    #     print(" SDF Valid gaussian count:", valid_grad_count, " from ", sample_bs)

                    sdf_consistency_loss = torch.abs(sampled_guassians_sdf[valid_grad_mask]).mean() # gaussians should better lie on the surface

                    sampled_guassians_sdf_grad = sampled_guassians_sdf_grad / (grad_norm.unsqueeze(-1) + 1e-7) # world frame # pointing out of the surface

                    # print(sampled_guassians_sdf_grad)                

                    # gaussian normals should better align with the sdf gradient direction
                    gaussian_normal_error = (1.0 - (sampled_guassians_sdf_grad[valid_grad_mask] * sampled_guassians_normals[valid_grad_mask]).sum(dim=1))                           
                    sdf_normal_consistency_loss = gaussian_normal_error.mean()

                    if not self.silence:
                        print(" SDF cons loss:", sdf_consistency_loss.item(), " SDF normal cons loss:", sdf_normal_consistency_loss.item())

                    sdf_consistency_loss *= self.config.lambda_sdf_cons
                    sdf_normal_consistency_loss *= self.config.lambda_sdf_normal_cons
                # ----------------

                # batch_visbility_mask_neural_points = torch.any(batch_visbility_mask.view(-1, gaussian_count_per_point), dim=1)
                # batch_visbility_mask_neural_points = torch.cat((batch_visbility_mask_neural_points, torch.tensor([False]).to(batch_visbility_mask_neural_points)))
                
                # # self.neural_points.local_geo_features[~batch_visbility_mask_neural_points] # I don't want to optimize those not visible parts

                # SDF training loss
                sdf_loss = 0.0
                eikonal_loss = 0.0
                if sdf_loss_on and self.config.lambda_sdf > 0.0:
                    # with batch size bs
                    coord, sdf_label, ts, _, sem_label, color_label, weight = self.get_batch(global_coord=not self.ba_done_flag)
                        
                    poses = self.used_poses[ts]
                    origins = poses[:, :3, 3]

                    # transformed to global frame if ba is done
                    if self.ba_done_flag:
                        coord = transform_batch_torch(coord, poses)
                        
                    if self.require_gradient:
                        coord.requires_grad_(True)
                        
                    geo_feature, _, weight_knn, _, certainty = self.neural_points.query_feature(coord, ts)
                    
                    # predict the scaled sdf with the feature
                    sdf_pred = self.sdf_mlp.sdf(geo_feature) # [N, K, 1]  

                    if not self.config.weighted_first:
                        sdf_pred = torch.sum(sdf_pred * weight_knn, dim=1).squeeze(1)  # N

                    # weight's sign indicate the sample is around the surface or in the free space
                    weight = torch.abs(weight).detach() 
                    # calculate the sdf bce loss
                    sdf_loss = sdf_bce_loss(sdf_pred,sdf_label, self.sdf_scale, weight, self.config.loss_weight_on)

                    if self.config.weight_e > 0:
                        if self.require_gradient:
                            g = get_gradient(coord, sdf_pred)  # to unit m
                        elif self.config.numerical_grad:
                            g = self.get_numerical_gradient(
                                coord[:: self.config.gradient_decimation],
                                sdf_pred[:: self.config.gradient_decimation],
                                self.config.voxel_size_m * self.config.num_grad_step_ratio)
                        eikonal_loss = self.config.weight_e * ((g.norm(2, dim=-1) - 1.0) ** 2).mean() 
                        if not self.silence:
                            print(" SDF BCE loss:", sdf_loss.item(), " SDF Eikonal loss:", eikonal_loss.item())
                        eikonal_loss *= self.config.lambda_sdf

                    sdf_loss *= self.config.lambda_sdf
                # ----------------

                # if not self.silence:
                #     visible_count = torch.sum(batch_visbility_mask).item()
                #     # print("# Visible local gaussians in this batch:", visible_count)

                # total loss
                # TODO: monitor losses by wandb
                total_loss = (rgb_loss_batch + depth_loss_batch + distort_loss_batch + normal_loss_batch + mono_normal_loss_batch + sky_loss_batch) / gs_bs \
                    + isotropic_loss + area_loss + opacity_loss \
                    + sdf_consistency_loss + sdf_normal_consistency_loss \
                    + sdf_loss + eikonal_loss

                total_loss.backward() 

                # print("Total loss:", total_loss.item())

                # update
                # self.neural_points.optimizer.step()
                # self.neural_points.optimizer.zero_grad(set_to_none=True) 

                opt.step()
                opt.zero_grad(set_to_none=True) 

                # print(torch.mean(self.neural_points.local_geo_features))

                T3 = get_time()

                # print("Optimization iter time (ms):", (T3-T2)*1e3) # still, this backpropagation is slow, but better to do this in batch
            
            # # filter dynamic gaussains (TODO)
            filter_gaussian_on = False
            # this may has some issue
            if filter_gaussian_on:
                local_gaussian_position = self.neural_points.get_local_xyz
                nonfree_local_gaussian_position = local_gaussian_position[~self.neural_points.local_free_gs_mask]
                nonfree_local_gaussians_static_mask = self.dynamic_filter(nonfree_local_gaussian_position, type_2_on=False)
                local_gaussians_static_mask = self.neural_points.local_free_gs_mask.clone()
                local_gaussians_static_mask[local_gaussians_static_mask==0] = nonfree_local_gaussians_static_mask # non free part according to this static mask, free part all static

                self.neural_points.local_valid_gs_mask = local_gaussians_static_mask 
                # self.neural_points.local_valid_gs_mask = self.neural_points.local_valid_gs_mask & local_gaussians_static_mask # TODO

            self.neural_points.assign_local_gaussians_to_global() # set back gaussians (and also neural points), better don't do it twice
            self.neural_points.assign_local_to_global() # set back pin feature

            self.gs_total_iter += (self.config.gs_bs * iter_count)

        # rendered the last frame for vis
        if eval_on:
            
            T1_v = get_time()

            with torch.no_grad():
                
                # current values
                gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, _ = self.spawn_gaussians()

                vis_cam_name = self.dataset.cam_names[0] # TODO # -1 

                # if self.config.gs_batch_training_on: 
                #     rand_idx = random.randint(0, len(self.cam_img_train_pool)-1) # random frame
                #     cur_viewpoint_cam: CamImage = self.cam_img_train_pool[rand_idx]
                # else: # lastest frame
                #     cur_viewpoint_cam: CamImage = self.dataset.cur_cam_img[vis_cam_name]

                # only use testing views
                if len(self.cam_img_test_pool) >= self.config.img_test_pool_size-1:
                    cur_viewpoint_cam: CamImage = self.cam_img_test_pool[0]
                else:
                    # use the last one in the pool (for single cam mode)
                    cur_viewpoint_cam: CamImage = self.cam_img_train_pool[0] # training view
                
                # now we just use the lastest training view for a sanity test (FIXME)
                # cur_viewpoint_cam: CamImage = self.cam_img_train_pool[-1]

                # print("Used cam id:", cur_viewpoint_cam.uid)

                cam_name = cur_viewpoint_cam.cam_id
                val_frame_id = cur_viewpoint_cam.frame_id 
                vis_down_rate = self.config.gs_vis_down_rate
                vis_down_scale = 2**(vis_down_rate)

                gaussian_vis_scale = self.config.gaussian_vis_scale 

                original_img = cur_viewpoint_cam.original_image_list[vis_down_rate]
        
                original_img_np = original_img.detach().cpu().numpy() # C, H, W
                original_img_int8 = (np.transpose(original_img_np, (1, 2, 0))[:,:,:3] * 255.0).astype(np.uint8) # H, W, 3
                original_img_int8 = np.ascontiguousarray(original_img_int8) 
                original_img_rgb = cv2.cvtColor(original_img_int8, cv2.COLOR_RGB2BGR)
                if self.config.o3d_vis_on and self.config.vis_in_cv2:
                    cv2.imshow(cam_name + ": Observed RGB", original_img_rgb)

                if cur_viewpoint_cam.depth_on: # how to convert a depth map # TODO
                    # print(np.shape(original_img_depth))
                    # print(original_img_np[3]) # why all 1?
                    original_img_depth = original_img_np[3]
                    depth_valid_mask = (original_img_depth > 0)
                    original_img_depth_color = (colorize_depth_maps(original_img_depth, 0.1, self.config.max_range*0.9)*255.0).astype(np.uint8) # 1, 3, H, W 
                    # print(np.shape(original_img_depth))
                    original_img_depth_color = np.transpose(original_img_depth_color[0], (1, 2, 0)) # H, W, 3 # colorized the depth map here
                    original_img_depth_color = cv2.cvtColor(original_img_depth_color, cv2.COLOR_RGB2BGR)
                    if self.config.o3d_vis_on and self.config.vis_in_cv2:
                        cv2.imshow(cam_name + ": Observed Depth", original_img_depth_color)

                # for this validation render frame
                T_w_l = self.used_poses[val_frame_id] # already in torch tensor, lidar pose for current frame
                T_c_l = torch.tensor(self.dataset.T_c_l_mats[cur_viewpoint_cam.cam_id], device=self.device) 
                T_w_c = T_w_l @ T_c_l.inverse() # need to convert to cam frame

                self.T_w_c_cur_view = T_w_c.detach().cpu().numpy()

                T1_r = get_time()

                render_pkg = render(cur_viewpoint_cam, T_w_c, gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, background, scaling_modifier=gaussian_vis_scale, down_rate=vis_down_rate) # render gaussians 

                # render_pkg = render(cur_viewpoint_cam, T_w_c, self.neural_points, background, scaling_modifier=gaussian_vis_scale, down_rate=vis_down_rate) # render gaussians
                T2_r = get_time()
                if not self.silence:
                    print("Render time per frame (ms):", (T2_r-T1_r)*1e3, " [", 1.0/(T2_r-T1_r), " Hz ]")
                
                renderd_image, rend_normal, surf_depth, depth_normal, rend_alpha = render_pkg["render"], render_pkg["rend_normal"], render_pkg["surf_depth"], render_pkg["surf_normal"], render_pkg["rend_alpha"]
                
                # normalize the normals to norm == 1
                if rend_normal is not None:
                    rend_normal = torch.nn.functional.normalize(rend_normal, dim=0) 
                if depth_normal is not None:
                    depth_normal = torch.nn.functional.normalize(depth_normal, dim=0) 

                if cur_viewpoint_cam.sky_mask_on:
                    cur_sky_mask = cur_viewpoint_cam.sky_mask_list[vis_down_rate] # still torch
                    non_sky_mask = ~ cur_sky_mask
                    if surf_depth is not None:
                        surf_depth = surf_depth * non_sky_mask
                    if depth_normal is not None:
                        depth_normal = depth_normal * non_sky_mask
                    if rend_normal is not None:
                        rend_normal = rend_normal * non_sky_mask

                renderd_image = torch.clamp(renderd_image, 0.0, 1.0) # rule out extreme value for vis
                renderd_image_np = (renderd_image.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                renderd_image_np = np.ascontiguousarray(renderd_image_np) 
                renderd_image_rgb_np = cv2.cvtColor(renderd_image_np, cv2.COLOR_RGB2BGR)
                if self.config.o3d_vis_on and self.config.vis_in_cv2:
                    cv2.imshow(cam_name + ": Rendered RGB", renderd_image_rgb_np)

                if surf_depth is not None:
                    rendered_depth_np = surf_depth.detach().cpu().numpy()
                    rendered_depth_color = (colorize_depth_maps(rendered_depth_np, 0.1, self.config.max_range*0.9)*255.0).astype(np.uint8) # 1, 3, H, W 
                    rendered_depth_np = rendered_depth_np[0] # H, W
                    rendered_depth_np = np.ascontiguousarray(rendered_depth_np)
                    rendered_depth_color = np.transpose(rendered_depth_color[0], (1, 2, 0)) # H, W, 3
                    rendered_depth_color = cv2.cvtColor(rendered_depth_color, cv2.COLOR_RGB2BGR)
                    if self.config.o3d_vis_on and self.config.vis_in_cv2:
                        cv2.imshow(cam_name + ": Rendered Depth", rendered_depth_color)
                
                if rend_normal is not None:
                    rend_normal_vis = 0.5 - rend_normal * 0.5  # convert to the normal vis color # surf_normal
                    rendered_normal_np = (rend_normal_vis.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
                    rendered_normal_np = cv2.cvtColor(rendered_normal_np, cv2.COLOR_RGB2BGR)
                    if self.config.o3d_vis_on and self.config.vis_in_cv2:
                        cv2.imshow(cam_name + ": Rendered Normal", rendered_normal_np)

                if depth_normal is not None:
                    depth_normal_vis = 0.5 - depth_normal * 0.5 # convert to the normal vis color # depth_normal
                    depth_normal_np = (depth_normal_vis.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
                    depth_normal_np = cv2.cvtColor(depth_normal_np, cv2.COLOR_RGB2BGR)
                    if self.config.o3d_vis_on and self.config.vis_in_cv2:
                        cv2.imshow(cam_name + ": Depth Normal", depth_normal_np)

                if cur_viewpoint_cam.mono_normal_on:
                    img_mono_normal = cur_viewpoint_cam.normal_img_list[vis_down_rate]
                    if cur_viewpoint_cam.sky_mask_on:
                        img_mono_normal = img_mono_normal * non_sky_mask
                    mono_normal_np = img_mono_normal.permute(1,2,0).detach().cpu().numpy()
                    mono_normal_np = 0.5 - mono_normal_np * 0.5 # convert to the normal vis color
                    mono_normal_vis_np = (mono_normal_np * 255.0).astype(np.uint8)  
                    mono_normal_vis_np = cv2.cvtColor(mono_normal_vis_np, cv2.COLOR_RGB2BGR)
                    if self.config.o3d_vis_on and self.config.vis_in_cv2:
                        cv2.imshow(cam_name + ": Mono Normal", mono_normal_vis_np)

                # print("Max alpha value:", torch.max(rend_alpha).item()) # <= 1
                # rendered_alpha_np = (rend_alpha.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
                # rendered_alpha_np = cv2.cvtColor(rendered_alpha_np, cv2.COLOR_GRAY2BGR)  
                # cv2.imshow(cam_name + ": Rendered Alpha", rendered_alpha_np)

                cv2.waitKey(1)

                if render_pcd and surf_depth is not None: # vis with "J"

                    # rendered_rgb_image_o3d = o3d.geometry.Image(renderd_image_np)  # with rendered RGB
                    observed_rgb_image_o3d = o3d.geometry.Image(original_img_int8)   # with original RGB (not availbale sometimes)         
                    rendered_depth_image_o3d = o3d.geometry.Image(rendered_depth_np)

                    rendered_rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(observed_rgb_image_o3d, 
                                                                                        rendered_depth_image_o3d, 
                                                                                        depth_scale=1.0, 
                                                                                        depth_trunc=self.config.max_range*0.9, 
                                                                                        convert_rgb_to_intensity=False)

                    
                    original_intrinsic = self.dataset.loader.intrinsic
                    resized_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=int(original_intrinsic.width/vis_down_scale), 
                        height=int(original_intrinsic.height/vis_down_scale), 
                        intrinsic_matrix=original_intrinsic.intrinsic_matrix/vis_down_scale)

                    
                    # rendered point cloud in the world frame
                    self.rendered_pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rendered_rgbd_o3d, 
                        resized_intrinsic, np.linalg.inv(self.T_w_c_cur_view))

                
                # cur_lidar_pose_np = self.used_poses[-1].detach().cpu().numpy() 
                # T_cr = np.linalg.inv(cur_lidar_pose_np) @ T_w_l.detach().cpu().numpy() 
                # self.rendered_pcd_o3d.transform(T_cr) # convert to the coordinate system of current lidar frame

                # cur psnr
                original_rgb = original_img[:3]
                cur_pnsr = psnr(renderd_image, original_rgb).mean().item()
                cur_ssim = ssim(renderd_image, original_rgb).item()

                if lpips_eval_on:
                    cur_lpips = self.lpips(renderd_image.unsqueeze(0), original_rgb.unsqueeze(0)).item()
                else:
                    cur_lpips = 0.0

                if not self.silence:
                    if cur_viewpoint_cam.train_view:
                        print("Eval (train view)") 
                    else: # we only eval the test views
                        print("Eval (test view)") 
                    print("Current PSNR ↑ :", cur_pnsr, ", SSIM ↑ :", cur_ssim, ", LPIPS ↓  :", cur_lpips)

                cur_depth_l1 = cur_depth_rmse = 0.0
                if cur_viewpoint_cam.depth_on and surf_depth is not None:
                    # print(np.shape(original_img_depth), np.shape(rendered_depth_np))
                    depth_valid_mask = (original_img_depth > eval_depth_min) & (rendered_depth_np > eval_depth_min) & (original_img_depth < eval_depth_max) & (rendered_depth_np < eval_depth_max)
                    diff_depth = np.abs(original_img_depth - rendered_depth_np) # already abs
                    diff_depth[~depth_valid_mask] = 0.0
                    diff_depth_masked = diff_depth[depth_valid_mask]
                    cur_depth_l1 = np.mean(diff_depth_masked)
                    cur_depth_rmse = np.sqrt(np.mean(diff_depth_masked**2))

                    diff_depth_color = (colorize_depth_maps(diff_depth, 0.0, self.config.max_range*0.05)*255.0).astype(np.uint8) # 1, 3, H, W 
                    diff_depth_color = np.transpose(diff_depth_color[0], (1, 2, 0)) # H, W, 3
                    diff_depth_color = cv2.cvtColor(diff_depth_color, cv2.COLOR_RGB2BGR)
                    if self.config.o3d_vis_on and self.config.vis_in_cv2:
                        cv2.imshow(cam_name + ": Rendered Depth Error", diff_depth_color)

                    if not self.silence:
                        print("Depth L1 (m) ↓ :", cur_depth_l1, ", Depth RMSE (m) ↓ :", cur_depth_rmse)

                if not cur_viewpoint_cam.train_view or self.config.gs_keyframe_interval==1: # if only train view, we also eval the train views
                    self.val_psnr_list.append(cur_pnsr)
                    self.val_ssim_list.append(cur_ssim)
                    self.val_lpips_list.append(cur_lpips)
                    if cur_viewpoint_cam.depth_on:
                        self.val_depthl1_list.append(cur_depth_l1)
                        self.val_depth_rmse_list.append(cur_depth_rmse)
                
            T2_v = get_time()
            if not self.silence:
                print("GS evaluation time (ms):", (T2_v-T1_v)*1e3)

        return 

    def init_gs_eval(self):
        # clear the lists for evaluation
        self.val_psnr_list = []
        self.val_ssim_list = []
        self.val_lpips_list = []
        self.val_depthl1_list = []
        self.val_depth_rmse_list = []

    def gs_eval_offline(self, eval_down_rate=0, test_view_only: bool = False, train_view_only: bool = False):

        with torch.no_grad():
            cam_name = self.dataset.loader.main_cam_name
            K_mat = self.dataset.K_mats[cam_name]
            height = self.dataset.loader.cam_heights[cam_name]
            width = self.dataset.loader.cam_widths[cam_name] 
            T_c_l = torch.tensor(self.dataset.T_c_l_mats[cam_name], device=self.device) 

            background = torch.tensor(self.config.bg_color, dtype=self.dtype, device=self.device)
            bg_3d = background.view(3, 1, 1)

            for frame_id in tqdm(range(0, self.dataset.processed_frame, 1), desc="GS evaluation"):
                
                if test_view_only and (frame_id % self.config.gs_keyframe_interval == 0):
                    continue

                if train_view_only and (frame_id % self.config.gs_keyframe_interval != 0):
                    continue

                T_w_l = self.used_poses[frame_id] # already in torch tensor, lidar pose for current frame
                T_w_c = T_w_l @ T_c_l.inverse() # need to convert to cam frame

                assert self.config.use_dataloader, "Only data loader version is supported currently"
                self.dataset.read_frame_with_loader(frame_id, init_pose = False, monodepth_on=True) # because we want to use the sky mask here

                cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                cur_view_cam.set_pose(T_w_c)

                self.neural_points.reset_local_map(T_w_l[:3,3], None, cur_ts=frame_id) # for larger map, you even need to firstly recreate hash

                # current values
                gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, _ = self.spawn_gaussians(alpha_filter_on=True)

                render_pkg = render(cur_view_cam, None, gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, background, down_rate=eval_down_rate) # render gaussians 

                # render_pkg = render(cur_view_cam, None, self.neural_points, background, down_rate=eval_down_rate) # render gaussians 

                # rendered results
                renderd_rgb_image, surf_depth = render_pkg["render"], render_pkg["surf_depth"] # 3, H, W / 1, H, W

                renderd_rgb_image = torch.clamp(renderd_rgb_image, 0, 1)
                # print(torch.max(renderd_rgb_image), torch.min(renderd_rgb_image)) # why there are value larger than 1?

                original_img = cur_view_cam.original_image_list[eval_down_rate]

                original_rgb_image = original_img[:3]

                if cur_view_cam.sky_mask_on:
                    cur_sky_mask = cur_view_cam.sky_mask_list[eval_down_rate] # still torch
                    mask_broadcasted = cur_sky_mask.repeat(3,1,1)
                    original_rgb_image[mask_broadcasted] = bg_3d.expand_as(original_rgb_image)[mask_broadcasted]

                cur_pnsr = psnr(renderd_rgb_image, original_rgb_image).mean().item()
                cur_ssim = ssim(renderd_rgb_image, original_rgb_image).item()
                cur_lpips = self.lpips(renderd_rgb_image.unsqueeze(0), original_rgb_image.unsqueeze(0)).item()

                self.val_psnr_list.append(cur_pnsr)
                self.val_ssim_list.append(cur_ssim)
                self.val_lpips_list.append(cur_lpips)

                if cur_view_cam.depth_on and surf_depth is not None: 
                    eval_depth_max = self.config.max_range
                    eval_depth_min = self.config.min_range
                    original_img_depth = original_img[3] # torch.tensor
                    depth_valid_mask = (original_img_depth > eval_depth_min) & (surf_depth > eval_depth_min) & (original_img_depth < eval_depth_max) & (surf_depth < eval_depth_max)
                    diff_depth = torch.abs(original_img_depth - surf_depth) # already abs
                    diff_depth[~depth_valid_mask] = 0.0
                    diff_depth_masked = diff_depth[depth_valid_mask].detach().cpu().numpy()
                    cur_depth_l1 = np.mean(diff_depth_masked)
                    cur_depth_rmse = np.sqrt(np.mean(diff_depth_masked**2))

                    self.val_depthl1_list.append(cur_depth_l1)
                    self.val_depth_rmse_list.append(cur_depth_rmse)

                    # TODO: add depth rendering eval

                # if not self.silence:
                #     print("Current PSNR ↑ :", cur_pnsr, ", SSIM ↑ :", cur_ssim, ", LPIPS ↓  :", cur_lpips)

    def gs_eval_out(self, gs_time_table = None):
        
        val_pnsr_np = val_ssim_np = val_lpips_np = val_depthl1_np = val_depth_rmse_np = gs_time_mean = 0.0

        if len(self.val_psnr_list) > 0:
            val_pnsr_np = np.mean(np.array(self.val_psnr_list))
            val_ssim_np = np.mean(np.array(self.val_ssim_list))
            val_lpips_np = np.mean(np.array(self.val_lpips_list))
            
            print(f"Calculated on {len(self.val_psnr_list)} frames")

            print("Average validation view PSNR  ↑ :", f"{val_pnsr_np:.3f}")
            print("Average validation view SSIM  ↑ :", f"{val_ssim_np:.3f}")
            print("Average validation view LPIPS ↓ :", f"{val_lpips_np:.3f}")


        if len(self.val_depth_rmse_list) > 0:
            val_depthl1_np = np.mean(np.array(self.val_depthl1_list))
            val_depth_rmse_np = np.mean(np.array(self.val_depth_rmse_list))
            print("Average validation view Depth L1 (m) ↓ :", f"{val_depthl1_np:.3f}")
            print("Average validation view Depth RMSE (m) ↓ :", f"{val_depth_rmse_np:.3f}")

        if gs_time_table is not None:
            gs_time_mean = np.mean(np.array(gs_time_table))

        gs_csv_columns = [
                "PSNR ↑",
                "SSIM ↑",
                "LPIPS ↓",
                "Depth L1 (m)",
                "Depth RMSE (m) ↓",
                "Consuming time per frame [s]",
                "Frame count",
            ]
        gs_eval = [
                {
                    gs_csv_columns[0]: val_pnsr_np,
                    gs_csv_columns[1]: val_ssim_np,
                    gs_csv_columns[2]: val_lpips_np,
                    gs_csv_columns[3]: val_depthl1_np,
                    gs_csv_columns[4]: val_depth_rmse_np,
                    gs_csv_columns[5]: gs_time_mean,
                    gs_csv_columns[6]: len(self.val_psnr_list),
                }
            ]
        gs_output_csv_path = os.path.join(self.config.run_path, "gs_eval.csv")
        try:
            with open(gs_output_csv_path, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=gs_csv_columns)
                writer.writeheader()
                for data in gs_eval:
                    writer.writerow(data)
        except IOError:
            print("I/O error")

        # if config.save_mesh:
        #     output_mc_res_m = config.mc_res_m*0.6
        #     mc_cm_str = str(round(output_mc_res_m*1e2))
        #     gs_tsdf_mesh_path = os.path.join(run_path, "mesh", "gs_rendered_tsdf_fusion_mesh_" + mc_cm_str + "cm.ply")
        #     gs_rendered_tsdf_mesh = self.gs_tsdf_fusion(vox_size=output_mc_res_m, depth_trunc=config.max_range*0.9, output_path=gs_tsdf_mesh_path)


    # TODO: deal with local and global map
    def gs_tsdf_fusion(self, render_frame_step = 1, vox_size = 0.1, down_rate = 0, depth_trunc = 10.0, output_path = None):
        # render depth and color from GS map and do tsdf fusion to build mesh

        cam_name = self.dataset.loader.main_cam_name
        K_mat = self.dataset.K_mats[cam_name]
        height = self.dataset.loader.cam_heights[cam_name]
        width = self.dataset.loader.cam_widths[cam_name] 
        T_c_l = torch.tensor(self.dataset.T_c_l_mats[cam_name], device=self.device) 

        cam_intrinsic_o3d = self.dataset.loader.intrinsic # main cam

        background = torch.tensor(self.config.bg_color, dtype=self.dtype, device=self.device)

        trunc_dist = 4 * vox_size

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=vox_size, # unit: m
            sdf_trunc=trunc_dist, # unit: m
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        for frame_id in tqdm(range(0, self.dataset.processed_frame, render_frame_step), desc="TSDF fusion"):

            cur_view_cam = CamImage(frame_id, None, K_mat, self.config.min_range*0.5, self.config.max_range*1.1, 
                cam_name, device=self.device, img_width=width, img_height=height)
            
            T_w_l = self.used_poses[frame_id] # already in torch tensor, lidar pose for current frame
            T_w_c = T_w_l @ T_c_l.inverse() # need to convert to cam frame

            # TODO: change to the new setup
            render_pkg = render(cur_view_cam, T_w_c, self.neural_points, background, down_rate=down_rate) # render gaussians 

            # rendered results
            renderd_rgb_image, surf_depth = render_pkg["render"], render_pkg["surf_depth"] # 3, H, W / 1, H, W

            renderd_rgb_image = torch.clamp(renderd_rgb_image, 0, 1)
            # print(torch.max(renderd_rgb_image), torch.min(renderd_rgb_image)) # why there are value larger than 1?

            renderd_image_np = (renderd_rgb_image.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 

            renderd_image_np = np.ascontiguousarray(renderd_image_np)
            rgb_image = o3d.geometry.Image(renderd_image_np)

            rendered_depth_np = surf_depth.squeeze(0).detach().cpu().numpy().astype(np.float32) 
            rendered_depth_np = np.ascontiguousarray(rendered_depth_np)
            depth_image = o3d.geometry.Image(rendered_depth_np)

            cur_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, 
                                                                        depth_image, 
                                                                        depth_scale=1.0, 
                                                                        depth_trunc=depth_trunc, 
                                                                        convert_rgb_to_intensity=False)

            T_c_w_np = torch.inverse(T_w_c).detach().cpu().numpy()

            volume.integrate(cur_rgbd, cam_intrinsic_o3d, T_c_w_np)

        tsdf_fusion_mesh = volume.extract_triangle_mesh()

        if output_path is not None:
            o3d.io.write_triangle_mesh(str(output_path), tsdf_fusion_mesh)
            print(f"Save the mesh resulting from TSDF fusion to {output_path}")

        return tsdf_fusion_mesh


    # joint optimization of PIN map and the poses in the sliding window
    # neural points are static in this case, we only fine-tune the neural point features for the map updating
    def bundle_adjustment(
        self, iter_count, window_size: int = 50, use_lie_group: bool = False
    ):

        import pypose as pp

        current_poses_mat = self.used_poses

        opt_window_size = min(current_poses_mat.shape[0], window_size)

        if use_lie_group:  # SE3
            current_poses_se3_opt = torch.nn.Parameter(
                pp.from_matrix(
                    current_poses_mat[-opt_window_size:], ltype=pp.SE3_type, check=False
                )
            )  # optimizable part
            poses_se3_fix = pp.from_matrix(
                current_poses_mat[:-opt_window_size], ltype=pp.SE3_type, check=False
            )  # fixed part
        else:  # se3
            current_poses_se3_opt = torch.nn.Parameter(pp.from_matrix(current_poses_mat[-opt_window_size:], ltype=pp.SE3_type, check=False).Log())    
            # optimizable part
            poses_se3_fix = pp.from_matrix(current_poses_mat[:-opt_window_size], ltype=pp.SE3_type, check=False).Log()    
            # fixed part

        # neural_point_feat = list(self.neural_points.parameters())
        neural_point_feat = [self.neural_points.local_geo_features, self.neural_points.local_color_features]

        # also add the poses as param here, for pose refinement (bundle ajustment)
        opt = setup_optimizer(
            self.config, neural_point_feat, 
            poses=current_poses_se3_opt, lr_ratio=self.config.lr_ba_map/self.config.lr
        )

        for iter in tqdm(range(iter_count), disable=self.silence):

            coord_ba, weight, ts = self.get_ba_samples(self.config.ba_bs)
            weight = weight.detach()

            current_poses_se3 = torch.cat([poses_se3_fix, current_poses_se3_opt], dim=0)

            if use_lie_group:
                poses = current_poses_se3[ts]  # SE3
            else:
                poses = (current_poses_se3[ts]).Exp()  # se3 -> SE3

            coord = poses.to(coord_ba) @ coord_ba

            sdf_pred = self.sdf(coord)[0]

            # if not self.config.loss_weight_on:
            weight = 1.0

            # calculate the weighted l2 loss
            # cur_loss = (weight * (sdf_pred**2)).mean()

            cur_loss = ((sdf_pred) ** 2).mean()  # don't weight

            # print(torch.sqrt(cur_loss))

            opt.zero_grad(set_to_none=True)
            cur_loss.backward(retain_graph=False)
            opt.step()

        # update the global map
        self.neural_points.assign_local_to_global()

        # update the poses after ba
        current_poses_se3 = torch.cat([poses_se3_fix, current_poses_se3_opt], dim=0)

        updated_poses_mat = current_poses_se3.detach().matrix()

        self.used_poses = updated_poses_mat

        # diff_pose = torch.matmul(torch.inverse(current_poses_mat), updated_poses_mat)
        # print(diff_pose[-opt_window_size:])

        updated_poses_np = updated_poses_mat.cpu().numpy()

        if self.config.pgo_on:
            self.dataset.pgo_poses[:self.dataset.processed_frame+1] = updated_poses_np
            # odom pose would not be changed in this case (odom pose is without ba)
            # update pgo odom edge
        elif self.config.track_on:
            self.dataset.odom_poses[:self.dataset.processed_frame+1] = updated_poses_np

        # FIXME
        self.dataset.cur_pose_ref = updated_poses_np[-1]
        self.dataset.last_pose_ref = updated_poses_np[-1]

        self.ba_done_flag = True

    # short-hand function
    def sdf(self, x, get_std=False):
        geo_feature, _, weight_knn, _, _ = self.neural_points.query_feature(x)
        sdf_pred = self.sdf_mlp.sdf(
            geo_feature
        )  # predict the scaled sdf with the feature # [N, K, 1]
        sdf_std = None
        if not self.config.weighted_first:
            sdf_pred_mean = torch.sum(sdf_pred * weight_knn, dim=1)  # N
            if get_std:
                sdf_var = torch.sum(
                    (weight_knn * (sdf_pred - sdf_pred_mean.unsqueeze(-1)) ** 2), dim=1
                )
                sdf_std = torch.sqrt(sdf_var).squeeze(1)
            sdf_pred = sdf_pred_mean.squeeze(1)
        return sdf_pred, sdf_std

    # TODO
    def get_gaussians(self, x):
        # directly predict using this parameter, there's even no interpolation
        # so should be even faster
        # for x,y,z and rotation, update them together with the neural points after the pgo update
        # define the number of gaussians here, we can even do level of details, small K for faraway gaussians during rendering
        # then maybe you can have different decoders for the prediction of different number of gaussians
        # get up ealier tomorrow to finish this

        return 0

    # get numerical gradient (smoother than analytical one) with a fixed step
    def get_numerical_gradient(self, x, sdf_x=None, eps=0.02, two_side=True):

        N = x.shape[0]

        eps_x = torch.tensor([eps, 0.0, 0.0], dtype=x.dtype, device=x.device)  # [3]
        eps_y = torch.tensor([0.0, eps, 0.0], dtype=x.dtype, device=x.device)  # [3]
        eps_z = torch.tensor([0.0, 0.0, eps], dtype=x.dtype, device=x.device)  # [3]

        if two_side:
            x_pos = x + eps_x
            x_neg = x - eps_x
            y_pos = x + eps_y
            y_neg = x - eps_y
            z_pos = x + eps_z
            z_neg = x - eps_z

            x_posneg = torch.concat((x_pos, x_neg, y_pos, y_neg, z_pos, z_neg), dim=0)
            sdf_x_posneg = self.sdf(x_posneg)[0].unsqueeze(-1)

            sdf_x_pos = sdf_x_posneg[:N]
            sdf_x_neg = sdf_x_posneg[N : 2 * N]
            sdf_y_pos = sdf_x_posneg[2 * N : 3 * N]
            sdf_y_neg = sdf_x_posneg[3 * N : 4 * N]
            sdf_z_pos = sdf_x_posneg[4 * N : 5 * N]
            sdf_z_neg = sdf_x_posneg[5 * N :]

            gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
            gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
            gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)

        else:
            x_pos = x + eps_x
            y_pos = x + eps_y
            z_pos = x + eps_z

            x_all = torch.concat((x_pos, y_pos, z_pos), dim=0)
            sdf_x_all = self.sdf(x_all)[0].unsqueeze(-1)

            sdf_x = sdf_x.unsqueeze(-1)

            sdf_x_pos = sdf_x_all[:N]
            sdf_y_pos = sdf_x_all[N : 2 * N]
            sdf_z_pos = sdf_x_all[2 * N :]

            gradient_x = (sdf_x_pos - sdf_x) / eps
            gradient_y = (sdf_y_pos - sdf_x) / eps
            gradient_z = (sdf_z_pos - sdf_x) / eps

        gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=1)  # [...,3]

        return gradient

    # as the strategy in Neuralangelo, [not used]
    def get_numerical_gradient_multieps(self, x, sdf_x, certainty, eps, two_side=True):

        N = x.shape[0]

        eps_vec = torch.ones_like(certainty) * eps
        eps_vec[certainty > 2.0] *= 0.5
        eps_vec[certainty > 20.0] *= 0.5
        eps_vec[certainty > 100.0] *= 0.5

        eps_vec = eps_vec.unsqueeze(1)

        zeros_vector = torch.zeros_like(eps_vec)
        eps_x = torch.cat([eps_vec, zeros_vector, zeros_vector], dim=1)
        eps_y = torch.cat([zeros_vector, eps_vec, zeros_vector], dim=1)
        eps_z = torch.cat([zeros_vector, zeros_vector, eps_vec], dim=1)

        if two_side:
            x_pos = x + eps_x
            x_neg = x - eps_x
            y_pos = x + eps_y
            y_neg = x - eps_y
            z_pos = x + eps_z
            z_neg = x - eps_z

            x_posneg = torch.concat((x_pos, x_neg, y_pos, y_neg, z_pos, z_neg), dim=0)
            sdf_x_posneg = self.sdf(x_posneg)[0].unsqueeze(-1)

            sdf_x_pos = sdf_x_posneg[:N]
            sdf_x_neg = sdf_x_posneg[N : 2 * N]
            sdf_y_pos = sdf_x_posneg[2 * N : 3 * N]
            sdf_y_neg = sdf_x_posneg[3 * N : 4 * N]
            sdf_z_pos = sdf_x_posneg[4 * N : 5 * N]
            sdf_z_neg = sdf_x_posneg[5 * N :]

            gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps_vec)
            gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps_vec)
            gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps_vec)

        gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=1)  # [...,3]

        return gradient
