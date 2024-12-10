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
    voxel_down_sample_torch,
    remove_gpu_cache,
    slerp_pose,
)

from eval.eval_mesh_utils import eval_pair

from gaussian_splatting.gaussian_renderer import render, spawn_gaussians
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, sky_bce_loss, sky_mask_loss, normal_smooth_loss, tukey_loss, opacity_entropy_loss
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.general_utils import rotation2normal
from gaussian_splatting.utils.sh_utils import RGB2SH, SH2RGB
from gaussian_splatting.scene.cameras import CamImage

from fused_ssim import fused_ssim

from gs_gui.gui_utils import VisPacket

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

        self.decoders = decoders

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
        # short-term memory
        self.cam_short_term_train_pool = []
        # self.cam_short_term_train_ids = []

        self.train_cam_uid = [] 

        # long-term memory
        self.cam_long_term_train_pool = []

        self.cam_img_test_pool = []
        self.test_cam_uid = [] 

        self.gs_train_frame_count: int = 0 # only consider the time frame (so if it's a multi-cam system, multi-cam images belong to a single frame)
        self.sdf_train_frame_count: int = 0

        # used training views in this frame # for visualization
        self.cur_frame_train_views = None

        # current exposure parameters for each camera
        self.cams_exposure_ab = {}
        self.per_cam_exposure_ab = {} # dict of dict, contains dict per cam
        
        # camera pose minor adjustment
        self.cams_pose_delta_rt = {}
        self.per_cam_pose_delta_rt = {}

        self.gs_total_iter = 0
        self.gs_iter_window = config.gs_bs * config.gs_iters * config.img_pool_size

        self.T_w_c_cur_view = None # validate render view camera pose
        
        # evaluation
        self.rendered_pcd_o3d = None # rerendered point cloud

        # init the lists for gs evaluation
        self.init_gs_eval()

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device) 


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

        # better to project the camera frame here (better also with the image timestamp)

        # dynamic filtering
        self.static_mask = torch.ones(
            frame_point_torch.shape[0], dtype=torch.bool, device=self.config.device
        )

        if filter_dynamic:
            # reset local map (consider the frame description for loop with latency) 
            self.neural_points.reset_local_map(frame_origin_torch, frame_orientation_torch, frame_id)

            # transformed to the global frame
            frame_point_torch_global = transform_torch(frame_point_torch, cur_pose_torch)

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

        self.dataset.static_mask = self.static_mask

        T1 = get_time()

        # sampling data for training (only using the actually measured points, no mono priors)
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

        self.sdf_train_frame_count += 1 # sample points from point cloud for sdf training

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
            
        # prune map and recreate hash (not used)
        if self.config.prune_map_on and ((frame_id + 1) % self.config.prune_freq_frame == 0):
            if self.neural_points.prune_map(self.config.max_prune_certainty):
                self.neural_points.recreate_hash(None, None, True, True, frame_id)
        # TODO: we can prune those free gaussians that has a very small opacity? # TODO: there's some floating gaussians in the sky due to wrong mono depth initialization
        

        # update neural point map
        self.neural_points.update(
            update_points, update_colors, update_normals, frame_origin_torch, frame_orientation_torch, frame_id
        )

        # update gaussians using mono depth predictions # TODO
        if mono_depth_point_cloud_torch is not None and self.config.monodepth_on: 
            # use the mono depth estimation results to do the initialization
            mono_depth_point_cloud_torch[:, :3] = transform_torch(mono_depth_point_cloud_torch[:, :3], cur_pose_torch)

            # also need to transform the normal
            
            # we currently use a easy fix for ground robot to use only the points with large height value
            if self.dataset.loader.mono_depth_for_high_z:
                update_points_min_z_quantile = torch.quantile(update_points[:, 2], 0.98) + self.config.voxel_size_m # TODO # height ?
                mono_depth_point_used_mask = mono_depth_point_cloud_torch[:, 2] > update_points_min_z_quantile
            else: # mono depth for low z
                update_points_max_z_quantile = torch.quantile(update_points[:, 2], 0.1) + self.config.voxel_size_m
                mono_depth_point_used_mask = mono_depth_point_cloud_torch[:, 2] < update_points_max_z_quantile

            mono_depth_point_cloud_torch = mono_depth_point_cloud_torch[mono_depth_point_used_mask]

            # voxel downsampling (make it sparse) # TODO: but how sparse
            down_voxel_size = self.config.monodepth_gaussian_res # add to config # TODO
            
            if mono_depth_point_cloud_torch.shape[0] > 0:
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

        # record the current map memory
        self.neural_points.record_memory(verbose=(not self.silence), record_footprint=True)

        T3 = get_time()

        # concat with current observations
        # self.coord_pool = torch.cat((self.coord_pool, coord), 0)
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

        # else:  # used when ba is not enabled
        global_coord = transform_torch(coord, cur_pose_torch)
        self.global_coord_pool = torch.cat(
            (self.global_coord_pool, global_coord), 0
        )
            # why so slow

        T3_1 = get_time()

        if (frame_id + 1) % self.config.pool_filter_freq == 0:
            pool_relatve = self.global_coord_pool - frame_origin_torch
            # print(pool_relatve.shape)
            if self.config.range_filter_2d:
                pool_relative_dist = torch.norm(pool_relatve[:,:2], p=2, dim=1)
            else:
                pool_relative_dist = torch.norm(pool_relatve, p=2, dim=1)
                
            dist_mask = pool_relative_dist < self.config.window_radius

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
            # self.coord_pool = self.coord_pool[filter_mask]
            # make global here
            self.global_coord_pool = self.global_coord_pool[filter_mask]   
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
            self.pool_sample_count = self.global_coord_pool.shape[0]

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

        # print("time for dynamic filtering     (ms):", (T1-T0)*1e3)
        # print("time for sampling              (ms):", (T2-T1)*1e3)
        # print("time for map updating          (ms):", (T3-T2)*1e3)
        # print("time for pool updating         (ms):", (T4-T3)*1e3) # mainly spent here
        # print("time for pool transforming     (ms):", (T3_1-T3_0)*1e3) # mainly spent here
        # print("time for filtering             (ms):", (T3_2-T3_1)*1e3)
    
    def dynamic_filter(self, points_torch, type_2_on: bool = True):

        if type_2_on:
            points_torch.requires_grad_(True)

        geo_feature, _, weight_knn, _, certainty = self.neural_points.query_feature(
            points_torch, accumulate_stability=False
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
            self.neural_points.local_neural_points, accumulate_stability=False
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

    def update_cam_pool(self, frame_id: int):

        if self.dataset.cur_cam_img is None:
            return

        # set camera poses
        cur_cam_names = list(self.dataset.cur_cam_img.keys())
        for cam_name in cur_cam_names: # for each cam in this frame
            cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]

            T_w_l = self.used_poses[frame_id] # already in torch tensor, lidar pose (in the lidar deskewed reference frame)

            T_c_l = torch.tensor(self.dataset.T_c_l_mats[cam_name], device=self.device) 

            diff_pose_l_c_ts = torch.eye(4).to(T_w_l)

            # relative transformation between the lidar reference timestamp and the camera triggering timestamp
            if frame_id > 0 and self.dataset.cur_sensor_ts is not None:
                cur_cam_ref_ts_ratio = self.dataset.get_cur_cam_ref_ts_ratio(cam_name)
                # print(cur_cam_ref_ts_ratio)
                diff_pose_l_c_ts = slerp_pose(self.dataset.last_odom_tran_torch, cur_cam_ref_ts_ratio, self.config.deskew_ref_ratio).to(T_w_l)
                # print(diff_pose_l_c_ts)

            T_w_l_cam_ts = T_w_l @ diff_pose_l_c_ts
            T_w_c = T_w_l_cam_ts @ torch.linalg.inv(T_c_l) # need to convert to cam frame # Here there could be different cameras, support this
            cur_view_cam.set_pose(T_w_c) # set camera pose

            # print(T_w_c)

            cur_view_cam.free_memory_under_levels(min(self.config.gs_down_rate, self.config.gs_vis_down_rate)-1)
        
        keyframe_on = (self.gs_train_frame_count == 0) or \
            (self.dataset.accu_travel_dist_for_keyframe > self.config.gs_keyframe_accu_travel_dist) or \
            (self.dataset.accu_travel_degree_for_keyframe > self.config.gs_keyframe_accu_travel_degree)

        # training views
        if keyframe_on and frame_id % self.config.gs_keyframe_interval==0:
            
            self.gs_train_frame_count += 1
            self.dataset.accu_travel_dist_for_keyframe = 0.0 # set back to zero
            self.dataset.accu_travel_degree_for_keyframe = 0.0 # set back to zero

            if not self.silence:
                print("New training frame added for frame {}".format(frame_id))

            # better to use the newly added gaussians ratio (FIXME)
            # move oldest short-term memory to long-term memory
            while len(self.cam_short_term_train_pool) > self.config.img_pool_size:
                oldest_short_term_train_cam = self.cam_short_term_train_pool[0]
                if self.config.long_term_train_down:
                    oldest_short_term_train_cam.free_memory_at_level(self.config.gs_down_rate)
                oldest_short_term_train_cam.in_long_term_memory = True
                self.cam_long_term_train_pool.append(oldest_short_term_train_cam)
                self.cam_short_term_train_pool.pop(0) # pop the oldest cam

            # add new observations to short-term memory
            for cam_name in cur_cam_names:
                cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                cur_view_cam.train_view = True
                self.cam_short_term_train_pool.append(cur_view_cam)
                self.train_cam_uid.append(cur_view_cam.uid)
            
            # range filter
            view_range = 0.5 * (self.config.sorrounding_map_radius + self.config.max_range)
            self.cam_long_term_train_pool = [cam_long_term for cam_long_term in self.cam_long_term_train_pool \
             if (torch.norm(self.used_poses[cam_long_term.frame_id,:3,3]-self.used_poses[frame_id,:3,3], 2) \
                < view_range)] 

            # capacity filter
            if len(self.cam_long_term_train_pool) > self.config.long_term_pool_size:
                self.cam_long_term_train_pool = random.sample(self.cam_long_term_train_pool, self.config.long_term_pool_size)
                # make sure the memory are freed

        # also add some testing views (all the others are then testing views), now it's deprecated, we do not do online evaluation
        else:
            
            if self.config.img_test_pool_size > 0:
                for cam_name in cur_cam_names:
                    cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                    self.test_cam_uid.append(cur_view_cam.uid)

                if len(self.cam_img_test_pool) > self.config.img_test_pool_size:
                    self.cam_img_test_pool.pop(0) # pop the oldest cam
                    
                cam_name = self.dataset.loader.main_cam_name
                cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                cur_view_cam.train_view = False

                self.cam_img_test_pool.append(cur_view_cam)

                # print(self.cam_short_term_train_pool_id)

    # update cam poses after loop pgo
    def update_poses_cam_pool(self, poses_after_pgo):

        for cam in self.cam_short_term_train_pool:
            cam_pose_after_pgo = poses_after_pgo[cam.frame_id]
            if isinstance(cam_pose_after_pgo, np.ndarray):
                cam_pose_after_pgo = torch.tensor(cam_pose_after_pgo, device=self.device, dtype=self.dtype)
            cam.set_pose(cam_pose_after_pgo)

        for cam in self.cam_long_term_train_pool:
            cam_pose_after_pgo = poses_after_pgo[cam.frame_id]
            if isinstance(cam_pose_after_pgo, np.ndarray):
                cam_pose_after_pgo = torch.tensor(cam_pose_after_pgo, device=self.device, dtype=self.dtype)
            cam.set_pose(cam_pose_after_pgo)

    # get a batch of training samples and labels for map optimization
    def get_batch(self, global_coord=True):

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

        coord = self.global_coord_pool[index, :]

        # if global_coord:
        #     coord = self.global_coord_pool[index, :]
        # else:
        #     coord = self.coord_pool[index, :]

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

    # transform the data pool after pgo pose correction
    def transform_data_pool(self, pose_diff_torch: torch.tensor):
        # pose_diff_torch [N,4,4]
        self.global_coord_pool = transform_batch_torch(
            self.global_coord_pool, pose_diff_torch[self.time_pool]
        )

    def free_pool(self):
        self.coord_pool = None
        self.global_coord_pool = None
        self.weight_pool = None
        self.sdf_label_pool = None
        self.time_pool = None
        self.sem_label_pool = None
        self.color_pool = None
        self.normal_label_pool = None

    # PIN map online training (mapping) given the fixed pose
    # the main training function
    def mapping(self, iter_count):

        iter_count += self.adaptive_iter_offset

        if iter_count <= 0 or self.neural_points.is_empty():
            return # skip the mapping

        # neural_point_feat = [self.neural_points.local_geo_features, self.neural_points.local_color_features]

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
            self.neural_points.local_geo_features,
            self.neural_points.local_color_features,
            sdf_mlp_param,
            sem_mlp_param,
            color_mlp_param,
        )

        for iter in tqdm(range(iter_count), disable=self.silence, desc="SDF training"):
            # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)

            T00 = get_time()
            # we do not use the ray rendering loss here for the incremental mapping
            coord, sdf_label, ts, _, sem_label, color_label, weight = self.get_batch(
                global_coord=True
            )  # coord here is in global frame if no ba pose update

            T01 = get_time()

            poses = self.used_poses[ts]
            origins = poses[:, :3, 3]

            surface_mask = torch.abs(sdf_label) < self.config.surface_sample_range_m

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
                color_pred = self.color_mlp.regress_color(color_feature[surface_mask])  # [N, K, C]
                if not self.config.weighted_first:
                    color_pred = torch.sum(color_pred * weight_knn[surface_mask], dim=1)  # N, C

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
                    color_pred,
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


    # jointly optimize the neural point features and gaussian parameters
    def joint_gsdf_mapping(self, iter_count: int, sdf_loss_on = True,
         online_eval_on = False, lpips_eval_on = False, render_pcd = False):
        
        # neural_point_feat = [self.neural_points.local_geo_features, self.neural_points.local_color_features]

        cams_param = self.cam_short_term_train_pool
        # cams_param = self.cam_short_term_train_pool if self.config.exposure_correction_on else None

        mlp_color_param = list(self.color_mlp.parameters()) if self.color_mlp is not None else None

        opt = setup_optimizer(
            self.config,
            self.neural_points.local_geo_features,
            self.neural_points.local_color_features,
            mlp_sdf_param=list(self.sdf_mlp.parameters()),
            mlp_color_param=mlp_color_param,
            mlp_gs_xyz_param=list(self.gaussian_xyz_mlp.parameters()),
            mlp_gs_scale_param=list(self.gaussian_scale_mlp.parameters()),
            mlp_gs_rot_param=list(self.gaussian_rot_mlp.parameters()),
            mlp_gs_alpha_param=list(self.gaussian_alpha_mlp.parameters()),
            mlp_gs_color_param=list(self.gaussian_color_mlp.parameters()),
            cams = cams_param,
        )

        self.cur_frame_train_views = {} # set back to empty

        background = torch.tensor(self.config.bg_color, dtype=self.dtype, device=self.device)
        bg_3d = background.view(3, 1, 1)

        cur_local_map_center = self.used_poses[-1,:3,3]

        neural_points_data, sorrounding_neural_points_data = self.neural_points.gather_local_data()
        
        # FIXME
        # or for sorrounding neural points, we still feed it through the mlp, rendering might be slower a bit, but these points are not optimizable, I think this idea is better
        # determine the sorrounding neural points, pre-filter them, and also feed them through the MLP
        # then maybe we can detach them, or ? the features are not optimizable

        # spawn only once
        sorrounding_spawn_results = None
        if self.config.decoder_freezed: # decoder already freezed
            # print("Spawn sorrounding gaussians !!! ")
            sorrounding_spawn_results = spawn_gaussians(sorrounding_neural_points_data, 
                        self.decoders, None, cur_local_map_center.float(),
                        dist_concat_on=self.config.dist_concat_on, 
                        view_concat_on=self.config.view_concat_on, 
                        scale_filter_on=True,
                        z_far=self.config.sorrounding_map_radius,
                        learn_color_residual=self.config.learn_color_residual,
                        gs_type=self.config.gs_type,
                        displacement_range_ratio=self.config.displacement_range_ratio,
                        max_scale_ratio=self.config.max_scale_ratio,
                        unit_scale_ratio=self.config.unit_scale_ratio)        

        # also have a valid mask for the neural points

        # better to also include the global map
        # then render them in a more efficient way (their features are not optimizable)
        
        short_term_img_pool_size = len(self.cam_short_term_train_pool)
        long_term_img_pool_size = len(self.cam_long_term_train_pool)

        if iter_count > 0 and short_term_img_pool_size > 0:

            # print("GS fitting on ")
        
            # TODO: For original GS
            # fastest speed: 20 ms / iter (bs=1) including the gaussian parameter loss
            # fastest speed: 10 ms / iter (bs=1) excluding the gaussian parameter loss (but we need to constriant these gaussians)

            # self.neural_points.training_setup_gs(with_pin_feature=self.config.pin_gs_opt_on)

            renderd_image = None

            # still too slow, figure it out how to make the process faster

            down_rate_short_term = self.config.gs_down_rate # TODO: add to config. img downsample rate 2**down_rate, if down_rate=0, then use original img
            
            if self.config.long_term_train_down:
                down_rate_long_term = down_rate_short_term + 1
            else:
                down_rate_long_term = down_rate_short_term

            eval_depth_max = self.config.max_range
            eval_depth_min = self.config.min_range

            assert short_term_img_pool_size > 0, "At least one frame for training is required"

            cam_count = len(self.dataset.cam_names) # TODO

            for iter in tqdm(range(iter_count), disable=self.silence, desc="GSDF training"):    

                # camera poses already set
                T1 = get_time()
                
                cur_min_visible_neural_point_ratio = 0.01 # don't restrict this to much

                short_term_train_prob = 1 - (1-self.config.short_term_train_prob)*(long_term_img_pool_size/self.config.long_term_pool_size)

                # 50 % short term (10% latest), 50 % long term history
                dice_number = random.random()
                if dice_number < short_term_train_prob: # [ 0, 1 ], 0.5 then means 50 % prob.
                    # short-term memory 
                    cur_img_idx = torch.randperm(short_term_img_pool_size)[0]
                    if dice_number < self.config.lastest_train_prob: # train more on the most recent imgs
                        cur_img_idx = -torch.randperm(cam_count)[0]
                    viewpoint_cam: CamImage = self.cam_short_term_train_pool[cur_img_idx]
                    train_down_rate = down_rate_short_term
                    weight_down_rate = 1.0

                    is_replay_mode = False
                    # if not self.silence:
                    #     print(" Train on a cam from short-term memory")
                    #     print(" Used cam id:", viewpoint_cam.uid)
                else:
                    # long-term memory
                    
                    cur_img_idx = torch.randperm(long_term_img_pool_size)[0]
                    viewpoint_cam: CamImage = self.cam_long_term_train_pool[cur_img_idx]
                    train_down_rate = down_rate_long_term
                    weight_down_rate = 4^(down_rate_long_term-down_rate_short_term)
                    dist_to_cur_frame = torch.norm((viewpoint_cam.camera_center - cur_local_map_center), 2)
                    if dist_to_cur_frame > self.config.local_map_radius: # out of the local map, then we have a min threshold for visual neural point in the local map
                        cur_min_visible_neural_point_ratio = self.config.min_visible_neural_point_ratio
                    is_replay_mode = True

                    # if not self.silence:
                    #     print(" Train on a cam from long-term memory")
                    #     print(" Used cam id:", viewpoint_cam.uid)

                cam_name = viewpoint_cam.cam_id

                gt_rgb_image = viewpoint_cam.rgb_image_list[train_down_rate] # 3, H, W

                # if self.config.exposure_correction_on and not self.silence:
                #     cur_exposure_a = viewpoint_cam.exposure_a.item()
                #     cur_exposure_b = viewpoint_cam.exposure_b.item()
                #     print("Cur cam view exposure coefficients {:.3f}, {:.3f}".format(cur_exposure_a, cur_exposure_b))
                
                if viewpoint_cam.depth_on:
                    gt_depth_image = viewpoint_cam.depth_image_list[train_down_rate] # 1, H, W
                else:
                    gt_depth_image = None

                T2 = get_time()

                # render gaussians
                render_pkg = render(viewpoint_cam, None, neural_points_data, 
                    self.decoders, sorrounding_spawn_results, background, down_rate=train_down_rate, 
                    min_visible_neural_point_ratio=cur_min_visible_neural_point_ratio,
                    verbose=(not self.silence),
                    replay_mode=is_replay_mode, 
                    dist_concat_on=self.config.dist_concat_on, 
                    view_concat_on=self.config.view_concat_on, 
                    correct_exposure=self.config.exposure_correction_on,
                    correct_exposure_affine=self.config.affine_exposure_correction,
                    learn_color_residual=self.config.learn_color_residual,
                    front_only_on=self.config.train_front_only,
                    d2n_on=(self.config.lambda_normal_depth_consist > 0.0),
                    gs_type=self.config.gs_type,
                    displacement_range_ratio=self.config.displacement_range_ratio,
                    max_scale_ratio=self.config.max_scale_ratio,
                    unit_scale_ratio=self.config.unit_scale_ratio)
                    
                if render_pkg is None:
                    continue
                
                # record the cam views used for training at this timestep
                self.cur_frame_train_views[viewpoint_cam.uid] = viewpoint_cam

                T3 = get_time()

                if "local_view_gaussian_count" in list(render_pkg.keys()):
                    local_visible_mask = visible_mask[:render_pkg["local_view_gaussian_count"]]
                else:
                    continue

                # rendered results
                rendered_rgb_image = render_pkg["render"] # 3, H, W 
                rendered_normal = render_pkg['rend_normal'] # 3, H, W # rendered normal
                rendered_depth = render_pkg["surf_depth"] # 1, H, W # rendered depth
                depth_normal = render_pkg['surf_normal'] # 3, H, W # calculated from the depth map (depth --> normal), D2N
                rendered_alpha = render_pkg["rend_alpha"] # 1, H, W accumulated opacity 
                dist_distortion = render_pkg["rend_dist"] # depth distortion # 1, H, W 

                # Gaussians information
                visible_mask = render_pkg["visibility_filter"] # gaussian visibility mask, this is for the spawned gaussians (include those sorrounding part)
                
                # these are only for those spawned gaussians in the local map
                gaussian_xyz = render_pkg["gaussian_xyz"]
                gaussian_scale = render_pkg["gaussian_scale"]
                gaussian_rot = render_pkg["gaussian_rot"]
                gaussian_alpha = render_pkg["gaussian_alpha"]
                gaussian_color = render_pkg["gaussian_color"]
                gaussian_free_mask = render_pkg["gaussian_free_mask"]
                visible_neural_point_ratio = render_pkg["visible_neural_point_ratio"]

                # print(" Visible neural point ratio: {:.2f}".format(visible_neural_point_ratio))

                alpha_all = render_pkg["alpha_all"]
                if alpha_all is not None:
                    gaussian_alpha_mask = alpha_all > 0.0

                local_gaussian_count = gaussian_xyz.shape[0]
                gaussian_count_per_point = self.gaussian_xyz_mlp.out_k

                # alpha_valid_part = alpha_all[gaussian_alpha_mask]

                T3_1 = get_time()

                # better to don't do this, the norm actually represent the opacity
                # # normalize the normals to norm == 1
                # if rendered_normal is not None:
                #     rendered_normal = torch.nn.functional.normalize(rendered_normal, dim=0) 
                # if depth_normal is not None:
                #     depth_normal = torch.nn.functional.normalize(depth_normal, dim=0) 
                
                
                T3_2 = get_time()

                # ----------------
                # Sky mask loss
                sky_loss = 0.0
                if viewpoint_cam.sky_mask_on: 
                    cur_sky_mask = viewpoint_cam.sky_mask_list[train_down_rate]
                    
                    sky_pixel_count = torch.sum(cur_sky_mask).item()

                    # print(cur_sky_mask)
                    non_sky_mask = ~cur_sky_mask
                    
                    # TODO: something wrong here
                    if self.config.lambda_sky > 0 and sky_pixel_count > 0 and rendered_alpha is not None:
                        sky_loss = sky_mask_loss(cur_sky_mask, rendered_alpha) # sky part has 0 alpha
                        # sky_loss = sky_bce_loss(cur_sky_mask, rendered_alpha) # let the sky part has small opacity, the others have a large opacity
                        if not self.silence:
                            print(" Sky loss:", sky_loss.item())
                        sky_loss *= self.config.lambda_sky                  # sky mask loss
                    
                    if rendered_normal is not None:
                        rendered_normal = rendered_normal * non_sky_mask
                    if depth_normal is not None:
                        depth_normal = depth_normal * non_sky_mask
                    if dist_distortion is not None:
                        dist_distortion = dist_distortion * non_sky_mask
                    
                    # masked the sky part as the background color
                    # mask_broadcasted = cur_sky_mask.repeat(3,1,1)
                    # gt_rgb_image[mask_broadcasted] = bg_3d.expand_as(gt_rgb_image)[mask_broadcasted]

                T3_3 = get_time()

                # ----------------
                # RGB rendering loss (combining L1 and SSIM), slow part

                # special issue for ipb car dataset (rear camera, remove the ego-car part for loss calculation)
                if cam_name == "rear": # only for ipb car dataset (FIXME), use mask in the future, now it's just a ugly quick fix
                    pixel_h_used = int(910/1024*gt_rgb_image.shape[1])
                elif cam_name == "front":
                    pixel_h_used = int(990/1024*gt_rgb_image.shape[1])
                else:  
                    pixel_h_used = -1

                rendered_rgb_image_for_loss = rendered_rgb_image[:,:pixel_h_used,:]
                gt_rgb_image_for_loss = gt_rgb_image[:,:pixel_h_used,:]

            
                loss_rgb_l1 = l1_loss(rendered_rgb_image_for_loss, gt_rgb_image_for_loss)
                # loss_rgb_robust = tukey_loss(rendered_rgb_image_for_loss, gt_rgb_image_for_loss) # when you set c=0, it would be just as the original l1 loss
                if self.config.lambda_ssim > 0.0:
                    # ssim_value = ssim(rendered_rgb_image_for_loss, gt_rgb_image_for_loss)
                    ssim_value = fused_ssim(rendered_rgb_image_for_loss.unsqueeze(0), gt_rgb_image_for_loss.unsqueeze(0)) # have to be 4 dim
                    rgb_loss = (1.0 - self.config.lambda_ssim) * loss_rgb_l1 + self.config.lambda_ssim * (1.0 - ssim_value)
                else:
                    rgb_loss = loss_rgb_l1 # l1 only, ssim might take a long time

                T3_4 = get_time()

                # ----------------
                # Depth rendering loss
                depth_loss = 0.0
                valid_depth_mask = None
                if rendered_depth is not None and gt_depth_image is not None and self.config.lambda_depth > 0:
                    valid_depth_mask = (gt_depth_image > eval_depth_min) & (gt_depth_image < eval_depth_max)
                    if rendered_alpha is not None:
                        accu_alpha_mask = rendered_alpha.detach() > self.config.depth_min_accu_alpha
                        valid_depth_mask = valid_depth_mask & accu_alpha_mask
                    # gt_depth_image = gt_depth_image[valid_depth_mask]
                    # # print(gt_depth_image)
                    # rendered_depth = rendered_depth[valid_depth_mask]
                    if self.config.inverse_depth_loss:
                        depth_loss = l1_loss(1.0/gt_depth_image, 1.0/rendered_depth, valid_depth_mask) # use inverse depth (then we will care more about the close range part)
                        # depth_loss = tukey_loss(1.0/gt_depth_image, 1.0/rendered_depth) 
                        # if not self.silence:
                        #     print(" Inverse depth rendering loss:", depth_loss.item())
                    else:
                        depth_loss = l1_loss(gt_depth_image, rendered_depth, valid_depth_mask)
                        # if not self.silence:
                        #     print(" Depth rendering loss (m):", depth_loss.item())
                    depth_loss *= (weight_down_rate * self.config.lambda_depth)

                T3_5 = get_time()

                # ----------------

                # Regularization losses
                # this normal consistency regularization loss seems to have some problem, figure it out (FIXME)
                # if valid_depth_mask is not None:
                #     if rendered_normal is not None:
                #         rendered_normal = rendered_normal[:, valid_depth_mask]
                #     if depth_normal is not None:
                #         depth_normal = depth_normal[:, valid_depth_mask]

                if rendered_normal is not None:
                    rendered_normal_norm = rendered_normal.norm(2, dim=0).detach()

                # ----------------
                # Normal-Depth consistency regularization loss
                normal_depth_consist_loss = 0.0
                if rendered_normal is not None and depth_normal is not None and self.config.lambda_normal_depth_consist > 0.0:
                    depth_normal_norm = depth_normal.norm(2, dim=0).detach()
                    normal_valid_mask = (rendered_normal_norm > 0) & (depth_normal_norm > 0)
                    if self.config.gs_consist_normal_fixed:
                        dot_product = (rendered_normal.detach() * depth_normal).sum(dim=0) # we detach here to use the normal to supervise depth
                    elif self.config.gs_consist_depth_fixed:
                        dot_product = (rendered_normal * depth_normal.detach()).sum(dim=0) # we detach here to use the depth to supervise normal
                    else:
                        dot_product = (rendered_normal * depth_normal).sum(dim=0)

                    # dot_product = (rendered_normal.detach() * depth_normal).sum(dim=0) # H, W 
                    normal_error = (depth_normal_norm * rendered_normal_norm) - dot_product # dot product  # kind of weighted with alpha
                    normal_error_valid = torch.masked_select(normal_error, normal_valid_mask)
                    normal_depth_consist_loss = normal_error_valid.mean() # still does not work well, disable it
                    if not self.silence:
                        print(" Normal-depth consistency loss:", normal_depth_consist_loss.item())
                    normal_depth_consist_loss *= self.config.lambda_normal_depth_consist
                
                # ----------------
                # Normal smoothness loss
                normal_smoothness_loss = 0.0
                if rendered_normal is not None and rendered_depth is not None and self.config.lambda_normal_smooth > 0.0:
                    normal_valid_mask = (rendered_normal_norm > 0)
                    normal_smoothness_loss = normal_smooth_loss(rendered_normal, rendered_depth, normal_valid_mask, depth_jump_thre_m=self.config.vox_down_m)
                    if not self.silence:
                        print(" Normal smoothness loss:", normal_smoothness_loss.item())
                    normal_smoothness_loss *= self.config.lambda_normal_smooth
                
                # ----------------
                # Mono normal regularization loss
                mono_normal_loss = 0
                if rendered_normal is not None and viewpoint_cam.mono_normal_on and self.config.lambda_mono_normal > 0:
                    mono_normal = viewpoint_cam.normal_img_list[train_down_rate]
                    mono_normal_norm = mono_normal.norm(2, dim=0) 
                    normal_valid_mask = (rendered_normal_norm > 0) & (mono_normal_norm > 0)
                    dot_product = (rendered_normal * mono_normal).sum(dim=0) # H, W
                    mono_normal_error = 1.0 - dot_product # dot product 
                    mono_normal_error = torch.masked_select(mono_normal_error, normal_valid_mask)
                    mono_normal_loss = mono_normal_error.mean()
                    # if not self.silence:
                    #     print(" Mono normal loss:", mono_normal_loss.item())
                    mono_normal_loss *= self.config.lambda_mono_normal  

                # ----------------
                # Normal along ray interval distance regularization loss
                distort_loss = 0
                if dist_distortion is not None and self.config.lambda_distort > 0:
                    distort_loss = dist_distortion.mean()
                    # if not self.silence:
                    #     print(" Depth distortion loss:", distort_loss.item())
                    distort_loss *= self.config.lambda_distort

                # ----------------
                # Opacity regularization loss (prefer large value, prefer positive value)
                # let the opacity to be ideally larger
                opacity_loss = 0.0
                opacity_ent_loss = 0.0
                constraint_min_alpha = self.config.min_alpha
                if self.config.lambda_opacity > 0 and alpha_all is not None: # better to use the distance to weight this value (smaller distance, larger weight) 
                    masked_alpha_mask = (alpha_all<constraint_min_alpha) # now only let those gaussians has negative opacity to increase their opacity # TODO: something weird
                    if torch.sum(masked_alpha_mask) > 0: 
                        opacity_loss = 0.0 - (alpha_all[masked_alpha_mask]).mean() # alpha value between [0, 2]
                        # if not self.silence:
                        #     print(" Opacity loss:", opacity_loss.item())
                        opacity_loss *= self.config.lambda_opacity
                
                # opacity entropy loss
                if self.config.lambda_opacity_ent > 0 and alpha_all is not None:
                    opacity_ent_loss = opacity_entropy_loss(torch.abs(alpha_all)) 
                    if not self.silence:
                        print(" Opacity entropy loss:", opacity_ent_loss.item())

                    opacity_ent_loss *= self.config.lambda_opacity_ent # TODO: change to lambda_opacity_ent


                # # actually we only need to use the points in the field of view (but this might already been handeled in CUDA)
                # # only these gaussians would be optimized
                # # also with the visible gaussians
                # # batch_visbility_mask is only for the valid gs
                # local_valid_gs_mask = self.neural_points.local_valid_gs_mask

                T3_6 = get_time()

                # only for the local map
                constraint_mask = local_visible_mask # also use the free ones now (FIXME)

                # also only restrict the gaussians with large alpha
                large_alpha_mask = (gaussian_alpha > constraint_min_alpha).squeeze(-1)
                # print(large_alpha_mask.shape)
                constraint_mask = constraint_mask & large_alpha_mask

                # TODO: consider to add again the depth distortion loss

                if gaussian_free_mask is not None:
                    constraint_mask = constraint_mask & (~gaussian_free_mask) # non-free visible gaussians

                constraint_count = torch.sum(constraint_mask).item() # TODO: for isotropic and area loss, we actually also need to consider free gaussians

                # if not self.silence:
                #     print(" # Gaussians for 3D losses in current view:", constraint_count) # non-free visible gaussians

                isotropic_loss = area_loss = sdf_consistency_loss = sdf_normal_consistency_loss = 0.0
                valid_opacity_loss = invalid_opacity_loss = 0.0

                if constraint_count > 10:
                    true_indices = torch.where(constraint_mask)[0]
                    gaussian_bs = int(self.config.bs * self.config.gaussian_bs_ratio) # TODO
                    sample_bs = min(constraint_count, gaussian_bs)  # Number of indices to sample # infer_bs is a bit too large here, TODO: add to config
                    # print("Sampled neural point count: " , sample_bs)
                    # don't use all the points here (random sample some of them as a batch)
                    sampled_indices = true_indices[torch.randperm(constraint_count)[:sample_bs]] # this is already the idx in all the local gaussians

                    # how to find those gaussians only in the training field of views?

                    # ----------------
                    # Gaussian scale istropic loss
                    if self.config.lambda_isotropic > 0 or self.config.lambda_area > 0:
                        scaling = gaussian_scale[sampled_indices]

                    if self.config.lambda_isotropic > 0:
                        if self.config.gs_type == "3d_gs":
                            scaling = scaling[:,:3]
                        else:
                            scaling = scaling[:,:2] # do not use the last one for Gaussian surfels
                        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean()
                        if not self.silence:
                            print(" Gaussian isotropic loss:", isotropic_loss.item())
                        isotropic_loss *= self.config.lambda_isotropic
                    
                    # ----------------
                    # Surfel area regularization loss (test this)
                    if self.config.lambda_area > 0:
                        if self.config.gs_type == "3d_gs":
                            area_loss = (scaling[:,0] * scaling[:,1] * scaling[:,2]).mean() # then it's volume loss
                            area_loss /= (self.config.voxel_size_m**3)
                        else:
                            area_loss = (scaling[:,0] * scaling[:,1]).mean() # but this is already mean value
                            area_loss /= (self.config.voxel_size_m**2) # normalize by the voxel area
                        # if not self.silence:
                        #     print(" Gaussian area loss:", area_loss.item())
                        area_loss *= self.config.lambda_area

                    T4 = get_time()

                    sampled_guassians_alpha = gaussian_alpha[sampled_indices]
                    
                    # Gaussian SDF consistency loss
                    if self.config.lambda_sdf_normal_cons > 0 or self.config.lambda_sdf_cons > 0:
                        
                        sampled_guassians_xyz = gaussian_xyz[sampled_indices]
                        sampled_guassians_normals = rotation2normal(gaussian_rot[sampled_indices]) # N, 3 # this is definitely normalized

                        sampled_count = sampled_guassians_xyz.shape[0]
                        shift_sample_count = self.config.gs_consist_shift_count

                        shift_range = self.config.gs_consist_shift_range_m # TODO: add to config

                        sampled_guassians_xyz_repeat = sampled_guassians_xyz.repeat(shift_sample_count, 1) # RK, 3
                        sampled_guassians_normals_repeat = sampled_guassians_normals.repeat(shift_sample_count, 1) # RK, 3

                        random_shift = (torch.randn(sampled_count * shift_sample_count, device=self.device)-0.5) * 2.0 * shift_range # -shift_range, +shift_range

                        random_shift_vec = sampled_guassians_normals_repeat * random_shift[:, None]  # RK, 3
                        
                        sampled_guassians_shifted_xyz = sampled_guassians_xyz_repeat + random_shift_vec # RK, 3

                        sampled_guassians_shifted_xyz_all = torch.cat((sampled_guassians_xyz, sampled_guassians_shifted_xyz), 0) # (1+R)K, 3

                        sampled_guassians_normals_all = torch.cat((sampled_guassians_normals, sampled_guassians_normals_repeat), 0) # (1+R)K, 3

                        sdf_label_all = torch.cat((torch.zeros(sampled_count, device=self.device), random_shift), 0) # (1+R)K, 3

                        # sampled_guassians_xyz.requires_grad_(True)
                        sampled_guassians_shifted_xyz_all.requires_grad_(True)

                        sampled_guassians_sdf, _, valid_nnk_mask = self.sdf(sampled_guassians_shifted_xyz_all, min_nn_count=3) # self.config.query_nn_k sdf, sdf_std, valid_mask
                        sampled_guassians_sdf_grad = get_gradient(sampled_guassians_shifted_xyz_all, sampled_guassians_sdf) # N, 3 # analytical one # how could the gradient to be zero (if it has no nearby neural points, then maybe)
                        grad_norm = sampled_guassians_sdf_grad.norm(dim=-1, keepdim=True).squeeze()  # unit: m # normalize 
                        
                        # print(grad_norm)
                        # print("Mean SDF grad norm:", grad_norm.mean().item()) # why there are more and more 0 here

                        # maybe relax this a bit

                        valid_grad_mask = (grad_norm < 1.5) & (grad_norm > 0.5) & (valid_nnk_mask)
                        valid_grad_mask_no_shift = valid_grad_mask[:sampled_count] # the original gaussian samples (without shift)

                        # valid_opacity_loss = (1.0 - sampled_guassians_alpha[valid_grad_mask_no_shift].mean()) 
                        invalid_opacity_loss = (sampled_guassians_alpha[~valid_grad_mask_no_shift].mean())
                        
                        if not self.silence:
                            print(" Invalid part opacity loss:", invalid_opacity_loss.item())

                        invalid_opacity_loss *= self.config.lambda_invalid_opacity

                        # TODO: does this really work?
                        # we let these part to be more transparent, but there's seems to have some problem with the poles
                        # Or we'd better use the entropy loss
                        
                        # sampled_opacity_loss *= (10.0 * self.config.lambda_opacity)
                        # opacity_loss += sampled_opacity_loss

                        # valid_cons_mask = valid_grad_mask


                        # valid_grad_mask = valid_grad_mask.detach()
                        valid_grad_count = torch.sum(valid_grad_mask).item()
                        if not self.silence:
                            print(" SDF Valid gaussian count:", valid_grad_count, " from ", valid_grad_mask.shape[0])

                        sdf_consistency_loss = torch.abs(sampled_guassians_sdf[valid_grad_mask] - sdf_label_all[valid_grad_mask]).mean() # gaussians should better lie on the surface

                        sampled_guassians_sdf_grad = sampled_guassians_sdf_grad / (grad_norm.unsqueeze(-1) + 1e-7) # world frame # pointing out of the surface

                        # print(sampled_guassians_sdf_grad)                

                        # gaussian normals should better align with the sdf gradient direction
                        gaussian_normal_error = (1.0 - (sampled_guassians_sdf_grad[valid_grad_mask] * sampled_guassians_normals_all[valid_grad_mask]).sum(dim=1))                           
                        sdf_normal_consistency_loss = gaussian_normal_error.mean()

                        if not self.silence:
                            print(" SDF cons loss:", sdf_consistency_loss.item(), " SDF normal cons loss:", sdf_normal_consistency_loss.item())

                        sdf_consistency_loss *= self.config.lambda_sdf_cons
                        sdf_normal_consistency_loss *= self.config.lambda_sdf_normal_cons
                # ----------------


                T5 = get_time()

                # batch_visbility_mask_neural_points = torch.any(batch_visbility_mask.view(-1, gaussian_count_per_point), dim=1)
                # batch_visbility_mask_neural_points = torch.cat((batch_visbility_mask_neural_points, torch.tensor([False]).to(batch_visbility_mask_neural_points)))
                
                # # self.neural_points.local_geo_features[~batch_visbility_mask_neural_points] # I don't want to optimize those not visible parts

                # ----------------
                # SDF training loss (BCE + Ekional)
                # TODO: add 3D color loss
                sdf_loss = 0.0
                eikonal_loss = 0.0
                color_loss = 0.0

                if sdf_loss_on and self.config.lambda_sdf > 0.0:
                    # with batch size bs (this is done for all the sdf samples in the local map)
                    coord, sdf_label, ts, _, sem_label, color_label, weight = self.get_batch()

                    surface_mask = torch.abs(sdf_label) < 0.5 * self.config.surface_sample_range_m
                    close_to_surface_mask = torch.abs(sdf_label) < self.config.free_sample_end_dist_m

                    poses = self.used_poses[ts]
                    origins = poses[:, :3, 3]
                        
                    if self.require_gradient:
                        coord.requires_grad_(True)
                        
                    geo_feature, color_feature, weight_knn, _, certainty = self.neural_points.query_feature(coord, ts, query_color_feature=self.config.color_on)
                    
                    # predict the scaled sdf with the feature
                    sdf_pred = self.sdf_mlp.sdf(geo_feature) # [N, K, 1]  

                    if not self.config.weighted_first:
                        sdf_pred = torch.sum(sdf_pred * weight_knn, dim=1).squeeze(1)  # N
                    
                    if self.config.color_on:
                        surface_color_pred = self.color_mlp.regress_color(color_feature[surface_mask])  # [N, K, C]
                        if not self.config.weighted_first:
                            surface_color_pred = torch.sum(surface_color_pred * weight_knn[surface_mask], dim=1)  # N, C

                    # weight's sign indicate the sample is around the surface or in the free space
                    weight = torch.abs(weight).detach() 
                    # calculate the sdf bce loss
                    sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, self.sdf_scale, weight, self.config.loss_weight_on)

                    if self.config.weight_e > 0:
                        coord_for_eikonal = coord[close_to_surface_mask]
                        sdf_pred_for_eikonal = sdf_pred[close_to_surface_mask]
                        if self.require_gradient:
                            g = get_gradient(coord_for_eikonal, sdf_pred_for_eikonal)  # to unit m
                        elif self.config.numerical_grad:
                            g = self.get_numerical_gradient(
                                coord_for_eikonal[:: self.config.gradient_decimation],
                                sdf_pred_for_eikonal[:: self.config.gradient_decimation],
                                self.config.voxel_size_m * self.config.num_grad_step_ratio)
                        eikonal_loss = ((g.norm(2, dim=-1) - 1.0) ** 2).mean() 
                        
                    if self.config.color_on and self.config.weight_i > 0:
                        color_loss = color_diff_loss(surface_color_pred, color_label[surface_mask])
                        
                    if not self.silence:
                        print(" SDF BCE loss:", sdf_loss.item(), " SDF Eikonal loss:", eikonal_loss.item())

                    sdf_loss *= self.config.lambda_sdf
                    eikonal_loss *= self.config.weight_e
                    color_loss *= self.config.weight_i 

                # ----------------

                # Total loss
                # TODO: monitor losses by wandb
                total_loss = rgb_loss + depth_loss \
                    + normal_depth_consist_loss + normal_smoothness_loss \
                    + mono_normal_loss + sky_loss + distort_loss \
                    + isotropic_loss + area_loss + opacity_loss + opacity_ent_loss \
                    + sdf_consistency_loss + sdf_normal_consistency_loss + invalid_opacity_loss \
                    + sdf_loss + eikonal_loss + color_loss

                # print("Total loss:", total_loss.item())
                
                # tqdm.set_postfix(loss=f"{total_loss:.4f}")

                T6 = get_time()

                # Update

                total_loss.backward() 
                
                opt.step()
                opt.zero_grad(set_to_none=True) 

                T7 = get_time()
                
                # if not self.silence:
                #     print(" Prepare              iter time (ms):", (T2-T1)*1e3) 
                #     print(" Rendering            iter time (ms):", (T3-T2)*1e3)
                #     print(" GS render loss       iter time (ms):", (T4-T3)*1e3) 

                #     # almost no way to further speed up
                #     # print(" part 0:", (T3_1-T3)*1e3)
                #     # print(" part 1:", (T3_2-T3_1)*1e3)
                #     # print(" part 2:", (T3_3-T3_2)*1e3)
                #     # print(" part 3:", (T3_4-T3_3)*1e3) # ||||
                #     # print(" part 4:", (T3_5-T3_4)*1e3)
                #     # print(" part 5:", (T3_6-T3_5)*1e3) # |||
                #     # print(" part 6:", (T4-T3_6)*1e3)   # ||

                #     print(" GS/SDF consist loss  iter time (ms):", (T5-T4)*1e3) 
                #     print(" SDF loss             iter time (ms):", (T6-T5)*1e3) 
                #     print(" Backward propagation iter time (ms):", (T7-T6)*1e3) # still, this backpropagation is slow, but better to do this in batch

            # self.neural_points.assign_local_gaussians_to_global() # set back gaussians (and also neural points), better don't do it twice
            self.neural_points.assign_local_to_global() # set back pin feature

            if cams_param is not None:
                for cam_param in cams_param:
                    if self.config.affine_exposure_correction:
                        self.cams_exposure_ab[cam_param.uid] = (cam_param.exposure_mat, cam_param.exposure_offset)
                    else:
                        self.cams_exposure_ab[cam_param.uid] = (cam_param.exposure_a, cam_param.exposure_b)
                    
                    self.cams_pose_delta_rt[cam_param.uid] = (cam_param.cam_rot_delta, cam_param.cam_trans_delta)

                    # print(cam_param.cam_rot_delta)
                    # print(cam_param.cam_trans_delta)

            self.gs_total_iter += (self.config.gs_bs * iter_count)

        return 

    def check_invalid_neural_points(self, stability_threshold = 1.0, 
            render_min_nn_count: int = 5, 
            max_z_thre = None):

        with torch.no_grad():  # eval step
            local_neural_points = self.neural_points.local_neural_points
            stable_neural_points_mask = self.neural_points.local_point_certainties > stability_threshold

            # print("Begin to check the validity")
            # print("Stable count {:d} from total local count {:d}".format(torch.sum(stable_neural_points_mask).item(),
            #     self.neural_points.local_count()))

            stable_neural_points = local_neural_points[stable_neural_points_mask]

            # this is a bit too much large, better to do it in batch
            stable_neural_points_sdf, _, valid_nnk_mask = self.sdf_batch(stable_neural_points, self.config.infer_bs, min_nn_count=render_min_nn_count) # self.config.query_nn_k

            static_mask = torch.abs(stable_neural_points_sdf) < self.config.dynamic_sdf_ratio_thre * self.config.voxel_size_m

            valid_stable_mask = static_mask & valid_nnk_mask

            # a little heuristic
            if max_z_thre is not None:
                stable_neural_points_local_frame = transform_torch(stable_neural_points, torch.inverse(self.used_poses[-1]))

                large_z_mask = stable_neural_points_local_frame[:,2] > max_z_thre # nein, should be current frame
                valid_stable_mask = valid_stable_mask | large_z_mask # large z also would be regarded as static here

            self.neural_points.local_valid_gs_mask[stable_neural_points_mask] = valid_stable_mask # start with all True

            # set back the mask to the global map
            local_mask = self.neural_points.local_mask
            self.neural_points.valid_gs_mask[local_mask[:-1]] = self.neural_points.local_valid_gs_mask
        

    def init_gs_eval(self):
        # clear the lists for evaluation
        self.train_psnr_list = []
        self.train_ssim_list = []
        self.train_lpips_list = []
        self.train_depthl1_list = []
        self.train_depth_rmse_list = []
        self.train_cd_list = []
        self.train_f1_list = []

        self.test_psnr_list = []
        self.test_ssim_list = []
        self.test_lpips_list = []
        self.test_depthl1_list = []
        self.test_depth_rmse_list = []
        self.test_cd_list = []
        self.test_f1_list = []

    def record_per_cam_param(self):

        exposure_record_uids = list(self.cams_exposure_ab.keys())
        for exposure_record_uid in exposure_record_uids:
            exposure_record = exposure_record_uid.split('_') # uid: frame_camname
            exposure_record_cam = exposure_record[1]  # camname
            exposure_record_frame = int(exposure_record[0]) # frame
            if exposure_record_cam not in list(self.per_cam_exposure_ab.keys()):
                self.per_cam_exposure_ab[exposure_record_cam] = {}
                self.per_cam_pose_delta_rt[exposure_record_cam] = {}
            else:
                (self.per_cam_exposure_ab[exposure_record_cam])[exposure_record_frame] = self.cams_exposure_ab[exposure_record_uid]
                (self.per_cam_pose_delta_rt[exposure_record_cam])[exposure_record_frame] = self.cams_pose_delta_rt[exposure_record_uid]


    def gs_eval_offline(self, q_main2vis=None, q_vis2main=None, 
                        eval_down_rate=0, skip_end_count: int = 0, 
                        sorrounding_map_radius = None,
                        lpips_eval_on: bool = False,
                        pc_cd_eval_on: bool = False):
        
        # NOTE: there are some randomness of Guassian Splatting's optimization even with random seed fixed
        # This is mainly due to the randomness in GPU schedule in the differentiable rasterizer (according to the author of 3DGS)
        # For PSNR, it may have a difference of 0.1-0.2 PSNR

        assert self.config.use_dataloader, "Only data loader version is supported currently"

        eval_cam_name = self.dataset.cam_names # use all the cams
        # eval_cam_name = [self.dataset.loader.main_cam_name] # front cam

        # with torch.no_grad():
            
        self.record_per_cam_param()

        if sorrounding_map_radius is not None:
            self.neural_points.sorrounding_map_radius = sorrounding_map_radius

        background = torch.tensor(self.config.bg_color, dtype=self.dtype, device=self.device)
        bg_3d = background.view(3, 1, 1)

        eval_down_scale = 2**(eval_down_rate)

        # skip_end_count means that we will skip the last n frames because the incremental mapping haven't done much mapping in such areas
        for frame_id in tqdm(range(0, self.dataset.processed_frame - skip_end_count, 1), desc="GS evaluation"):

            remove_gpu_cache()
            
            T_w_l = self.used_poses[frame_id] #
            
            if frame_id % 100 == 0:
                self.neural_points.recreate_hash(T_w_l[:3,3], kept_points=True, with_ts=True, cur_ts=frame_id) # and at the same time reset local map
            else:
                self.neural_points.reset_local_map(T_w_l[:3,3], cur_ts=frame_id)
            
            neural_points_data, sorrounding_neural_points_data = self.neural_points.gather_local_data()

            sorrounding_spawn_results = spawn_gaussians(sorrounding_neural_points_data, 
                self.decoders, None, T_w_l[:3,3],
                dist_concat_on=self.config.dist_concat_on, 
                view_concat_on=self.config.view_concat_on, 
                scale_filter_on=True,
                z_far=self.config.sorrounding_map_radius,
                learn_color_residual=self.config.learn_color_residual,
                gs_type=self.config.gs_type,
                displacement_range_ratio=self.config.displacement_range_ratio,
                max_scale_ratio=self.config.max_scale_ratio,
                unit_scale_ratio=self.config.unit_scale_ratio)

            # load the cam datas to cur_cam_img
            self.dataset.read_frame_with_loader(frame_id, init_pose = False, use_image=True, monodepth_on=self.config.monodepth_on) # because we want to use the sky mask here

            # crop frames and possibly do LiDAR intrinsic corrections
            self.dataset.filter_and_correct()

            # deskew and reset depth map
            tran_in_frame = None
            if self.config.deskew and frame_id > 0:
                tran_in_frame = self.dataset.get_tran_in_frame(frame_id)
                self.dataset.deskew_at_frame(tran_in_frame)
            
            if not self.dataset.is_rgbd:
                self.dataset.project_pointcloud_to_cams(use_only_colorized_points=True, tran_in_frame=tran_in_frame) # self.config.learn_color_residual)

            if pc_cd_eval_on:
                cur_frame_measured_pcd_o3d = o3d.geometry.PointCloud()

                cur_frame_measured_xyz_np = (
                    self.dataset.cur_point_cloud_torch[:,:3].detach().cpu().numpy().astype(np.float64)
                )

                cur_frame_measured_color_np = (
                    self.dataset.cur_point_cloud_torch[:,3:].detach().cpu().numpy().astype(np.float64)
                )
                cur_frame_measured_pcd_o3d.points = o3d.utility.Vector3dVector(cur_frame_measured_xyz_np)
                cur_frame_measured_pcd_o3d.colors = o3d.utility.Vector3dVector(cur_frame_measured_color_np)

                cur_frame_rendered_pcd_o3d = o3d.geometry.PointCloud()

            cur_cam_names = list(self.dataset.cur_cam_img.keys())
            
            eval_depth_max = self.config.max_range * 0.8
            eval_depth_min = self.config.min_range
            
            for cam_name in cur_cam_names:

                K_mat = self.dataset.K_mats[cam_name]
                T_c_l_np = self.dataset.T_c_l_mats[cam_name]
                T_c_l = torch.tensor(T_c_l_np, device=self.device) 
                height = self.dataset.cam_heights[cam_name]
                width = self.dataset.cam_widths[cam_name] 

                cur_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()

                cur_intrinsic_o3d.set_intrinsics(
                                height=int(height/eval_down_scale),
                                width=int(width/eval_down_scale),
                                fx=K_mat[0,0]/eval_down_scale,
                                fy=K_mat[1,1]/eval_down_scale,
                                cx=K_mat[0,2]/eval_down_scale,
                                cy=K_mat[1,2]/eval_down_scale)

                diff_pose_l_c_ts = torch.eye(4).to(T_w_l)

                # relative transformation between the lidar reference timestamp and the camera triggering timestamp
                if frame_id > 0 and self.dataset.cur_sensor_ts is not None:
                    T_last_cur_lidar = torch.linalg.inv(self.used_poses[frame_id-1]) @ T_w_l   
                    cur_cam_ref_ts_ratio = self.dataset.get_cur_cam_ref_ts_ratio(cam_name)
                    diff_pose_l_c_ts = slerp_pose(T_last_cur_lidar, cur_cam_ref_ts_ratio, self.config.deskew_ref_ratio).to(T_w_l)

                T_w_l_cam_ts = T_w_l @ diff_pose_l_c_ts
                T_w_c = T_w_l_cam_ts @ torch.linalg.inv(T_c_l) # need to convert to cam frame

                # you need to also load the camera exposure coefficients here
                cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                cur_view_cam.set_pose(T_w_c)

                # print(T_w_c)
                
                cur_uid = cur_view_cam.uid
                cur_cam_id = cur_view_cam.cam_id # cam_name
                cur_frame_id = cur_view_cam.frame_id # frame_id

                # find the closest train view exposure and delta pose
                if self.config.exposure_correction_on:
                    closest_train_frame_id = min(self.per_cam_exposure_ab[cur_cam_id].keys(), key=lambda k: (abs(k - cur_frame_id), k))
                    cur_exposure = (self.per_cam_exposure_ab[cur_cam_id])[closest_train_frame_id]
                    if self.config.affine_exposure_correction:
                        cur_view_cam.set_exposure_affine(cur_exposure[0], cur_exposure[1])
                    else:
                        cur_view_cam.set_exposure_ab(cur_exposure[0], cur_exposure[1])
                    
                    cur_delta_pose = (self.per_cam_pose_delta_rt[cur_cam_id])[closest_train_frame_id]
                    if closest_train_frame_id == cur_frame_id:
                        cur_view_cam.set_delta_pose(cur_delta_pose[0], cur_delta_pose[1])                

                # gs_cam_refine_iter_count = 50 # in config now

                if cam_name in eval_cam_name:

                    opt = setup_optimizer(
                        self.config,
                        cams = [cur_view_cam],
                    )
                    
                    gt_rgb_img = cur_view_cam.rgb_image_list[eval_down_rate]

                    # if cur_view_cam.sky_mask_on:
                    #     # mask the sky part for eval
                    #     cur_sky_mask = cur_view_cam.sky_mask_list[eval_down_rate] # still torch
                    #     mask_broadcasted = cur_sky_mask.repeat(3,1,1)
                    #     gt_rgb_img[mask_broadcasted] = bg_3d.expand_as(gt_rgb_img)[mask_broadcasted]

                    if cam_name == "rear": # only for ipb car dataset (FIXME), use mask in the future, now it's just a ugly quick fix
                        pixel_h_used = int(910/1024*gt_rgb_img.shape[1])
                    elif cam_name == "front":
                        pixel_h_used = int(990/1024*gt_rgb_img.shape[1])
                    else:  
                        pixel_h_used = -1

                    gt_rgb_image_for_eval = gt_rgb_img[:,:pixel_h_used,:]

                    gt_depth_img = None
                    if cur_view_cam.depth_on is not None: 
                        gt_depth_img = cur_view_cam.depth_image_list[eval_down_rate] # torch.tensor
                        valid_depth_mask = (gt_depth_img > eval_depth_min) & (gt_depth_img < eval_depth_max)

                    for iter in tqdm(range(self.config.gs_cam_refine_iter_count+1), disable=self.silence, desc="Camera refinement"):    

                        # current values
                        render_pkg = render(cur_view_cam, None, neural_points_data, 
                            self.decoders, sorrounding_spawn_results, background, 
                            down_rate=eval_down_rate, 
                            dist_concat_on=self.config.dist_concat_on, 
                            view_concat_on=self.config.view_concat_on, 
                            correct_exposure=self.config.exposure_correction_on, 
                            correct_exposure_affine=self.config.affine_exposure_correction,
                            learn_color_residual=self.config.learn_color_residual,
                            front_only_on=self.config.train_front_only,
                            gs_type=self.config.gs_type,
                            displacement_range_ratio=self.config.displacement_range_ratio,
                            max_scale_ratio=self.config.max_scale_ratio,
                            unit_scale_ratio=self.config.unit_scale_ratio)

                        # rendered results
                        rendered_rgb_image, rendered_depth, rendered_alpha = render_pkg["render"], render_pkg["surf_depth"], render_pkg["rend_alpha"] # 3, H, W / 1, H, W

                        rendered_rgb_image = torch.clamp(rendered_rgb_image, 0, 1)

                        rendered_rgb_image_for_eval = rendered_rgb_image[:,:pixel_h_used,:]

                        if not self.config.gs_eval_cam_refine_on:
                            break

                        loss_rgb_robust = tukey_loss(rendered_rgb_image_for_eval, gt_rgb_image_for_eval, c=0.0) # now just l1 loss

                        if self.config.lambda_ssim > 0.0:
                            ssim_value = fused_ssim(rendered_rgb_image_for_eval.unsqueeze(0), gt_rgb_image_for_eval.unsqueeze(0)) # have to be 4 dim
                            rgb_loss = (1.0 - self.config.lambda_ssim) * loss_rgb_robust + self.config.lambda_ssim * (1.0 - ssim_value)
                        else:
                            rgb_loss = loss_rgb_robust # l1 only, ssim might take a long time

                        depth_loss = 0.0 
                        if rendered_depth is not None and gt_depth_img is not None and self.config.lambda_depth > 0:
                            if rendered_alpha is not None:
                                accu_alpha_mask = rendered_alpha.detach() > self.config.depth_min_accu_alpha
                                valid_depth_mask = valid_depth_mask & accu_alpha_mask
                            depth_loss = l1_loss(gt_depth_img, rendered_depth, valid_depth_mask)
                            depth_loss *= self.config.lambda_depth

                        total_loss = rgb_loss + depth_loss

                        # print("Camera refinement loss:", total_loss.item())

                        opt.zero_grad(set_to_none=True) 
                        total_loss.backward(retain_graph=True) 
                        opt.step()    
                    
                    cur_psnr = psnr(rendered_rgb_image_for_eval, gt_rgb_image_for_eval).mean().item()
                    cur_ssim = fused_ssim(rendered_rgb_image_for_eval.unsqueeze(0), gt_rgb_image_for_eval.unsqueeze(0), train=False).item()
                    # cur_ssim = ssim(rendered_rgb_image_for_eval, gt_rgb_image_for_eval).item()
                    if lpips_eval_on:
                        cur_lpips = self.lpips(rendered_rgb_image_for_eval.unsqueeze(0), gt_rgb_image_for_eval.unsqueeze(0)).item()
                    else:
                        cur_lpips = -1.0 # not available

                    if not self.silence:
                        print("Camera id: {}".format(cur_view_cam.uid))
                        print("Current view PSNR  ↑ :", f"{cur_psnr:.3f}")
                        print("Current view SSIM  ↑ :", f"{cur_ssim:.3f}")
                        print("Current view LPIPS ↓ :", f"{cur_lpips:.3f}")
                        if self.config.exposure_correction_on and not self.config.affine_exposure_correction:
                            print("Current view exposure coefficients {:.3f}, {:.3f}".format(cur_exposure[0].item(), cur_exposure[1].item()))
                        

                    if gt_depth_img is not None and rendered_depth is not None: 
                        valid_depth_mask = (gt_depth_img > eval_depth_min) & (rendered_depth > eval_depth_min) & (gt_depth_img < eval_depth_max) & (rendered_depth < eval_depth_max)
                        
                        accu_alpha_mask = None
                        if rendered_alpha is not None:
                            accu_alpha_mask = rendered_alpha > self.config.depth_min_accu_alpha
                            valid_depth_mask = valid_depth_mask & accu_alpha_mask
                                        
                        diff_depth = torch.abs(gt_depth_img - rendered_depth) # already abs
                        # diff_depth[~valid_depth_mask] = 0.0
                        diff_depth_masked = diff_depth[valid_depth_mask].detach().cpu().numpy()
                        cur_depth_l1 = np.mean(diff_depth_masked)
                        cur_depth_rmse = np.sqrt(np.mean(diff_depth_masked**2))
                        if not self.silence:
                            print("Current view Depth L1 (m) ↓ :", f"{cur_depth_l1:.3f}")
                            print("Current view Depth RMSE (m) ↓ :", f"{cur_depth_rmse:.3f}")


                        if pc_cd_eval_on: 

                            rendered_rgb_np = (rendered_rgb_image * 255).byte().permute(1, 2, 0).detach().contiguous().cpu().numpy().astype(np.uint8) 
                            rgb_img_o3d = o3d.geometry.Image(rendered_rgb_np)
                            
                            if accu_alpha_mask is not None:
                                rendered_depth[~accu_alpha_mask] = 0.0

                            rendered_depth_np = rendered_depth.detach().cpu().numpy().astype(np.float32) 
                            rendered_depth_np = np.transpose(rendered_depth_np, (1, 2, 0))

                            depth_img_o3d = o3d.geometry.Image(rendered_depth_np)

                            cur_rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, 
                                                                                    depth_img_o3d, 
                                                                                    depth_scale=1.0, 
                                                                                    depth_trunc=eval_depth_max, 
                                                                                    convert_rgb_to_intensity=False)

                            cur_cam_rendered_pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(
                                                                cur_rgbd_o3d, 
                                                                cur_intrinsic_o3d, 
                                                                T_c_l_np)

                            cur_frame_rendered_pcd_o3d += cur_cam_rendered_pcd_o3d # already under lidar frame


                    if cur_view_cam.uid in self.train_cam_uid:
                        # as train views
                        if not self.silence:
                            print("Evalualted as a train view")
                        self.train_psnr_list.append(cur_psnr)
                        self.train_ssim_list.append(cur_ssim)
                        self.train_lpips_list.append(cur_lpips)
                        if cur_view_cam.depth_on and rendered_depth is not None: 
                            self.train_depthl1_list.append(cur_depth_l1)
                            self.train_depth_rmse_list.append(cur_depth_rmse)
                    
                    else:
                        # as test views
                        if not self.silence:
                            print("Evaluated as a test view")
                        self.test_psnr_list.append(cur_psnr)
                        self.test_ssim_list.append(cur_ssim)
                        self.test_lpips_list.append(cur_lpips)
                        if cur_view_cam.depth_on and rendered_depth is not None: 
                            self.test_depthl1_list.append(cur_depth_l1)
                            self.test_depth_rmse_list.append(cur_depth_rmse)

            # compute cd with regards to original lidar pc
            if pc_cd_eval_on: 
                cd_metrics = eval_pair(cur_frame_rendered_pcd_o3d, cur_frame_measured_pcd_o3d, 
                    down_sample_res=0.05, threshold=0.1, 
                    truncation_acc=1.0, truncation_com=1.0) # FIXME
                
                cur_cd = cd_metrics['Chamfer_L1 (m)']
                cur_f1 = cd_metrics['F-score (%)']

                if not self.silence:
                    print("Current frame Chamfer Distance L1 (m) ↓ :", f"{cur_cd:.3f}")
                    print("Current frame F1-score (%) ↑ :", f"{cur_f1:.3f}")

                if cur_view_cam.uid in self.train_cam_uid:
                    self.train_cd_list.append(cur_cd)
                    self.train_f1_list.append(cur_f1)
                else:
                    self.test_cd_list.append(cur_cd)
                    self.test_f1_list.append(cur_f1)

            if q_main2vis is not None:
                # add the eval frame to vis
                
                packet_to_vis= VisPacket(frame_id=frame_id,
                    current_frames=self.dataset.cur_cam_img, 
                    img_down_rate=self.config.gs_vis_down_rate)
                
                packet_to_vis.add_neural_points_data(self.neural_points)

                if pc_cd_eval_on: 
                    packet_to_vis.add_scan(np.array(cur_frame_rendered_pcd_o3d.points, dtype=np.float64), np.array(cur_frame_rendered_pcd_o3d.colors, dtype=np.float64))

                odom_poses, gt_poses, pgo_poses = self.dataset.get_poses_np_for_vis(frame_id)
                packet_to_vis.add_traj(odom_poses, gt_poses, pgo_poses)

                q_main2vis.put(packet_to_vis)

            if q_vis2main is not None:
                if not q_vis2main.empty():
                    while q_vis2main.get().flag_pause:
                        continue


    def gs_eval_out(self):
        
        train_psnr_np = train_ssim_np = train_lpips_np = train_depthl1_np = train_depth_rmse_np = train_cd_np = train_f1_np = 0.0

        cam_count = len(self.dataset.cam_names) # better to also compute for each cam
        # TODO: fix 

        train_frame_count = len(self.train_psnr_list) 

        # TODO: it's even better to print the results for each camera, it's possible
        if train_frame_count > 0:
            train_psnr_np = np.mean(np.array(self.train_psnr_list))
            train_ssim_np = np.mean(np.array(self.train_ssim_list))
            train_lpips_np = np.mean(np.array(self.train_lpips_list))
            
            print(f"Calculated on {train_frame_count} train views")

            print("Average train view PSNR  ↑ :", f"{train_psnr_np:.3f}")
            print("Average train view SSIM  ↑ :", f"{train_ssim_np:.3f}")
            print("Average train view LPIPS ↓ :", f"{train_lpips_np:.3f}")


        if len(self.train_depth_rmse_list) > 0:
            train_depthl1_np = np.mean(np.array(self.train_depthl1_list))
            train_depth_rmse_np = np.mean(np.array(self.train_depth_rmse_list))
            print("Average train view Depth L1 (m) ↓ :", f"{train_depthl1_np:.3f}")
            print("Average train view Depth RMSE (m) ↓ :", f"{train_depth_rmse_np:.3f}")

        if len(self.train_cd_list) > 0:
            train_cd_np = np.mean(np.array(self.train_cd_list))
            train_f1_np = np.mean(np.array(self.train_f1_list))
            print("Average train frame CD (m) ↓ :", f"{train_cd_np:.3f}")
            print("Average train frame F1 (%) ↑ :", f"{train_f1_np:.3f}")

        test_psnr_np = test_ssim_np = test_lpips_np = test_depthl1_np = test_depth_rmse_np = test_cd_np = test_f1_np = 0.0

        test_frame_count = len(self.test_psnr_list) 
        if test_frame_count > 0:
            test_psnr_np = np.mean(np.array(self.test_psnr_list))
            test_ssim_np = np.mean(np.array(self.test_ssim_list))
            test_lpips_np = np.mean(np.array(self.test_lpips_list))
            
            print(f"Calculated on {test_frame_count} test frames")

            print("Average test view PSNR  ↑ :", f"{test_psnr_np:.3f}")
            print("Average test view SSIM  ↑ :", f"{test_ssim_np:.3f}")
            print("Average test view LPIPS ↓ :", f"{test_lpips_np:.3f}")


        if len(self.test_depth_rmse_list) > 0:
            test_depthl1_np = np.mean(np.array(self.test_depthl1_list))
            test_depth_rmse_np = np.mean(np.array(self.test_depth_rmse_list))
            print("Average test view Depth L1 (m) ↓ :", f"{test_depthl1_np:.3f}")
            print("Average test view Depth RMSE (m) ↓ :", f"{test_depth_rmse_np:.3f}")
        
        if len(self.test_cd_list) > 0:
            test_cd_np = np.mean(np.array(self.test_cd_list))
            test_f1_np = np.mean(np.array(self.test_f1_list))
            print("Average test frame CD (m) ↓ :", f"{test_cd_np:.3f}")
            print("Average test frame F1 (%) ↑ :", f"{test_f1_np:.3f}")

        gs_csv_columns = [
                "Frame-Type",
                "PSNR↑",
                "SSIM↑",
                "LPIPS↓",
                "Depth-L1(m)↓",
                "Depth-RMSE(m)↓",
                "Recon-CD(m)↓",
                "Recon-F1(%)↑",
                "Frame-count",
        ]
        gs_eval = [
            {
                gs_csv_columns[0]: "train",
                gs_csv_columns[1]: train_psnr_np,
                gs_csv_columns[2]: train_ssim_np,
                gs_csv_columns[3]: train_lpips_np,
                gs_csv_columns[4]: train_depthl1_np,
                gs_csv_columns[5]: train_depth_rmse_np,
                gs_csv_columns[6]: train_cd_np,
                gs_csv_columns[7]: train_f1_np,
                gs_csv_columns[8]: train_frame_count,
            },
            {
                gs_csv_columns[0]: "test",
                gs_csv_columns[1]: test_psnr_np,
                gs_csv_columns[2]: test_ssim_np,
                gs_csv_columns[3]: test_lpips_np,
                gs_csv_columns[4]: test_depthl1_np,
                gs_csv_columns[5]: test_depth_rmse_np,
                gs_csv_columns[6]: test_cd_np,
                gs_csv_columns[7]: test_f1_np,
                gs_csv_columns[8]: test_frame_count,
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
    # better to use vdb fusion instead
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

            # current values
            render_pkg = render(cur_view_cam, None, neural_points_data, 
                self.decoders, sorrounding_spawn_results, background, 
                down_rate=eval_down_rate, 
                dist_concat_on=self.config.dist_concat_on, 
                view_concat_on=self.config.view_concat_on, 
                correct_exposure=self.config.exposure_correction_on, 
                correct_exposure_affine=self.config.affine_exposure_correction,
                learn_color_residual=self.config.learn_color_residual,
                front_only_on=self.config.train_front_only,
                gs_type=self.config.gs_type,
                displacement_range_ratio=self.config.displacement_range_ratio,
                max_scale_ratio=self.config.max_scale_ratio,
                unit_scale_ratio=self.config.unit_scale_ratio)

            # rendered results
            rendered_rgb_image, rendered_depth = render_pkg["render"], render_pkg["surf_depth"] # 3, H, W / 1, H, W

            rendered_rgb_image = torch.clamp(rendered_rgb_image, 0, 1)
            # print(torch.max(rendered_rgb_image), torch.min(rendered_rgb_image)) # why there are value larger than 1?

            renderd_image_np = (rendered_rgb_image.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 

            renderd_image_np = np.ascontiguousarray(renderd_image_np)
            rgb_image = o3d.geometry.Image(renderd_image_np)

            rendered_depth_np = rendered_depth.squeeze(0).detach().cpu().numpy().astype(np.float32) 
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

    # for visualization
    def get_data_pool_o3d(self, down_rate=1, only_cur_data=False):

        if self.global_coord_pool is None:
            return None

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


    # joint optimization of PIN map and the poses in the sliding window
    # neural points are static in this case, we only fine-tune the neural point features for the map updating
    # deprecated
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
        # neural_point_feat = [self.neural_points.local_geo_features, self.neural_points.local_color_features]

        # also add the poses as param here, for pose refinement (bundle ajustment)
        opt = setup_optimizer(
            self.config, 
            self.neural_points.local_geo_features, 
            self.neural_points.local_color_features,
            poses=current_poses_se3_opt
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
    def sdf(self, x, get_std=False, min_nn_count=1, accumulate_stability=False):
        geo_feature, _, weight_knn, nn_count, _ = self.neural_points.query_feature(x, accumulate_stability=accumulate_stability) # we do not add stability here
        sdf_pred = self.sdf_mlp.sdf(geo_feature)    
        # predict the scaled sdf with the feature # [N, K, 1]
        sdf_std = None
        if not self.config.weighted_first:
            sdf_pred_mean = torch.sum(sdf_pred * weight_knn, dim=1)  # N
            if get_std:
                sdf_var = torch.sum(
                    (weight_knn * (sdf_pred - sdf_pred_mean.unsqueeze(-1)) ** 2), dim=1
                )
                sdf_std = torch.sqrt(sdf_var).squeeze(1)
            sdf_pred = sdf_pred_mean.squeeze(1)

        valid_mask = (nn_count >= min_nn_count)

        return sdf_pred, sdf_std, valid_mask

    # short-hand function
    def sdf_batch(self, x, bs, get_std=False, min_nn_count=1, accumulate_stability=False):

        count = x.shape[0]
        iter_n = math.ceil(count / bs)

        sdf_pred = torch.zeros(count, dtype=self.dtype, device=self.device)

        if get_std:
            sdf_std = torch.zeros(count, dtype=self.dtype, device=self.device)
        else:
            sdf_std = None

        valid_mask = torch.ones(count, dtype=bool, device=self.device)
        
        for n in tqdm(range(iter_n), disable=self.silence):
            head = n * bs
            tail = min((n + 1) * bs, count)
            batch_x = x[head:tail, :]
            batch_sdf, batch_sdf_std, batch_valid_mask = self.sdf(batch_x, get_std, min_nn_count, accumulate_stability)
            sdf_pred[head:tail] = batch_sdf
            if batch_sdf_std is not None and sdf_std is not None:
                sdf_std[head:tail] = batch_sdf_std
            valid_mask[head:tail] = batch_valid_mask

        return sdf_pred, sdf_std, valid_mask

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

            # TODO: consider valid mask

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
