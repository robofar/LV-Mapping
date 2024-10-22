#!/usr/bin/env python3
# @file      pin_slam.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import argparse
import csv
import os
import sys
import time

import rerun as rr
import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
import wandb
from rich import print
from tqdm import tqdm

from dataset.dataset_indexing import set_dataset_path
from dataset.slam_dataset import SLAMDataset
from model.decoder import Decoder
from model.neural_gaussians import NeuralPoints
from utils.config import Config
from utils.loop_detector import (
    NeuralPointMapContextManager,
    detect_local_loop,
)
from utils.mapper import Mapper
from utils.mesher import Mesher
from utils.pgo import PoseGraphManager
from utils.tools import (
    freeze_decoders,
    get_time,
    load_decoder,
    save_implicit_map,
    setup_experiment,
    split_chunks,
    transform_torch,
    remove_gpu_cache,
)
from utils.multiprocessing_utils import clone_obj
from utils.tracker import Tracker
from utils.visualizer import MapVisualizer

from gs_gui import slam_gui
from gs_gui.gui_utils import VisPacket, ParamsGUI

'''
    📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency
     Y. Pan et al. from IPB
'''

# or maybe we directly add the inspection function here?

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, nargs='?', default='config/lidar_slam/run.yaml', help='[Optional] Path to *.yaml config file, if not set, default config would be used')
parser.add_argument('dataset_name', type=str, nargs='?', help='[Optional] Name of a specific dataset, example: kitti, mulran, or rosbag (when -d is set)')
parser.add_argument('sequence_name', type=str, nargs='?', help='[Optional] Name of a specific data sequence or the rostopic for point cloud (when -d is set)')
parser.add_argument('--seed', type=int, default=42, help='Set the random seed (default 42)')
parser.add_argument('--input_path', '-i', type=str, default=None, help='Path to the point cloud input directory (this will override the pc_path in config file)')
parser.add_argument('--output_path', '-o', type=str, default=None, help='Path to the result output directory (this will override the output_root in config file)')
parser.add_argument('--range', nargs=3, type=int, metavar=('START', 'END', 'STEP'), default=None, help='Specify the start, end and step of the processed frame, for example: --range 10 1000 1')
parser.add_argument('--data_loader_on', '-d', action='store_true', default=True, help='Use specific data loader (you can use the rosbag, pcap, mcap dataloaders and some typical supported datasets)')
parser.add_argument('--visualize', '-v', action='store_true', help='Turn on the GS visualizer, note that this would make the SLAM processing slower')
parser.add_argument('--cpu_only', '-c', action='store_true', help='Run only on CPU')
parser.add_argument('--log_on', '-l', action='store_true', help='Turn on the logs printing')
parser.add_argument('--wandb_on', '-w', action='store_true', help='Turn on the weight & bias logging')
parser.add_argument('--save_map', '-s', action='store_true', help='Save the PIN map after SLAM')
parser.add_argument('--save_mesh', '-m', action='store_true', help='Save the reconstructed mesh after SLAM')
parser.add_argument('--save_merged_pc', '-p', action='store_true', help='Save the merged point cloud after SLAM')
parser.add_argument('--deskew', action='store_true', help='Try to deskew the LiDAR scans')

args, unknown = parser.parse_known_args()

def run_pin_slam(config_path=None, dataset_name=None, sequence_name=None, seed=None):

    config = Config()
    if config_path is not None: # use as a function
        config.load(config_path)
        if dataset_name is not None:
            set_dataset_path(config, dataset_name, sequence_name)
        if seed is not None:
            config.seed = seed
        argv = ['pin_slam.py', config_path, dataset_name, sequence_name, str(seed)]
        run_path = setup_experiment(config, argv)
    else: # from args
        argv = sys.argv
        config.load(args.config_path)
        config.use_dataloader = args.data_loader_on
        config.seed = args.seed
        config.silence = not args.log_on
        config.wandb_vis_on = args.wandb_on
        config.gs_vis_on = args.visualize
        config.save_map = args.save_map
        config.save_mesh = args.save_mesh
        config.save_merged_pc = args.save_merged_pc
        if not config.deskew and args.deskew: # set to True if not set in the config file but set upon running
            config.deskew = True
        if args.range is not None:
            config.begin_frame, config.end_frame, config.step_frame = args.range
        if args.cpu_only:
            config.device = 'cpu'
        if args.input_path is not None:
            config.pc_path = args.input_path
        if args.output_path is not None:
            config.output_root = args.output_path
        if args.dataset_name is not None: # specific dataset [optional]
            set_dataset_path(config, args.dataset_name, args.sequence_name)
        run_path = setup_experiment(config, argv)
        print("[bold green]PIN-SLAM starts[/bold green]","📍" )

    mp.set_start_method("spawn")

    # initialize the mlp decoder
    # geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)

    geo_feature_dim = config.feature_dim
    sem_feature_dim = config.sem_feature_dim
    color_feature_dim = config.color_feature_dim
    # print("colro feature dim:", color_feature_dim)

    geo_mlp = Decoder(config, geo_feature_dim, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    sem_mlp = Decoder(config, sem_feature_dim, config.sem_mlp_hidden_dim, config.sem_mlp_level, config.sem_class_count + 1) if config.semantic_on else None
    color_mlp = Decoder(config, color_feature_dim, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel) if config.color_on else None

    # # Load the decoder model
    # if config.load_model: # not used
    #     load_decoder(config, geo_mlp, sem_mlp, color_mlp)

    n_gaussian = config.spawn_n_gaussian # almost 2D, then 4 already means 1/2 resolution

    dist_concat_dim = 1 if config.dist_concat_on else 0
    view_concat_dim = 3 if config.view_concat_on else 0

    # gs_2d = True
    # # scale_dim = 2 if gs_2d else 3
    # scale_dim = 3

    # current value
    # gs_mlp_hidden_dim 64
    # gs_mlp_level 1

    gaussian_xyz_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, n_gaussian, 0)
    gaussian_rot_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 4, n_gaussian, 0)
    gaussian_scale_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, n_gaussian, 0)
    gaussian_alpha_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 1, n_gaussian, dist_concat_dim) # concat distance
    gaussian_color_mlp = Decoder(config, color_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, n_gaussian, view_concat_dim) # concat view direction

    mlp_dict = {}
    
    mlp_dict["sdf"] = geo_mlp
    mlp_dict["semantic"] = sem_mlp
    mlp_dict["color"] = color_mlp

    mlp_dict["gauss_xyz"] = gaussian_xyz_mlp
    mlp_dict["gauss_scale"] = gaussian_scale_mlp
    mlp_dict["gauss_rot"] = gaussian_rot_mlp
    mlp_dict["gauss_alpha"] = gaussian_alpha_mlp
    mlp_dict["gauss_color"] = gaussian_color_mlp

    # initialize the neural point features
    neural_points = NeuralPoints(config)

    # non-blocking visualizer
    if config.o3d_vis_on:
        o3d_vis = MapVisualizer(config) 
    
    q_main2vis = q_vis2main = None
    if config.gs_vis_on:
        # communicator between the processes
        q_main2vis = mp.Queue() 
        q_vis2main = mp.Queue()

        params_gui = ParamsGUI(
            decoders=mlp_dict,
            background=torch.tensor(config.bg_color, dtype=config.dtype, device=config.device),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            config=config,
        )

        gui_process = mp.Process(target=slam_gui.run, args=(params_gui,)) # TODO: something wrong here
        gui_process.start()
        time.sleep(2) # second

    # dataset
    dataset = SLAMDataset(config)

    # odometry tracker
    tracker = Tracker(config, neural_points, mlp_dict)

    # mapper
    mapper = Mapper(config, dataset, neural_points, mlp_dict)

    # mesh reconstructor
    mesher = Mesher(config, neural_points, mlp_dict)
    cur_mesh = None

    # pose graph manager (for back-end optimization) initialization
    pgm = PoseGraphManager(config) 
    init_pose = dataset.gt_poses[0] if dataset.gt_pose_provided else np.eye(4)  
    pgm.add_pose_prior(0, init_pose, fixed=True)

    # loop closure detector
    lcd_npmc = NeuralPointMapContextManager(config) # npmc: neural point map context

    last_frame = dataset.total_pc_count-1
    loop_reg_failed_count = 0

    # save merged point cloud map from gt pose as a reference map
    if config.save_merged_pc and dataset.gt_pose_provided:
        dataset.write_merged_point_cloud(use_gt_pose=True, out_file_name='merged_gt_pc', 
            frame_step=5, merged_downsample=True)
    
    gs_time_table = []

    # TODO: test TSDF fusion (pass)
    # tsdf_mesh_path = os.path.join(run_path, "mesh", "tsdf_fusion_mesh.ply")
    # tsdf_mesh = dataset.o3d_tsdf_fusion(frame_step=20, output_path=tsdf_mesh_path)
    # return 
        
    # for each frame
    for frame_id in tqdm(range(dataset.total_pc_count)): # frame id as the processed frame, possible skipping done in data loader
        
        remove_gpu_cache()

        # judge pause
        if config.gs_vis_on:
            if not q_vis2main.empty():
                while q_vis2main.get().flag_pause:
                    continue

        # I. Load data and preprocessing
        T0 = get_time()

        if config.use_dataloader:
            dataset.read_frame_with_loader(frame_id, use_image=config.gs_on, monodepth_on=config.monodepth_on)
        else:
            dataset.read_frame(frame_id)

        T1 = get_time()
        
        valid_frame_flag = dataset.preprocess_frame()
        if not valid_frame_flag:
            dataset.processed_frame += 1
            continue 

        T2 = get_time()
        
        # II. Odometry
        if frame_id > 0: 
            if config.track_on:
                tracking_result = tracker.tracking(dataset.cur_source_points, dataset.cur_pose_guess_torch, 
                                                   dataset.cur_source_colors, dataset.cur_source_normals,
                                                   vis_result=config.o3d_vis_on and not config.o3d_vis_raw)
                cur_pose_torch, cur_odom_cov, weight_pc_o3d, valid_flag = tracking_result
                dataset.lose_track = not valid_flag
                dataset.update_odom_pose(cur_pose_torch) # update dataset.cur_pose_torch
                
                if not valid_flag and config.o3d_vis_on and o3d_vis.debug_mode > 0:
                    o3d_vis.stop()
                
            else: # incremental mapping with gt pose
                if dataset.gt_pose_provided:
                    dataset.update_odom_pose(dataset.cur_pose_guess_torch) 
                else:
                    sys.exit("You are using the mapping mode, but no pose is provided.")

        travel_dist = dataset.travel_dist[:frame_id+1]
        neural_points.travel_dist = torch.tensor(travel_dist, device=config.device, dtype=config.dtype) # always update this
                                                                                                                                                            
        T3 = get_time()

        # III. Loop detection and pgo
        if config.pgo_on: 
            if config.global_loop_on:
                if config.local_map_context and frame_id >= config.local_map_context_latency: # local map context
                    local_map_frame_id = frame_id-config.local_map_context_latency
                    local_map_pose = torch.tensor(dataset.pgo_poses[local_map_frame_id], device=config.device, dtype=torch.float64)
                    if config.local_map_context_latency > 0:
                        neural_points.reset_local_map(local_map_pose[:3,3], None, local_map_frame_id, False, config.loop_local_map_time_window)
                    context_pc_local = transform_torch(neural_points.local_neural_points.detach(), torch.linalg.inv(local_map_pose)) # transformed back into the local frame
                    neural_points_feature = neural_points.local_geo_features[:-1].detach() if config.loop_with_feature else None
                    lcd_npmc.add_node(local_map_frame_id, context_pc_local, neural_points_feature)
                else: # first frame not yet have local map, use scan context
                    lcd_npmc.add_node(frame_id, dataset.cur_point_cloud_torch)
            pgm.add_frame_node(frame_id, dataset.pgo_poses[frame_id]) # add new node and pose initial guess
            pgm.init_poses = dataset.pgo_poses[:frame_id+1]
            if frame_id > 0:
                cur_edge_cov = cur_odom_cov if config.use_reg_cov_mat else None
                pgm.add_odometry_factor(frame_id, frame_id-1, dataset.last_odom_tran, cov = cur_edge_cov) # T_p<-c
                pgm.estimate_drift(travel_dist, frame_id, correct_ratio=0.01) # estimate the current drift
                if config.pgo_with_pose_prior: # add pose prior
                    pgm.add_pose_prior(frame_id, dataset.pgo_poses[frame_id])
            local_map_context_loop = False
            if frame_id - pgm.last_loop_idx > config.pgo_freq and not dataset.stop_status:
                # detect candidate local loop, find the nearest history pose and activate certain local map
                loop_candidate_mask = ((travel_dist[-1] - travel_dist) > (config.min_loop_travel_dist_ratio*config.local_map_radius)) # should not be too close
                loop_id = None
                if np.any(loop_candidate_mask): # have at least one candidate
                    # firstly try to detect the local loop by checking the distance
                    loop_id, loop_dist, loop_transform = detect_local_loop(dataset.pgo_poses[:frame_id+1], loop_candidate_mask, pgm.drift_radius, frame_id, loop_reg_failed_count, config.local_loop_dist_thre, config.local_loop_dist_thre*3.0, config.silence)
                    if loop_id is None and config.global_loop_on: # global loop detection (large drift)
                        loop_id, loop_cos_dist, loop_transform, local_map_context_loop = lcd_npmc.detect_global_loop(dataset.pgo_poses[:frame_id+1], pgm.drift_radius*config.loop_dist_drift_ratio_thre, loop_candidate_mask, neural_points) # latency has been considered here     
                if loop_id is not None:
                    if config.loop_z_check_on and abs(loop_transform[2,3]) > config.voxel_size_m*4.0: # for multi-floor buildings, z may cause ambiguilties
                        loop_id = None # delta z check failed
                if loop_id is not None: # if a loop is found, we refine loop closure transform initial guess with a scan-to-map registration                    
                    pose_init_torch = torch.tensor((dataset.pgo_poses[loop_id] @ loop_transform), device=config.device, dtype=torch.float64) # T_w<-c = T_w<-l @ T_l<-c 
                    neural_points.recreate_hash(pose_init_torch[:3,3], None, True, True, loop_id) # recreate hash and local map at the loop candidate frame for registration, this is the reason why we'd better to keep the duplicated neural points until the end
                    loop_reg_source_point = dataset.cur_source_points.clone()
                    pose_refine_torch, loop_cov_mat, weight_pcd, reg_valid_flag = tracker.tracking(loop_reg_source_point, pose_init_torch, loop_reg=True, vis_result=config.o3d_vis_on)
                    # visualize the loop closure and loop registration
                    if config.o3d_vis_on and o3d_vis.debug_mode > 1:
                        points_torch_init = transform_torch(loop_reg_source_point, pose_init_torch) # apply transformation
                        points_o3d_init = o3d.geometry.PointCloud()
                        points_o3d_init.points = o3d.utility.Vector3dVector(points_torch_init.detach().cpu().numpy().astype(np.float64))
                        loop_neural_pcd = neural_points.get_neural_points_o3d(query_global=False, color_mode=o3d_vis.neural_points_vis_mode, random_down_ratio=1)
                        o3d_vis.update(points_o3d_init, neural_points=loop_neural_pcd, pause_now=True)
                        o3d_vis.update(weight_pcd, neural_points=loop_neural_pcd, pause_now=True)
                    # only conduct pgo when the loop and loop constraint is correct
                    if reg_valid_flag: # refine succeed
                        pose_refine_np = pose_refine_torch.detach().cpu().numpy()
                        loop_transform = np.linalg.inv(dataset.pgo_poses[loop_id]) @ pose_refine_np # T_l<-c = T_l<-w @ T_w<-c # after refinement
                        cur_edge_cov = loop_cov_mat if config.use_reg_cov_mat else None
                        reg_valid_flag = pgm.add_loop_factor(frame_id, loop_id, loop_transform, cov = cur_edge_cov)
                    if reg_valid_flag:
                        if not config.silence:
                            print("[bold green]Refine loop transformation succeed [/bold green]")
                        pgm.optimize_pose_graph() # conduct pgo
                        cur_loop_vis_id = frame_id-config.local_map_context_latency if local_map_context_loop else frame_id
                        pgm.loop_edges_vis.append(np.array([loop_id, cur_loop_vis_id],dtype=np.uint32)) # only for vis
                        pgm.loop_edges.append(np.array([loop_id, frame_id],dtype=np.uint32))
                        pgm.loop_trans.append(loop_transform)
                        # update the neural points and poses
                        pose_diff_torch = torch.tensor(pgm.get_pose_diff(), device=config.device, dtype=config.dtype)
                        dataset.cur_pose_torch = torch.tensor(pgm.cur_pose, device=config.device, dtype=config.dtype)
                        neural_points.adjust_map(pose_diff_torch) # transform neural points (position and orientation) along with associated frame poses # time consuming part
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, (not config.pgo_merge_map), config.rehash_with_time, frame_id) # recreate hash from current time
                        mapper.transform_data_pool(pose_diff_torch) # transform global pool
                        dataset.update_poses_after_pgo(pgm.pgo_poses)
                        pgm.last_loop_idx = frame_id
                        pgm.min_loop_idx = min(pgm.min_loop_idx, loop_id)
                        loop_reg_failed_count = 0
                        if config.o3d_vis_on:
                            o3d_vis.before_pgo = False
                    else:
                        if not config.silence:
                            print("[bold red]Registration failed, reject the loop candidate [/bold red]")
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, True, True, frame_id) # if failed, you need to reset the local map back to current frame
                        loop_reg_failed_count += 1
                        if config.o3d_vis_on and o3d_vis.debug_mode > 1:
                            o3d_vis.stop()

        T4 = get_time()
        
        # IV: Mapping and bundle adjustment
        
        # Re-generate colorized point cloud and correct depth map after point cloud deskewing
        # if config.deskew: # only needed for LiDAR datasets
        dataset.project_pointcloud_to_cams(use_only_colorized_points=config.learn_color_residual) # True # config.learn_color_residual
        
        # if lose track, we will not update the map and data pool (don't let the wrong pose to corrupt the map)
        # if the robot stop, also don't process this frame, since there's no new oberservations
        dataset.voxel_downsample_points_for_mapping()
        
        if (not dataset.lose_track and not dataset.stop_status) or frame_id < 5:
            mapper.process_frame(dataset.cur_point_cloud_torch, dataset.cur_sem_labels_torch, dataset.cur_point_normals,
                                 dataset.cur_pose_torch, frame_id, (config.dynamic_filter_on and frame_id > 0),
                                 dataset.cur_point_cloud_mono_depth, dataset.cur_point_normals_mono_depth)
        else:
            mapper.determine_used_pose()
            neural_points.reset_local_map(dataset.cur_pose_torch[:3,3], None, frame_id) # not efficient for large map
                                
        T5 = get_time()

        # for the first frame, we need more iterations to do the initialization (warm-up)
        if config.gs_on:
            # when train gs we do not do SDF training seperately except for the first frame
            cur_iter_num = config.iters * config.init_iter_ratio if frame_id == 0 else 0 
        else:
            cur_iter_num = config.iters * config.init_iter_ratio if frame_id == 0 else config.iters
        if dataset.stop_status:
            cur_iter_num = max(1, cur_iter_num-10)
        if frame_id == config.freeze_after_frame: # freeze the decoder after certain frame 
            freeze_decoders(mlp_dict, config)
            config.decoder_freezed = True

        # # conduct local bundle adjustment (with lower frequency)
        # if config.track_on and config.ba_freq_frame > 0 and (frame_id+1) % config.ba_freq_frame == 0:
        #     mapper.bundle_adjustment(config.ba_iters, config.ba_frame)
        
        # mapping with fixed poses (every frame) # now only done for the first frame
        if frame_id % config.mapping_freq_frame == 0:
            mapper.mapping(cur_iter_num)

        T5_1 = get_time()

        # gaussian splatting mapping (fitting)
        if config.gs_on: # only when color available
            if config.gs_batch_training_on:
                gs_iter_num = config.gs_iters if frame_id == (config.gs_batch_frame-1) else 0
            else:
                gs_iter_num = config.gs_iters

            mapper.update_cam_pool(frame_id)

            mapper.joint_gsdf_mapping(gs_iter_num, online_eval_on=config.gs_eval_on, render_pcd=False) # only when sdf field is learned well 
            
            # TODO: check its time consuming, can be done once per x frames 
            if frame_id > 5 and frame_id % 2 == 0:
                with torch.no_grad():  # eval step
                    mapper.check_invalid_neural_points(render_min_nn_count=config.query_nn_k)

        T6 = get_time()

        # regular saving logs
        if config.log_freq_frame > 0 and (frame_id+1) % config.log_freq_frame == 0:
            dataset.write_results_log()

        if not config.silence:
            print("time for frame reading          (ms):", (T1-T0)*1e3)
            print("time for frame preprocessing    (ms):", (T2-T1)*1e3)
            if config.track_on:
                print("time for odometry               (ms):", (T3-T2)*1e3)
            if config.pgo_on:
                print("time for loop detection and PGO (ms):", (T4-T3)*1e3)
            print("time for mapping preparation    (ms):", (T5-T4)*1e3)
            print("time for mapping (SDF)          (ms):", (T5_1-T5)*1e3)
            if config.gs_on:
                print("time for mapping (Gaussian+SDF) (ms):", (T6-T5_1)*1e3)


        # V: Mesh reconstruction and visualization
        cur_mesh = None
        cur_sdf_slice = None

        # set the point cloud for visualization
        dataset.update_o3d_map()
        frame_point_cloud_for_vis = dataset.cur_frame_o3d # already in world frame

        odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis(dataset.processed_frame)
        loop_edges = pgm.loop_edges_vis if config.pgo_on else None

        if config.o3d_vis_on: # if visualizer is off, there's no need to reconstruct the mesh

            o3d_vis.cur_frame_id = frame_id # frame id in the data folder

            T7 = get_time()

            if frame_id == last_frame:
                o3d_vis.vis_global = True
                o3d_vis.ego_view = False
                mapper.free_pool()

            neural_pcd = None
            if o3d_vis.render_neural_points or (frame_id == last_frame): # last frame also vis
                neural_pcd = neural_points.get_neural_points_o3d(query_global=o3d_vis.vis_global, color_mode=o3d_vis.neural_points_vis_mode, 
                                                                 random_down_ratio=1, cur_sensor_position=dataset.cur_pose_ref[:3,3], 
                                                                 vis_normals=o3d_vis.vis_gaussian_normal,
                                                                 vis_free_gaussians=o3d_vis.vis_free_gaussian) # select from geo_feature, ts and certainty

            # reconstruction by marching cubes
            if config.mesh_freq_frame > 0:
                if (frame_id == 0 or frame_id == last_frame or (frame_id+1) % config.mesh_freq_frame == 0 or pgm.last_loop_idx == frame_id):              
                    # update map bbx
                    global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=31) # prime number
                    dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
                    
                    mesh_path = None # no need to save the mesh
                    if frame_id == last_frame and config.save_mesh: # save the mesh at the last frame
                        mc_cm_str = str(round(o3d_vis.mc_res_m*1e2))
                        mesh_path = os.path.join(run_path, "mesh", 'mesh_frame_' + str(frame_id) + "_" + mc_cm_str + "cm.ply")
                    
                    # figure out how to do it efficiently
                    if not o3d_vis.vis_global: # only build the local mesh
                        # cur_mesh = mesher.recon_aabb_mesh(dataset.cur_bbx, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)
                        used_local_pcd = global_neural_pcd_down if neural_pcd is None else neural_pcd
                        chunks_aabb = split_chunks(used_local_pcd, dataset.cur_bbx, o3d_vis.mc_res_m*100) # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
                    else:
                        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(global_neural_pcd_down, aabb, o3d_vis.mc_res_m*100) # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
            
            if config.sdfslice_freq_frame > 0:
                if o3d_vis.render_sdf and (frame_id == 0 or frame_id == last_frame or (frame_id + 1) % config.sdfslice_freq_frame == 0):
                    slice_res_m = config.voxel_size_m * 0.5 # better be larger (to save time) # TODO: add to config
                    sdf_bound = config.surface_sample_range_m * 4.0
                    query_sdf_locally = True
                    if o3d_vis.vis_global:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.map_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, False, -sdf_bound, sdf_bound) # horizontal slice
                    else:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.cur_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # horizontal slice (local)
                    if config.vis_sdf_slice_v:
                        cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(dataset.cur_bbx, dataset.cur_pose_ref[0,3], slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # vertical slice (local)
                        cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                    else:
                        cur_sdf_slice = cur_sdf_slice_h
                                
            # pool_pcd = mapper.get_data_pool_o3d(down_rate=17, only_cur_data=o3d_vis.vis_only_cur_samples) if o3d_vis.render_data_pool else None # down rate should be a prime number
            
            pool_pcd = mapper.get_data_pool_o3d(down_rate=31, only_cur_data=o3d_vis.vis_only_cur_samples)

            o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)

            if o3d_vis.vis_mono_depth_frame:
                frame_point_cloud_for_vis = dataset.cur_frame_mono_depth_o3d 
                if o3d_vis.debug_mode == 1: # show mono depth and lidar together, lidar as red color
                    dataset.cur_frame_o3d.paint_uniform_color(np.array([1.0, 0, 0])) # RED
                    frame_point_cloud_for_vis += dataset.cur_frame_o3d 
                
            o3d_vis.update(frame_point_cloud_for_vis, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd, mapper.T_w_c_cur_view, mapper.rendered_pcd_o3d)

            T8 = get_time()

            if not config.silence:
                print("time for o3d update             (ms):", (T7-T6)*1e3)
                print("time for visualization          (ms):", (T8-T7)*1e3)

        if config.gs_vis_on:
        
            T9 = get_time()

            # add the most recent train frame for vis
            packet_to_vis: VisPacket = VisPacket(frame_id=dataset.processed_frame,
                current_frames=dataset.cur_cam_img, 
                keyframes=None, # mapper.cur_frame_train_views, # None
                img_down_rate=config.gs_vis_down_rate)

            # spawn gaussians in the current local map
            # gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, _ = mapper.spawn_gaussians()
            # packet_to_vis.add_gaussians(gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color)

            packet_to_vis.add_neural_points_data(neural_points, only_local_map=True)

            if frame_point_cloud_for_vis is not None:
                packet_to_vis.add_scan(np.array(frame_point_cloud_for_vis.points, dtype=np.float64), np.array(frame_point_cloud_for_vis.colors, dtype=np.float64))

            if cur_mesh is not None:
                packet_to_vis.add_mesh(np.array(cur_mesh.vertices, dtype=np.float64), np.array(cur_mesh.triangles), np.array(cur_mesh.vertex_colors, dtype=np.float64))

            if cur_sdf_slice is not None:
                packet_to_vis.add_sdf_slice(np.array(cur_sdf_slice.points, dtype=np.float64), np.array(cur_sdf_slice.colors, dtype=np.float64))
            
            if pool_pcd is not None:
                packet_to_vis.add_sdf_training_pool(np.array(pool_pcd.points, dtype=np.float64), np.array(pool_pcd.colors, dtype=np.float64))

            packet_to_vis.add_traj(odom_poses, gt_poses, pgo_poses)

            q_main2vis.put(packet_to_vis)

            T10 = get_time()
            if not config.silence:
                print("time for gs visualizer update   (ms):", (T10-T9)*1e3)

        cur_frame_process_time = np.array([T2-T1, T3-T2, T5-T4, T6-T5, T4-T3]) # loop & pgo in the end, visualization and I/O time excluded
        dataset.time_table.append(cur_frame_process_time) # in s
        gs_time_table.append(T6-T5_1)

        if config.wandb_vis_on:
            wandb_log_content = {'frame': frame_id, 'timing(s)/preprocess': T2-T1, 'timing(s)/tracking': T3-T2, 'timing(s)/pgo': T4-T3, 'timing(s)/mapping': T6-T4} 
            wandb.log(wandb_log_content)
        
        dataset.processed_frame += 1
    
    # VI. Save results
    pose_eval_results = None
    if config.track_on:
        pose_eval_results = dataset.write_results()
    if config.pgo_on and pgm.pgo_count>0:
        print("# Loop corrected: ", pgm.pgo_count)
        pgm.write_g2o(os.path.join(run_path, "final_pose_graph.g2o"))
        pgm.write_loops(os.path.join(run_path, "loop_log.txt"))
        if config.o3d_vis_on:
            pgm.plot_loops(os.path.join(run_path, "loop_plot.png"), vis_now=False)  
    
    # gs eval 
    if config.gs_on: # TODO: add to a function inside mapper or dataset 
        
        print("Begin rendering evaluation")
        # don't do visualization
        mapper.gs_eval_offline(None, q_vis2main, eval_down_rate=config.gs_vis_down_rate, skip_end_count=10) 
        # visualize the results
        # mapper.gs_eval_offline(q_main2vis, q_vis2main, eval_down_rate=config.gs_vis_down_rate, skip_end_count=10)
        mapper.gs_eval_out()

    neural_points.prune_map(config.max_prune_certainty, 0) # prune uncertain points for the final output     
    neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, False, False) # merge the final neural point map
    
    neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode = 0, vis_normals=True, vis_free_gaussians=True)
    if config.save_map:
        neural_points_path = os.path.join(run_path, "map", "neural_points.ply")
        o3d.io.write_point_cloud(neural_points_path, neural_pcd) # write the neural point cloud
        print(f"save the neural point map to {neural_points_path}")
    if config.save_mesh and cur_mesh is None:
        output_mc_res_m = config.mc_res_m*0.6
        chunks_aabb = split_chunks(neural_pcd, neural_pcd.get_axis_aligned_bounding_box(), output_mc_res_m * 100) # reconstruct in chunks
        mc_cm_str = str(round(output_mc_res_m*1e2))
        mesh_path = os.path.join(run_path, "mesh", "mesh_" + mc_cm_str + "cm.ply")
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, output_mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=config.mesh_min_nn)
    neural_points.clear_temp() # clear temp data for output
    if config.save_map:
        # if config.gs_on:
        #     gs_map = os.path.join(run_path, "map", "gaussians.ply") # global gaussian map is also saved here (now we don't save gaussian map anymore)
        #     neural_points.save_gaussian_ply(gs_map)
        # else:
        save_implicit_map(run_path, neural_points, mlp_dict)
    if config.save_merged_pc:
        dataset.write_merged_point_cloud() # replay: save merged point cloud map
    
    if config.o3d_vis_on:
        while True:
            o3d_vis.ego_view = False
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
            odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis(dataset.processed_frame)
            o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)
    
    return pose_eval_results

if __name__ == "__main__":

    run_pin_slam()