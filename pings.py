#!/usr/bin/env python3
# @file      pin_slam.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import csv
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
# import torch.multiprocessing as mp
import multiprocessing as mp
import wandb
from rich import print
from tqdm import tqdm

import dtyper as typer
from typing import Optional, Tuple

from dataset.dataset_indexing import set_dataset_path
from dataset.slam_dataset import SLAMDataset
from dataset.dataloaders import available_dataloaders
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
from utils.tracker import Tracker

from gs_gui import slam_gui
from gs_gui.gui_utils import VisPacket, ParamsGUI, ControlPacket

'''
    📍PINGS
     Y. Pan et al. from IPB
'''

app = typer.Typer(add_completion=False, rich_markup_mode="rich", context_settings={"help_option_names": ["-h", "--help"]})

_available_dl_help = available_dataloaders()

docstring = f"""
:round_pushpin: PINGS: joint distance field and radiance field mapping using a unified neural representation\n

[bold green]Examples: [/bold green]

# Run on IPB Car data sequence
$ python3 pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -vmsg

# Run on Oxford Spires dataset
$ python3 pings.py ./config/lidar_slam/run_oxford_gs.yaml oxford -i ./data/Oxford-Spires-Dataset/2024-03-12-keble-college-04/ -vmsg
"""

@app.command(help=docstring)
def run_pin_slam(
    config_path: str = typer.Argument('config/lidar_slam/run.yaml', help='Path to *.yaml config file'),
    dataset_name: Optional[str] = typer.Argument(None, help='Name of a specific dataset, example: kitti, mulran, or rosbag (when data_loader_on is set)'),
    sequence_name: Optional[str] = typer.Argument(None, help='Name of a specific data sequence or the rostopic for point cloud (when data_loader_on is set)'),
    seed: int = typer.Option(42, help='Set the random seed'),
    input_path: Optional[str] = typer.Option(None, '--input-path', '-i', help='Path to the point cloud input directory (overrides pc_path in config file)'),
    output_path: Optional[str] = typer.Option(None, '--output-path', '-o', help='Path to the result output directory (overrides output_root in config file)'),
    frame_range: Optional[Tuple[int, int, int]] = typer.Option(None, '--range', help='Specify the start, end and step of the processed frame, e.g. "10 1000 1"'),
    data_loader_on: bool = typer.Option(True, '--data-loader-on', '-d', help='Use specific data loader (rosbag, pcap, mcap dataloaders and typical supported datasets)'),
    visualize: bool = typer.Option(False, '--visualize', '-v', help='Turn on the GS visualizer (could make the SLAM processing slower)'),
    cpu_only: bool = typer.Option(False, '--cpu-only', '-c', help='Run only on CPU'),
    log_on: bool = typer.Option(False, '--log-on', '-l', help='Turn on the logs printing'),
    wandb_on: bool = typer.Option(False, '--wandb-on', '-w', help='Turn on the weight & bias logging'),
    save_map: bool = typer.Option(False, '--save-map', '-s', help='Save the PIN map after SLAM'),
    save_mesh: bool = typer.Option(False, '--save-mesh', '-m', help='Save the reconstructed mesh after SLAM'),
    save_merged_pc: bool = typer.Option(False, '--save-merged-pc', '-p', help='Save the merged point cloud after SLAM'),
    gs_on: bool = typer.Option(False, '--gs-on', '-g', help='Turn on GS'),
    deskew: bool = typer.Option(False, '--deskew', help='Try to deskew the LiDAR scans'),
    tag: Optional[str] = typer.Option(None, '--tag', help='A tag for this experiment')
) -> None:

    config = Config()
    config.load(config_path)
    config.use_dataloader = data_loader_on
    config.seed = seed
    config.silence = not log_on
    config.wandb_vis_on = wandb_on
    config.gs_on = gs_on
    config.o3d_vis_on = visualize
    config.save_map = save_map
    config.save_mesh = save_mesh
    config.save_merged_pc = save_merged_pc

    if not config.deskew and deskew:
        config.deskew = True
    
    if frame_range:
        config.begin_frame, config.end_frame, config.step_frame = frame_range
        
    if cpu_only:
        config.device = 'cpu'
        
    if input_path:
        config.pc_path = input_path
        
    if output_path:
        config.output_root = output_path
        
    if dataset_name:
        set_dataset_path(config, dataset_name, sequence_name)

    if tag:
        config.name = "{}_{}".format(tag, config.name)  

    argv = sys.argv
    run_path = setup_experiment(config, argv)
    print("[bold green]PIN-SLAM starts[/bold green]")

    mp.set_start_method("spawn")

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
    # gaussian_alpha_mlp = Decoder(config, color_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 1, n_gaussian, view_concat_dim) # concat distance
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

    # save merged point cloud map from gt pose as a reference map
    if config.save_merged_pc and dataset.gt_pose_provided:
        dataset.write_merged_point_cloud(use_gt_pose=True, out_file_name='merged_gt_pc', 
            frame_step=1, merged_downsample=True, tsdf_fusion_on=True)
    
    gs_time_table = []

    # TODO: test TSDF fusion (pass)
    # tsdf_mesh_path = os.path.join(run_path, "mesh", "tsdf_fusion_mesh.ply")
    # tsdf_mesh = dataset.o3d_tsdf_fusion(frame_step=20, output_path=tsdf_mesh_path)
    # return 

    q_main2vis = q_vis2main = None
    if config.o3d_vis_on:
        # communicator between the processes
        q_main2vis = mp.Queue() 
        q_vis2main = mp.Queue(maxsize=1)

        params_gui = ParamsGUI(
            decoders=mlp_dict,
            background=torch.tensor(config.bg_color, dtype=config.dtype, device=config.device),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            config=config,
            is_rgbd=dataset.is_rgbd,
            neural_point_vis_down_rate=config.neural_point_vis_down_rate,
        )
        gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
        gui_process.start()
        time.sleep(3) # second

        # visualizer configs
        vis_visualize_on = True
        vis_source_pc_weight = False
        vis_global_on = not config.local_map_default_on
        vis_mesh_on = config.mesh_default_on
        vis_mesh_freq_frame = config.mesh_freq_frame
        vis_mesh_mc_res_m = config.mc_res_m
        vis_mesh_min_nn = config.mesh_min_nn
        vis_sdf_on = config.sdf_default_on
        vis_sdf_freq_frame = config.sdfslice_freq_frame
        vis_sdf_slice_height = config.sdf_slice_height
        vis_sdf_res_m = config.vis_sdf_res_m

    cur_mesh = None
    cur_sdf_slice = None
    pool_pcd = None

        
    # for each frame
    for frame_id in tqdm(range(dataset.total_pc_count)): # frame id as the processed frame, possible skipping done in data loader
        
        remove_gpu_cache()

        # I. Load data and preprocessing
        T0 = get_time()

        dataset.init_temp_data() # init cur frame temp data

        if config.use_dataloader:
            dataset.read_frame_with_loader(frame_id, use_image=config.gs_on, monodepth_on=config.monodepth_on)
        else:
            dataset.read_frame(frame_id)

        T1 = get_time()
        
        valid_lidar_frame_flag = dataset.preprocess_frame()

        T2 = get_time()

        if valid_lidar_frame_flag: # no input point cloud for this frame

            # II. Odometry
            if frame_id > 0: 
                if config.track_on:
                    tracking_result = tracker.tracking(dataset.cur_source_points, dataset.cur_pose_guess_torch, 
                                                    dataset.cur_source_colors, dataset.cur_source_normals,
                                                    vis_result=config.o3d_vis_on)
                    cur_pose_torch, cur_odom_cov, weight_pc_o3d, valid_flag = tracking_result
                    dataset.lose_track = not valid_flag
                    dataset.update_odom_pose(cur_pose_torch) # update dataset.cur_pose_torch
                    
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
                detect_correct_loop(config, pgm, dataset, neural_points, lcd_npmc, mapper, tracker, frame_id)

            T4 = get_time()
            
            # IV: Mapping
            
            # Re-generate colorized point cloud and correct depth map after point cloud deskewing
            # if config.deskew: # only needed for LiDAR datasets
            # TODO: turn on
            if config.gs_on and (not dataset.is_rgbd):
                dataset.project_pointcloud_to_cams(use_only_colorized_points=config.learn_color_residual, tran_in_frame=dataset.last_odom_tran_torch) # True # config.learn_color_residual
            
            # if lose track, we will not update the map and data pool (don't let the wrong pose to corrupt the map)
            # if the robot stop, also don't process this frame, since there's no new oberservations
            dataset.voxel_downsample_points_for_mapping()
            
            # update neural point map and sample data for sdf training
            if (not dataset.lose_track and not dataset.stop_status) or frame_id < 5:
                mapper.process_frame(dataset.cur_point_cloud_torch, dataset.cur_sem_labels_torch, dataset.cur_point_normals,
                                    dataset.cur_pose_torch, frame_id, (config.dynamic_filter_on and frame_id > 0),
                                    dataset.cur_point_cloud_mono_depth, dataset.cur_point_normals_mono_depth)
            else:
                neural_points.reset_local_map(dataset.cur_pose_torch[:3,3], None, frame_id) # not efficient for large map


        T5 = get_time()

        mapper.determine_used_pose()

        # for the first frame, we need more iterations to do the initialization (warm-up)
        if config.gs_on:
            # when train gs we do not do SDF training seperately except for the first frame
            cur_iter_num = config.iters * config.init_iter_ratio if mapper.sdf_train_frame_count == 1 else 0 
            frame_count_for_freeze_check = mapper.gs_train_frame_count
        else:
            cur_iter_num = config.iters * config.init_iter_ratio if mapper.sdf_train_frame_count == 1 else config.iters
            frame_count_for_freeze_check = mapper.sdf_train_frame_count
        if dataset.stop_status:
            cur_iter_num = max(1, cur_iter_num-10)

        # freeze the decoder after certain frame 
        if not config.decoder_freezed and (frame_count_for_freeze_check == config.freeze_after_frame):
            freeze_decoders(mlp_dict, config)
            config.decoder_freezed = True
            neural_points.compute_feature_principle_components(down_rate = 17)

        # mapping with fixed poses # now only done for the first frame for the gs mode
        if valid_lidar_frame_flag:
            mapper.mapping(cur_iter_num)

        T5_1 = get_time()

        # gaussian splatting mapping (fitting)
        if config.gs_on: # only when color available
            if dataset.cur_cam_img is not None and not neural_points.is_empty(): # when there are new imgs, do training
                mapper.update_cam_pool(frame_id)
                mapper.joint_gsdf_mapping(config.gs_iters) # only when sdf field is learned well 
            
        # TODO: check its time consuming, can be done once per x frames 
        if valid_lidar_frame_flag and mapper.sdf_train_frame_count > 5 and mapper.sdf_train_frame_count % 2 == 0 and config.gs_invalid_check_on:
            mapper.check_invalid_neural_points() # render_min_nn_count=config.query_nn_k

        T6 = get_time()

        # regular saving logs (not used)
        if config.log_freq_frame > 0 and (frame_id+1) % config.log_freq_frame == 0:
            dataset.write_results_log()

        if not config.silence:
            print("time for frame reading          (ms): {:.2f}".format((T1-T0)*1e3))
            if valid_lidar_frame_flag:
                print("time for frame preprocessing    (ms): {:.2f}".format((T2-T1)*1e3))
                if config.track_on:
                    print("time for odometry               (ms): {:.2f}".format((T3-T2)*1e3))
                if config.pgo_on:
                    print("time for loop detection and PGO (ms): {:.2f}".format((T4-T3)*1e3))
                print("time for mapping preparation    (ms): {:.2f}".format((T5-T4)*1e3))
                print("time for mapping (SDF)          (ms): {:.2f}".format((T5_1-T5)*1e3))
            if config.gs_on:
                print("time for mapping (Gaussian+SDF) (ms): {:.2f}".format((T6-T5_1)*1e3))

        # V: Mesh reconstruction and visualization

        if valid_lidar_frame_flag or dataset.cur_cam_img is not None: # lidar or cameras are avilable

            if config.o3d_vis_on: # if visualizer is off, there's no need to reconstruct the mesh

                if not q_vis2main.empty():
                    control_packet: ControlPacket = q_vis2main.get()

                    vis_visualize_on = control_packet.flag_vis
                    vis_global_on = control_packet.flag_global
                    vis_mesh_on = control_packet.flag_mesh   
                    vis_sdf_on = control_packet.flag_sdf
                    vis_source_pc_weight = control_packet.flag_source
                    vis_mesh_mc_res_m = control_packet.mc_res_m
                    vis_mesh_min_nn = control_packet.mesh_min_nn
                    vis_mesh_freq_frame = control_packet.mesh_freq_frame
                    vis_sdf_slice_height = control_packet.sdf_slice_height
                    vis_sdf_freq_frame = control_packet.sdf_freq_frame
                    vis_sdf_res_m = control_packet.sdf_res_m

                    while control_packet.flag_pause:
                        time.sleep(0.1)
                        if not q_vis2main.empty():
                            control_packet = q_vis2main.get()
                            if not control_packet.flag_pause:
                                break

                # set the point cloud for visualization
                dataset.update_o3d_map() # this is after downsampling
                frame_point_cloud_for_vis = dataset.cur_frame_o3d # already in world frame

                T7 = get_time()

                neural_pcd = None
                # if o3d_vis.render_neural_points or (frame_id == last_frame): # last frame also vis
                #     neural_pcd = neural_points.get_neural_points_o3d(query_global=o3d_vis.vis_global, color_mode=o3d_vis.neural_points_vis_mode, 
                #                                                     random_down_ratio=1, cur_sensor_position=dataset.cur_pose_ref[:3,3], 
                #                                                     vis_normals=o3d_vis.vis_gaussian_normal,
                #                                                     vis_free_gaussians=o3d_vis.vis_free_gaussian) # select from geo_feature, ts and certainty
                
                sdf_train_frame_id = mapper.sdf_train_frame_count

                # reconstruction by marching cubes
                if vis_mesh_on and (frame_id == 0 or frame_id == last_frame or (frame_id+1) % vis_mesh_freq_frame == 0 or pgm.last_loop_idx == frame_id):            
                    # update map bbx
                    global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=31) # prime number
                    dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
                    
                    # figure out how to do it efficiently
                    if not vis_global_on: # only build the local mesh
                        used_local_pcd = global_neural_pcd_down if neural_pcd is None else neural_pcd
                        cur_bbx = used_local_pcd.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(used_local_pcd, cur_bbx, vis_mesh_mc_res_m*100) # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, vis_mesh_mc_res_m, None, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=vis_mesh_min_nn)    
                    else:
                        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(global_neural_pcd_down, aabb, vis_mesh_mc_res_m*100) # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, vis_mesh_mc_res_m, None, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=vis_mesh_min_nn)    
                
                # if config.sdfslice_freq_frame > 0:
                #     if o3d_vis.render_sdf and (sdf_train_frame_id == 1 or frame_id == last_frame or sdf_train_frame_id % config.sdfslice_freq_frame == 0):
                #         slice_res_m = config.voxel_size_m * 0.6 # better be larger (to save time) # TODO: add to config
                #         sdf_bound = config.surface_sample_range_m * 4.0
                #         query_sdf_locally = True
                #         if o3d_vis.vis_global:
                #             cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.map_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, False, -sdf_bound, sdf_bound) # horizontal slice
                #         else:
                #             cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.cur_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # horizontal slice (local)
                #         if config.vis_sdf_slice_v:
                #             cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(dataset.cur_bbx, dataset.cur_pose_ref[0,3], slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # vertical slice (local)
                #             cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                #         else:
                #             cur_sdf_slice = cur_sdf_slice_h
                                    
                pool_pcd = mapper.get_data_pool_o3d(down_rate=37)

                odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis(dataset.processed_frame)
                loop_edges = pgm.loop_edges_vis if config.pgo_on else None

                # if o3d_vis.vis_mono_depth_frame:
                #     frame_point_cloud_for_vis = dataset.cur_frame_mono_depth_o3d 
                #     if o3d_vis.debug_mode == 1: # show mono depth and lidar together, lidar as red color
                #         dataset.cur_frame_o3d.paint_uniform_color(np.array([1.0, 0, 0])) # RED
                #         frame_point_cloud_for_vis += dataset.cur_frame_o3d

                # add the most recent train frame for vis
                # we have either Lidar point cloud or camera images loaded
                packet_to_vis: VisPacket = VisPacket(frame_id=dataset.processed_frame,
                    current_frames=dataset.cur_cam_img, 
                    keyframes=mapper.cur_frame_train_views, # None
                    img_down_rate=config.gs_vis_down_rate)

                # spawn gaussians in the current local map
                # gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, _ = mapper.spawn_gaussians()
                # packet_to_vis.add_gaussians(gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color)
                if not neural_points.is_empty():
                    packet_to_vis.add_neural_points_data(neural_points, only_local_map=True, add_sorrounding_points=config.gs_on, pca_color_on=config.decoder_freezed)

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

                T8 = get_time()

                if not config.silence:
                    print("time for o3d update             (ms): {:.2f}".format((T7-T6)*1e3))
                    print("time for gs visualizer update   (ms): {:.2f}".format((T8-T7)*1e3))

                
        if valid_lidar_frame_flag:
            cur_frame_process_time = np.array([T2-T1, T3-T2, T5-T4, T6-T5, T4-T3]) # loop & pgo in the end, visualization and I/O time excluded
            dataset.time_table.append(cur_frame_process_time) # in s
            if config.gs_on:
                gs_time_table.append(T6-T5_1)

            if config.wandb_vis_on:
                wandb_log_content = {'frame': frame_id, 'timing(s)/preprocess': T2-T1, 'timing(s)/tracking': T3-T2, 'timing(s)/pgo': T4-T3, 'timing(s)/mapping': T6-T4} 
                wandb.log(wandb_log_content)
        
        dataset.processed_frame += 1
    
    # VI. Save results
    mapper.free_pool()
    pose_eval_results = dataset.write_results()
    if config.pgo_on and pgm.pgo_count>0:
        print("# Loop corrected: ", pgm.pgo_count)
        pgm.write_g2o(os.path.join(run_path, "final_pose_graph.g2o"))
        pgm.write_loops(os.path.join(run_path, "loop_log.txt"))
        pgm.plot_loops(os.path.join(run_path, "loop_plot.png"), vis_now=False)  
    
    # gs eval # TODO: better put after map pruning and hash recreating 
    if config.gs_on and config.gs_eval_on: 
        print("Begin rendering evaluation")
        mapper.gs_eval_offline(None, q_vis2main, eval_down_rate=config.gs_vis_down_rate, skip_end_count=10, 
                               lpips_eval_on=True, pc_cd_eval_on=config.rendered_pc_eval_on, rerender_tsdf_fusion_on=config.rerender_tsdf_fusion_on) # FIXME
        mapper.gs_eval_out()

    neural_points.prune_map(config.max_prune_certainty, 0) # prune uncertain points for the final output     
    neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, False, False) # merge the final neural point map
    print("Final Gaussian count:", neural_points.count(valid_gs_only=True)) # FIXME

    color_mode_for_neural_point_output = 1 # 0: original rgb, 1: geo_feature pca, 2: color_feature_pca, 3: ts, 4: certainty, 5: random
    neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode = color_mode_for_neural_point_output, vis_free_gaussians=False)
    if config.save_map:
        neural_points_path = os.path.join(run_path, "map", "neural_points.ply")
        o3d.io.write_point_cloud(neural_points_path, neural_pcd) # write the neural point cloud
        print(f"save the neural point map to {neural_points_path}")
    if config.save_mesh and cur_mesh is None:
        chunks_aabb = split_chunks(neural_pcd, neural_pcd.get_axis_aligned_bounding_box(), config.mc_res_m * 100) # reconstruct in chunks
        mc_cm_str = str(round(config.mc_res_m*1e2))
        mesh_path = os.path.join(run_path, "mesh", "mesh_" + mc_cm_str + "cm.ply")
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, config.mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=config.mesh_min_nn)
        print(f"save the reconstructed mesh to {mesh_path}")
    neural_points.clear_temp() # clear temp data for output
    if config.save_map:
        save_implicit_map(run_path, neural_points, mlp_dict)
    # if config.save_merged_pc:
    #     dataset.write_merged_point_cloud() # replay: save merged point cloud map
    
    # if config.o3d_vis_on:
    #     while True:
    #         o3d_vis.ego_view = False
    #         o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
    #         odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis(dataset.processed_frame)
    #         o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)
    
    return pose_eval_results



def detect_correct_loop(config, pgm: PoseGraphManager, dataset: SLAMDataset, neural_points: NeuralPoints, lcd_npmc: NeuralPointMapContextManager,
                        mapper: Mapper, tracker: Tracker, frame_id: int):

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
        travel_dist = dataset.travel_dist[:frame_id+1]
        pgm.add_odometry_factor(frame_id, frame_id-1, dataset.last_odom_tran) # T_p<-c
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
            loop_id, loop_dist, loop_transform = detect_local_loop(dataset.pgo_poses[:frame_id+1], loop_candidate_mask, pgm.drift_radius, frame_id, dataset.loop_reg_failed_count, config.local_loop_dist_thre, config.local_loop_dist_thre*3.0, config.silence)
            if loop_id is None and config.global_loop_on: # global loop detection (large drift)
                loop_id, loop_cos_dist, loop_transform, local_map_context_loop = lcd_npmc.detect_global_loop(dataset.pgo_poses[:frame_id+1], pgm.drift_radius*config.loop_dist_drift_ratio_thre, loop_candidate_mask, neural_points) # latency has been considered here     
        if loop_id is not None:
            if config.loop_z_check_on and abs(loop_transform[2,3]) > config.voxel_size_m*4.0: # for multi-floor buildings, z may cause ambiguilties
                loop_id = None # delta z check failed
        if loop_id is not None: # if a loop is found, we refine loop closure transform initial guess with a scan-to-map registration                    
            pose_init_torch = torch.tensor((dataset.pgo_poses[loop_id] @ loop_transform), device=config.device, dtype=torch.float64) # T_w<-c = T_w<-l @ T_l<-c 
            neural_points.recreate_hash(pose_init_torch[:3,3], None, True, True, loop_id) # recreate hash and local map at the loop candidate frame for registration, this is the reason why we'd better to keep the duplicated neural points until the end
            loop_reg_source_point = dataset.cur_source_points.clone()
            pose_refine_torch, loop_cov_mat, weight_pcd, reg_valid_flag = tracker.tracking(loop_reg_source_point, pose_init_torch, loop_reg=True)
            # only conduct pgo when the loop and loop constraint is correct
            if reg_valid_flag: # refine succeed
                pose_refine_np = pose_refine_torch.detach().cpu().numpy()
                loop_transform = np.linalg.inv(dataset.pgo_poses[loop_id]) @ pose_refine_np # T_l<-c = T_l<-w @ T_w<-c # after refinement
                reg_valid_flag = pgm.add_loop_factor(frame_id, loop_id, loop_transform)
            if reg_valid_flag:
                if not config.silence:
                    print("[bold green]Refine loop transformation succeed [/bold green]")
                pgm.optimize_pose_graph() # conduct pgo
                cur_loop_vis_id = frame_id-config.local_map_context_latency if local_map_context_loop else frame_id
                pgm.loop_edges_vis.append(np.array([loop_id, cur_loop_vis_id],dtype=np.uint32)) # only for vis
                pgm.loop_edges.append(np.array([loop_id, frame_id],dtype=np.uint32))
                pgm.loop_trans.append(loop_transform)
                # update the neural points and poses after pgo
                pose_diff_torch = torch.tensor(pgm.get_pose_diff(), device=config.device, dtype=config.dtype)
                dataset.cur_pose_torch = torch.tensor(pgm.cur_pose, device=config.device, dtype=config.dtype)
                neural_points.adjust_map(pose_diff_torch) # transform neural points (position and orientation) along with associated frame poses # time consuming part
                neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, (not config.pgo_merge_map), config.rehash_with_time, frame_id) # recreate hash from current time
                mapper.transform_data_pool(pose_diff_torch) # transform global pool
                dataset.update_poses_after_pgo(pgm.pgo_poses)
                if config.gs_on:
                    mapper.update_poses_cam_pool(pgm.pgo_poses) # transform cameras in the pool
                pgm.last_loop_idx = frame_id
                pgm.min_loop_idx = min(pgm.min_loop_idx, loop_id)
                dataset.loop_reg_failed_count = 0
            else:
                if not config.silence:
                    print("[bold red]Registration failed, reject the loop candidate [/bold red]")
                neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, True, True, frame_id) # if failed, you need to reset the local map back to current frame
                dataset.loop_reg_failed_count += 1


if __name__ == "__main__":
    app()