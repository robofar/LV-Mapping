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
import torch.multiprocessing as mp
# import multiprocessing as mp
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
    create_bbx_o3d,
    save_implicit_map,
    setup_experiment,
    split_chunks,
    transform_torch,
    remove_gpu_cache,
)
from utils.tracker import Tracker

from gs_gui import slam_gui
from gs_gui.gui_utils import VisPacket, ParamsGUI, ControlPacket, get_latest_queue

'''
    📍PINGS
     Y. Pan et al. from IPB
'''

app = typer.Typer(add_completion=False, rich_markup_mode="rich", context_settings={"help_option_names": ["-h", "--help"]})

_available_dl_help = available_dataloaders()

docstring = f"""
:round_pushpin: PINGS: joint distance field and radiance field mapping using a unified neural representation\n

[bold green]Examples: [/bold green]

# Use a more specific dataloader: select from {", ".join(_available_dl_help)}

# Run on IPB Car data sequence
$ python3 pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/ipbcar_test_subset/ -vmsg

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
    print("[bold green]PINGS starts[/bold green]")

    mp.set_start_method("spawn")

    geo_feature_dim = config.feature_dim
    sem_feature_dim = config.sem_feature_dim
    color_feature_dim = config.color_feature_dim

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

    # mapper
    mapper = Mapper(config, dataset, neural_points, mlp_dict)

    # mesh reconstructor
    mesher = Mesher(config, neural_points, mlp_dict)
    cur_mesh = None

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
        q_vis2main = mp.Queue()

        params_gui = ParamsGUI(
            decoders=mlp_dict,
            background=torch.tensor(config.bg_color, dtype=config.dtype, device=config.device),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            config=config,
            is_rgbd=dataset.is_rgbd,
            neural_point_vis_down_rate=config.neural_point_vis_down_rate,
            gs_default_on=gs_on,
            frustum_size=config.vis_frame_axis_len,
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

    print("Some config info:")
    print(f"Dynamic Filtering using SDF: {config.dynamic_filter_on}")
    print(f"GS Invalid Check (neural_points.valid_gs_mask): {config.gs_invalid_check_on}")
    print(f"Estimating Normal for Input PointCloud: {config.estimate_normal}")
    print(f"Use only colorized points: {config.learn_color_residual}")
    print(f"Temporal Local Map: {config.temporal_local_map_off == False}")
    print(f"Use Local Pool for SDF Training: {config.use_local_pool_sdf}")
    print(f"Freeze decoder after {config.freeze_after_iter} iterations")

    mapper.load_gt_poses()
        
    # for each frame
    for frame_id in tqdm(range(dataset.total_pc_count)): # frame id as the processed frame, possible skipping done in data loader
        remove_gpu_cache()
        print(f"Frame ID: {frame_id}")

        # I. Load data (pointclouds and poses) and preprocessing
        dataset.init_temp_data() # init cur frame temp data
        dataset.read_frame_with_loader(frame_id, use_image=config.gs_on) # read pointcloud + all poses (cur_pose, prev_pose and odometry since we know all poses in advance)
        dataset.preprocess_frame() # preprocess frame + downsampling + deskewing
        if (not dataset.is_rgbd): # if it is rgbd dataset dont override gt depth (but change such that I obtain foundation masks)
            dataset.project_pointcloud_to_cams(use_only_colorized_points = config.learn_color_residual, use_odom_tran=False)
        

        mapper.process_frame(dataset.cur_point_cloud_torch, dataset.cur_sem_labels_torch, dataset.cur_point_normals, dataset.cur_pose_torch, frame_id, (config.dynamic_filter_on and frame_id > 0))

        if config.gs_on: # only when color available
            if dataset.cur_cam_img is not None:
                mapper.update_cam_pool(frame_id)





        # last
        dataset.processed_frame += 1
    

    print("SDF Training...")
    mapper.mapping(5000)


    # VI. Save results
    mapper.free_pool()

    color_mode_for_neural_point_output = 1 # 0: original rgb, 1: geo_feature pca, 2: color_feature_pca, 3: ts, 4: certainty, 5: random
    neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode = color_mode_for_neural_point_output)
    if config.save_mesh and cur_mesh is None:
        print("Saving mesh...")
        chunks_aabb = split_chunks(neural_pcd, neural_pcd.get_axis_aligned_bounding_box(), config.mc_res_m * 100) # reconstruct in chunks
        mc_cm_str = str(round(config.mc_res_m*1e2))
        mesh_path = os.path.join(run_path, "mesh", "mesh_" + mc_cm_str + "cm.ply")
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, config.mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=config.mesh_min_nn)
        print(f"save the reconstructed mesh to {mesh_path}")


if __name__ == "__main__":
    app()
