#!/usr/bin/env python3
# @file      inspect_pings.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import argparse
import glob
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
from rich import print

from dataset.slam_dataset import SLAMDataset, read_kitti_format_poses
from model.decoder import Decoder
from model.neural_gaussians import NeuralPoints
from utils.config import Config
from utils.mesher import Mesher
from utils.tools import setup_experiment, split_chunks, load_decoders
from utils.visualizer import MapVisualizer

from gaussian_splatting.scene.cameras import CamImage

from gs_gui import slam_gui
from gs_gui.gui_utils import VisPacket, ParamsGUI

# TODO

'''
    load the pings and pose and render the video 
'''

parser = argparse.ArgumentParser()
parser.add_argument('experiment_path', type=str, help='Path to a certain experiment folder storing the PINGS map')
parser.add_argument('--center_frame_id', '-f', type=int, default=0, help='PINGS local map center frame id')
parser.add_argument('--show_mesh', '-m', action='store_true', default=False, help='Show the PINGS mesh')
parser.add_argument('--show_global', '-g', action='store_true', default=False, help='Show the global map instead of the local map (might cost a lot of memory and not very fast during inferencing)')
parser.add_argument('--mesh_mc_m', type=float, default=-1, help='Marching cubes resolution (in meter) for mesh reconstruction')
parser.add_argument('--mesh_min_nn_k', type=int, default=-1, help='SDF querying min neighbor neural point count for mesh reconstruction')
args, unknown = parser.parse_known_args()

def inspect_pings_map():

    experiment_path = args.experiment_path

    yaml_files = glob.glob(f"{experiment_path}/*.yaml")
    if len(yaml_files) > 1: # Check if there is exactly one YAML file
        sys.exit("There are multiple YAML files. Please handle accordingly.")
    elif len(yaml_files) == 0:  # If no YAML files are found
        sys.exit("No YAML files found in the specified path.")
    
    config = Config()
    config.load(yaml_files[0])

    model_path = os.path.join(experiment_path, "model", "pin_map.pth")
    pose_path = os.path.join(experiment_path, "slam_poses_kitti.txt")
    config.model_path = model_path
    config.pose_path = pose_path

    center_frame_id = int(args.center_frame_id)

    print("Please provide the path to the result folder.\n\
            Try: python inspect_pings.py xxx/result/path\
            [optional: -f center_frame]")

    # example: python inspect_pings.py ./experiments/test_ipbcar_gs_ipb_car__2024-10-25_11-46-52 -g
    # example: python inspect_pings.py ./experiments/test_ipbcar_gs_ipb_car__2024-10-25_13-31-54 -f 10
    # example: python inspect_pings.py ./experiments/test_ipbcar_gs_ipb_car__2024-10-25_15-36-50
    # example: (good) roundabout inspect_pings.py ./experiments/test_ipbcar_gs_ipb_car__2024-10-25_18-59-40 -g


    print("[bold green]Load PINGS Map[/bold green]","📍" )

    run_path = setup_experiment(config, sys.argv, debug_mode=True)
    config.use_dataloader = True

    mp.set_start_method("spawn") # don't forget this
    
    # initialize the mlp decoder
    geo_feature_dim = config.feature_dim
    color_feature_dim = config.color_feature_dim

    geo_mlp = Decoder(config, geo_feature_dim, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    color_mlp = Decoder(config, color_feature_dim, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel) if config.color_on else None

    dist_concat_dim = 1 if config.dist_concat_on else 0
    view_concat_dim = 3 if config.view_concat_on else 0

    gaussian_xyz_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, 0)
    gaussian_rot_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 4, config.spawn_n_gaussian, 0)
    gaussian_scale_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, 0)
    gaussian_alpha_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 1, config.spawn_n_gaussian, dist_concat_dim) # concat distance
    gaussian_color_mlp = Decoder(config, color_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, view_concat_dim) # concat view direction
    
    mlp_dict = {}
    
    mlp_dict["sdf"] = geo_mlp
    mlp_dict["color"] = color_mlp
    mlp_dict["semantic"] = None

    mlp_dict["gauss_xyz"] = gaussian_xyz_mlp
    mlp_dict["gauss_scale"] = gaussian_scale_mlp
    mlp_dict["gauss_rot"] = gaussian_rot_mlp
    mlp_dict["gauss_alpha"] = gaussian_alpha_mlp
    mlp_dict["gauss_color"] = gaussian_color_mlp

    # initialize the neural point features
    neural_points = NeuralPoints(config)

    # Load the map, decoders are then freezed

    loaded_model = torch.load(model_path)
    neural_points = loaded_model["neural_points"]

    # load decoders
    load_decoders(loaded_model, mlp_dict) 

    slam_poses = read_kitti_format_poses(config.pose_path)
    frame_count = len(slam_poses)
    slam_poses = np.array(slam_poses)

    # # dataset
    # dataset = SLAMDataset(config)
    # dataset.read_frame_with_loader(0, init_pose = True, use_image=True) 

    # reset neural points
    center_frame_id = min(center_frame_id, frame_count-1)
    ref_pose = torch.tensor(slam_poses[center_frame_id], device=config.device, dtype=config.dtype)
    ref_position = ref_pose[:3,3]
    # ref_position = neural_points.neural_points[0]
    
    neural_points.recreate_hash(ref_position, with_ts=False)

    # mesh reconstructor
    mesher = Mesher(config, neural_points, mlp_dict)

    cur_mesh = None
    if args.show_mesh:

        print("Reconstruct mesh from the SDF")

        down_rate = 31 # prime number
        mesh_vox_size_m = args.mesh_mc_m
        mesh_min_nn_k_used = args.mesh_min_nn_k
        if args.mesh_mc_m < 0:
            mesh_vox_size_m = config.voxel_size_m*0.6 # use the default value
        if args.mesh_min_nn_k < 0:
            mesh_min_nn_k_used = config.mesh_min_nn # use the default value
        
        neural_pcd = neural_points.get_neural_points_o3d(query_global=args.show_global, color_mode=2, random_down_ratio=down_rate)
        mesh_aabb = neural_pcd.get_axis_aligned_bounding_box()
        chunks_aabb = split_chunks(neural_pcd, mesh_aabb, mesh_vox_size_m*100) 
        print("Number of chunks for reconstruction:", len(chunks_aabb))
        print("Marching cubes resolution: {:.2f} m".format(mesh_vox_size_m))

        out_mesh_path = None
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, mesh_vox_size_m, out_mesh_path, False, False, \
                                                    config.color_on, filter_isolated_mesh=True, mesh_min_nn=mesh_min_nn_k_used)

    # GS visualizer
    # I really don't know why this does not work
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
            gs_default_on=True,
            robot_default_on=False,
            neural_point_default_on=False,
            mesh_default_on=True,
        )

        gui_process = mp.Process(target=slam_gui.run, args=(params_gui,)) # TODO: something is wrong here
        gui_process.start()
        # time.sleep(2) # second

        packet_to_vis: VisPacket = VisPacket(frame_id=center_frame_id, img_down_rate=config.gs_vis_down_rate)
        packet_to_vis.add_neural_points_data(neural_points, only_local_map=(not args.show_global))
        packet_to_vis.add_traj(slam_poses=slam_poses)
        if cur_mesh is not None:
            packet_to_vis.add_mesh(np.array(cur_mesh.vertices, dtype=np.float64), np.array(cur_mesh.triangles), np.array(cur_mesh.vertex_colors, dtype=np.float64))
        
        q_main2vis.put(packet_to_vis)

    while True:
        # print("what's wrong")
        if config.gs_vis_on:
            if not q_vis2main.empty():
                while q_vis2main.get().flag_pause:
                    continue

    # mesh_vox_size_m = None
    # if len(sys.argv) > 2:
    #     mesh_vox_size_m = float(sys.argv[2])
    #     print("Marching cubes resolution: ", mesh_vox_size_m, " m")

    # down_rate = 1

    # crop_file_name = "neural_points.ply" # default name
    # if len(sys.argv) > 3: # only use cropped bbx for meshing
    #     crop_file_name = sys.argv[3]
    #     cropped_ply_path = os.path.join(result_folder, "map", crop_file_name)
    #     cropped_pc = o3d.io.read_point_cloud(cropped_ply_path)
    #     mesh_aabb = cropped_pc.get_axis_aligned_bounding_box()
    #     chunks_aabb = split_chunks(cropped_pc, mesh_aabb, mesh_vox_size_m*300) 
    #     print("Load cropped region")
    # else:
    #     neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=2, random_down_ratio=down_rate)
    #     mesh_aabb = neural_points.get_map_o3d_bbx()
    #     if mesh_vox_size_m is not None:
    #         chunks_aabb = split_chunks(neural_pcd, mesh_aabb, mesh_vox_size_m*300) 
    #     # print("AABB for meshing: ", mesh_aabb)
    
    # print("Number of chunks for reconstruction:", len(chunks_aabb))
    
    # neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=2, random_down_ratio=down_rate)
    # neural_pcd_cropped = neural_pcd.crop(mesh_aabb)
    # cropped_np_out_path = os.path.join(result_folder, "map", "out_ts_" + crop_file_name)
    # o3d.io.write_point_cloud(cropped_np_out_path, neural_pcd_cropped)

    # neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=0, random_down_ratio=down_rate)
    # neural_pcd_cropped = neural_pcd.crop(mesh_aabb)
    # cropped_np_out_path = os.path.join(result_folder, "map", "out_feature_" + crop_file_name)
    # o3d.io.write_point_cloud(cropped_np_out_path, neural_pcd_cropped)

    # print("Neural point count:", neural_points.count())
    # # neural_points_vis_mode = 2

    # if len(sys.argv) > 4:
    #     out_mesh_path = os.path.join(result_folder, "mesh", sys.argv[4])
    #     print("Output the mesh to: ", out_mesh_path)
    # else:
    #     out_mesh_path = None   
    #     print("Do not output mesh")
    
    # mesh_min_nn_used = 9
    # if len(sys.argv) > 5:
    #     mesh_min_nn_used = int(sys.argv[5])

    # cur_mesh = None
    # if mesh_vox_size_m is not None:
    #     cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, mesh_vox_size_m, out_mesh_path, False, config.semantic_on, 
    #                                                  config.color_on, filter_isolated_mesh=True, mesh_min_nn=mesh_min_nn_used)
    
    # if config.o3d_vis_on:
    #     while True:
    #         if vis.render_neural_points:
    #             neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=vis.neural_points_vis_mode, random_down_ratio=down_rate)
    #         vis.update(mesh=cur_mesh, neural_points=neural_pcd)
            
    
if __name__ == "__main__":
    inspect_pings_map()
