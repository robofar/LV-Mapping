#!/usr/bin/env python3
# @file      inspect_pings.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import argparse
import glob
import os
import sys
import time
import yaml

from typing import Dict, List

import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from rich import print

from dataset.slam_dataset import SLAMDataset, read_kitti_format_poses
from model.decoder import Decoder
from model.neural_gaussians import NeuralPoints
from utils.config import Config
from utils.mesher import Mesher
from utils.tools import setup_experiment, split_chunks, load_decoders, save_video_np, remove_gpu_cache, colorize_depth_maps
from utils.visualizer import MapVisualizer

from gaussian_splatting.scene.cameras import CamImage
from gaussian_splatting.gaussian_renderer import render, spawn_gaussians

from gs_gui import slam_gui
from gs_gui.gui_utils import VisPacket, ParamsGUI

# TODO

'''
    load the pings and pose and render the video 
'''

parser = argparse.ArgumentParser()
parser.add_argument('experiment_path', type=str, help='Path to a certain experiment folder storing the PINGS map')
parser.add_argument('--center_frame_id', '-f', type=int, default=0, help='PINGS local map center frame id')
parser.add_argument('--render_video', '-v', action='store_true', default=False, help='Render video with pre-defined trajectory in the PINGS map')
parser.add_argument('--show_mesh', '-m', action='store_true', default=False, help='Show the PINGS mesh')
parser.add_argument('--show_global', '-g', action='store_true', default=False, help='Show the global map instead of the local map (might cost a lot of memory and not very fast during inferencing)')
parser.add_argument('--mesh_mc_m', type=float, default=-1, help='Marching cubes resolution (in meter) for mesh reconstruction')
parser.add_argument('--mesh_min_nn_k', type=int, default=-1, help='SDF querying min neighbor neural point count for mesh reconstruction')
args, unknown = parser.parse_known_args()

def inspect_pings_map():

    experiment_path = args.experiment_path

    yaml_files = glob.glob(f"{experiment_path}/*.yaml")
    # this might not be a clever way, fix this later (FIXME)
    if len(yaml_files) > 1: # Check if there is exactly one YAML file
        sys.exit("There are multiple YAML files. Please handle accordingly.")
    elif len(yaml_files) == 0:  # If no YAML files are found
        sys.exit("No YAML files found in the specified path.")
    
    config = Config()
    config.load(yaml_files[0])

    model_path = os.path.join(experiment_path, "model", "pin_map.pth")
    pose_path = os.path.join(experiment_path, "slam_poses_kitti.txt")
    full_config_path = os.path.join(experiment_path, "meta", "config_all.yaml")
    config.model_path = model_path
    config.pose_path = pose_path

    if os.path.exists(full_config_path):
        full_config_args = yaml.safe_load(open(full_config_path))
        config.pc_path = full_config_args["pc_path"]
        config.data_loader_name = full_config_args["data_loader_name"]
        config.data_loader_seq = full_config_args["data_loader_seq"]

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
    neural_points.temporal_local_map_on = False

    # load decoders
    load_decoders(loaded_model, mlp_dict) 

    slam_poses = read_kitti_format_poses(config.pose_path)
    frame_count = len(slam_poses)

    # # dataset
    dataset = SLAMDataset(config)
    # print(dataset.cam_names)

    if args.render_video:
        video_folder_path = os.path.join(experiment_path, "video")
        os.makedirs(video_folder_path, 0o755, exist_ok=True)
        render_to_video(video_folder_path, config, dataset, neural_points, mlp_dict, slam_poses, dataset.cam_names)

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
        packet_to_vis.add_traj(slam_poses=np.array(slam_poses))
        if cur_mesh is not None:
            packet_to_vis.add_mesh(np.array(cur_mesh.vertices, dtype=np.float64), np.array(cur_mesh.triangles), np.array(cur_mesh.vertex_colors, dtype=np.float64))
        
        q_main2vis.put(packet_to_vis)

    while True:
        # print("what's wrong")
        if config.gs_vis_on:
            if not q_vis2main.empty():
                while q_vis2main.get().flag_pause:
                    continue


def render_to_video(video_save_base_path: str, config: Config, dataset: SLAMDataset, 
                    neural_points: NeuralPoints, decoders: Dict[str, Decoder], 
                    lidar_poses: Dict[str, np.array], cam_list: List[str],
                    normal_in_world_frame: bool = True):

    # lidar_poses as list of np array

    background = torch.tensor(config.bg_color, dtype=config.dtype, device=config.device)
    bg_3d = background.view(3, 1, 1)

    eval_down_rate = 0

    rendered_rgb_cam_dict = {}
    rendered_depth_cam_dict = {}
    rendered_normal_cam_dict = {}

    # initialize lists
    for cur_cam_name in cam_list: 
        rendered_rgb_cam_dict[cur_cam_name] = []
        rendered_depth_cam_dict[cur_cam_name] = []
        rendered_normal_cam_dict[cur_cam_name] = []

    frame_count = len(lidar_poses)

    frame_begin = 0
    frame_end = frame_count
    frame_step = 1

    for frame_id in tqdm(range(frame_begin, frame_end, frame_step), desc="Render views along the trajectory"):
        remove_gpu_cache()

        T_w_l_np = lidar_poses[frame_id]
        T_w_l = torch.tensor(T_w_l_np, dtype=config.dtype, device=config.device)

        if frame_id % 100 == 0:
            neural_points.recreate_hash(T_w_l[:3,3], kept_points=True, with_ts=False) # and at the same time reset local map
        else:
            neural_points.reset_local_map(T_w_l[:3,3], cur_ts=frame_id)

        neural_points_data, sorrounding_neural_points_data = neural_points.gather_local_data()
    
        sorrounding_spawn_results = spawn_gaussians(sorrounding_neural_points_data, 
                    decoders, None, T_w_l[:3,3],
                    dist_concat_on=config.dist_concat_on, 
                    view_concat_on=config.view_concat_on, 
                    scale_filter_on=True,
                    z_far=config.sorrounding_map_radius,
                    learn_color_residual=config.learn_color_residual)
        
        dataset.read_frame_with_loader(frame_id, init_pose = False, use_image=True, monodepth_on=config.monodepth_on) 

        for cur_cam_name in cam_list: 

            rendered_rgb_list = []
            rendered_depth_list = []
            rendered_normal_list = []

            K_mat = dataset.K_mats[cur_cam_name]

            T_c_l = torch.tensor(dataset.T_c_l_mats[cur_cam_name], dtype=config.dtype, device=config.device) 

            T_w_c = T_w_l @ T_c_l.inverse() # need to convert to cam frame

            # you need to also load the camera exposure coefficients here
            cur_view_cam: CamImage = dataset.cur_cam_img[cur_cam_name]
            cur_view_cam.set_pose(T_w_c)

            # current values
            render_pkg = render(cur_view_cam, None, neural_points_data, 
                decoders, sorrounding_spawn_results, background, 
                down_rate=eval_down_rate, 
                dist_concat_on=config.dist_concat_on, 
                view_concat_on=config.view_concat_on, 
                correct_exposure=config.exposure_correction_on, 
                learn_color_residual=config.learn_color_residual,
                front_only_on=config.train_front_only)
            
            # rendered results
            rendered_rgb_image, rendered_depth, rendered_normal = render_pkg["render"], render_pkg["surf_depth"], render_pkg["rend_normal"] # 3, H, W / 1, H, W
            
            # rgb 
            rendered_rgb_image = torch.clamp(rendered_rgb_image, 0, 1)
            rendered_rgb_np = (rendered_rgb_image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy() # value 0-255
            rendered_rgb_cam_dict[cur_cam_name].append(rendered_rgb_np)

            # depth
            if rendered_depth is not None:
                color_map_used = "inferno_r"
                rendered_depth_np = rendered_depth.detach().cpu().numpy() 
                rendered_depth_color_np = (colorize_depth_maps(rendered_depth_np, 0.1, config.max_range, cmap=color_map_used)[0]*255.0).astype(np.uint8) # 1, 3, H, W 
                rendered_depth_color_np = np.ascontiguousarray(np.transpose(rendered_depth_color_np, (1, 2, 0))) # H, W, 3
                rendered_depth_cam_dict[cur_cam_name].append(rendered_depth_color_np)
            
            if rendered_normal is not None:
                if normal_in_world_frame: 
                    rendered_normal = -1.0 * (rendered_normal.permute(1,2,0) @ (cur_view_cam.world_view_transform[:3,:3].T)).permute(2,0,1)

                normal_norm = rendered_normal.norm(2, dim=0) 
                rendered_normal_show = 0.5 * (normal_norm - rendered_normal) #   # convert to the normal vis color
                # rendered_normal_show = 0.5 * (1 - rendered_normal)
                rendered_normal_np = (rendered_normal_show.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
                rendered_normal_np = np.ascontiguousarray(rendered_normal_np)
                rendered_normal_cam_dict[cur_cam_name].append(rendered_normal_np)


    for cur_cam_name in cam_list: 
        cur_rendered_rgb_list = rendered_rgb_cam_dict[cur_cam_name]
        cur_rendered_depth_list = rendered_depth_cam_dict[cur_cam_name]
        cur_rendered_normal_list = rendered_normal_cam_dict[cur_cam_name]

        cur_rgb_video_save_path = os.path.join(video_save_base_path, "rendered_rgb_{}.mp4".format(cur_cam_name))
        save_video_np(cur_rendered_rgb_list, cur_rgb_video_save_path)

        if len(cur_rendered_depth_list) > 0:
            cur_depth_video_save_path = os.path.join(video_save_base_path, "rendered_depth_{}.mp4".format(cur_cam_name))
            save_video_np(cur_rendered_depth_list, cur_depth_video_save_path)

        if len(cur_rendered_normal_list) > 0:    
            cur_normal_video_save_path = os.path.join(video_save_base_path, "rendered_normal_{}.mp4".format(cur_cam_name))
            save_video_np(cur_rendered_normal_list, cur_normal_video_save_path)

    # dataset.filter_and_correct()

    # if config.deskew and frame_id > 0:
    #     dataset.deskew_at_frame(frame_id)
    
    # dataset.project_pointcloud_to_cams(use_only_colorized_points=True) 

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
