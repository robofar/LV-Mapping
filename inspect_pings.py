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
import vdbfusion

from dataset.slam_dataset import SLAMDataset, read_kitti_format_poses
from model.decoder import Decoder
from model.neural_gaussians import NeuralPoints
from utils.config import Config
from utils.mesher import Mesher, filter_isolated_vertices
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
parser.add_argument('--recon_3d', '-r', action='store_true', default=False, help='Reconstruct 3D by rendering the PINGS map')
parser.add_argument('--show_mesh', '-m', action='store_true', default=False, help='Show the PINGS mesh')
parser.add_argument('--show_global', '-g', action='store_true', default=False, help='Show the global map instead of the local map (might cost a lot of memory and not very fast during inferencing)')
parser.add_argument('--mesh_mc_m', type=float, default=-1, help='Marching cubes resolution (in meter) for mesh reconstruction')
parser.add_argument('--mesh_min_nn_k', type=int, default=-1, help='SDF querying min neighbor neural point count for mesh reconstruction')
parser.add_argument('--sorrounding_map_r_m', type=float, default=-1, help='Radius of the sorrounding map in meter for far-away stuff rendering')
parser.add_argument('--tsdf_fusion_max_range_m', type=float, default=-1, help='Maximum range for doing the TSDF fusion')
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
    if not os.path.exists(pose_path):
        pose_path = os.path.join(experiment_path, "gt_poses_kitti.txt")
    # TODO

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

    # best for now (ipb_car 2050-2150) ./pings_experiments/test_ipbcar_gs_ipb_car__2024-10-30_17-02-47/ 

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

    loaded_model = torch.load(model_path)
    neural_points = loaded_model["neural_points"] # neural_points config are also loaded
    neural_points.temporal_local_map_on = False
    neural_points.compute_feature_principle_components(down_rate=31)

    if args.sorrounding_map_r_m > 0:
        config.sorrounding_map_radius = args.sorrounding_map_r_m
        neural_points.sorrounding_map_radius = args.sorrounding_map_r_m

    # Load the map, decoders are then freezed
    # load decoders
    load_decoders(loaded_model, mlp_dict) 

    # # dataset
    dataset = SLAMDataset(config)
    # print(dataset.cam_names)


    poses_for_render = read_kitti_format_poses(config.pose_path)
    frame_count = len(poses_for_render)
    
    video_folder_path = None
    if args.render_video:
        video_folder_path = os.path.join(experiment_path, "video")
        os.makedirs(video_folder_path, 0o755, exist_ok=True)
    
    mesh_folder_path = None
    if args.recon_3d:
        mesh_folder_path = os.path.join(experiment_path, "mesh")
        os.makedirs(mesh_folder_path, 0o755, exist_ok=True)

    if args.render_video or args.recon_3d:
        render_with_poses(config, dataset, neural_points, mlp_dict, poses_for_render, dataset.cam_names, 
            recon_3d_on=args.recon_3d, 
            video_save_base_path=video_folder_path, 
            mesh_save_base_path=mesh_folder_path,
            eval_down_rate=config.gs_vis_down_rate)
        
    # reset neural points
    center_frame_id = min(center_frame_id, frame_count-1)
    ref_pose = torch.tensor(poses_for_render[center_frame_id], device=config.device, dtype=config.dtype)
    ref_position = ref_pose[:3,3]
    # ref_position = neural_points.neural_points[0]
    
    neural_points.recreate_hash(ref_position, with_ts=False)

    # print(geo_feature_3d.shape)

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
            neural_point_color_default_mode=1, # 0: original rgb, 1: geo feature pca, 2: photo feature pca, 3: time, 4: stability
        )

        gui_process = mp.Process(target=slam_gui.run, args=(params_gui,)) # TODO: something is wrong here
        gui_process.start()
        # time.sleep(2) # second

        packet_to_vis: VisPacket = VisPacket(frame_id=center_frame_id, img_down_rate=config.gs_vis_down_rate)
        packet_to_vis.add_neural_points_data(neural_points, only_local_map=(not args.show_global), pca_color_on=True)
        packet_to_vis.add_traj(slam_poses=np.array(poses_for_render))
        if cur_mesh is not None:
            packet_to_vis.add_mesh(np.array(cur_mesh.vertices, dtype=np.float64), np.array(cur_mesh.triangles), np.array(cur_mesh.vertex_colors, dtype=np.float64))
        
        q_main2vis.put(packet_to_vis)

    while True:
        # print("what's wrong")
        if config.gs_vis_on:
            if not q_vis2main.empty():
                while q_vis2main.get().flag_pause:
                    continue


def render_with_poses(config: Config, dataset: SLAMDataset,
                      neural_points: NeuralPoints, decoders: Dict[str, Decoder], 
                      lidar_poses: Dict[str, np.array], 
                      cam_list: List[str],
                      video_save_base_path: str = None,
                      recon_3d_on: bool = False,
                      mesh_save_base_path: str = None,
                      eval_down_rate: int = 0, 
                      normal_in_world_frame: bool = True,
                      tsdf_fusion_voxel_size: float = None,
                      tsdf_fusion_max_range: float = None,
                      tsdf_fusion_space_carving_on: bool = False):
    
    """
        Inspection of the PINGS map, conduct rendering with given poses
    """

    # lidar_poses as list of np array

    background = torch.tensor(config.bg_color, dtype=config.dtype, device=config.device)
    bg_3d = background.view(3, 1, 1)

    save_video_on = False
    if video_save_base_path is not None:
        save_video_on = True

    # TODO:
    if recon_3d_on:
        if tsdf_fusion_voxel_size is None:
            tsdf_fusion_voxel_size = config.voxel_size_m*0.6 # use the default value
        sdf_trunc = tsdf_fusion_voxel_size * 3.0
        space_carving_on = tsdf_fusion_space_carving_on # False: fast, cannot deal with dynamics, True: slow, can deal with dynamics, may also remove thin objects
        vdb_volume = vdbfusion.VDBVolume(tsdf_fusion_voxel_size,
                                        sdf_trunc,
                                        space_carving_on)


    rendered_rgb_cam_dict = {}
    rendered_depth_cam_dict = {}
    rendered_normal_cam_dict = {}

    intrinsic_o3d_cam_dict = {}
    extrinsic_o3d_cam_dict = {}

    eval_down_scale = 2**(eval_down_rate)

    for cur_cam_name in cam_list: 
        # initialize lists for video
        rendered_rgb_cam_dict[cur_cam_name] = []
        rendered_depth_cam_dict[cur_cam_name] = []
        rendered_normal_cam_dict[cur_cam_name] = []

        # initialize o3d intrinsics and extrinsics

        cur_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()

        cur_K_mat = dataset.K_mats[cur_cam_name]
        # this is for eval_down_rate = 0, if not 1, then you need to change K_mat accordingly
        cur_intrinsic_o3d.set_intrinsics(
                                    height=int(dataset.cam_heights[cur_cam_name]/eval_down_scale),
                                    width=int(dataset.cam_widths[cur_cam_name]/eval_down_scale),
                                    fx=cur_K_mat[0,0]/eval_down_scale,
                                    fy=cur_K_mat[1,1]/eval_down_scale,
                                    cx=cur_K_mat[0,2]/eval_down_scale,
                                    cy=cur_K_mat[1,2]/eval_down_scale)
        
        intrinsic_o3d_cam_dict[cur_cam_name] = cur_intrinsic_o3d
        extrinsic_o3d_cam_dict[cur_cam_name] = dataset.T_c_l_mats[cur_cam_name]

    frame_count = len(lidar_poses)

    # add to input args
    frame_begin = 2400
    frame_end = 3200
    # frame_end = 20
    frame_step = 2

    for frame_id in tqdm(range(frame_begin, frame_end, frame_step), desc="Render views along the trajectory"):
        remove_gpu_cache()

        T_w_l_np = lidar_poses[frame_id]
        T_w_l = torch.tensor(T_w_l_np, dtype=config.dtype, device=config.device)
        
        cur_frame_position_np = T_w_l_np[:3,3]
        cur_frame_position_torch = T_w_l[:3,3]

        if frame_id % 100 == 0:
            neural_points.recreate_hash(cur_frame_position_torch, kept_points=True, with_ts=False) # and at the same time reset local map
        else:
            neural_points.reset_local_map(cur_frame_position_torch, cur_ts=frame_id)

        neural_points_data, sorrounding_neural_points_data = neural_points.gather_local_data()
    
        sorrounding_spawn_results = spawn_gaussians(sorrounding_neural_points_data, 
                    decoders, None, cur_frame_position_torch,
                    dist_concat_on=config.dist_concat_on, 
                    view_concat_on=config.view_concat_on, 
                    scale_filter_on=True,
                    z_far=config.sorrounding_map_radius,
                    learn_color_residual=config.learn_color_residual,
                    gs_type=config.gs_type)
        
        # may not load images
        # print("Begin data loading")
        dataset.read_frame_with_loader(frame_id, init_pose = False, use_image=True, monodepth_on=config.monodepth_on) 
        # print("Data loading done")

        cur_frame_pcd_o3d = o3d.geometry.PointCloud()

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
                front_only_on=config.train_front_only,
                gs_type=config.gs_type)
            
            # rendered results
            rendered_rgb_image, rendered_depth, rendered_normal = render_pkg["render"], render_pkg["surf_depth"], render_pkg["rend_normal"] # 3, H, W / 1, H, W
            
            # rgb 
            rendered_rgb_image = torch.clamp(rendered_rgb_image, 0, 1)
            rendered_rgb_np = (rendered_rgb_image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)  # value 0-255
            if save_video_on:
                rendered_rgb_cam_dict[cur_cam_name].append(rendered_rgb_np)

            # depth
            if rendered_depth is not None:
                color_map_used = "inferno_r"
                rendered_depth_np = rendered_depth.detach().cpu().numpy().astype(np.float32) 
                rendered_depth_color_np = (colorize_depth_maps(rendered_depth_np, 0.1, config.max_range, cmap=color_map_used)[0]*255.0).astype(np.uint8) # 1, 3, H, W 
                rendered_depth_color_np = np.ascontiguousarray(np.transpose(rendered_depth_color_np, (1, 2, 0))) # H, W, 3
                if save_video_on:
                    rendered_depth_cam_dict[cur_cam_name].append(rendered_depth_color_np)
            
            if rendered_normal is not None:
                if normal_in_world_frame: 
                    rendered_normal = -1.0 * (rendered_normal.permute(1,2,0) @ (cur_view_cam.world_view_transform[:3,:3].T)).permute(2,0,1)

                normal_norm = rendered_normal.norm(2, dim=0) 
                rendered_normal_show = 0.5 * (normal_norm - rendered_normal) #   # convert to the normal vis color
                # rendered_normal_show = 0.5 * (1 - rendered_normal)
                rendered_normal_np = (rendered_normal_show.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
                rendered_normal_np = np.ascontiguousarray(rendered_normal_np)
                if save_video_on:
                    rendered_normal_cam_dict[cur_cam_name].append(rendered_normal_np)

            if rendered_depth is not None and recon_3d_on:
                
                if args.tsdf_fusion_max_range_m > 0:
                    depth_trunc = args.tsdf_fusion_max_range_m     
                else:
                    depth_trunc = config.max_range * 0.8 # default value

                rgb_img_o3d = o3d.geometry.Image(rendered_rgb_np)

                rendered_depth_np = np.transpose(rendered_depth_np, (1, 2, 0))

                depth_img_o3d = o3d.geometry.Image(rendered_depth_np)

                cur_rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, 
                                                                        depth_img_o3d, 
                                                                        depth_scale=1.0, 
                                                                        depth_trunc=depth_trunc, 
                                                                        convert_rgb_to_intensity=False)

                cur_cam_pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(
                            cur_rgbd_o3d, 
                            intrinsic_o3d_cam_dict[cur_cam_name], 
                            extrinsic_o3d_cam_dict[cur_cam_name])

                cur_frame_pcd_o3d += cur_cam_pcd_o3d # add visualizer # TODO

        if recon_3d_on:
            print("Begin TSDF fusion")
            cur_frame_pcd_o3d.transform(T_w_l_np) # convert to world frame

            # downsample a bit
            cur_frame_pcd_o3d = cur_frame_pcd_o3d.voxel_down_sample(config.vox_down_m)

            # better to visualize together with the LiDAR point cloud (TODO)
            # o3d.visualization.draw_geometries([cur_frame_pcd_o3d]) 

            # better do the downsampling first (TODO), too time consuming here
            vdb_volume.integrate(np.array(cur_frame_pcd_o3d.points), cur_frame_position_np)
            print("TSDF fusion done")
                

    if recon_3d_on:

        # Extract triangle mesh (numpy arrays)
        vert, tri = vdb_volume.extract_triangle_mesh()

        mesh_tsdf_fusion = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vert),
            o3d.utility.Vector3iVector(tri),
        )

        mesh_tsdf_fusion = filter_isolated_vertices(mesh_tsdf_fusion, config.min_cluster_vertices)

        mesh_tsdf_fusion.compute_vertex_normals()

        # save the mesh
        if mesh_save_base_path is not None:
            mesh_path = os.path.join(mesh_save_base_path, "mesh_tsdf_fusion_{}cm.ply".format(str(round(tsdf_fusion_voxel_size*1e2))))
            o3d.io.write_triangle_mesh(mesh_path, mesh_tsdf_fusion)
            print("Save the mesh results from TSDF fusion to {}".format(mesh_path))

        # free memory
        vdb_volume = None
        mesh_tsdf_fusion = None

    if save_video_on:
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

            # free the lists
            rendered_rgb_cam_dict[cur_cam_name] = []
            rendered_depth_cam_dict[cur_cam_name] = []
            rendered_normal_cam_dict[cur_cam_name] = []

    # CPU memory might not be enough for all of these videos, maybe output it one by one


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
