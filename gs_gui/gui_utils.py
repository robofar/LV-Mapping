import queue

import cv2
import numpy as np
import open3d as o3d
import torch

from gaussian_splatting.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)

cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size # original z -30.0
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up

# camera frustum
def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02, h_w_ratio = 0.5, z_ratio = 1.5): 
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -h_w_ratio, z_ratio],
                [-1.0, -h_w_ratio, z_ratio],
                [1.0, h_w_ratio, z_ratio],
                [-1.0, h_w_ratio, z_ratio],
            ]
        )
        * size # too small
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum

# actually not only gaussians, we also send the cameras, point cloud and mesh data
class VisPacket:
    def __init__(
        self,
        frame_id = None,
        current_frames=None,
        keyframe=None,
        gaussian_xyz=None,
        gaussian_scale=None,
        gaussian_rot=None,
        gaussian_alpha=None,
        gaussian_color=None,
        gtcolor=None,
        gtdepth=None,
        gtnormal=None,
        keyframes=None,
        finish=False,
        kf_window=None,
        current_pointcloud_xyz=None,
        current_pointcloud_rgb=None,
        sdf_slice_xyz=None,
        sdf_slice_rgb=None,
        mesh_verts=None,
        mesh_faces=None,
        mesh_verts_rgb=None,
        odom_poses=None,
        gt_poses=None,
        slam_poses=None,
        local_only=True,
        img_down_rate=0,
    ):
        self.has_gaussians = False
        self.has_neural_points = False

        self.neural_points_data = None

        self.local_gaussian_count = 0

        self.frame_id = frame_id

        if gaussian_xyz is not None:
            self.has_gaussians = True

            self.dtype = gaussian_xyz.dtype
            self.device = gaussian_xyz.device

            # now set to 0 (TODO)
            self.max_sh_degree = 0
            self.active_sh_degree = 0

            self.gaussian_xyz = gaussian_xyz.detach()
            self.gaussian_scale = gaussian_scale.detach()
            self.gaussian_rot = gaussian_rot.detach()
            self.gaussian_alpha = gaussian_alpha.detach()
            self.gaussian_color = gaussian_color.detach()

            self.local_gaussian_count = self.gaussian_xyz.shape[0]

        self.keyframe = keyframe
        self.current_frames = current_frames
        self.cam_list = []

        self.gtcolor = {}
        self.gtdepth = {}
        self.gtnormal = {}

        self.img_resize_width = 600 # resized for vis
        if current_frames is not None:
            self.cam_list = list(current_frames.keys())
            for cam in self.cam_list:
                current_frame = current_frames[cam]
                if current_frame.rgb_image_list[img_down_rate] is not None:
                    gtcolor = current_frame.rgb_image_list[img_down_rate]
                    if current_frame.sky_mask_on:
                        # mask the sky part
                        cur_sky_mask = current_frame.sky_mask_list[img_down_rate] # still torch
                        cur_sky_mask_used = cur_sky_mask.expand(3, -1, -1)
                        gtcolor[cur_sky_mask_used] = 1.0
                    
                    if current_frame.depth_on:
                        gtdepth = current_frame.depth_image_list[img_down_rate]
                    if current_frame.mono_normal_on:
                        gtnormal = current_frame.normal_img_list[img_down_rate]
                        if current_frame.sky_mask_on:
                            # mask the sky part
                            gtnormal[cur_sky_mask_used] = 0.0
            
                gtcolor = self.resize_img(gtcolor)

                # exposure correction for vis
                with torch.no_grad():
                    gtcolor = (gtcolor - current_frame.exposure_b) /  torch.exp(current_frame.exposure_a)

                self.gtcolor[cam] = gtcolor

                self.gtdepth[cam] = self.resize_img(gtdepth, is_sparse=True)

                self.gtnormal[cam] = self.resize_img(gtnormal)

        self.keyframes = keyframes
        self.finish = finish
        self.kf_window = kf_window

        self.current_pointcloud_xyz = current_pointcloud_xyz
        self.current_pointcloud_rgb = current_pointcloud_rgb

        self.sdf_slice_xyz = sdf_slice_xyz
        self.sdf_slice_rgb = sdf_slice_rgb

        self.mesh_verts = mesh_verts
        self.mesh_faces = mesh_faces
        self.mesh_verts_rgb = mesh_verts_rgb

        self.odom_poses = odom_poses
        self.gt_poses = gt_poses
        self.slam_poses = slam_poses

        self.img_down_rate = img_down_rate

    def add_neural_points_data(self, neural_points):
        if neural_points is not None:
            self.has_neural_points = True
            self.neural_points_data = {}
            self.neural_points_data["position"] = neural_points.local_neural_points
            self.neural_points_data["color"] = neural_points.local_point_colors
            self.neural_points_data["geo_feature"] = neural_points.local_geo_features
            self.neural_points_data["color_feature"] = neural_points.local_color_features
            self.neural_points_data["resolution"] = neural_points.resolution
            self.neural_points_data["free_mask"] = neural_points.local_free_gs_mask
            self.neural_points_data["valid_mask"] = neural_points.local_valid_gs_mask
            self.neural_points_data["count"] = neural_points.count()
            self.neural_points_data["valid_count"] = neural_points.count(valid_gs_only=True)
            self.neural_points_data["local_count"] = neural_points.local_count()
            self.neural_points_data["valid_local_count"] = neural_points.local_count(valid_gs_only=True)
            self.neural_points_data["map_memory_mb"] = neural_points.cur_memory_mb

    def add_gaussians(self,  
                    gaussian_xyz=None,
                    gaussian_scale=None,
                    gaussian_rot=None,
                    gaussian_alpha=None,
                    gaussian_color=None):

        if gaussian_xyz is not None:
            self.has_gaussians = True

            self.dtype = gaussian_xyz.dtype
            self.device = gaussian_xyz.device

            # now set to 0 (TODO)
            self.max_sh_degree = 0
            self.active_sh_degree = 0

            self.gaussian_xyz = gaussian_xyz.detach()
            self.gaussian_scale = gaussian_scale.detach()
            self.gaussian_rot = gaussian_rot.detach()
            self.gaussian_alpha = gaussian_alpha.detach()
            self.gaussian_color = gaussian_color.detach()

            self.local_count = self.gaussian_xyz.shape[0]

    def add_scan(self, current_pointcloud_xyz=None, current_pointcloud_rgb=None):
        self.current_pointcloud_xyz = current_pointcloud_xyz
        self.current_pointcloud_rgb = current_pointcloud_rgb
        # TODO: add normal later

    def add_sdf_slice(self, sdf_slice_xyz=None, sdf_slice_rgb=None):
        self.sdf_slice_xyz = sdf_slice_xyz
        self.sdf_slice_rgb = sdf_slice_rgb

    def add_mesh(self, mesh_verts=None, mesh_faces=None, mesh_verts_rgb=None):
        self.mesh_verts = mesh_verts
        self.mesh_faces = mesh_faces
        self.mesh_verts_rgb = mesh_verts_rgb

    # TODO: add loop edges
    def add_traj(self, odom_poses=None, gt_poses=None, slam_poses=None):
        self.odom_poses = odom_poses
        self.gt_poses = gt_poses
        self.slam_poses = slam_poses

    def resize_img(self, img, resize_width = None, is_sparse: bool = False):
        if img is None:
            return None
        
        if resize_width is None:
            return img

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(resize_width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (resize_width, height))
        # or as torch
        resize_height = int(resize_width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        if is_sparse:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(resize_height, resize_width), mode='nearest-exact')
        else:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(resize_height, resize_width), mode="bilinear", align_corners=False)

        return img.squeeze(0)

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(
            self.get_xyz, self.get_scaling, scaling_modifier, self.rotation
        )

    # this is for 3D GS, update the version for 2D GS
    def build_covariance_from_scaling_rotation(
        self, center, scaling, scaling_modifier, rotation
    ): # center not used
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm


def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.qsize() < 1:
                break
    return message


class Packet_vis2main:
    flag_pause = None
    flag_nextbatch = None
    cur_cam = None


class ParamsGUI:
    def __init__(
        self,
        decoders=None,
        background=None,
        q_main2vis=None,
        q_vis2main=None,
        config=None, # PINGS configs
    ):
        self.decoders = decoders # dict of MLPs
        
        self.background = background
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.config = config
