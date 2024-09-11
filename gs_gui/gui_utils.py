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

        base_behind = np.array([[0.0, -2.5, -20.0]]) * self.size # original z -30.0
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
def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02): 
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
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
        gaussians=None,
        keyframe=None,
        current_frame=None,
        gtcolor=None,
        gtdepth=None,
        gtnormal=None,
        keyframes=None,
        finish=False,
        kf_window=None,
        current_pointcloud_xyz=None,
        current_pointcloud_rgb=None,
        mesh_verts=None,
        mesh_faces=None,
        mesh_verts_rgb=None,
        odom_poses=None,
        gt_poses=None,
        pgo_poses=None,
        img_down_rate=0,
    ):
        self.has_gaussians = False
        if gaussians is not None:
            self.has_gaussians = True

            self.max_sh_degree = gaussians.max_sh_degree
            self.active_sh_degree = gaussians.active_sh_degree

            self.get_xyz = gaussians.get_xyz.clone()
            self.get_opacity = gaussians.get_opacity.clone()
            self.get_scaling = gaussians.get_scaling.clone()
            self.get_rotation = gaussians.get_rotation.clone()
            self.get_features = gaussians.get_features.clone()

            self.count = gaussians.get_xyz.shape[0]
    
            self.get_local_xyz = gaussians.get_local_xyz.detach().clone()
            self.get_local_opacity = gaussians.get_local_opacity.detach().clone()
            self.get_local_scaling = gaussians.get_local_scaling.detach().clone()
            self.get_local_rotation = gaussians.get_local_rotation.detach().clone()
            self.get_local_features = gaussians.get_local_features.detach().clone()

            self.local_count = self.get_local_xyz.shape[0]

            self.local_valid_gs_mask = gaussians.local_valid_gs_mask

            self._rotation = gaussians.rotation.clone()
            self.rotation_activation = torch.nn.functional.normalize

            if gaussians.unique_kfIDs is None:
                self.unique_kfIDs = torch.ones(gaussians.count()).to(self._rotation)
            else:
                self.unique_kfIDs = gaussians.unique_kfIDs.clone()
            # self.n_obs = gaussians.n_obs.clone()

            self.dtype = gaussians.dtype
            self.device = gaussians.device

        self.keyframe = keyframe
        self.current_frame = current_frame
        if current_frame is not None:
            if current_frame.original_image_list[img_down_rate] is not None:
                cur_gt_img = current_frame.original_image_list[img_down_rate]
                gtcolor = cur_gt_img[:3]
                if current_frame.depth_on:
                    gtdepth = cur_gt_img[3].unsqueeze(0)
                if current_frame.mono_normal_on:
                    gtnormal = current_frame.normal_img_list[img_down_rate]
        
        self.img_resize_width = 480

        self.gtcolor = self.resize_img(gtcolor, self.img_resize_width)
        self.gtdepth = self.resize_img(gtdepth, self.img_resize_width)
        self.gtnormal = self.resize_img(gtnormal, self.img_resize_width)

        self.keyframes = keyframes
        self.finish = finish
        self.kf_window = kf_window

        self.current_pointcloud_xyz = current_pointcloud_xyz
        self.current_pointcloud_rgb = current_pointcloud_rgb

        self.mesh_verts = mesh_verts
        self.mesh_faces = mesh_faces
        self.mesh_verts_rgb = mesh_verts_rgb

        self.odom_poses = odom_poses
        self.gt_poses = gt_poses
        self.pgo_poses = pgo_poses

    def add_scan(self, current_pointcloud_xyz=None, current_pointcloud_rgb=None):
        self.current_pointcloud_xyz = current_pointcloud_xyz
        self.current_pointcloud_rgb = current_pointcloud_rgb
        # TODO: add normal later

    def add_mesh(self, mesh_verts=None, mesh_faces=None, mesh_verts_rgb=None):
        self.mesh_verts = mesh_verts
        self.mesh_faces = mesh_faces
        self.mesh_verts_rgb = mesh_verts_rgb

    # TODO: add loop edges
    def add_traj(self, odom_poses=None, gt_poses=None, pgo_poses=None):
        self.odom_poses = odom_poses
        self.gt_poses = gt_poses
        self.pgo_poses = pgo_poses

    def resize_img(self, img, width):
        if img is None:
            return None

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (width, height))
        # or as torch
        height = int(width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
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


class ParamsGUI:
    def __init__(
        self,
        pipe=None,
        background=None,
        gaussians=None,
        q_main2vis=None,
        q_vis2main=None,
        config=None,
    ):
        self.pipe = pipe
        self.background = background
        self.gaussians = gaussians
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.config = config
