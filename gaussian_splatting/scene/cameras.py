#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from gaussian_splatting.utils.graphics_utils import getWorld2View, getWorld2View2, getProjectionMatrix, focal2fov


# this is important
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id # this is not necessary
        # extrinsic
        self.R = R # rotation
        self.T = T # translation

        # you can use this two functions
        # FovY = focal2fov(focal_length_x, height)
        # FovX = focal2fov(focal_length_x, width)

        self.FoVx = FoVx 
        self.FoVy = FoVy
        self.image_name = image_name

        # we may need to handle the sky mask

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # image as 3,H,W
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda() # T_wg
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda() # T_gi
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0) # T_wi
        self.camera_center = self.world_view_transform.inverse()[3, :3]

# used by us
class CamImage:
    def __init__(self, frame_id: int, image, K_mat, z_min, z_max, cam_id: str = "cam", img_down_rate = 0, sky_mask = None, device = "cuda"):
        
        self.frame_id = frame_id
        self.cam_id = cam_id
        self.uid = f"{frame_id:05d}_{cam_id}"

        self.train_view = False # is used as train view or test view

        image[:3] = image[:3].clamp(0.0, 1.0) # only for the RGB part
        self.image_width = image.shape[2]
        self.image_height = image.shape[1]

        self.fx = K_mat[0,0]
        self.fy = K_mat[1,1]

        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)

        # pyramid of images
        self.original_image_list = []

        original_image = image.to(device)

        self.channel_count = original_image.shape[0]
        if self.channel_count == 4:
            self.depth_on = True
        else:
            self.depth_on = False

        # TODO: may add normal
        # TODO: the issue of the depth rendering loss lie in the depth image downsampling, bilinear may not be a good idea, update it 
        
        # C can be either 3 or 4
        # NOTE: F.interpolate require 4D input
        # Downsample to Cx(H/2)x(W/2)
        down_level1_image = F.interpolate(original_image[:3].unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
        # Downsample to Cx(H/4)x(W/4)
        down_level2_image = F.interpolate(down_level1_image[:3].unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
        # Downsample to Cx(H/8)x(W/8)
        down_level3_image = F.interpolate(down_level2_image[:3].unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)

        if self.depth_on:
            down_level1_depth = F.interpolate(original_image[3].unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)
            down_level1_image = torch.cat((down_level1_image, down_level1_depth), dim=0)

            down_level2_depth = F.interpolate(down_level1_image[3].unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)
            down_level2_image = torch.cat((down_level2_image, down_level2_depth), dim=0)

            down_level3_depth = F.interpolate(down_level2_image[3].unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)
            down_level3_image = torch.cat((down_level3_image, down_level3_depth), dim=0)

        # TODO: also downsample sky mask
        self.sky_mask = sky_mask

        if img_down_rate > 0:
            original_image = None
        self.original_image_list.append(original_image)

        if img_down_rate > 1:
            down_level1_image = None
        self.original_image_list.append(down_level1_image)

        if img_down_rate > 2:
            down_level2_image = None
        self.original_image_list.append(down_level2_image)

        self.original_image_list.append(down_level3_image)

        self.zfar = z_max # 100.0
        self.znear = z_min # 0.1

        # GL
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).T # T_gi

# what does this mean?
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
