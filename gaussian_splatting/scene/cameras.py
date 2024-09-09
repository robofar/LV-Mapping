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

import random

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from gaussian_splatting.utils.graphics_utils import getWorld2View, getWorld2View2, getProjectionMatrix, focal2fov


# used by us
class CamImage:
    def __init__(self, frame_id: int, image, K_mat, z_min=0.01, z_max=100.0,
        cam_id: str = "cam", img_down_rate = 0, normal_img = None, sky_mask = None, 
        device = "cuda", cam_pose = None, img_width = None, img_height = None):
        
        self.frame_id = frame_id
        self.cam_id = cam_id
        self.uid = f"{frame_id:05d}_{cam_id}"

        self.device = device
        self.dtype = torch.float32

        self.train_view = False # is used as train view or test view

        if image is not None:
            image[:3] = image[:3].clamp(0.0, 1.0) # only for the RGB part
            self.image_width = image.shape[2]
            self.image_height = image.shape[1]
        else:
            self.image_width = img_width
            self.image_height = img_height

        self.fx = K_mat[0,0]
        self.fy = K_mat[1,1]
        self.cx = K_mat[0,2]
        self.cy = K_mat[1,2]

        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)

        # principle point (not always at the center) as a ratio, like 0.5, 0.5
        self.prcppoint = torch.tensor([self.cx / self.image_width, self.cy / self.image_height]).to(dtype=self.dtype, device=self.device)

        self.zfar = z_max # 100.0
        self.znear = z_min # 0.1

        # GL
        self.projection_matrix = (getProjectionMatrix(znear=self.znear, zfar=self.zfar,
             fovX=self.FoVx, fovY=self.FoVy,
              W=self.image_width, H=self.image_height, prcp=self.prcppoint).T).to(dtype=self.dtype, device=self.device) # T_gi        
        
        self.world_view_transform = None
        if cam_pose is not None: # we also directly load the camera pose here
            self.world_view_transform = (camera_pose.inverse().T).to(dtype=self.dtype, device=self.device) 
            self.camera_center = self.world_view_transform.inverse()[3, :3]
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix 

        # pyramid of images
        self.original_image_list = []
        self.sky_mask_list = []
        self.normal_img_list = []

        if image is not None:
            original_image = image.to(self.device)

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
                down_level1_depth = F.interpolate(original_image[3].unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
                down_level1_image = torch.cat((down_level1_image, down_level1_depth), dim=0)

                down_level2_depth = F.interpolate(down_level1_image[3].unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
                down_level2_image = torch.cat((down_level2_image, down_level2_depth), dim=0)

                down_level3_depth = F.interpolate(down_level2_image[3].unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
                down_level3_image = torch.cat((down_level3_image, down_level3_depth), dim=0)

            if sky_mask is not None: # sky_mask 1, H, W
                self.sky_mask_on = True
                down_level1_sky_mask = F.interpolate(sky_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
                down_level2_sky_mask = F.interpolate(down_level1_sky_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
                down_level3_sky_mask = F.interpolate(down_level2_sky_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
            else:
                down_level1_sky_mask = down_level2_sky_mask = down_level3_sky_mask = None
                self.sky_mask_on = False

            if normal_img is not None: # normal already in device
                self.mono_normal_on = True
                down_level1_normal = F.interpolate(normal_img.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
                down_level2_normal = F.interpolate(down_level1_normal.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
                down_level3_normal = F.interpolate(down_level2_normal.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            else:
                down_level1_normal = down_level2_normal = down_level3_normal = None
                self.mono_normal_on = False

            if img_down_rate > 0:
                original_image = None
                sky_mask = None
                normal_img = None
            self.original_image_list.append(original_image)
            self.sky_mask_list.append(sky_mask)
            self.normal_img_list.append(normal_img)

            if img_down_rate > 1:
                down_level1_image = None
                down_level1_sky_mask = None
                down_level1_normal = None
            self.original_image_list.append(down_level1_image)
            self.sky_mask_list.append(down_level1_sky_mask)
            self.normal_img_list.append(down_level1_normal)

            if img_down_rate > 2:
                down_level2_image = None
                down_level2_sky_mask = None
                down_level2_normal = None
            self.original_image_list.append(down_level2_image)
            self.sky_mask_list.append(down_level2_sky_mask)
            self.normal_img_list.append(down_level2_normal)

            self.original_image_list.append(down_level3_image)
            self.sky_mask_list.append(down_level3_sky_mask)
            self.normal_img_list.append(down_level3_normal)
    
    def random_patch(self, h_size=float('inf'), w_size=float('inf')):
        # just use part (a random patch) of the image
        h = self.image_height
        w = self.image_width
        h_size = min(h_size, h) # h
        w_size = min(w_size, w) # w
        h0 = random.randint(0, h - h_size) # 0
        w0 = random.randint(0, w - w_size) # 0
        h1 = h0 + h_size
        w1 = w0 + w_size
        return torch.tensor([h0, w0, h1, w1]).to(torch.float32).to(self.device)
    
    # @staticmethod
    # def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
    #     projection_matrix = getProjectionMatrix2(
    #         znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
    #     ).transpose(0, 1)
    #     return Camera(
    #         uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
    #     )



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
