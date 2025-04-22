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
    def __init__(
        self, 
        frame_id, 
        rgb_image, 
        K_mat, 
        z_min=0.1, 
        z_max=100.0,
        cam_id: str = "cam", 
        depth_image = None, 
        normal_img = None, 
        sky_mask = None, 
        foundation_mask = None,
        binary_mask = None,
        device = "cuda", 
        image_device = "cuda",
        cam_pose = None, 
        img_width = None, 
        img_height = None
    ):
        
        self.frame_id = frame_id
        self.cam_id = cam_id
        self.uid = f"{frame_id:05d}_{cam_id}"

        self.device = device
        self.image_device = image_device # device for the image
        self.dtype = torch.float32

        self.train_view: bool = False # is used as train view or test view
        self.in_long_term_memory: bool = False

        if rgb_image is not None:
            rgb_image = rgb_image.clamp(0.0, 1.0) # only for the RGB part
            self.image_width = rgb_image.shape[2]
            self.image_height = rgb_image.shape[1]
        else:
            self.image_width = img_width
            self.image_height = img_height
        
        # if input rgb_image is None, then you need to input valid img_width and height

        # numpy array
        self.K_mat = K_mat
        self.fx = K_mat[0,0]
        self.fy = K_mat[1,1]
        self.cx = K_mat[0,2]
        self.cy = K_mat[1,2]

        self.K_mat_torch = torch.tensor(K_mat, dtype=self.dtype, device=self.device)

        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)

        # principle point (not always at the center) as a ratio, like 0.5, 0.5
        self.prcppoint = torch.tensor([self.cx / self.image_width, self.cy / self.image_height]).to(dtype=self.dtype, device=self.device)

        self.zfar = z_max # 100.0
        self.znear = z_min # 0.1

        # GL
        # OpenGL projection matrix
        self.projection_matrix = (getProjectionMatrix(znear=self.znear, zfar=self.zfar,
             fovX=self.FoVx, fovY=self.FoVy,
              W=self.image_width, H=self.image_height, prcp=self.prcppoint).T).to(dtype=self.dtype, device=self.device) # T_gi        
        
        self.world_view_transform = None # as (T_cw.T)
        self.camera_center = None
        self.full_proj_transform = None 
        
        # set the poses related transformations

        # init value
        self.R = torch.eye(3, dtype=self.dtype, device=self.device)
        # this is not the camera center in world frame, use camera_center instead
        self.T = torch.zeros(3, dtype=self.dtype, device=self.device) 

        self.set_pose(cam_pose)

        # camera pose optimization
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        # exposure correction parameters
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        # exposure correction affine transformation parameters
        self.exposure_mat = nn.Parameter(
            torch.eye(3, requires_grad=True, device=device)
        )
        self.exposure_offset = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        if rgb_image is not None:
            rgb_image = rgb_image.to(self.image_device)
            
            if depth_image is not None: # 1, H, W
                depth_image = depth_image.to(self.image_device)
                self.depth_on = True
            else:
                self.depth_on = False



            if sky_mask is not None: # sky_mask 1, H, W
                sky_mask = sky_mask.to(self.image_device)
                self.sky_mask_on = True
            else:
                self.sky_mask_on = False

            if normal_img is not None: # normal already in device  # sky_mask 3, H, W
                normal_img = normal_img.to(self.image_device)
                self.mono_normal_on = True
            else:
                self.mono_normal_on = False

            if foundation_mask is not None:
                foundation_mask = foundation_mask.to(self.image_device)
                self.foundation_mask_on = True
            else:
                self.foundation_mask_on = False
            
            if binary_mask is not None:
                binary_mask = binary_mask.to(self.image_device)
                self.binary_mask_on = True
            else:
                self.binary_mask_on = False


        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.sky_mask = sky_mask
        self.normal_img = normal_img
        self.foundation_mask = foundation_mask # uint16
        self.binary_mask = binary_mask # bool

            
    
    # deprecated
    def random_patch(self, h_size=float('inf'), w_size=float('inf')):
        # just use part (a random patch) of the image
        h = self.image_height
        w = self.image_width
        h_size = min(h_size, h) # h
        w_size = min(w_size, w) # w
        h0 = random.randint(0, h - h_size) # 0
        w0 = random.randint(0, w - w_size) # 0
        h1 = h0 + h_size - 1 
        w1 = w0 + w_size - 1
        return torch.tensor([h0, w0, h1, w1]).to(dtype=self.dtype, device=self.device)

    # leave this on device, not image_device, because it is used in renderer only
    # before using renderer I will anyways move image to gpu (i.e. device)
    def full_patch(self):
        h1 = int(self.image_height) - 1
        w1 = int(self.image_width) - 1
        return torch.tensor([0, 0, h1, w1]).to(dtype=self.dtype, device=self.device)

    def set_pose(self, cam_pose):
        # input pose is torch tensor, and is T_w_c (c to w)

        if cam_pose is not None: # we also directly load the camera pose here
                    
            T_cw = torch.linalg.inv(cam_pose).to(dtype=self.dtype, device=self.device) 

            self.world_view_transform = (T_cw.T)
            self.camera_center = torch.linalg.inv(self.world_view_transform)[3, :3]
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix 
            
            self.R = T_cw[:3, :3] # rotation part
            self.T = T_cw[:3, 3] # translation part

    def set_exposure_ab(self, exposure_a, exposure_b):
        self.exposure_a = exposure_a
        self.exposure_b = exposure_b

    def set_exposure_affine(self, exposure_mat, exposure_offset):
        self.exposure_mat = exposure_mat
        self.exposure_offset = exposure_offset

    def set_delta_pose(self, delta_r, delta_t):
        self.cam_rot_delta = delta_r
        self.cam_trans_delta = delta_t

    def set_depth_img(self, depth_img_torch):
        if depth_img_torch is not None:  # 1, H, W
            self.depth_on = True
            depth_img_torch = depth_img_torch.to(self.image_device)   
            self.depth_image = depth_img_torch

    def free_memory(self):
        self.rgb_image = None
        self.depth_image = None
        self.normal_img = None
        self.sky_mask = None
        self.foundation_mask = None
        self.binary_mask = None
    

    def move_to_device(self):
        self.rgb_image = self.rgb_image.to(self.device)
        if self.depth_on:
            self.depth_image = self.depth_image.to(self.device)
        if self.foundation_mask_on:
            self.foundation_mask = self.foundation_mask.to(self.device)
        if self.binary_mask_on:
            self.binary_mask = self.binary_mask.to(self.device)


# this is not used
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


# # what does this mean?
# class MiniCam:
#     def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
#         self.image_width = width
#         self.image_height = height    
#         self.FoVy = fovy
#         self.FoVx = fovx
#         self.znear = znear
#         self.zfar = zfar
#         self.world_view_transform = world_view_transform
#         self.full_proj_transform = full_proj_transform
#         view_inv = torch.inverse(self.world_view_transform)
#         self.camera_center = view_inv[3][:3]
