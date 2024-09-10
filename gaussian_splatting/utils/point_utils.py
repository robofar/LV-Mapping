import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

from gaussian_splatting.utils.graphics_utils import fov2focal


# used by 2D GS
def depths_to_points(camera, depth):
    """
        camera: view camera
        depth: depthmap 
    """
    # device = view.device
    c2w = (camera.world_view_transform.T).inverse()
    W, H = depth.shape[2], depth.shape[1]
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ camera.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depth.reshape(-1, 1) * rays_d + rays_o
    return points

# used by 2D GS
def depth_to_normal(camera, depth):
    """
        camera: view camera
        depth: depthmap 
    """
    # print(depth.shape)
    points = depths_to_points(camera, depth).reshape(*depth.shape[1:], 3) # already in world frame
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.linalg.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    # as the gradient of depth 
    # pointing towards the surface
    return output

# used by Gaussian Surfels
def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)
    
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    

    p_c = (p[:, 1:-1, 1:-1, :]      ) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:,  :-2, 1:-1, :] - p_c) * mask[:,  :-2, 1:-1, :]
    p_l = (p[:, 1:-1,  :-2, :] - p_c) * mask[:, 1:-1,  :-2, :]
    p_b = (p[:, 2:  , 1:-1, :] - p_c) * mask[:, 2:  , 1:-1, :]
    p_r = (p[:, 1:-1, 2:  , :] - p_c) * mask[:, 1:-1, 2:  , :]

    n_ul = torch.linalg.cross(p_u, p_l) # changed for torch.cross [FIXME](yue)
    n_ur = torch.linalg.cross(p_r, p_u)
    n_br = torch.linalg.cross(p_b, p_r)
    n_bl = torch.linalg.cross(p_l, p_b)

    # n_ul = torch.nn.functional.normalize(torch.linalg.cross(p_u, p_l), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.linalg.cross(p_r, p_u), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.linalg.cross(p_b, p_r), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.linalg.cross(p_l, p_b), dim=-1)

    # n_ul = torch.nn.functional.normalize(torch.linalg.cross(p_l, p_u), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.linalg.cross(p_u, p_r), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.linalg.cross(p_r, p_b), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.linalg.cross(p_b, p_l), dim=-1)
    
    n = n_ul + n_ur + n_br + n_bl
    n = n[0]
    
    # n *= -torch.sum(camVDir * camN, -1, True).sign() # no cull back

    mask = mask[0, 1:-1, 1:-1, :]

    # n = gaussian_blur(n, filter_size, 1) * mask

    n = torch.nn.functional.normalize(n, dim=-1)
    # n[..., 1] *= -1
    # n *= -1

    n = (n * mask).permute([2, 0, 1])
    return n