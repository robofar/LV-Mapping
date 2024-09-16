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

import math
import numpy as np
import torch

# we support multiple GS variants: 3d_gs, 2d_gs, gaussian_surfel
gs_zoo = ["3d_gs", "2d_gs", "gaussian_surfel"]
gs_dim = [3, 2, 3]

gs_type = "gaussian_surfel"
# gs_type = "3d_gs"

# 2DGS
if gs_type == "2d_gs":
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# Gaussian Surfel
elif gs_type == "gaussian_surfel":
    from diff_gaussian_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
elif gs_type == "3d_gs":
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
else:
    print("select from a kind of gs variants")

from model.neural_gaussians import NeuralPoints
from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.utils.point_utils import depth_to_normal, depth2normal
from gaussian_splatting.utils.graphics_utils import getWorld2View
from gaussian_splatting.scene.cameras import CamImage

# the mian gaussain rendering function
def render(viewpoint_camera: CamImage, cam_pose: torch.Tensor,
           neural_gaussians: NeuralPoints, bg_color: torch.Tensor, 
           scaling_modifier = 1.0, override_color = None, 
           down_rate=0, verbose: bool = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    scaling_modifier: You can use the Scaling Modifier to control the size of the displayed Gaussians, or show the initial point cloud. (suggested value, 0.001 to 1.0)
    """
    
    dtype = torch.float32
    device = neural_gaussians.device

    if neural_gaussians.get_local_opacity.shape[0] == 0: # not yet started
        return None

    means3D = neural_gaussians.get_local_xyz.to(dtype)
    opacity = neural_gaussians.get_local_opacity.to(dtype)

    # only use those valid ones
    valid_gs_mask = neural_gaussians.local_valid_gs_mask
    means3D = means3D[valid_gs_mask]
    opacity = opacity[valid_gs_mask]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # here we need to use neural_point coordinate + (optimizable) displacement 
    # better to use only the points in the local map
    screenspace_points = torch.zeros_like(means3D, requires_grad=True, dtype=dtype, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means2D = screenspace_points

    img_scale = 2**down_rate

    # Set up rasterization configuration (scalar value)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # TODO: document this part, figure out why
    if cam_pose is not None:
        cam_pose = cam_pose.to(dtype=dtype, device=device)
        T_cw = torch.linalg.inv(cam_pose)
        cam_world_view_tran = T_cw.T # first inverse, then transpose
        projection_matrix = viewpoint_camera.projection_matrix # P_mat.T
        cam_center = torch.linalg.inv(cam_world_view_tran)[3, :3]
        full_proj_transform = cam_world_view_tran @ projection_matrix 

        viewpoint_camera.world_view_transform = cam_world_view_tran
        viewpoint_camera.full_proj_transform = full_proj_transform
        viewpoint_camera.camera_center = cam_center

        viewpoint_camera.R = T_cw[:3, :3] # rotation part
        viewpoint_camera.T = T_cw[:3, 3] # translation part

    resolution_width = int(viewpoint_camera.image_width/img_scale)
    resolution_height = int(viewpoint_camera.image_height/img_scale)

    # used by gaussian surfels
    patch_size = [float('inf'), float('inf')]
    surface_on = True
    normalize_depth_on = True
    perpix_depth_on = True
    default_on = True
    front_only_on = False # TODO: (false) does not work, but why? # don't cull those gaussians with back normals, optimize all the gaussians in the fov

    gaussian_surfel_train_config = torch.tensor([surface_on, normalize_depth_on, perpix_depth_on, default_on, front_only_on], dtype=dtype, device=device) # surface_on, normalize_depth_on, perpix_depth_on

    # print(resolution_height, resolution_width)

    if gs_type == "2d_gs":
        # 2D GS
        raster_settings = GaussianRasterizationSettings(
            image_height=resolution_height,
            image_width=resolution_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier, # Scaling Modifier to control the size of the displayed Gaussians
            viewmatrix=viewpoint_camera.world_view_transform, # from world frame to camera space 
            projmatrix=viewpoint_camera.full_proj_transform, 
            sh_degree=neural_gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
    elif gs_type == "gaussian_surfel":
        # Gaussian Surfel
        raster_settings = GaussianRasterizationSettings(
            image_height=resolution_height,
            image_width=resolution_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            patch_bbox=viewpoint_camera.random_patch(), # original image size
            prcppoint=viewpoint_camera.prcppoint, # principle point
            sh_degree=neural_gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            config=gaussian_surfel_train_config,
        )
    elif gs_type == "3d_gs":
        # 3D GS
        raster_settings = GaussianRasterizationSettings(
            image_height=resolution_height,
            image_width=resolution_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=neural_gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    compute_cov3D_python = False 

    if compute_cov3D_python: # False now (do not use precomputed covariance)
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = neural_gaussians.get_local_covariance(scaling_modifier)
        # print(splat2world)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:  # this is used now
        scales = neural_gaussians.get_local_scaling.to(dtype) 
        rotations = neural_gaussians.get_local_rotation.to(dtype)

        scales = scales[valid_gs_mask]
        rotations = rotations[valid_gs_mask]

        contains_nan = torch.isnan(rotations).any()
        assert ~contains_nan, "NaN in rotation"

    # if verbose:
    #     print(means3D)
    #     print(scales)
    
    # TODO
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if convert_SHs_python: # what's the difference, python or cuda?
            shs_view = neural_gaussians.get_local_features.transpose(1, 2).view(-1, 3, (neural_gaussians.max_sh_degree+1)**2)
            dir_pp = (neural_gaussians.get_local_xyz - viewpoint_camera.camera_center.repeat(neural_gaussians.local_count(), 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(neural_gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:  # this is used currently
            shs = neural_gaussians.get_local_features.to(dtype)
            shs = shs[valid_gs_mask]
    else:
        colors_precomp = override_color
    
    # main rasterization function
    if gs_type == "2d_gs":
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp # none
        ) 

        rendered_image = torch.nan_to_num(rendered_image, 0, 0)
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rets =  {"render": rendered_image,
                "viewspace_points": means2D,
                "visibility_filter" : radii > 0,
                "radii": radii,
        }

        # additional regularizations
        render_alpha = allmap[1:2]

        # print(render_alpha.shape)

        # print(torch.max(render_alpha)) # not rendering anything

        # get normal map
        # transform normal from view space to world space
        # this is the normal of the gaussian at the rendered surface
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
        # render_normal = render_normal / render_alpha
        render_normal = torch.nan_to_num(render_normal, 0, 0)
        
        # get median depth map # what does this mean? # TODO
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0) # gaussian depth (camera to ray-splat intersection) when aplha (most close to) = 0.5

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha) # alpha blending of the gaussian depth (camera to ray-splat intersection)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map (this is depth distortion instead of depth) (something like distortion?)
        render_dist = allmap[6:7]

        # print(render_dist)

        # pseudo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1; 
        # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk anliasing.

        # what's the depth_ratio? TODO
        # TODO: read the paper again

        depth_ratio = 0 # unbounded scene, in this case, just render_depth_expected
        # depth_ratio = 1 # bounded scene, in this case, just render_depth_median
        surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
        
        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        # surf_normal = surf_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        # surf_normal = surf_normal * (render_alpha).detach()  # pointing toward the surface

        mask_vis = (render_alpha.detach() > 1e-5)
        surf_normal = depth2normal(surf_depth, mask_vis, viewpoint_camera) # normal computed from rendered depth

        # rendered result
        rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist, # distortion
            'surf_depth': surf_depth, # rendered depth
            'surf_normal': surf_normal, # normal calculated from rendered depth
        })

        return rets
    
    elif gs_type == "gaussian_surfel":
        # gaussian surfels
        # Rasterize visible Gaussians to image, obtain their radii (on screen [unit: pixel]). 
        rendered_image, rendered_normal, rendered_depth, rendered_opac, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # rendered_image = torch.nan_to_num(rendered_image, 0, 0)
        # rendered_normal = torch.nan_to_num(rendered_normal, 0, 0)
        # rendered_depth = torch.nan_to_num(rendered_depth, 0, 0)
        # rendered_opac = torch.nan_to_num(rendered_opac, 0, 0)
        # radii = torch.nan_to_num(radii, 0, 0)

        # here the rendered_normal is already normalized?

        # print(viewpoint_camera.world_view_transform)

        # this small part is from 2D GS
        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        
        mask_vis = (rendered_opac.detach() > 1e-5)
        surf_normal = depth2normal(rendered_depth, mask_vis, viewpoint_camera) # pointing inward the surface

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        
        # if verbose:
        #     print(screenspace_points)
        #     print(radii)
        
        return {"render": rendered_image, "rend_normal": rendered_normal, "surf_depth": rendered_depth,
                "rend_alpha": rendered_opac, 'surf_normal': surf_normal, 'rend_dist': None,
                "viewspace_points": screenspace_points, "visibility_filter": radii > 0, "radii": radii} # > 1 or > 0


    elif gs_type == "3d_gs":

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        
        return {"render": rendered_image, "rend_normal": None, "surf_depth": None,
                "rend_alpha": None, 'surf_normal': None, 'rend_dist': None,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}


    return None