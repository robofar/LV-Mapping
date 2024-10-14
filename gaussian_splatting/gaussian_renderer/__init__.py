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

from typing import Dict

import math
import numpy as np
import torch

# we support multiple GS variants: 3d_gs, 2d_gs, gaussian_surfel
gs_zoo = ["3d_gs", "2d_gs", "gaussian_surfel"]

gs_type = "gaussian_surfel"
# gs_type = "2d_gs"
# gs_type = "3d_gs"

# 2DGS
if gs_type == "2d_gs":
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# Gaussian Surfel
elif gs_type == "gaussian_surfel":
    from diff_gaussian_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# 3DGS
elif gs_type == "3d_gs":
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
else:
    print("select from a kind of gs variants")

from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.utils.point_utils import depth_to_normal, depth2normal
from gaussian_splatting.utils.graphics_utils import getWorld2View
from gaussian_splatting.scene.cameras import CamImage

from model.decoder import Decoder
from model.neural_gaussians import NeuralPoints

from utils.tools import get_time

# the mian gaussain rendering function
def render(viewpoint_camera: CamImage, 
           cam_pose: torch.Tensor,
           neural_points_data: Dict,
           decoders: Dict[str, Decoder],
           gaussians: Dict[str, torch.Tensor], # input already spwaned gaussians 
           bg_color: torch.Tensor, 
           scaling_modifier: float = 1.0, 
           down_rate: int = 0, 
           verbose: bool = False,
           replay_mode: bool = False,
           dist_concat_on: bool = False, 
           view_concat_on: bool = False, 
           alpha_filter_on: bool = True,
           correct_exposure: bool = True,
           learn_color_residual: bool = True):

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    scaling_modifier: You can use the Scaling Modifier to control the size of the displayed Gaussians, or show the initial point cloud. (suggested value, 0.001 to 1.0)
    """
    
    # if neural_points.count() == 0: # not yet started
    #     return None

    dtype = torch.float32
    device = viewpoint_camera.device

    img_scale = 2**down_rate

    active_sh_degree = 0 # we use view dependent color prediction, no SH involved

    # Set up rasterization configuration (scalar value)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    z_far = viewpoint_camera.zfar

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
    surface_on = True # render normal
    normalize_depth_on = True # render normalized depth (with D = D/opacity)
    perpix_depth_on = True
    default_on = True
    front_only_on = False # TODO: (false) does not work, but why? # don't cull those gaussians with back normals, optimize all the gaussians in the fov

    gaussian_surfel_train_config = torch.tensor([surface_on, normalize_depth_on, perpix_depth_on, default_on, front_only_on], dtype=dtype, device=device) # surface_on, normalize_depth_on, perpix_depth_on

    # print(resolution_height, resolution_width)


    # Rasterizer settings
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
            sh_degree=active_sh_degree,
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
            sh_degree=active_sh_degree,
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
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    visible_neural_point_ratio = 0.0

    # Use already predicted gaussians
    if neural_points_data is None and decoders is None and gaussians is not None:
        gaussian_xyz = gaussians["gaussian_xyz"]
        gaussian_scale = gaussians["gaussian_scale"]
        gaussian_rot = gaussians["gaussian_rot"]
        gaussian_alpha = gaussians["gaussian_alpha"]
        gaussian_color = gaussians["gaussian_color"]
        alpha_all = None

    else: 

        # get only the visible local neural points
        visible_neural_point_mask = rasterizer.markVisible(neural_points_data["position"])
        # print(visible_neural_point_mask.shape)
        # print(visible_neural_point_mask.sum().item())

        visible_neural_point_ratio = visible_neural_point_mask.sum() / visible_neural_point_mask.shape[0]

        if visible_neural_point_ratio < 0.05 and replay_mode: # is 0.05 too small?
            print("Too small ratio of visible neural points, skip this frame ")
            return None

        # Spawn Gaussians
        spawn_results = spawn_gaussians(neural_points_data,
            decoders, visible_neural_point_mask, viewpoint_camera.camera_center, 
            dist_concat_on, view_concat_on, 
            alpha_filter_on, z_far, learn_color_residual=learn_color_residual)

        if spawn_results is None: # in the case when there's no visible neural points in current FOV
            return None

        gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, alpha_all, gaussian_free_mask = spawn_results
    

    means3D = gaussian_xyz
    opacity = gaussian_alpha

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # here we need to use neural_point coordinate + (optimizable) displacement 
    # better to use only the points in the local map
    screenspace_points = torch.zeros_like(means3D, requires_grad=True, dtype=dtype, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means2D = screenspace_points

    scales = gaussian_scale
    rotations = gaussian_rot

    gaussian_count = gaussian_xyz.shape[0]

    contains_nan = torch.isnan(rotations).any()
    assert ~contains_nan, "NaN in rotation"

    # shs = gaussian_sh # currently let sh degree as 0

    colors = gaussian_color

    # Spawned gaussians
    # if train_mode:
    results = {
        "gaussian_xyz": gaussian_xyz, 
        "gaussian_scale": gaussian_scale, 
        "gaussian_rot": gaussian_rot, 
        "gaussian_alpha": gaussian_alpha, 
        "gaussian_color": gaussian_color, 
        "alpha_all": alpha_all,
        "gaussian_free_mask": gaussian_free_mask,
        "view_gaussian_count": gaussian_count,
        "visible_neural_point_ratio": visible_neural_point_ratio
    }

    # main rasterization function
    if gs_type == "2d_gs":
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            colors_precomp = colors,
            opacities = opacity,
            scales = scales,
            rotations = rotations
        ) 

        rendered_image = torch.nan_to_num(rendered_image, 0, 0)
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.

        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        # this is the normal of the gaussian at the rendered surface
        render_normal = allmap[2:5] # in camera frame # normalized render normal, the same as gaussian surfel
        # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1) # to world frame
        # render_normal = render_normal / render_alpha
        render_normal = torch.nan_to_num(render_normal, 0, 0)
        
        # get median depth map # what does this mean? # TODO
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0) # gaussian depth (camera to ray-splat intersection) when aplha (most close to) = 0.5

        # get expected depth map
        render_depth_expected = allmap[0:1] # this is normalized depth
        
        render_depth_expected = render_depth_expected / render_alpha # alpha blending of the gaussian depth (camera to ray-splat intersection), this is denormalized
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

        depth_ratio = 0 # unbounded scene, in this case, just render_depth_expected (mean)
        # depth_ratio = 1 # bounded scene, in this case, just render_depth_median
        surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
        
        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        # surf_normal = surf_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        # surf_normal = surf_normal * (render_alpha).detach()  # pointing toward the surface

        mask_vis = (render_alpha.detach() > 1e-3)
        d2n = depth2normal(surf_depth, mask_vis, viewpoint_camera) # normal computed from rendered depth # in camera frame

        # d2n = depth_to_normal(viewpoint_camera, surf_depth) # in world frame

        # d2n = None 

        # rendered result
        results.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist, # distortion
            'surf_depth': surf_depth, # rendered depth
            'surf_normal': d2n, # normal calemoculated from rendered depth # in cam frame
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii
        })

    
    elif gs_type == "gaussian_surfel":
        # gaussian surfels
        # Rasterize visible Gaussians to image, obtain their radii (on screen [unit: pixel]). 
        # rendered color, depth and normal are all calculated by alpha blending
        # depth and normal are normalized by rendered opacity
        rendered_image, rendered_normal, rendered_depth, rendered_alpha, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            colors_precomp = colors,
            opacities = opacity,
            scales = scales,
            rotations = rotations)
        
        # rendered_depth is normalized by (alpha_blending depth / rendered opacity 1-T)
        # but rendered_normal is not normalized ?
        
        # rendered_normal = rendered_normal / rendered_alpha # don't do this, we can just use the unnormalized version

        rendered_normal_norm = rendered_normal.norm(2, dim=0)  # 3, H, W

        # print(rendered_normal_norm) # why is this not 1?, how is it calculated

        # print(rendered_alpha)

        # d2n = None

        # tic_d2n = get_time()
        mask_vis = (rendered_alpha.detach() > 1e-3)
        d2n = depth2normal(rendered_depth, mask_vis, viewpoint_camera) # pointing inward the surface # in camera frame
        
        # # d2n = depth_to_normal(viewpoint_camera, rendered_depth) # in world frame
        # toc_d2n = get_time()
        # print("D2N time:", (toc_d2n-tic_d2n) * 1000) # could be more than 1ms, disable for now

        results.update({
            "rend_normal": rendered_normal, # in cam frame
            "surf_depth": rendered_depth,
            "rend_alpha": rendered_alpha,
            'surf_normal': d2n, # in cam frame
            'rend_dist': None,
            "viewspace_points": screenspace_points, 
            "visibility_filter": radii > 0, 
            "radii": radii}) # > 1 or > 0


    elif gs_type == "3d_gs":

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            colors_precomp = colors,
            opacities = opacity,
            scales = scales,
            rotations = rotations)
        
        results.update({
            "rend_normal": None, 
            "surf_depth": depth,
            "rend_alpha": None, 
            'surf_normal': None, 
            'rend_dist': None,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii})        

    if correct_exposure:
        rendered_image = (torch.exp(viewpoint_camera.exposure_a)) * rendered_image + viewpoint_camera.exposure_b # apply this for now
        # but when evaluating, how to set the values for these parameters

    results.update({"render": rendered_image})

    return results


def spawn_gaussians(neural_points_data: Dict,
                    decoders: Dict[str, Decoder],
                    visble_mask: torch.tensor = None,
                    cam_origin: torch.tensor = None, 
                    dist_concat_on: bool = False, 
                    view_concat_on: bool = False,
                    alpha_filter_on: bool = True,
                    z_far: float = 100.0,
                    dist_adaptive_scale: bool = False,
                    learn_color_residual: bool = True):

    neural_point_position = neural_points_data["position"]
    neural_point_color = neural_points_data["color"]
    neural_point_geo_features = neural_points_data["geo_feature"]
    neural_point_color_features = neural_points_data["color_feature"]
    neural_point_resolution = neural_points_data["resolution"]
    
    neural_point_free_mask = None
    gaussian_free_mask = None
    if "free_mask" in list(neural_points_data.keys()):
        neural_point_free_mask = neural_points_data["free_mask"]

    neural_point_valid_mask = None
    if "valid_mask" in list(neural_points_data.keys()):
        neural_point_valid_mask = neural_points_data["valid_mask"]


    if visble_mask is not None:
        if neural_point_valid_mask is not None:
            visble_mask = visble_mask & neural_point_valid_mask # only visble and also valid neural points will be used 

        neural_point_position = neural_point_position[visble_mask]
        neural_point_color = neural_point_color[visble_mask]

        if neural_point_free_mask is not None:
            neural_point_free_mask = neural_point_free_mask[visble_mask]

        visible_idx = torch.nonzero(visble_mask)
        visible_idx = torch.cat((visible_idx.view(-1), torch.tensor([-1]).to(visible_idx)))

        # print(" Current Visible neural point count: {:d}".format(torch.sum(visble_mask).item()))

        neural_point_geo_features = neural_point_geo_features[visible_idx]
        neural_point_color_features = neural_point_color_features[visible_idx]

    visible_neural_point_count = neural_point_position.shape[0]
    if visible_neural_point_count < 10: 
        return None

    # if too much neural points, you'd better to feed them to networks in batch
    gaussian_xyz_mlp = decoders["gauss_xyz"] 
    gaussian_scale_mlp = decoders["gauss_scale"] 
    gaussian_rot_mlp = decoders["gauss_rot"] 
    gaussian_alpha_mlp = decoders["gauss_alpha"] 
    gaussian_color_mlp = decoders["gauss_color"] 

    # sdf_mlp = decoders["sdf"] #not used yet

    # after visible filtering    
    neural_point_count = neural_point_position.shape[0]

    view_direction = None
    view_distance = None
    if cam_origin is not None:
        view_direction = neural_point_position - cam_origin # N, 3
        view_distance = view_direction.norm(dim=1, keepdim=True) # N, 1
        # normalize
        view_direction = view_direction / view_distance

    geo_feature_in = neural_point_geo_features[:-1]

    # ------------------
    # Position (view independent)

    displacement_range = 1.0 * neural_point_resolution * torch.ones((neural_point_count, 1)).to(neural_point_position) # 1.0 might be too small maybe
    if neural_point_free_mask is not None:
        displacement_range[neural_point_free_mask] = 5.0 * neural_point_resolution

    # test this scale here, better to not be too large (FIXME)
    xyz_displacement = displacement_range * torch.tanh(gaussian_xyz_mlp.mlp_batch(geo_feature_in)) # N, 3K # [-1,1]        
    # print(xyz_displacement)

    local_point_count = xyz_displacement.shape[0] # N
    gaussian_count_per_point = gaussian_xyz_mlp.out_k # K
    local_gaussian_count = local_point_count * gaussian_count_per_point

    gaussian_xyz = neural_point_position.repeat(1, gaussian_count_per_point) + xyz_displacement # N, 3K
    gaussian_xyz = gaussian_xyz.view(local_gaussian_count, -1) # NK, 3

    # ------------------
    # Rotation (view independent)
    gaussian_rot = gaussian_rot_mlp.mlp_batch(geo_feature_in) # N, 4K
    gaussian_rot = gaussian_rot.view(local_gaussian_count, -1) # NK , 4
    gaussian_rot = torch.nn.functional.normalize(gaussian_rot) # normalize (after activation)
    gaussian_rot = torch.nan_to_num(gaussian_rot, 0, 0)


    # ------------------
    # Scale (view dependent or not) ? # TODO
    max_gaussian_scale = 2.0 * neural_point_resolution
    dist_ratio = 0.0
    if view_distance is not None and dist_adaptive_scale:
        dist_ratio = view_distance / z_far # N, 1
        dist_ratio = dist_ratio.repeat(1, gaussian_scale_mlp.mlp_out_dim)

    gaussian_scale = 0.5 * neural_point_resolution * torch.exp(gaussian_scale_mlp.mlp_batch(geo_feature_in) + dist_ratio) # N, 2K
    gaussian_scale = torch.clamp(gaussian_scale, max=max_gaussian_scale)
    # FIXME
    # what should be the maximum size here? $ TODO
    # gaussian_scale = max_gaussian_scale * torch.sigmoid(gaussian_scale_mlp.mlp_batch(geo_feature_in)) # N, 2K
    
    gaussian_scale = gaussian_scale.view(local_gaussian_count, -1) # NK, 3 (2) # positive (after activation)

    if gs_type == "gaussian_surfel":
        gaussian_scale = gaussian_scale[:,:2] # NK, 2 #
        thin_dim_scale = torch.full((local_gaussian_count, 1), 1e-7).to(gaussian_scale) # already after activation, last dim, very thin
        gaussian_scale = torch.cat((gaussian_scale, thin_dim_scale), dim=1) # NK, 3

    elif gs_type == "2d_gs": # support only 2 dim
        gaussian_scale = gaussian_scale[:,:2] # NK, 2 #

    # else: # 3dgs use this 3dim version
    
    if dist_concat_on and view_distance is not None:
        # print(view_distance)
        geo_feature_in = torch.concat((geo_feature_in, view_distance), dim=1)

    # ------------------
    # Opacity (view dependent)

    # gaussian_alpha = torch.sigmoid(gaussian_alpha_mlp.mlp_batch(geo_feature_in) 
    gaussian_alpha = torch.tanh(gaussian_alpha_mlp.mlp_batch(geo_feature_in)) 
    # gaussian_alpha = 0.9 + 0.1 * torch.sigmoid(gaussian_alpha_mlp.mlp_batch(geo_feature_in)) 
    # gaussian_alpha = 0.5-0.5*torch.tanh(gaussian_alpha_mlp.mlp_batch(geo_feature_in)) # N, K  #[-1,1] --> [0,1]

    gaussian_alpha = gaussian_alpha.view(local_gaussian_count, -1) # NK, 1 # [0-1] (after activation)
    
    # print("mean opacity:", gaussian_alpha.mean().item()) # the opacity is too low, may have some problem, better to have either 0 or 1 opacity

    # ------------------
    # Color (view dependent)

    color_feature_in = neural_point_color_features[:-1]
    if view_concat_on and view_direction is not None:
        color_feature_in = torch.concat((color_feature_in, view_direction), dim=1) # no high freq positional embedding yet
    
    # try to now use only one single feature vector
    
    ## learn residual now
    # TODO: compare, but it seems that there's no much difference
    if learn_color_residual:
        # by doing so, we can somehow restrict the color to not diverge much from the initial guess, so that the view-dependent color would not give very random results
        residual_range = 0.1
        gaussian_rgb_residual = residual_range * torch.tanh(gaussian_color_mlp.mlp_batch(color_feature_in)) # N, 3K [-residual_range, residual_range]
        # gaussian_rgb_residual = gaussian_color_mlp.mlp_batch(color_feature_in) # N, 3K # better restrict this to a very samll value
        gaussian_color = neural_point_color.repeat(1, gaussian_count_per_point) + gaussian_rgb_residual # N, 3K
        gaussian_color = torch.clamp(gaussian_color, 0.0, 1.0)
    else: 
        # or we directly learn the color value (instead of residual)
        gaussian_color = torch.sigmoid(gaussian_color_mlp.mlp_batch(color_feature_in)) # N, 3K

    gaussian_color = gaussian_color.view(local_gaussian_count, -1) # NK, 3 # not SH anymore

    # gaussian_rgb_base = neural_point_color.repeat(1, gaussian_count_per_point)
    # gaussian_color = gaussian_color.view(local_gaussian_count, 1, -1) # NK, 1, 3
    # gaussian_sh = RGB2SH(gaussian_rgb_base)

    # ------------------
    # Mask

    alpha_all = gaussian_alpha.clone()

    if neural_point_free_mask is not None:
        gaussian_free_mask = neural_point_free_mask.repeat(1, gaussian_count_per_point).view(-1)

    # alpha threshold # but this cannot let the gradients to backpropagate (FIXME) # what's the better way to set an self-adpative mask
    if alpha_filter_on:
        before_size = gaussian_alpha.shape[0]

        alpha_thre = 0.0 # tanh [-1,1]
        alpha_mask_idx = torch.nonzero(gaussian_alpha.squeeze(-1) > alpha_thre).view(-1)

        gaussian_xyz = gaussian_xyz[alpha_mask_idx]
        gaussian_scale = gaussian_scale[alpha_mask_idx]
        gaussian_rot = gaussian_rot[alpha_mask_idx]
        gaussian_alpha = gaussian_alpha[alpha_mask_idx]
        gaussian_color = gaussian_color[alpha_mask_idx]

        after_shape = gaussian_alpha.shape[0]

        if gaussian_free_mask is not None:
            gaussian_free_mask = gaussian_free_mask[alpha_mask_idx]

        # print("Gaussian count:", before_size, "-->", after_shape) # it's downsampled a bit too much, shall we have some inductive bias

    # also consider the entropy loss, let the opacity to be either 0 or 1

    return gaussian_xyz, gaussian_scale, gaussian_rot, gaussian_alpha, gaussian_color, alpha_all, gaussian_free_mask
