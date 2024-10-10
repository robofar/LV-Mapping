import pathlib
import threading
import time
from datetime import datetime

import cv2
import glfw
import numpy as np
import copy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import torch.nn.functional as F
from OpenGL import GL as gl

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2
from gs_gui.gl_render import util, util_gau
from gs_gui.gl_render.render_ogl import OpenGLRenderer
from gs_gui.gui_utils import (
    VisPacket,
    Packet_vis2main,
    create_frustum,
    cv_gl,
    get_latest_queue,
)
# from utils.camera_utils import Camera
from gaussian_splatting.scene.cameras import CamImage
# from utils.logging_utils import Log

from utils.tools import colorize_depth_maps, setup_seed, get_time, remove_gpu_cache

# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

YELLOW = np.array([1, 0.706, 0])
RED = np.array([255, 0, 0]) / 255.0
PURPLE = np.array([238, 130, 238]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
GOLDEN = np.array([1.0, 0.843, 0.0])
GREEN = np.array([0, 128, 0]) / 255.0
BLUE = np.array([0, 0, 128]) / 255.0
LIGHTBLUE = np.array([0.00, 0.65, 0.93])


class SLAM_GUI:
    def __init__(self, params_gui=None):
        self.step = 0
        self.process_finished = False
        self.device = "cuda"

        self.frustum_dict = {}
        self.model_dict = {}

        self.q_main2vis = None
        self.gaussian_cur = None

        self.decoders = None

        self.background = None
        self.config = None

        self.init = False
        self.kf_window = None
        self.render_img = None

        if params_gui is not None:
            self.decoders = params_gui.decoders
            self.background = params_gui.background
            self.init = True
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main
            self.config = params_gui.config
        
        if self.config is not None:
            setup_seed(self.config.seed)

        self.init_widget()

        self.gaussian_nums = []

        # these are only used for the elliopsoid rendering 
      
        self.g_camera = util.Camera(self.window_h, self.window_w)
        self.window_gl = self.init_glfw() # this has no issue

        # TODO: something wrong here with the glfw (just crash) after I use mini-forge
        # Maybe a pyQT issue
        # exactly this line here
        # self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)  

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.gaussians_gl = util_gau.GaussianData(0, 0, 0, 0, 0)

        # screenshot saving path
        self.save_path = "."
        self.save_path = pathlib.Path(self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        threading.Thread(target=self._update_thread).start()

    # has some issue here
    def init_widget(self):
        # self.window_w, self.window_h = 1600, 900
        self.window_w, self.window_h = 2560, 1600

        self.window = gui.Application.instance.create_window(
           "PINGS Viewer", self.window_w, self.window_h
        ) # open3d gui #FIXME, now this is crashing
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)

        self.widget3d.scene.show_skybox(False)

        self.window.add_child(self.widget3d)

        # not used now
        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"
        self.lit.line_width = 3 * self.window.scaling  # note that this is scaled with respect to pixels,

        self.lit_geo = rendering.MaterialRecord()
        self.lit_geo.shader = "defaultUnlit"

        # scan
        self.scan_render = rendering.MaterialRecord()
        self.scan_render.shader = "defaultLit" # "defaultUnlit", "normals", "depth"
        self.scan_render.point_size = 3 * self.window.scaling
        self.scan_render.base_color = [0.9, 0.9, 0.9, 1.0]

        # neural points
        self.neural_points_render = rendering.MaterialRecord()
        self.neural_points_render.shader = "defaultLit"
        self.neural_points_render.point_size = 3 * self.window.scaling
        self.neural_points_render.base_color = [0.9, 0.9, 0.9, 1.0]

        # sdf slice
        self.sdf_render = rendering.MaterialRecord()
        self.sdf_render.shader = "defaultLit"
        self.sdf_render.point_size = 10 * self.window.scaling
        self.sdf_render.base_color = [1.0, 1.0, 1.0, 1.0]

        # mesh 
        self.mesh_render = rendering.MaterialRecord()
        self.mesh_render.shader = "normals"
        # self.mesh_render.base_color = [0.5, 0.5, 0.5, 0.5]


        # trajectory
        self.traj_render = rendering.MaterialRecord()
        self.traj_render.shader = "unlitLine"
        self.traj_render.line_width = 4 * self.window.scaling  # note that this is scaled with respect to pixels,


        self.cad_render = rendering.MaterialRecord()
        self.cad_render.shader = "defaultLit"
        self.cad_render.base_color = [0.9, 0.9, 0.9, 1.0]

        # how to apply different materials (TODO)
        self.clay_geo = rendering.MaterialRecord()
        self.clay_geo.shader = "defaultLit"

        # self.line_mat = rendering.MaterialRecord()
        # self.line_mat.shader = "unlitLine"
        # self.line_mat.line_width = 5  # note that this is scaled with respect to pixels,

        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        # other geometry entities
        self.mesh = o3d.geometry.TriangleMesh()
        self.scan = o3d.geometry.PointCloud()
        self.sdf_slice = o3d.geometry.PointCloud()
        self.neural_points = o3d.geometry.PointCloud()
        self.sensor_cad = o3d.geometry.TriangleMesh()
        self.sensor_cad_origin = o3d.geometry.TriangleMesh()

        if self.config.sensor_cad_path is not None:
            self.sensor_cad_origin = o3d.io.read_triangle_mesh(self.config.sensor_cad_path)
            self.sensor_cad_origin.compute_vertex_normals()

        self.odom_traj = o3d.geometry.LineSet()
        self.slam_traj = o3d.geometry.LineSet()
        self.gt_traj = o3d.geometry.LineSet()

        self.last_used_pose = np.eye(4)

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        em = self.window.theme.font_size
        margin = 0.5 * em
        
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        
        self.button = gui.ToggleSwitch("Resume / Pause SLAM")
        self.button.is_on = True
        self.button.set_on_clicked(self._on_button)
        self.panel.add_child(self.button)

        self.button_render = gui.ToggleSwitch("Resume / Pause Rendering")
        self.button_render.is_on = True # default off
        # self.button_render.set_on_clicked(self._on_button_render)
        self.panel.add_child(self.button_render)


        self.panel.add_child(gui.Label("Viewpoint Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))
        
        # h = gui.Horiz(0.25 * em, gui.Margins(margin)) 
        # self._arcball_button = gui.Button("Arcball")
        # self._arcball_button.horizontal_padding_em = 0.5
        # self._arcball_button.vertical_padding_em = 0
        # self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)

        # self._fly_button = gui.Button("Fly")
        # self._fly_button.horizontal_padding_em = 0.5
        # self._fly_button.vertical_padding_em = 0
        # self._fly_button.set_on_clicked(self._set_mouse_mode_fly)

        # h.add_child(self._arcball_button)
        # h.add_child(self._fly_button)

        # self.panel.add_child(h)

        ##Check boxes
        vp_subtile1.add_child(gui.Label("Camera follow options"))
        chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        
        self.followcam_chbox = gui.Checkbox("Follow Camera")
        self.followcam_chbox.checked = True
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("From Behind")
        self.staybehind_chbox.checked = True
        chbox_tile.add_child(self.staybehind_chbox)

        self.fly_chbox = gui.Checkbox("Fly Mode")
        self.fly_chbox.checked = False
        self.fly_chbox.set_on_checked(self._set_mouse_mode)
        chbox_tile.add_child(self.fly_chbox)
        
        vp_subtile1.add_child(chbox_tile)

        ##Combo panels
        combo_tile = gui.Vert(0.5 * em, gui.Margins(margin))

        ## Jump to the camera viewpoint
        # self.combo_kf = gui.Combobox()
        # self.combo_kf.set_on_selection_changed(self._on_combo_kf)
        # combo_tile.add_child(gui.Label("Camera list"))
        # combo_tile.add_child(self.combo_kf)
        # vp_subtile2.add_child(combo_tile)

        self.combo_cams = gui.Combobox()
        self.combo_cams.set_on_selection_changed(self._on_combo_cams)
        combo_tile.add_child(gui.Label("Camera list"))
        combo_tile.add_child(self.combo_cams)
        vp_subtile2.add_child(combo_tile)

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        self.panel.add_child(viewpoint_tile)

        self.panel.add_child(gui.Label("3D Objects"))

        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.gs_chbox = gui.Checkbox("GS Rendering")
        self.gs_chbox.checked = False
        # self.gs_chbox.set_on_checked(self._on_gs_chbox)
        chbox_tile_3dobj.add_child(self.gs_chbox)

        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        # disable this for now
        # self.kf_window_chbox = gui.Checkbox("Active window")
        # self.kf_window_chbox.set_on_checked(self._on_kf_window_chbox)
        # chbox_tile_3dobj.add_child(self.kf_window_chbox)

        # disable this for now
        # self.axis_chbox = gui.Checkbox("Axis")
        # self.axis_chbox.checked = False
        # self.axis_chbox.set_on_checked(self._on_axis_chbox)
        # chbox_tile_3dobj.add_child(self.axis_chbox)

        self.mesh_chbox = gui.Checkbox("PIN Mesh")
        self.mesh_chbox.checked = False
        self.mesh_chbox.set_on_checked(self._on_mesh_chbox)
        chbox_tile_3dobj.add_child(self.mesh_chbox)
        self.mesh_name = "pin_mesh"

        self.scan_chbox = gui.Checkbox("Scan")
        self.scan_chbox.checked = True
        self.scan_chbox.set_on_checked(self._on_scan_chbox)
        chbox_tile_3dobj.add_child(self.scan_chbox)
        self.scan_name = "cur_scan"

        # TODO
        self.neural_point_chbox = gui.Checkbox("Neural Points")
        self.neural_point_chbox.checked = False
        self.neural_point_chbox.set_on_checked(self._on_neural_point_chbox)
        chbox_tile_3dobj.add_child(self.neural_point_chbox)
        self.neural_point_name = "neural_points"

        # self.sky_chbox = gui.Checkbox("Sky")
        # self.sky_chbox.checked = False
        # self.sky_chbox.set_on_checked(self._on_sky_chbox)
        # chbox_tile_3dobj.add_child(self.sky_chbox)

        chbox_tile_3dobj_2 = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.sdf_chbox = gui.Checkbox("SDF")
        self.sdf_chbox.checked = False
        self.sdf_chbox.set_on_checked(self._on_sdf_chbox)
        chbox_tile_3dobj_2.add_child(self.sdf_chbox)
        self.sdf_name = "cur_sdf_slice"


        self.cad_chbox = gui.Checkbox("Robot")
        self.cad_chbox.checked = True
        self.cad_chbox.set_on_checked(self._on_cad_chbox)
        chbox_tile_3dobj_2.add_child(self.cad_chbox)
        self.cad_name = "sensor_cad"

        self.gt_traj_chbox = gui.Checkbox("GT Trajectory")
        self.gt_traj_chbox.checked = False
        self.gt_traj_chbox.set_on_checked(self._on_gt_traj_chbox)
        chbox_tile_3dobj_2.add_child(self.gt_traj_chbox)
        self.gt_traj_name = "gt_trajectory"

        self.slam_traj_chbox = gui.Checkbox("SLAM Trajectory")
        self.slam_traj_chbox.checked = False
        self.slam_traj_chbox.set_on_checked(self._on_slam_traj_chbox)
        chbox_tile_3dobj_2.add_child(self.slam_traj_chbox)
        self.slam_traj_name = "slam_trajectory"

        self.panel.add_child(chbox_tile_3dobj)

        self.panel.add_child(chbox_tile_3dobj_2)

        self.panel.add_child(gui.Label("GS Rendering options"))
        chbox_tile_geometry = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.depth_chbox = gui.Checkbox("Depth")
        self.depth_chbox.checked = False
        chbox_tile_geometry.add_child(self.depth_chbox)

        self.normal_chbox = gui.Checkbox("Normal")
        self.normal_chbox.checked = False
        chbox_tile_geometry.add_child(self.normal_chbox)

        self.d2n_chbox = gui.Checkbox("D2N")
        self.d2n_chbox.checked = False
        chbox_tile_geometry.add_child(self.d2n_chbox)

        self.opacity_chbox = gui.Checkbox("Opacity")
        self.opacity_chbox.checked = False
        chbox_tile_geometry.add_child(self.opacity_chbox)

        # self.time_shader_chbox = gui.Checkbox("Time Shader")
        # self.time_shader_chbox.checked = False
        # chbox_tile_geometry.add_child(self.time_shader_chbox)

        self.elliopsoid_chbox = gui.Checkbox("Ellipsoid")
        self.elliopsoid_chbox.checked = False
        chbox_tile_geometry.add_child(self.elliopsoid_chbox)

        self.panel.add_child(chbox_tile_geometry)

        slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        slider_label = gui.Label("Gaussian Scale (0-1)")
        self.scaling_slider = gui.Slider(gui.Slider.DOUBLE)
        # Scaling Modifier to control the size of the displayed Gaussians
        self.scaling_slider.set_limits(0.001, 1.0)
        self.scaling_slider.double_value = 1.0
        slider_tile.add_child(slider_label)
        slider_tile.add_child(self.scaling_slider)
        self.panel.add_child(slider_tile)

        # screenshot buttom
        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        self.panel.add_child(self.screenshot_btn)

        ## Info Tab
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()
        tab_info = gui.Vert(0, tab_margins)

        self.frame_info = gui.Label("Frame: ")
        tab_info.add_child(self.frame_info)

        self.neural_points_info = gui.Label("# Neural points: ")
        tab_info.add_child(self.neural_points_info)

        self.gaussian_info = gui.Label("# Current view Gaussians: ")
        tab_info.add_child(self.gaussian_info)

        self.freq_info = gui.Label("Render FPS: ")
        tab_info.add_child(self.freq_info)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)


        ## Input Image Tab
        tabs2 = gui.TabControl()
        tab_input = gui.Vert(0, tab_margins)
        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()
        self.in_normal_widget = gui.ImageWidget()
        tab_input.add_child(gui.Label("Input Color/Depth/Normal"))
        tab_input.add_child(self.in_rgb_widget)
        tab_input.add_child(self.in_depth_widget)
        tab_input.add_child(self.in_normal_widget)
        tabs2.add_tab("Input", tab_input)
        self.panel.add_child(tabs2)

        self.window.add_child(self.panel)

    # something wrong here
    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        # check by: glxinfo | grep "OpenGL version"

        # set opengl version hint (FIXME)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        ) 

        glfw.make_context_current(window)
        glfw.swap_interval(0)

        if not window:
            glfw.terminate()
            exit(1)
        return window

    def update_activated_renderer_state(self, gaus, rend_mode=-4):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.scaling_slider.double_value)
        self.g_renderer.set_render_mod(rend_mode)
        self.g_renderer.update_camera_pose(self.g_camera)
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)

    def add_camera(self, camera, name, color=[0, 1, 0], gt=False, size=0.01):
        W2C = (
            getWorld2View2(camera.R_gt, camera.T_gt)
            if gt
            else getWorld2View2(camera.R, camera.T)
        )
        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color, size=size)
            self.combo_cams.add_item(name)
            self.frustum_dict[name] = frustum
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.traj_render) # add camera frame to visualizer
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        return frustum

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 0.7
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        self.is_done = True

        print("Received terminate signal")
        # clean up the pipe
        while not self.q_main2vis.empty():
            self.q_main2vis.get()
        while not self.q_vis2main.empty():
            self.q_vis2main.get()
        self.q_vis2main = None
        self.q_main2vis = None
        self.process_finished = True

        return True  # False would cancel the close

    def _on_combo_model(self, new_val, new_idx):
        model_idx = self.model_dict[new_val]
        self.global_map.active_map_idx = model_idx

    def _on_combo_kf(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = frustum.view_dir

        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def _on_combo_cams(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        selected_gtcolor = self.gaussian_cur.gtcolor[new_val]
        selected_gtdepth = self.gaussian_cur.gtdepth[new_val]
        selected_gtnormal = self.gaussian_cur.gtnormal[new_val]

        if selected_gtcolor is not None:
            rgb = torch.clamp(selected_gtcolor, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)

        if selected_gtdepth is not None:
            depth = selected_gtdepth.contiguous().cpu().numpy() 
            depth_color = (colorize_depth_maps(depth, 0.1, self.config.max_range*0.9)*255.0).astype(np.uint8)
            depth_color = np.transpose(depth_color[0], (1, 2, 0))
            depth_color = np.ascontiguousarray(depth_color)
            depth_color_o3d = o3d.geometry.Image(depth_color)
            self.in_depth_widget.update_image(depth_color_o3d)

        if selected_gtnormal is not None:
            normal = selected_gtnormal.contiguous().cpu().numpy() 
            normal_color = 0.5 - normal * 0.5
            normal_color = np.transpose(normal_color, (1, 2, 0))
            normal_color = np.ascontiguousarray(normal_color)
            normal_color_o3d = o3d.geometry.Image(normal_color)
            self.in_normal_widget.update_image(normal_color_o3d)

    # def _on_gs_chbox(self, is_checked, name=None):
    #     names = self.frustum_dict.keys() if name is None else [name]
    #     for name in names:
    #         self.widget3d.scene.show_geometry(name, is_checked)

    def _on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    # def _on_axis_chbox(self, is_checked):
    #     name = "axis"
    #     if is_checked:
    #         self.widget3d.scene.remove_geometry(name)
    #         self.widget3d.scene.add_geometry(name, self.axis, self.lit_geo)
    #     else:
    #         self.widget3d.scene.remove_geometry(name)

    def _on_cad_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.cad_name)
            self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)
        else:
            self.widget3d.scene.remove_geometry(self.cad_name)
    
    def _on_neural_point_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.neural_point_name)
            self.widget3d.scene.add_geometry(self.neural_point_name, self.neural_points, self.neural_points_render) # TODO: add pin-slam mesh
        else:
            self.widget3d.scene.remove_geometry(self.neural_point_name)

    # TODO: rendering shader is not good
    def _on_mesh_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.mesh_name)
            self.widget3d.scene.add_geometry(self.mesh_name, self.mesh, self.mesh_render) # TODO: add pin-slam mesh
        else:
            self.widget3d.scene.remove_geometry(self.mesh_name)

    def _on_scan_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.scan_name)
            self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)
        else:
            self.widget3d.scene.remove_geometry(self.scan_name)

    def _on_sdf_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.sdf_name)
            self.widget3d.scene.add_geometry(self.sdf_name, self.sdf_slice, self.sdf_render)
        else:
            self.widget3d.scene.remove_geometry(self.sdf_name)

    def _on_gt_traj_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.gt_traj_name)
            self.widget3d.scene.add_geometry(self.gt_traj_name, self.gt_traj, self.traj_render)
        else:
            self.widget3d.scene.remove_geometry(self.gt_traj_name)

    def _on_slam_traj_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.slam_traj_name)
            self.widget3d.scene.add_geometry(self.slam_traj_name, self.slam_traj, self.traj_render)
        else:
            self.widget3d.scene.remove_geometry(self.slam_traj_name)

    def _on_sky_chbox(self, is_checked):
        self.widget3d.scene.show_skybox(is_checked)


    def _on_kf_window_chbox(self, is_checked):
        if self.kf_window is None:
            return
        edge_cnt = 0
        for key in self.kf_window.keys():
            for kf_idx in self.kf_window[key]:
                name = "kf_edge_{}".format(edge_cnt)
                edge_cnt += 1
                if "keyframe_{}".format(key) not in self.frustum_dict.keys():
                    continue
                test1 = self.frustum_dict["keyframe_{}".format(key)].view_dir[1]
                kf = self.frustum_dict["keyframe_{}".format(kf_idx)].view_dir[1]
                points = [test1, kf]
                lines = [[0, 1]]
                colors = [[0, 1, 0]] # green camera frame

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                if is_checked:
                    self.widget3d.scene.remove_geometry(name)
                    self.widget3d.scene.add_geometry(name, line_set, self.traj_render)
                else:
                    self.widget3d.scene.remove_geometry(name)

    def _on_button(self, is_on):
        packet = Packet_vis2main()
        packet.flag_pause = not self.button.is_on
        self.q_vis2main.put(packet)

    def _on_slider(self, value):
        packet = self.prepare_viz2main_packet()
        self.q_vis2main.put(packet)

    def _on_render_btn(self):
        packet = Packet_vis2main()
        packet.flag_nextbatch = True
        self.q_vis2main.put(packet)

    def _on_screenshot_btn(self):
        if self.render_img is None:
            return
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.save_path / "screenshots" / dt
        save_dir.mkdir(parents=True, exist_ok=True)
        # create the filename
        filename = save_dir / "screenshot"
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}-gui.png", img)
        img = np.asarray(self.render_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}.png", img)

    def _set_mouse_mode(self, is_on):
        if is_on:
            self.widget3d.set_view_controls(gui.SceneWidget.Controls.FLY)
        else:
            self.widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)

    @staticmethod
    def resize_img(img, width):
        height = int(width * img.shape[0] / img.shape[1])
        return cv2.resize(img, (width, height))

    # disable this now
    # def add_ids(self):
    #     indices = (
    #         torch.unique(self.gaussian_cur.unique_kfIDs).cpu().numpy().astype(int)
    #     ).tolist()
    #     for idx in indices:
    #         if idx in self.gaussian_id_dict.keys():
    #             continue

    #         self.gaussian_id_dict[idx] = 0
    #         self.combo_gaussian_id.add_item(str(idx))

    def receive_data(self, q):
        if q is None:
            return

        # TODO: is this slow?
        gaussian_packet = get_latest_queue(q)

        if gaussian_packet is None:
            return

        self.gaussian_cur = gaussian_packet
        self.init = True

        if gaussian_packet.frame_id is not None:
            self.frame_info.text = "Frame: {}".format(gaussian_packet.frame_id)
                
        if gaussian_packet.has_neural_points:
            self.neural_points_info.text = "# Neural points: {} (local {})  [PINGS Map size: {:.1f} MB]".format(
                gaussian_packet.neural_points_data["count"],
                gaussian_packet.neural_points_data["local_count"],
                gaussian_packet.neural_points_data["map_memory_mb"]
            )
            if self.neural_point_chbox.checked:
                self.neural_points.points = o3d.utility.Vector3dVector(gaussian_packet.neural_points_data["position"].detach().cpu().numpy())
                # self.neural_points.colors = o3d.utility.Vector3dVector(gaussian_packet.neural_points_data["color"].detach().cpu().numpy())
                self.widget3d.scene.remove_geometry(self.neural_point_name)
                self.widget3d.scene.add_geometry(self.neural_point_name, self.neural_points, self.neural_points_render)

            # show feature PCA color (TODO)

        frustum_size = self.config.max_range*0.008

        if gaussian_packet.current_frames is not None and len(gaussian_packet.cam_list)>0: # as Camera class
            
            for cam in gaussian_packet.cam_list:
                frustum = self.add_camera(
                    gaussian_packet.current_frames[cam], name=cam, color=[0, 1, 0], size=frustum_size
                )
            if self.followcam_chbox.checked:
                selected_cam = self.combo_cams.selected_text
                selected_frustum = self.frustum_dict[selected_cam]
                viewpoint = (
                    selected_frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else selected_frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        # not used yet (TODO)
        # if gaussian_packet.keyframe is not None: # as Camera class
        #     name = "keyframe_{}".format(gaussian_packet.keyframe.uid)
        #     frustum = self.add_camera(
        #         gaussian_packet.keyframe, name=name, color=[0, 0, 1], size=frustum_size
        #     )

        # if gaussian_packet.keyframes is not None:
        #     for keyframe in gaussian_packet.keyframes:
        #         name = "keyframe_{}".format(keyframe.uid)
        #         frustum = self.add_camera(keyframe, name=name, color=[0, 0, 1], size=frustum_size)

        # if gaussian_packet.kf_window is not None:
        #     self.kf_window = gaussian_packet.kf_window
        #     self._on_kf_window_chbox(is_checked=self.kf_window_chbox.checked)

        selected_cam = self.combo_cams.selected_text
        selected_frustum = self.frustum_dict[selected_cam]

        selected_gtcolor = self.gaussian_cur.gtcolor[selected_cam]
        selected_gtdepth = self.gaussian_cur.gtdepth[selected_cam]
        selected_gtnormal = self.gaussian_cur.gtnormal[selected_cam]

        if selected_gtcolor is not None:
            rgb = torch.clamp(selected_gtcolor, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)

        if selected_gtdepth is not None:
            depth = selected_gtdepth.contiguous().cpu().numpy() 
            depth_color = (colorize_depth_maps(depth, 0.1, self.config.max_range*0.9)*255.0).astype(np.uint8)
            depth_color = np.transpose(depth_color[0], (1, 2, 0))
            depth_color = np.ascontiguousarray(depth_color)
            depth_color_o3d = o3d.geometry.Image(depth_color)
            self.in_depth_widget.update_image(depth_color_o3d)

        if selected_gtnormal is not None:
            normal = selected_gtnormal.contiguous().cpu().numpy() 
            normal_color = 0.5 - normal * 0.5
            normal_color = np.transpose(normal_color, (1, 2, 0))
            normal_color = np.ascontiguousarray(normal_color)
            normal_color_o3d = o3d.geometry.Image(normal_color)
            self.in_normal_widget.update_image(normal_color_o3d)

        if gaussian_packet.current_pointcloud_xyz is not None:
            self.scan.points = o3d.utility.Vector3dVector(gaussian_packet.current_pointcloud_xyz)
            if gaussian_packet.current_pointcloud_rgb is not None:
                self.scan.colors = o3d.utility.Vector3dVector(gaussian_packet.current_pointcloud_rgb)
            if self.scan_chbox.checked:
                self.widget3d.scene.remove_geometry(self.scan_name)
                self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)

        if gaussian_packet.sdf_slice_xyz is not None:
            if self.sdf_chbox.checked:
                self.sdf_slice.points = o3d.utility.Vector3dVector(gaussian_packet.sdf_slice_xyz)
                if gaussian_packet.sdf_slice_rgb is not None:
                    self.sdf_slice.colors = o3d.utility.Vector3dVector(gaussian_packet.sdf_slice_rgb)

                self.widget3d.scene.remove_geometry(self.sdf_name)
                self.widget3d.scene.add_geometry(self.sdf_name, self.sdf_slice, self.sdf_render)

        if gaussian_packet.mesh_verts is not None and gaussian_packet.mesh_faces is not None:
            self.mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(gaussian_packet.mesh_verts),
                o3d.utility.Vector3iVector(gaussian_packet.mesh_faces),
                )
            if gaussian_packet.mesh_verts_rgb is not None:    
                self.mesh.vertex_colors = o3d.utility.Vector3dVector(gaussian_packet.mesh_verts_rgb)
            self.mesh.compute_vertex_normals()

            if self.mesh_chbox.checked:
                self.widget3d.scene.remove_geometry(self.mesh_name)
                self.widget3d.scene.add_geometry(self.mesh_name, self.mesh, self.mesh_render)

        if gaussian_packet.gt_poses is not None:
            gt_position_np = gaussian_packet.gt_poses[:, :3, 3]
            self.gt_traj.points = o3d.utility.Vector3dVector(gt_position_np)
            gt_edges = np.array([[i, i + 1] for i in range(gt_position_np.shape[0] - 1)])
            self.gt_traj.lines = o3d.utility.Vector2iVector(gt_edges)
            self.gt_traj.paint_uniform_color(BLACK)
            
            if self.gt_traj_chbox.checked:
                self.widget3d.scene.remove_geometry(self.gt_traj_name)
                self.widget3d.scene.add_geometry(self.gt_traj_name, self.gt_traj, self.traj_render)

            if gaussian_packet.slam_poses is None:
                if self.cad_chbox.checked:
                    self.sensor_cad = copy.deepcopy(self.sensor_cad_origin)
                    self.sensor_cad.transform(gaussian_packet.gt_poses[-1])
                    self.widget3d.scene.remove_geometry(self.cad_name)
                    self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)

        if gaussian_packet.slam_poses is not None:
            slam_position_np = gaussian_packet.slam_poses[:, :3, 3]
            self.slam_traj.points = o3d.utility.Vector3dVector(slam_position_np)
            slam_edges = np.array([[i, i + 1] for i in range(slam_position_np.shape[0] - 1)])
            self.slam_traj.lines = o3d.utility.Vector2iVector(slam_edges)
            self.slam_traj.paint_uniform_color(RED)

            if self.slam_traj_chbox.checked:
                self.widget3d.scene.remove_geometry(self.slam_traj_name)
                self.widget3d.scene.add_geometry(self.slam_traj_name, self.slam_traj, self.traj_render)
            
            if self.cad_chbox.checked:
                self.sensor_cad = copy.deepcopy(self.sensor_cad_origin)
                self.sensor_cad.transform(gaussian_packet.slam_poses[-1])
                self.widget3d.scene.remove_geometry(self.cad_name)
                self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)
        
        if gaussian_packet.finish:
            print("Received terminate signal")
            # clean up the pipe
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main = None
            self.q_main2vis = None
            self.process_finished = True

    @staticmethod
    def depth_to_normal(points, k=3, d_min=1e-3, d_max=10.0):
        k = (k - 1) // 2
        # points: (B, 3, H, W)
        b, _, h, w = points.size()
        points_pad = F.pad(
            points, (k, k, k, k), mode="constant", value=0
        )  # (B, 3, k+H+k, k+W+k)
        if d_max is not None:
            valid_pad = (points_pad[:, 2:, :, :] > d_min) & (
                points_pad[:, 2:, :, :] < d_max
            )  # (B, 1, k+H+k, k+W+k)
        else:
            valid_pad = points_pad[:, 2:, :, :] > d_min
        valid_pad = valid_pad.float()

        # vertical vector (top - bottom)
        vec_vert = (
            points_pad[:, :, :h, k : w + k]
            - points_pad[:, :, 2 * k : h + (2 * k), k : w + k]
        )

        # horizontal vector (left - right)
        vec_hori = (
            points_pad[:, :, k : h + k, :w]
            - points_pad[:, :, k : h + k, 2 * k : w + (2 * k)]
        )

        # valid_mask
        valid_mask = (
            valid_pad[:, :, k : h + k, k : w + k]
            * valid_pad[:, :, :h, k : w + k]
            * valid_pad[:, :, 2 * k : h + (2 * k), k : w + k]
            * valid_pad[:, :, k : h + k, :w]
            * valid_pad[:, :, k : h + k, 2 * k : w + (2 * k)]
        )
        valid_mask = valid_mask > 0.5

        # get cross product (B, 3, H, W)
        cross_product = -torch.linalg.cross(vec_vert, vec_hori, dim=1)
        normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)
        return normal, valid_mask

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )

    # done
    def get_current_cam(self):
        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()

        # print(w2c)

        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view()
        hfov_deg = self.vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        H = image_gui.shape[1]
        W = image_gui.shape[2]
        cx = W // 2
        cy = H // 2
        T = torch.from_numpy(w2c)

        K_mat = torch.eye(3)
        K_mat[0,0] = fx
        K_mat[1,1] = fy
        K_mat[0,2] = cx
        K_mat[1,2] = cy

        current_cam = CamImage(-1, None, K_mat, 0.01, 100.0, 
            img_width=W, img_height=H, cam_pose=torch.linalg.inv(T))

        # print(current_cam.camera_center)
                                                        
        return current_cam


    # main rendering function for the 3D visualizer
    def render_o3d_image(self, results, current_cam, normal_in_world_frame: bool = True):

        if not self.gs_chbox.checked:
            return None # don't show gs rendering results

        if self.depth_chbox.checked:
            depth = results["surf_depth"]
            if depth is None:
                return None # don't show gs rendering results

            depth = depth.detach().cpu().numpy()
            # max_depth = np.max(depth)
            depth_color = (colorize_depth_maps(depth, 0.1, self.config.max_range*0.9, cmap="inferno_r")[0]*255.0).astype(np.uint8) # 1, 3, H, W 
            depth_color = np.transpose(depth_color, (1, 2, 0)) # H, W, 3
            depth_color = np.ascontiguousarray(depth_color)
            render_img = o3d.geometry.Image(depth_color)

        elif self.normal_chbox.checked:
            normal = results["rend_normal"]
            if normal is None:
                return None # don't show gs rendering results

            if normal_in_world_frame: 
            # transform to world frame
                normal = -1.0 * (normal.permute(1,2,0) @ (current_cam.world_view_transform[:3,:3].T)).permute(2,0,1)
                
            normal = torch.nn.functional.normalize(normal, dim=0) # normalize to norm==1
            normal_color = 0.5 - normal * 0.5  # convert to the normal vis color
            normal_color = (normal_color.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
            normal_color = np.ascontiguousarray(normal_color)
            render_img = o3d.geometry.Image(normal_color)

        elif self.d2n_chbox.checked:
            d2n = results["surf_normal"]
            if d2n is None:
                return None # don't show gs rendering results
            d2n = torch.nn.functional.normalize(d2n, dim=0) # normalize to norm==1
            d2n_color = 0.5 - d2n * 0.5  # convert to the normal vis color
            d2n_color = (d2n_color.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
            d2n_color = np.ascontiguousarray(d2n_color)
            render_img = o3d.geometry.Image(d2n_color)

        elif self.opacity_chbox.checked:
            
            opacity = results["rend_alpha"]
            
            if opacity is None:
                return None # don't show gs rendering results

            opacity = opacity.detach().cpu().numpy()

            opacity_color = (colorize_depth_maps(opacity, 0.0, 1.0, cmap="jet")[0]*255.0).astype(np.uint8)

            opacity_color = np.transpose(opacity_color, (1, 2, 0)) # H, W, 3
            opacity_color = np.ascontiguousarray(opacity_color)
            
            render_img = o3d.geometry.Image(opacity_color)

        elif self.elliopsoid_chbox.checked: # important

            return None # TODO: currently has some issue

            if self.gaussian_cur is None:
                return
            glfw.poll_events()
            # gl.glClearColor(0, 0, 0, 1.0)
            gl.glClearColor(1.0, 1.0, 1.0, 0.0)
            gl.glClear(
                gl.GL_COLOR_BUFFER_BIT
                | gl.GL_DEPTH_BUFFER_BIT
                | gl.GL_STENCIL_BUFFER_BIT
            )

            w = int(self.window.size.width * self.widget3d_width_ratio)
            glfw.set_window_size(self.window_gl, w, self.window.size.height)
            self.g_camera.fovy = current_cam.FoVy
            self.g_camera.update_resolution(self.window.size.height, w)
            self.g_renderer.set_render_reso(w, self.window.size.height)
            frustum = create_frustum(
                np.linalg.inv(cv_gl @ self.widget3d.scene.camera.get_view_matrix())
            )

            self.g_camera.position = frustum.eye.astype(np.float32)
            self.g_camera.target = frustum.center.astype(np.float32)
            self.g_camera.up = frustum.up.astype(np.float32)

            # here all the gaussians in the global map
            # self.gaussians_gl.xyz = self.gaussian_cur.get_xyz.cpu().numpy()
            # self.gaussians_gl.opacity = self.gaussian_cur.get_opacity.cpu().numpy()
            # self.gaussians_gl.scale = self.gaussian_cur.get_scaling.cpu().numpy()
            # self.gaussians_gl.rot = self.gaussian_cur.get_rotation.cpu().numpy()
            # self.gaussians_gl.sh = self.gaussian_cur.get_features.cpu().numpy()[:, 0, :]

            # local map only
            gaussian_count = self.gaussian_cur.gaussian_xyz.shape[0]

            self.gaussians_gl.xyz = self.gaussian_cur.gaussian_xyz.cpu().numpy()
            self.gaussians_gl.opacity = self.gaussian_cur.gaussian_alpha.cpu().numpy() + 1.0
            self.gaussians_gl.scale = self.gaussian_cur.gaussian_scale.cpu().numpy()
            self.gaussians_gl.rot = self.gaussian_cur.gaussian_rot.cpu().numpy()

            gaussians_gl_rgb = self.gaussian_cur.gaussian_color.cpu().numpy()
            self.gaussians_gl.sh = (gaussians_gl_rgb - 0.5) / 0.28209479177387814 # C0

            # self.gaussians_gl.sh = self.gaussian_cur.gaussian_color.cpu().numpy()[:, 0, :]

            self.update_activated_renderer_state(self.gaussians_gl, -3) # -4 as gauss ball, -3 as flat gauss
            self.g_renderer.sort_and_update(self.g_camera)
            width, height = glfw.get_framebuffer_size(self.window_gl)
            self.g_renderer.draw()
            bufferdata = gl.glReadPixels(
                0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
            )
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
            img = cv2.flip(img, 0)
            render_img = o3d.geometry.Image(img)
            glfw.swap_buffers(self.window_gl)
        else:
            rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )

            # print(rgb)

            render_img = o3d.geometry.Image(rgb)
        return render_img

    def rasterise(self, current_cam):
        
        # TODO: subscribe to current camera, reset local map for rendering
        # Why the memory of the gaussians are not released (you need to remove the cache)

        if self.gaussian_cur is None:
            return None

        if self.gaussian_cur.neural_points_data is None:
            return None

        # print("# Local neural point:", self.neural_points.local_count())

        # print(self.decoders["gauss_xyz"])

        with torch.no_grad():
            # rendering_data = render(current_cam, None, self.gaussian_cur.gaussian_xyz, 
            #     self.gaussian_cur.gaussian_scale, self.gaussian_cur.gaussian_rot, 
            #     self.gaussian_cur.gaussian_alpha, self.gaussian_cur.gaussian_color, 
            #     self.background, scaling_modifier=self.scaling_slider.double_value, 
            #     down_rate=self.gaussian_cur.img_down_rate)

            # TODO: figure out why the GPU memory cannot be released

            render_tic = get_time()

            rendering_data = render(current_cam, None, self.gaussian_cur.neural_points_data, self.decoders, 
                None, self.background, scaling_modifier=self.scaling_slider.double_value, 
                down_rate=self.config.gs_vis_down_rate, 
                dist_concat_on=self.config.dist_concat_on, view_concat_on=self.config.view_concat_on, correct_exposure=False)
            
            render_toc = get_time()

            gaussians_all_count = rendering_data["alpha_all"].shape[0]
            gaussians_valid_count = rendering_data["view_gaussian_count"]
            valid_ratio = 1.0 * gaussians_valid_count / gaussians_all_count
            mean_valid_count = valid_ratio * self.config.spawn_n_gaussian
            cur_view_gaussian_count = rendering_data["visibility_filter"].shape[0]
            self.gaussian_info.text = "# Current view Gaussians: {} (valid: {:.1f} / {})".format(cur_view_gaussian_count, mean_valid_count, self.config.spawn_n_gaussian)

            render_time = render_toc - render_tic # s
            render_freq = 1.0/render_time
            
            self.freq_info.text = "Render FPS: {:.1f}".format(render_freq)
        
        return rendering_data

        # return None

    # main function here for render gs
    def render_gui(self):
        if not self.init:
            return

        current_cam = self.get_current_cam() # TODO, you can also send back it to main

        if not self.gs_chbox.checked:
            if self.render_img is None:
                return
            self.render_img = None
        else: # gs_chbox checked
            results = self.rasterise(current_cam)
            if results is None:
                return
            # print("Results get")
            self.render_img = self.render_o3d_image(results, current_cam)
            results = {} # free memory 
        ## self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)
        self.widget3d.scene.set_background([1, 1, 1, 1], self.render_img)


    # this is used
    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                print("Closing Visualization")
                break

            def update():
                if self.button_render.is_on:
                    # print("UPDATE scene")
                    if self.step % 3 == 0: # 0.03s # 30 Hz
                        # print("UPDATE scene happens")
                        # self.scene_update() # don't do it so frequently
                        self.render_gui()

                    if self.step % 50 == 0: # 0.5s # 2 Hz # receive latest data
                        self.receive_data(self.q_main2vis)

                    if self.step % 100 == 0:
                        remove_gpu_cache() # remove cache regularly

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)

    
    # # not used now
    # def scene_update(self):
    #     self.receive_data(self.q_main2vis)
    #     self.render_gui()


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI(params_gui)
    app.run()


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI()
    app.run()


if __name__ == "__main__":
    main()
