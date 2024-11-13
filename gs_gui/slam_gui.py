import pathlib
import threading
import time
from datetime import datetime

import cv2
import os
import glfw
import matplotlib.cm as cm
import numpy as np
import copy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import torch.nn.functional as F
from OpenGL import GL as gl
from brisque import BRISQUE

from pickle import load, dump

# import pycg # TODO

from gaussian_splatting.gaussian_renderer import render, spawn_gaussians
from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2
from gaussian_splatting.utils.image_utils import psnr
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

from utils.tools import colorize_depth_maps, seed_anything, get_time, remove_gpu_cache

# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

YELLOW = np.array([1, 0.706, 0])
RED = np.array([255, 0, 0]) / 255.0
PURPLE = np.array([238, 130, 238]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
GOLDEN = np.array([1.0, 0.843, 0.0])
GREEN = np.array([0, 128, 0]) / 255.0
BLUE = np.array([0, 0, 128]) / 255.0
LIGHTBLUE = np.array([0.00, 0.65, 0.93])

ToGLCamera = np.array([
    [1,  0,  0,  0],
    [0,  -1,  0,  0],
    [0,  0,  -1,  0],
    [0,  0,  0,  1]
])
FromGLGamera = np.linalg.inv(ToGLCamera)


os.environ["PYOPENGL_PLATFORM"] = "osmesa"

class SLAM_GUI:
    def __init__(self, params_gui=None):
        self.step = 0
        self.process_finished = False
        self.device = "cuda"

        self.frustum_dict = {}
        self.keyframe_dict = {}
        self.model_dict = {}

        self.q_main2vis = None
        self.gaussian_cur = None

        self.cur_base_gaussians = None # Dict: these are background gaussians stored in the visualizer

        self.decoders = None

        self.background = None
        self.config = None

        self.init = False
        self.kf_window = None
        self.render_img = None

        if params_gui is not None:
            self.decoders = params_gui.decoders
            self.background = params_gui.background
            # self.init = False
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main
            self.config = params_gui.config
            self.gs_default_on = params_gui.gs_default_on
            self.robot_default_on = params_gui.robot_default_on
            self.neural_point_default_on = params_gui.neural_point_default_on
            self.mesh_default_on = params_gui.mesh_default_on
            self.neural_point_color_default_mode = params_gui.neural_point_color_default_mode
        
        if self.config is not None:
            seed_anything(self.config.seed)

        self.init_widget()

        self.cur_frame_id = -1

        self.gaussian_nums = []

        self.brisque_scorer = BRISQUE(url=False)

        self.view_save_base_path = os.path.expanduser("~/.viewpoints")
        os.makedirs(self.view_save_base_path, 0o755, exist_ok=True)

        # these are only used for the elliopsoid rendering 
      
        self.g_camera = util.Camera(self.window_h, self.window_w)
        self.window_gl = self.init_glfw() # this has no issue

        # TODO: something wrong here with the glfw (just crash) after I use mini-forge
        # exactly this line here

        # solution:
        # os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        # or set in your conda environment
        # export PYOPENGL_PLATFORM=osmesa
        # reference: 
        # https://github.com/facebookresearch/AnimatedDrawings/issues/99

        self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)  

        # gl.glEnable(gl.GL_TEXTURE_2D)
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
           "PINGS Viewer for {}".format(self.config.name), self.window_w, self.window_h
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
        # self.widget3d.scene.show_skybox(False)

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
        self.scan_render.base_color = [0.9, 0.9, 0.9, 0.8]

        # neural points
        self.neural_points_render = rendering.MaterialRecord()
        self.neural_points_render.shader = "defaultLit"
        self.neural_points_render.point_size = 3 * self.window.scaling
        self.neural_points_render.base_color = [0.9, 0.9, 0.9, 0.8]

        # sdf slice
        self.sdf_render = rendering.MaterialRecord()
        self.sdf_render.shader = "defaultLit"
        self.sdf_render.point_size = 10 * self.window.scaling
        self.sdf_render.base_color = [1.0, 1.0, 1.0, 1.0]

        # sdf sample pool
        self.sdf_pool_render = rendering.MaterialRecord()
        self.sdf_pool_render.shader = "defaultLit"
        self.sdf_pool_render.point_size = 1 * self.window.scaling
        self.sdf_pool_render.base_color = [1.0, 1.0, 1.0, 1.0]

        # mesh 
        self.mesh_render = rendering.MaterialRecord()
        if self.mesh_default_on:
            self.mesh_render.shader = "defaultLit"
        else:
            self.mesh_render.shader = "normals" 
        
        # self.mesh_render.base_color = [0.5, 0.5, 0.5, 0.5]

        # trajectory
        self.traj_render = rendering.MaterialRecord()
        self.traj_render.shader = "unlitLine"
        self.traj_render.line_width = 4 * self.window.scaling  # note that this is scaled with respect to pixels,

        # cur frame frustrum
        self.cur_frame_render = rendering.MaterialRecord()
        self.cur_frame_render.shader = "unlitLine"
        self.cur_frame_render.line_width = 4 * self.window.scaling

        # train frame frustrum
        self.train_frame_render = rendering.MaterialRecord()
        self.train_frame_render.shader = "unlitLine"
        self.train_frame_render.line_width = 2 * self.window.scaling

        # range ring
        self.ring_render = rendering.MaterialRecord()
        self.ring_render.shader = "unlitLine"
        self.ring_render.line_width = 2 * self.window.scaling  # note that this is scaled with respect to pixels,

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
        self.rendered_scan = o3d.geometry.PointCloud()
        self.sdf_pool = o3d.geometry.PointCloud() # sample pool
        self.sdf_slice = o3d.geometry.PointCloud()
        self.neural_points = o3d.geometry.PointCloud()
        self.invalid_neural_points = o3d.geometry.PointCloud()
        self.sensor_cad = o3d.geometry.TriangleMesh()
        self.sensor_cad_origin = o3d.geometry.TriangleMesh()

        if self.config.sensor_cad_path is not None:
            self.sensor_cad_origin = o3d.io.read_triangle_mesh(self.config.sensor_cad_path)
            self.sensor_cad_origin.compute_vertex_normals()

        self.odom_traj = o3d.geometry.LineSet()
        self.slam_traj = o3d.geometry.LineSet()
        self.gt_traj = o3d.geometry.LineSet()

        # range circles
        self.range_circle = o3d.geometry.LineSet()
        circle_points_1 = generate_circle(radius=self.config.max_range/2, num_points=100)
        lines1 = [[i, (i + 1) % len(circle_points_1)] for i in range(len(circle_points_1))]
        range_circle1 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(circle_points_1),
            lines=o3d.utility.Vector2iVector(lines1),
        )
        circle_points_2 = generate_circle(radius=self.config.max_range, num_points=100)
        lines2 = [[i, (i + 1) % len(circle_points_2)] for i in range(len(circle_points_2))]
        range_circle2 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(circle_points_2),
            lines=o3d.utility.Vector2iVector(lines2),
        )
        self.range_circle_origin = range_circle1 + range_circle2
        self.range_circle_origin.paint_uniform_color(LIGHTBLUE)

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        
        em = self.window.theme.font_size
        margin = 0.8 * em
        
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        # tabs.add_tab("Setting", tab_info) # FIXME

        slider_line = gui.Horiz(1.0 * em, gui.Margins(margin))
        
        # these are not button, but rather switch
        self.slider_slam = gui.ToggleSwitch("Resume / Pause SLAM")
        self.slider_slam.is_on = True
        self.slider_slam.set_on_clicked(self._on_slam_slider)
        slider_line.add_child(self.slider_slam)

        self.slider_render = gui.ToggleSwitch("Resume / Pause Rendering")
        self.slider_render.is_on = True # default on
        slider_line.add_child(self.slider_render)

        self.panel.add_child(slider_line)

        # self.panel.add_child(gui.Label("Viewpoint Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile3 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile4 = gui.Vert(0.5 * em, gui.Margins(margin))
        
        # h = gui.Horiz(0.25 * em, gui.Margins(margin)) 
        # self._arcball_button = gui.Button("Arcball")
        # self._arcball_button.horizontal_padding_em = 0.5
        # self._arcball_button.vertical_padding_em = 0
        # self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)

        # h.add_child(self._arcball_button)
        # h.add_child(self._fly_button)

        # self.panel.add_child(h)

        ##Check boxes
        vp_subtile1.add_child(gui.Label("Camera view options"))
        chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        
        self.followcam_chbox = gui.Checkbox("Follow")
        self.followcam_chbox.checked = True
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("Behind")
        self.staybehind_chbox.checked = True
        chbox_tile.add_child(self.staybehind_chbox)

        self.fly_chbox = gui.Checkbox("Fly")
        # NOTE: in fly mode, you can control like a game using WASD,Q,Z,E,R, up, right, left, down
        self.fly_chbox.checked = False
        self.fly_chbox.set_on_checked(self._set_mouse_mode)
        chbox_tile.add_child(self.fly_chbox)
        
        vp_subtile1.add_child(chbox_tile)

        ##Combo panels for current frames
        combo_tile = gui.Vert(0.5 * em, gui.Margins(margin))

        self.combo_cams = gui.Combobox()
        self.combo_cams.set_on_selection_changed(self._on_combo_cams)
        combo_tile.add_child(gui.Label("Cur. cameras"))
        combo_tile.add_child(self.combo_cams)
        vp_subtile2.add_child(combo_tile)

        ##Combo panels for train frames
        combo_tile2 = gui.Vert(0.5 * em, gui.Margins(margin))
        self.combo_train_cams = gui.Combobox()
        self.combo_train_cams.set_on_selection_changed(self._on_combo_train_cams)
        combo_tile2.add_child(gui.Label("Train cameras"))
        combo_tile2.add_child(self.combo_train_cams)
        vp_subtile3.add_child(combo_tile2)

        ##Combo panels for preset views 
        combo_tile3 = gui.Vert(0.5 * em, gui.Margins(margin))
        self.combo_preset_cams = gui.Combobox()
        for i in range(10):
            self.combo_preset_cams.add_item(str(i))

        # self.combo_preset_cams.set_on_selection_changed(self._on_combo_preset_cams) 
        combo_tile3.add_child(gui.Label("Preset views"))
        combo_tile3.add_child(self.combo_preset_cams)
        vp_subtile4.add_child(combo_tile3)

        self.reset_view_btn = gui.Button("Reset")
        self.reset_view_btn.set_on_clicked(
            self._on_reset_view_btn
        )  # set the callback function

        self.save_view_btn = gui.Button("Save")
        self.save_view_btn.set_on_clicked(
            self._on_save_view_btn
        )  # set the callback function

        self.load_view_btn = gui.Button("Load")
        self.load_view_btn.set_on_clicked(
            self._on_load_view_btn
        )  # set the callback function

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        viewpoint_tile.add_child(vp_subtile3)
        viewpoint_tile.add_child(vp_subtile4)

        viewpoint_tile.add_child(self.save_view_btn)
        viewpoint_tile.add_child(self.load_view_btn)
        viewpoint_tile.add_child(self.reset_view_btn)
        
        self.panel.add_child(viewpoint_tile)

        self.panel.add_child(gui.Label("3D Objects"))

        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.gs_chbox = gui.Checkbox("GS")
        self.gs_chbox.checked = self.gs_default_on
        # self.gs_chbox.set_on_checked(self._on_gs_chbox)
        chbox_tile_3dobj.add_child(self.gs_chbox)

        self.backface_chbox = gui.Checkbox("Backface")
        self.backface_chbox.checked = False
        # self.backface_chbox.set_on_checked(self._on_backface_chbox)
        chbox_tile_3dobj.add_child(self.backface_chbox)

        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.keyframe_chbox = gui.Checkbox("Train Cams")
        self.keyframe_chbox.checked = True
        self.keyframe_chbox.set_on_checked(self._on_keyframes_chbox)
        chbox_tile_3dobj.add_child(self.keyframe_chbox)

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
        self.mesh_chbox.checked = self.mesh_default_on
        self.mesh_chbox.set_on_checked(self._on_mesh_chbox)
        chbox_tile_3dobj.add_child(self.mesh_chbox)
        self.mesh_name = "pin_mesh"

        self.cmesh_chbox = gui.Checkbox("Colorized Mesh")
        self.cmesh_chbox.checked = self.mesh_default_on
        self.cmesh_chbox.set_on_checked(self._on_cmesh_chbox)
        chbox_tile_3dobj.add_child(self.cmesh_chbox)

        self.scan_chbox = gui.Checkbox("Scan")
        self.scan_chbox.checked = True
        self.scan_chbox.set_on_checked(self._on_scan_chbox)
        chbox_tile_3dobj.add_child(self.scan_chbox)
        self.scan_name = "cur_scan"

        self.rendered_scan_chbox = gui.Checkbox("Rendered Points")
        self.rendered_scan_chbox.checked = True
        self.rendered_scan_chbox.set_on_checked(self._on_rendered_scan_chbox)
        chbox_tile_3dobj.add_child(self.rendered_scan_chbox)
        self.rendered_scan_name = "cur_rendered_scan"

        # self.sky_chbox = gui.Checkbox("Sky")
        # self.sky_chbox.checked = False
        # self.sky_chbox.set_on_checked(self._on_sky_chbox)
        # chbox_tile_3dobj.add_child(self.sky_chbox)
        # self.widget3d.scene.show_skybox(True) # does not work

        chbox_tile_3dobj_2 = gui.Horiz(0.5 * em, gui.Margins(margin))

        # TODO
        self.neural_point_chbox = gui.Checkbox("Neural Points")
        self.neural_point_chbox.checked = self.neural_point_default_on
        self.neural_point_chbox.set_on_checked(self._on_neural_point_chbox)
        chbox_tile_3dobj_2.add_child(self.neural_point_chbox)
        self.neural_point_name = "neural_points"

        self.invalid_neural_point_chbox = gui.Checkbox("Invalid Points")
        self.invalid_neural_point_chbox.checked = False
        self.invalid_neural_point_chbox.set_on_checked(self._on_invalid_neural_point_chbox)
        chbox_tile_3dobj_2.add_child(self.invalid_neural_point_chbox)
        self.invalid_neural_point_name = "invalid_neural_points"

        self.sdf_chbox = gui.Checkbox("SDF")
        self.sdf_chbox.checked = False
        self.sdf_chbox.set_on_checked(self._on_sdf_chbox)
        chbox_tile_3dobj_2.add_child(self.sdf_chbox)
        self.sdf_name = "cur_sdf_slice"

        self.cad_chbox = gui.Checkbox("Robot")
        self.cad_chbox.checked = self.robot_default_on
        self.cad_chbox.set_on_checked(self._on_cad_chbox)
        chbox_tile_3dobj_2.add_child(self.cad_chbox)
        self.cad_name = "sensor_cad"

        self.gt_traj_chbox = gui.Checkbox("GT Traj.")
        self.gt_traj_chbox.checked = False
        self.gt_traj_chbox.set_on_checked(self._on_gt_traj_chbox)
        chbox_tile_3dobj_2.add_child(self.gt_traj_chbox)
        self.gt_traj_name = "gt_trajectory"

        self.slam_traj_chbox = gui.Checkbox("SLAM Traj.")
        self.slam_traj_chbox.checked = False
        self.slam_traj_chbox.set_on_checked(self._on_slam_traj_chbox)
        chbox_tile_3dobj_2.add_child(self.slam_traj_chbox)
        self.slam_traj_name = "slam_trajectory"

        self.range_circle_chbox = gui.Checkbox("R Circle")
        self.range_circle_chbox.checked = False
        self.range_circle_chbox.set_on_checked(self._on_range_circle_chbox)
        chbox_tile_3dobj_2.add_child(self.range_circle_chbox)
        self.range_circle_name = "range_circle"

        self.sdf_pool_chbox = gui.Checkbox("SDF Samples")
        self.sdf_pool_chbox.checked = False
        self.sdf_pool_chbox.set_on_checked(self._on_sdf_pool_chbox)
        chbox_tile_3dobj_2.add_child(self.sdf_pool_chbox)
        self.sdf_pool_name = "sdf_sample_pool"

        self.panel.add_child(chbox_tile_3dobj)

        self.panel.add_child(chbox_tile_3dobj_2)


        self.panel.add_child(gui.Label("Neural Point Color Options"))
        chbox_tile_neuralpoint = gui.Horiz(0.5 * em, gui.Margins(margin))

        # default mode 0: original rgb color

        # mode 1
        self.neuralpoint_geofeature_chbox = gui.Checkbox("Geometric Feature")
        self.neuralpoint_geofeature_chbox.checked = (self.neural_point_color_default_mode==1)
        self.neuralpoint_geofeature_chbox.set_on_checked(self._on_neuralpoint_geofeature_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_geofeature_chbox)

        # mode 2
        self.neuralpoint_colorfeature_chbox = gui.Checkbox("Photometric Feature")
        self.neuralpoint_colorfeature_chbox.checked = (self.neural_point_color_default_mode==2)
        self.neuralpoint_colorfeature_chbox.set_on_checked(self._on_neuralpoint_colorfeature_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_colorfeature_chbox)

        # mode 3
        self.neuralpoint_ts_chbox = gui.Checkbox("Timestep")
        self.neuralpoint_ts_chbox.checked = (self.neural_point_color_default_mode==3)
        self.neuralpoint_ts_chbox.set_on_checked(self._on_neuralpoint_ts_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_ts_chbox)

        # mode 4
        self.neuralpoint_stability_chbox = gui.Checkbox("Stability")
        self.neuralpoint_stability_chbox.checked = (self.neural_point_color_default_mode==4)
        self.neuralpoint_stability_chbox.set_on_checked(self._on_neuralpoint_stability_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_stability_chbox)

        self.panel.add_child(chbox_tile_neuralpoint)

        self.panel.add_child(gui.Label("GS Rendering Options"))
        chbox_tile_gsrender = gui.Horiz(0.5 * em, gui.Margins(margin))

        # these cannot be on at the same time
        self.depth_chbox = gui.Checkbox("Depth")
        self.depth_chbox.checked = False
        self.depth_chbox.set_on_checked(self._on_depth_chbox)
        chbox_tile_gsrender.add_child(self.depth_chbox)

        self.normal_chbox = gui.Checkbox("Normal")
        self.normal_chbox.checked = False
        self.normal_chbox.set_on_checked(self._on_normal_chbox)
        chbox_tile_gsrender.add_child(self.normal_chbox)

        self.d2n_chbox = gui.Checkbox("D2N")
        self.d2n_chbox.checked = False
        self.d2n_chbox.set_on_checked(self._on_d2n_chbox)
        chbox_tile_gsrender.add_child(self.d2n_chbox)

        self.opacity_chbox = gui.Checkbox("Opacity")
        self.opacity_chbox.checked = False
        self.opacity_chbox.set_on_checked(self._on_opacity_chbox)
        chbox_tile_gsrender.add_child(self.opacity_chbox)

        # self.time_shader_chbox = gui.Checkbox("Time Shader")
        # self.time_shader_chbox.checked = False
        # chbox_tile_gsrender.add_child(self.time_shader_chbox)

        self.elliopsoid_chbox = gui.Checkbox("Ellipsoid")
        self.elliopsoid_chbox.checked = False
        self.elliopsoid_chbox.set_on_checked(self._on_elliopsoid_chbox)
        chbox_tile_gsrender.add_child(self.elliopsoid_chbox)

        self.elliopsoid_2d_chbox = gui.Checkbox("Surfel mode")
        if self.config.gs_type == "3d_gs":
            self.elliopsoid_2d_chbox.checked = False
        else:
            self.elliopsoid_2d_chbox.checked = True
        chbox_tile_gsrender.add_child(self.elliopsoid_2d_chbox)

        self.normal_in_world_chbox = gui.Checkbox("Normal in world")
        self.normal_in_world_chbox.checked = True
        chbox_tile_gsrender.add_child(self.normal_in_world_chbox)

        self.normal_with_alpha_chbox = gui.Checkbox("Normal with alpha")
        self.normal_with_alpha_chbox.checked = True
        chbox_tile_gsrender.add_child(self.normal_with_alpha_chbox)

        self.panel.add_child(chbox_tile_gsrender)

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

        self.brisque_score_info = gui.Label("Current view BRISQUE score: ")
        tab_info.add_child(self.brisque_score_info)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)


        ## Input/Eval Image Tab
        tabs2 = gui.TabControl()
        
        tab_input = gui.Vert(0, tab_margins)

        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()
        self.in_normal_widget = gui.ImageWidget()

        self.rendered_rgb_widget = gui.ImageWidget()
        self.rendered_depth_widget = gui.ImageWidget()
        self.rendered_depth_error_widget = gui.ImageWidget()

        view_info_tile = gui.Horiz(1.5 * em, gui.Margins(margin))
        # view_info_tile.add_stretch()

        self.cur_view_info = gui.Label("Camera: ")
        view_info_tile.add_child(self.cur_view_info)

        self.cur_exposure_info = gui.Label("Exposure: ")
        view_info_tile.add_child(self.cur_exposure_info)

        self.cur_view_psnr_info = gui.Label("PSNR: ")
        view_info_tile.add_child(self.cur_view_psnr_info)

        self.cur_view_depthl1_info = gui.Label("Depth L1 (m): ")
        view_info_tile.add_child(self.cur_view_depthl1_info)
    
        tab_input.add_child(view_info_tile)

        tab_input.add_child(gui.Label("GT Color | Rendered Color | GT Depth | Depth Error | Normal"))

        # view_info_tile2 = gui.Horiz(1.5 * em, gui.Margins(margin)) # empty one

        # tab_input.add_child(view_info_tile2)

        tab_input.add_child(self.in_rgb_widget)

        tab_input.add_child(self.rendered_rgb_widget)
        
        tab_input.add_child(self.in_depth_widget)

        tab_input.add_child(self.rendered_depth_error_widget)
        
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
        self.g_renderer.set_render_mod(rend_mode) #  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 flat ball, -4 gaussian ball
        self.g_renderer.update_camera_pose(self.g_camera) 
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)

    def add_camera(self, camera, name, color=[0, 1, 0], size=0.01):
        # only the cam geometry
        # img are not added

        W2C = getWorld2View2(camera.R, camera.T)

        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        
        if name not in self.frustum_dict.keys():
            # frustum = create_frustum(C2W, color, size=size)
            self.combo_cams.add_item(name)
        
        frustum.update_pose(C2W)
        self.frustum_dict[name] = frustum
        self.widget3d.scene.add_geometry(name, frustum.line_set, self.cur_frame_render) # add camera frame to visualizer
        
        # frustum = self.frustum_dict[name]
        # frustum.update_pose(C2W)
        # self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        return frustum

    def add_keyframe(self, camera, name, color=[0, 1, 0], size=0.01):
        # only the cam geometry
        # img are not added

        # here keyframe are actually the train frames

        W2C = getWorld2View2(camera.R, camera.T)
        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.keyframe_dict.keys():
            # frustum = create_frustum(C2W, color, size=size)
            self.combo_train_cams.add_item(name) # TODO
            self.keyframe_dict[name] = frustum
            frustum.update_pose(C2W)
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.train_frame_render) # add camera frame to visualizer
        # frustum = self.keyframe_dict[name]
        # frustum.update_pose(C2W)
        # self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.keyframe_chbox.checked)
        return frustum

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        # self.widget3d_width_ratio = 0.6 # 0.7 # FIXME
        self.widget3d_width_ratio = self.config.visualizer_split_width_ratio
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

        # look_at(center, eye, up): sets the camera view so that the camera is located at ‘eye’, pointing towards ‘center’, and oriented so that the up vector is ‘up’
        # both center, eye, up are 3x1 np arrays
        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def _on_combo_cams(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        self.update_img_show(new_val)

    def _on_combo_train_cams(self, new_val, new_idx):
        frustum = self.keyframe_dict[new_val]
        viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        self.update_img_show(new_val, from_cur_frame=False)

    # def _on_gs_chbox(self, is_checked, name=None):
    #     names = self.frustum_dict.keys() if name is None else [name]
    #     for name in names:
    #         self.widget3d.scene.show_geometry(name, is_checked)

    def _on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_keyframes_chbox(self, is_checked, name=None):
        names = self.keyframe_dict.keys() if name is None else [name]
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

    def _on_invalid_neural_point_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.invalid_neural_point_name)
            self.widget3d.scene.add_geometry(self.invalid_neural_point_name, self.invalid_neural_points, self.neural_points_render) # TODO: add pin-slam mesh
        else:
            self.widget3d.scene.remove_geometry(self.invalid_neural_point_name)

    # TODO: rendering shader is not good
    def _on_mesh_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.mesh_name)
            self.widget3d.scene.add_geometry(self.mesh_name, self.mesh, self.mesh_render) # TODO: add pin-slam mesh
        else:
            self.widget3d.scene.remove_geometry(self.mesh_name)

        # packet = Packet_vis2main()
        # packet.flag_mesh = is_checked
        # self.q_vis2main.put(packet)
    
    def _on_cmesh_chbox(self, is_checked):
        if is_checked:
            self.mesh_render.shader = "defaultLit"
        else:
            self.mesh_render.shader = "normals"
        if self.mesh_chbox.checked:
            self.widget3d.scene.remove_geometry(self.mesh_name)
            self.widget3d.scene.add_geometry(self.mesh_name, self.mesh, self.mesh_render)


    def _on_scan_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.scan_name)
            self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)
        else:
            self.widget3d.scene.remove_geometry(self.scan_name)

    def _on_rendered_scan_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.rendered_scan_name)
            self.widget3d.scene.add_geometry(self.rendered_scan_name, self.rendered_scan, self.scan_render)
        else:
            self.widget3d.scene.remove_geometry(self.rendered_scan_name)
    
    def _on_sdf_pool_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.sdf_pool_name)
            self.widget3d.scene.add_geometry(self.sdf_pool_name, self.sdf_pool, self.sdf_pool_render)
        else:
            self.widget3d.scene.remove_geometry(self.sdf_pool_name)

    # sdf slice
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

    def _on_range_circle_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.range_circle_name)
            self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)
        else:
            self.widget3d.scene.remove_geometry(self.range_circle_name)

    # only one can be selected at the same time
    def _on_elliopsoid_chbox(self, is_checked):
        if is_checked:
            self.depth_chbox.checked = False
            self.normal_chbox.checked = False
            self.d2n_chbox.checked = False
            self.opacity_chbox.checked = False

    def _on_depth_chbox(self, is_checked):
        if is_checked:
            self.elliopsoid_chbox.checked = False
            self.normal_chbox.checked = False
            self.d2n_chbox.checked = False
            self.opacity_chbox.checked = False

    def _on_normal_chbox(self, is_checked):
        if is_checked:
            self.elliopsoid_chbox.checked = False
            self.depth_chbox.checked = False
            self.d2n_chbox.checked = False
            self.opacity_chbox.checked = False

    def _on_d2n_chbox(self, is_checked):
        if is_checked:
            self.elliopsoid_chbox.checked = False
            self.normal_chbox.checked = False
            self.depth_chbox.checked = False
            self.opacity_chbox.checked = False

    def _on_opacity_chbox(self, is_checked):
        if is_checked:
            self.elliopsoid_chbox.checked = False
            self.normal_chbox.checked = False
            self.d2n_chbox.checked = False
            self.depth_chbox.checked = False

    def _on_neuralpoint_geofeature_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_colorfeature_chbox.checked = False
            self.neuralpoint_stability_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False

    def _on_neuralpoint_colorfeature_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_stability_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False

    def _on_neuralpoint_ts_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_stability_chbox.checked = False
            self.neuralpoint_colorfeature_chbox.checked = False

    def _on_neuralpoint_stability_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False
            self.neuralpoint_colorfeature_chbox.checked = False

    def _on_sky_chbox(self, is_checked):
        self.widget3d.scene.show_skybox(is_checked)

    # def _on_backface_chbox(self, is_checked):
    #     self.widget3d.enable_back_face_culling(is_checked) 
    #     # self.mesh_render.show_back_face = is_checked

    # def _on_kf_window_chbox(self, is_checked):
    #     if self.kf_window is None:
    #         return
    #     edge_cnt = 0
    #     for key in self.kf_window.keys():
    #         for kf_idx in self.kf_window[key]:
    #             name = "kf_edge_{}".format(edge_cnt)
    #             edge_cnt += 1
    #             if "keyframe_{}".format(key) not in self.frustum_dict.keys():
    #                 continue
    #             test1 = self.frustum_dict["keyframe_{}".format(key)].view_dir[1]
    #             kf = self.frustum_dict["keyframe_{}".format(kf_idx)].view_dir[1]
    #             points = [test1, kf]
    #             lines = [[0, 1]]
    #             colors = [[0, 1, 0]] # green camera frame

    #             line_set = o3d.geometry.LineSet()
    #             line_set.points = o3d.utility.Vector3dVector(points)
    #             line_set.lines = o3d.utility.Vector2iVector(lines)
    #             line_set.colors = o3d.utility.Vector3dVector(colors)

    #             if is_checked:
    #                 self.widget3d.scene.remove_geometry(name)
    #                 self.widget3d.scene.add_geometry(name, line_set, self.traj_render)
    #             else:
    #                 self.widget3d.scene.remove_geometry(name)

    def _on_slam_slider(self, is_on):
        packet = Packet_vis2main()
        packet.flag_pause = not self.slider_slam.is_on
        self.q_vis2main.put(packet)

    def _on_slider(self, value):
        packet = self.prepare_viz2main_packet()
        self.q_vis2main.put(packet)

    def _on_render_btn(self):
        packet = Packet_vis2main()
        packet.flag_nextbatch = True
        self.q_vis2main.put(packet)

    def _on_screenshot_btn(self):
        
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

        if self.render_img is None:
            return
        img = np.asarray(self.render_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}.png", img)

    def _on_reset_view_btn(self):
        self.center_bev()
        self.fly_chbox.checked = False
        self.widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
    
    def _on_save_view_btn(self):
        save_view_file_name = 'saved_view_{}.pkl'.format(self.combo_preset_cams.selected_text)
        save_view_file_path = os.path.join(self.view_save_base_path, save_view_file_name)
        if self.save_view(save_view_file_path):
            print("Camera view {} saved".format(self.combo_preset_cams.selected_text))
    
    def _on_load_view_btn(self):
        load_view_file_name = 'saved_view_{}.pkl'.format(self.combo_preset_cams.selected_text)
        load_view_file_path = os.path.join(self.view_save_base_path, load_view_file_name)
        if self.load_view(load_view_file_path):
            print("Camera view {} loaded".format(self.combo_preset_cams.selected_text))

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

    def save_view(self, fname='.saved_view.pkl'):
        try:
            model_matrix = np.asarray(self.widget3d.scene.camera.get_model_matrix())
            extrinsic = model_matrix_to_extrinsic_matrix(model_matrix)
            height, width = int(self.window.size.height), int(self.widget3d_width)
            intrinsic = create_camera_intrinsic_from_size(width, height)
            saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)
            with open(fname, 'wb') as pickle_file:
                dump(saved_view, pickle_file)
            return True
        except Exception as e:
            print(e)
            return False

    def load_view(self, fname=".saved_view.pkl"):
        try:
            with open(fname, 'rb') as pickle_file:
                saved_view = load(pickle_file)
            self.widget3d.setup_camera(saved_view['intrinsic'], saved_view['extrinsic'], saved_view['width'], saved_view['height'], self.widget3d.scene.bounding_box)
            # Looks like the ground plane gets messed up, no idea how to fix
            return True
        except Exception as e:
            print("Can't find file", e)
            return False
    

    def receive_data(self, q):
        if q is None:
            return

        # TODO: is this slow?
        gaussian_packet = get_latest_queue(q)

        if gaussian_packet is None:
            return

        if gaussian_packet.frame_id != self.cur_frame_id:
            
            # only update with new data (once)

            self.cur_frame_id = gaussian_packet.frame_id

            self.gaussian_cur = gaussian_packet

            if gaussian_packet.frame_id is not None:
                self.frame_info.text = "Frame: {}".format(gaussian_packet.frame_id)
                    
            if gaussian_packet.has_neural_points:
                self.neural_points_info.text = "# Neural points: {} (local {}), # Valid: {} (local {}) [PINGS Map size: {:.1f} MB]".format(
                    gaussian_packet.neural_points_data["count"],
                    gaussian_packet.neural_points_data["local_count"],
                    gaussian_packet.neural_points_data["valid_count"],
                    gaussian_packet.neural_points_data["valid_local_count"],
                    gaussian_packet.neural_points_data["map_memory_mb"]
                )
                # done every time, could be a bit time consuming here
                
                neural_point_position = gaussian_packet.neural_points_data["position"].detach().cpu().numpy()
                
                dict_keys = list(gaussian_packet.neural_points_data.keys())

                neural_point_colors = None

                if "color_pca_geo" in dict_keys and self.neuralpoint_geofeature_chbox.checked:
                    neural_point_colors = gaussian_packet.neural_points_data["color_pca_geo"].detach().cpu().numpy()
                elif "color_pca_color" in dict_keys and self.neuralpoint_colorfeature_chbox.checked:
                    neural_point_colors = gaussian_packet.neural_points_data["color_pca_color"].detach().cpu().numpy()
                elif "ts" in dict_keys and self.neuralpoint_ts_chbox.checked:
                    max_ts = torch.max(gaussian_packet.neural_points_data["ts"])* 1.0
                    ts_np = (gaussian_packet.neural_points_data["ts"]/max_ts).detach().cpu().numpy()
                    color_map = cm.get_cmap("jet")
                    neural_point_colors = color_map(ts_np)[:, :3].astype(np.float64)
                elif "stability" in dict_keys and self.neuralpoint_stability_chbox.checked:
                    stability_vis_np = (1.0 - gaussian_packet.neural_points_data["stability"]/1000.0).detach().cpu().numpy()
                    certainty_np = np.clip(stability_vis_np, 0.0, 1.0)
                    neural_point_colors = np.repeat(certainty_np.reshape(-1, 1), 3, axis=1)
                elif "color" in dict_keys:
                    neural_point_colors = gaussian_packet.neural_points_data["color"].detach().cpu().numpy()
                
                
                neural_point_valid_mask = gaussian_packet.neural_points_data["valid_mask"].detach().cpu().numpy()

                valid_neural_point_position = neural_point_position[neural_point_valid_mask]                
                invalid_neural_point_position = neural_point_position[~neural_point_valid_mask]

                self.neural_points.points = o3d.utility.Vector3dVector(valid_neural_point_position)
                self.invalid_neural_points.points = o3d.utility.Vector3dVector(invalid_neural_point_position)

                if neural_point_colors is not None:
                    valid_neural_point_color = neural_point_colors[neural_point_valid_mask]
                    invalid_neural_point_color = neural_point_colors[~neural_point_valid_mask]
                    invalid_neural_point_color[:,:] = (0, 0, 0) # invalid part set to black for vis

                    self.neural_points.colors = o3d.utility.Vector3dVector(valid_neural_point_color)                
                    self.invalid_neural_points.colors = o3d.utility.Vector3dVector(invalid_neural_point_color)

                self.widget3d.scene.remove_geometry(self.neural_point_name)
                self.widget3d.scene.add_geometry(self.neural_point_name, self.neural_points, self.neural_points_render)
                self.widget3d.scene.show_geometry(self.neural_point_name, self.neural_point_chbox.checked)
            
                if self.invalid_neural_point_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.invalid_neural_point_name)
                    self.widget3d.scene.add_geometry(self.invalid_neural_point_name, self.invalid_neural_points, self.neural_points_render)

                # FIXME

                # show feature PCA color (TODO)

            if gaussian_packet.has_sorrounding_points:
                cur_center_position = gaussian_packet.sorrounding_neural_points_data["center"]
                
                # spawn gaussians for the sorrounding map
                # self.cur_base_gaussians are stored in GPU, might take some memory

                self.cur_base_gaussians = spawn_gaussians(gaussian_packet.sorrounding_neural_points_data, 
                    self.decoders, None, cur_center_position,
                    dist_concat_on=self.config.dist_concat_on, 
                    view_concat_on=self.config.view_concat_on, 
                    scale_filter_on=True,
                    z_far=self.config.sorrounding_map_radius,
                    learn_color_residual=self.config.learn_color_residual,
                    gs_type=self.config.gs_type)
            
            frustum_size = self.config.max_range*0.008

            # load cameras
            if gaussian_packet.current_frames is not None and len(gaussian_packet.cam_list)>0: # as Camera class
                
                for cam_name in list(self.frustum_dict.keys()): 
                    self.widget3d.scene.remove_geometry(cam_name)

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

                    # show rgb / depth / normal imgs (also the rendered rgb / depth error, etc.)
                    self.update_img_show(selected_cam)            

            if gaussian_packet.keyframes is not None: # as Camera class
                
                # remove old stuff from last frame
                for keyframe_name in list(self.keyframe_dict.keys()): 
                    self.widget3d.scene.remove_geometry(keyframe_name)

                self.keyframe_dict = {} # set back to empty
                self.combo_train_cams.clear_items() # set back to empty

                # add new stuff from this frame
                for cam in gaussian_packet.keyframe_list:
                    cur_keyframe =  gaussian_packet.keyframes[cam]
                    if cur_keyframe.in_long_term_memory:
                        frustum_color = [0.5, 0.5, 0]
                    else:
                        frustum_color = [1, 1, 0]
                    frustum = self.add_keyframe(
                        cur_keyframe, name=cur_keyframe.uid, color=frustum_color, size=frustum_size
                    ) 

            # TODO: add evaluation online, visualize the error map here

            if gaussian_packet.current_pointcloud_xyz is not None:
                self.scan.points = o3d.utility.Vector3dVector(gaussian_packet.current_pointcloud_xyz)
                if gaussian_packet.current_pointcloud_rgb is not None:
                    self.scan.colors = o3d.utility.Vector3dVector(gaussian_packet.current_pointcloud_rgb)
                if self.scan_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.scan_name)
                    self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)

            if gaussian_packet.current_rendered_xyz is not None:
                self.rendered_scan.points = o3d.utility.Vector3dVector(gaussian_packet.current_rendered_xyz)
                if gaussian_packet.current_rendered_rgb is not None:
                    self.rendered_scan.colors = o3d.utility.Vector3dVector(gaussian_packet.current_rendered_rgb)
                if self.rendered_scan_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.rendered_scan_name)
                    self.widget3d.scene.add_geometry(self.rendered_scan_name, self.rendered_scan, self.scan_render)

            if gaussian_packet.sdf_slice_xyz is not None:
                if self.sdf_chbox.checked:
                    self.sdf_slice.points = o3d.utility.Vector3dVector(gaussian_packet.sdf_slice_xyz)
                    if gaussian_packet.sdf_slice_rgb is not None:
                        self.sdf_slice.colors = o3d.utility.Vector3dVector(gaussian_packet.sdf_slice_rgb)

                    self.widget3d.scene.remove_geometry(self.sdf_name)
                    self.widget3d.scene.add_geometry(self.sdf_name, self.sdf_slice, self.sdf_render)

            if gaussian_packet.sdf_pool_xyz is not None:
                if self.sdf_pool_chbox.checked:
                    self.sdf_pool.points = o3d.utility.Vector3dVector(gaussian_packet.sdf_pool_xyz)
                    if gaussian_packet.sdf_pool_rgb is not None:
                        self.sdf_pool.colors = o3d.utility.Vector3dVector(gaussian_packet.sdf_pool_rgb)

                    self.widget3d.scene.remove_geometry(self.sdf_pool_name)
                    self.widget3d.scene.add_geometry(self.sdf_pool_name, self.sdf_pool, self.sdf_pool_render)

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
                if gt_position_np.shape[0] > 1:
                    self.gt_traj.points = o3d.utility.Vector3dVector(gt_position_np)
                    gt_edges = np.array([[i, i + 1] for i in range(gt_position_np.shape[0] - 1)])
                    self.gt_traj.lines = o3d.utility.Vector2iVector(gt_edges)
                    self.gt_traj.paint_uniform_color(BLACK)
                
                if self.gt_traj_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.gt_traj_name)
                    self.widget3d.scene.add_geometry(self.gt_traj_name, self.gt_traj, self.traj_render)

                if gaussian_packet.slam_poses is None:
                    
                    self.sensor_cad = copy.deepcopy(self.sensor_cad_origin)
                    self.sensor_cad.transform(gaussian_packet.gt_poses[-1])
                    
                    if self.cad_chbox.checked:
                        self.widget3d.scene.remove_geometry(self.cad_name)
                        self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)
                    
                    self.range_circle = copy.deepcopy(self.range_circle_origin)
                    self.range_circle.transform(gaussian_packet.gt_poses[-1])  

                    if self.range_circle_chbox.checked: 
                        self.widget3d.scene.remove_geometry(self.range_circle_name)
                        self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)

            if gaussian_packet.slam_poses is not None:
                
                slam_position_np = gaussian_packet.slam_poses[:, :3, 3]
                if slam_position_np.shape[0] > 1:
                    self.slam_traj.points = o3d.utility.Vector3dVector(slam_position_np)
                    slam_edges = np.array([[i, i + 1] for i in range(slam_position_np.shape[0] - 1)])
                    self.slam_traj.lines = o3d.utility.Vector2iVector(slam_edges)
                    self.slam_traj.paint_uniform_color(RED)

                if self.slam_traj_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.slam_traj_name)
                    self.widget3d.scene.add_geometry(self.slam_traj_name, self.slam_traj, self.traj_render)
                
                self.sensor_cad = copy.deepcopy(self.sensor_cad_origin)
                self.sensor_cad.transform(gaussian_packet.slam_poses[-1])

                if self.cad_chbox.checked:
                    
                    self.widget3d.scene.remove_geometry(self.cad_name)
                    self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)

                self.range_circle = copy.deepcopy(self.range_circle_origin)
                self.range_circle.transform(gaussian_packet.slam_poses[-1])
                
                if self.range_circle_chbox.checked: 
                    
                    self.widget3d.scene.remove_geometry(self.range_circle_name)
                    self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)


        # set up inital camera # no camera
        if len(gaussian_packet.cam_list) == 0 and not self.init:
            self.center_bev()

        self.init = True

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

    def center_bev(self):
        # set the view point to BEV of the current 3d objects
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60, bounds, bounds.get_center())  # field of view, bound, center


    def update_img_show(self, cam_name, 
                        from_cur_frame: bool = True,
                        online_eval_on: bool = True, 
                        show_depth_error: bool = False):

        if cam_name in list(self.gaussian_cur.gtcolor.keys()):
            selected_gtcolor = self.gaussian_cur.gtcolor[cam_name]
            selected_gtdepth = self.gaussian_cur.gtdepth[cam_name]
            selected_gtnormal = self.gaussian_cur.gtnormal[cam_name]
        else:
            selected_gtcolor = selected_gtdepth = selected_gtnormal = None

        if selected_gtcolor is not None:
            rgb = torch.clamp(selected_gtcolor, min=0, max=1.0) 
            rgb_np = (rgb * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb_o3d = o3d.geometry.Image(rgb_np)
            self.in_rgb_widget.update_image(rgb_o3d)

        if selected_gtdepth is not None:
            depth_np = selected_gtdepth.contiguous().cpu().numpy() 
            depth_color_np = (colorize_depth_maps(depth_np, 0.1, self.config.max_range*0.8)*255.0).astype(np.uint8)
            depth_color_np = np.transpose(depth_color_np[0], (1, 2, 0))

            if selected_gtcolor is not None:
                depth_color_np = self.overlaid_img(depth_color_np, rgb_np)     
            
            depth_color_np = np.ascontiguousarray(depth_color_np)
            depth_color_o3d = o3d.geometry.Image(depth_color_np)
            self.in_depth_widget.update_image(depth_color_o3d)

        if selected_gtnormal is not None:
            normal = selected_gtnormal.contiguous().cpu().numpy() 
            normal_color = 0.5 - normal * 0.5
            normal_color = np.transpose(normal_color, (1, 2, 0))
            normal_color = np.ascontiguousarray(normal_color)
            normal_color_o3d = o3d.geometry.Image(normal_color)
            self.in_normal_widget.update_image(normal_color_o3d)

        cur_psnr = None
        cur_depthl1 = None

        if from_cur_frame:
            cur_frame_cam: CamImage = self.gaussian_cur.current_frames[cam_name]
        else:
            cur_frame_cam: CamImage = self.gaussian_cur.keyframes[cam_name]

        down_rate_used = max(self.config.gs_vis_down_rate, cur_frame_cam.cur_best_level)

        # render_mesh_on = True
        # if render_mesh_on:

        #     depth_image_mesh = self.widget3d.scene.render_to_depth_image(width=640, height=480) # FUCK
        #     depth_image_mesh_np = np.asarray(depth_image_mesh)

        #     print(depth_image_mesh_np.shape())

        #     # depth_image_mesh_np = (colorize_depth_maps(depth_image_mesh_np, 0.0, self.config.min_range, cmap="inferno_r")[0]*255.0).astype(np.uint8)


        if online_eval_on:

            with torch.no_grad():
                render_results = render(cur_frame_cam, 
                    None, self.gaussian_cur.neural_points_data, 
                    self.decoders, self.cur_base_gaussians, self.background,
                    scaling_modifier=self.scaling_slider.double_value, 
                    down_rate=down_rate_used, 
                    dist_concat_on=self.config.dist_concat_on, 
                    view_concat_on=self.config.view_concat_on, 
                    correct_exposure=self.config.exposure_correction_on,
                    learn_color_residual=self.config.learn_color_residual,
                    front_only_on=(not self.backface_chbox.checked),
                    d2n_on=False,
                    gs_type=self.config.gs_type,
                    displacement_range_ratio=self.config.displacement_range_ratio,
                    max_scale_ratio=self.config.max_scale_ratio,
                    unit_scale_ratio=self.config.unit_scale_ratio)
                    

            if render_results is not None:
                
                rendered_rgb = torch.clamp(render_results["render"], 0.0, 1.0)

                rendered_rgb_np = (
                    (rendered_rgb * 255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                rendered_rgb_o3d = o3d.geometry.Image(rendered_rgb_np)
                self.rendered_rgb_widget.update_image(rendered_rgb_o3d)

                cur_psnr = psnr(rendered_rgb, rgb).mean().item()

                eval_depth_max = self.config.max_range * 0.8
                eval_depth_min = self.config.min_range
                diff_depth_max_show = eval_depth_max * 0.05 # unit: m

                rendered_depth = render_results["surf_depth"]
                cur_gt_depth = selected_gtdepth
                if rendered_depth is not None and cur_gt_depth is not None:
                    
                    depth_valid_mask = (rendered_depth > eval_depth_min) & (cur_gt_depth > eval_depth_min) & (cur_gt_depth < eval_depth_max) & (rendered_depth < eval_depth_max)
                    if render_results["rend_alpha"] is not None:
                        depth_valid_mask = depth_valid_mask & (render_results["rend_alpha"] > self.config.depth_min_accu_alpha)

                    diff_depth = torch.abs(rendered_depth - cur_gt_depth)
                    diff_depth_masked = diff_depth[depth_valid_mask].detach().cpu().numpy()
                    cur_depthl1 = np.mean(diff_depth_masked)

                    if show_depth_error:

                        diff_depth[~depth_valid_mask] = 0.0
                        diff_depth_np = diff_depth.detach().cpu().numpy()
                        
                        diff_depth_color_np = (colorize_depth_maps(diff_depth_np, 0.0, diff_depth_max_show, cmap="inferno_r")[0]*255.0).astype(np.uint8)
                        diff_depth_color_np = np.transpose(diff_depth_color_np, (1, 2, 0)) # H, W, 3
                        diff_depth_color_np = np.ascontiguousarray(diff_depth_color_np)

                        if selected_gtcolor is not None:
                            diff_depth_color_np = self.overlaid_img(diff_depth_color_np, rgb_np) 

                        diff_depth_o3d = o3d.geometry.Image(diff_depth_color_np)

                        self.rendered_depth_error_widget.update_image(diff_depth_o3d)
        
        if cur_frame_cam.train_view:
            train_view_info = "train"
        else:
            train_view_info = "test"

        self.cur_view_info.text = "Camera: {} [{}]".format(cur_frame_cam.uid, train_view_info)
        self.cur_exposure_info.text = "Exposure: ({:.3f} , {:.3f})".format(cur_frame_cam.exposure_a.item(), cur_frame_cam.exposure_b.item())
        
        if cur_psnr is not None:
            self.cur_view_psnr_info.text = "PSNR: {:.3f}".format(cur_psnr)
        
        if cur_depthl1 is not None:
            self.cur_view_depthl1_info.text = "Depth L1 (m): {:.3f}".format(cur_depthl1)

    
    def overlaid_img(self, foreground_img_np, background_img_np, alpha_foreground: float = 0.7):
        overlaid_img_np = (1 - alpha_foreground) * background_img_np + alpha_foreground * foreground_img_np
        overlaid_img_np = overlaid_img_np.astype(np.uint8)

        return overlaid_img_np

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

        cur_view_mat = self.widget3d.scene.camera.get_view_matrix()

        has_nan = np.isnan(cur_view_mat).any()
        if has_nan:
            cur_view_mat = np.eye(4)

        w2c = cv_gl @ cur_view_mat # should not be NaN

        # print(w2c)

        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view() # Here there's problem

        hfov_deg = self.vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        H = image_gui.shape[1]
        W = image_gui.shape[2]
        cx = W // 2
        cy = H // 2
        T = torch.from_numpy(w2c) # T_cw

        K_mat = np.eye(3)
        K_mat[0,0] = fx
        K_mat[1,1] = fy
        K_mat[0,2] = cx
        K_mat[1,2] = cy

        current_cam = CamImage(-1, None, K_mat, self.config.min_range*0.2, self.config.local_map_radius*1.1,
            img_width=W, img_height=H, cam_pose=torch.linalg.inv(T)) # T_wc
    
        # print(current_cam.camera_center)
                                                        
        return current_cam


    # main rendering function for the 3D visualizer
    def render_o3d_image(self, results, current_cam, normal_in_world_frame: bool = True, normal_with_alpha: bool = True):

        if not self.gs_chbox.checked:
            return None # don't show gs rendering results

        rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )

        if self.step % 200 == 0:  # 2 second
            cur_brisque_score = self.brisque_scorer.score(img=rgb)
            self.brisque_score_info.text = ("Current view BRISQUE score: {:.3f}".format(cur_brisque_score))
        
        if self.depth_chbox.checked:
            depth = results["surf_depth"]
            if depth is None:
                return None # don't show gs rendering results
            
            if results["rend_alpha"] is not None:
                valid_depth_mask = (results["rend_alpha"] > self.config.depth_min_accu_alpha)
                depth[~valid_depth_mask] = 0.0

            depth = depth.detach().cpu().numpy()
            # max_depth = np.max(depth)
            depth_color = (colorize_depth_maps(depth, 0.1, self.config.max_range, cmap="inferno_r")[0]*255.0).astype(np.uint8) # 1, 3, H, W 
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

            if normal_with_alpha:
                normal_norm = normal.norm(2, dim=0) 
                normal_color = 0.5 * (normal_norm - normal) #   # convert to the normal vis color
            else:
                normal = torch.nn.functional.normalize(normal, dim=0) # normalize to norm==1 # don't do this, for small opacity region, we just downweight its normal
                normal_color = 0.5 * (1 - normal)

            normal_color = (normal_color.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
            normal_color = np.ascontiguousarray(normal_color)
            render_img = o3d.geometry.Image(normal_color)

        elif self.d2n_chbox.checked:
            d2n = results["surf_normal"]
            if d2n is None:
                return None # don't show gs rendering results
            
            if normal_in_world_frame: 
            # transform to world frame
                d2n = -1.0 * (d2n.permute(1,2,0) @ (current_cam.world_view_transform[:3,:3].T)).permute(2,0,1)

            if normal_with_alpha:
                d2n_norm = d2n.norm(2, dim=0) 
                d2n_color =  0.5 * (d2n_norm - d2n) # convert to the normal vis color
            else:
                d2n = torch.nn.functional.normalize(d2n, dim=0) # normalize to norm==1
                d2n_color = 0.5 * (1 - d2n)
            
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

        elif self.elliopsoid_chbox.checked: # important # TODO: try this still

            # return None # TODO: currently has some issue

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

            # neural gaussian version
            self.gaussians_gl.xyz = results["gaussian_xyz"].cpu().numpy()

            gaussian_scale = results["gaussian_scale"]
            if self.config.gs_type == "2d_gs":
                thin_dim_scale = torch.full((gaussian_scale.shape[0], 1), 1e-7).to(gaussian_scale) # already after activation, last dim, very thin
                gaussian_scale = torch.cat((gaussian_scale, thin_dim_scale), dim=1) # NK, 3

            self.gaussians_gl.scale = gaussian_scale.cpu().numpy()
            self.gaussians_gl.rot = results["gaussian_rot"].cpu().numpy()
            self.gaussians_gl.opacity = results["gaussian_alpha"].cpu().numpy()
            gaussians_gl_rgb = results["gaussian_color"].cpu().numpy()
            self.gaussians_gl.sh = (gaussians_gl_rgb - 0.5) / 0.28209479177387814

            # TODO
            # here all the gaussians in the global map
            # self.gaussians_gl.xyz = self.gaussian_cur.get_xyz.cpu().numpy()
            # self.gaussians_gl.opacity = self.gaussian_cur.get_opacity.cpu().numpy()
            # self.gaussians_gl.scale = self.gaussian_cur.get_scaling.cpu().numpy()
            # self.gaussians_gl.rot = self.gaussian_cur.get_rotation.cpu().numpy()
            # self.gaussians_gl.sh = self.gaussian_cur.get_features.cpu().numpy()[:, 0, :]

            # local map only
            # self.gaussians_gl.xyz = self.gaussian_cur.gaussian_xyz.cpu().numpy()
            # self.gaussians_gl.opacity = self.gaussian_cur.gaussian_alpha.cpu().numpy() + 1.0
            # self.gaussians_gl.scale = self.gaussian_cur.gaussian_scale.cpu().numpy()
            # self.gaussians_gl.rot = self.gaussian_cur.gaussian_rot.cpu().numpy()

            # gaussians_gl_rgb = self.gaussian_cur.gaussian_color.cpu().numpy()
            # self.gaussians_gl.sh = (gaussians_gl_rgb - 0.5) / 0.28209479177387814 # C0

            # # self.gaussians_gl.sh = self.gaussian_cur.gaussian_color.cpu().numpy()[:, 0, :]

            if self.elliopsoid_2d_chbox.checked:
                render_mode = -3 # 2D surfel
            else:
                render_mode = -4 # 3D elliopsoid

            self.update_activated_renderer_state(self.gaussians_gl, render_mode) # > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 flat ball (better fit with Gaussian Surfels), -4 gaussian ball
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

            render_tic = get_time()

            rendering_data = render(current_cam, None, 
                self.gaussian_cur.neural_points_data, 
                self.decoders, self.cur_base_gaussians, self.background, 
                scaling_modifier=self.scaling_slider.double_value, 
                down_rate=self.config.gs_vis_down_rate, 
                dist_concat_on=self.config.dist_concat_on, 
                view_concat_on=self.config.view_concat_on, 
                correct_exposure=False,
                learn_color_residual=self.config.learn_color_residual,
                front_only_on=(not self.backface_chbox.checked),
                d2n_on=self.d2n_chbox.checked,
                gs_type=self.config.gs_type,
                displacement_range_ratio=self.config.displacement_range_ratio,
                max_scale_ratio=self.config.max_scale_ratio,
                unit_scale_ratio=self.config.unit_scale_ratio)
            
            render_toc = get_time()

            if rendering_data is not None:
                if "local_view_gaussian_count" in list(rendering_data.keys()):
                    gaussians_all_count = rendering_data["alpha_all"].shape[0]
                    gaussians_valid_count = rendering_data["local_view_gaussian_count"]
                    valid_ratio = 1.0 * gaussians_valid_count / gaussians_all_count
                    mean_valid_count = valid_ratio * self.config.spawn_n_gaussian
                    cur_visble_gaussian_count = torch.sum(rendering_data["visibility_filter"]).item()
                    self.gaussian_info.text = "# Current view Gaussians: {} (valid: {:.1f} / {})".format(cur_visble_gaussian_count, mean_valid_count, self.config.spawn_n_gaussian)

            render_time = render_toc - render_tic # s
            render_freq = 1.0/render_time
            
            if self.step % 10 == 0:
                self.freq_info.text = "Render FPS: {:.1f}".format(render_freq)
        
        return rendering_data

        # return None

    # main function here for render gs
    def render_gui(self):
        if not self.init:
            return

        current_cam = self.get_current_cam()

        if not self.gs_chbox.checked:
            if self.render_img is None:
                return
            self.render_img = None
        else: # gs_chbox checked
            results = self.rasterise(current_cam)
            if results is None:
                return
            self.render_img = self.render_o3d_image(results, current_cam, 
                    self.normal_in_world_chbox.checked, 
                    self.normal_with_alpha_chbox.checked)
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
            
            # print(self.step)

            def update():
                if self.slider_render.is_on:
                    # print("UPDATE scene")
                    if self.step % 3 == 0: # per 0.03s # 30 Hz
                        self.render_gui() # stucked here

                    if self.step % 20 == 0: # per 0.2s # 5 Hz # receive latest data
                        self.receive_data(self.q_main2vis) # this is also slow

                    if self.step % 50 == 0: # per 0.5s
                        remove_gpu_cache() # remove cache regularly
                
                else:
                    while not self.q_main2vis.empty(): # free the queue
                        self.q_main2vis.get()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)


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

def generate_circle(radius=1.0, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    # Circle in the XY plane
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(num_points)  # Z-coordinates are 0 for a flat circle in XY-plane
    circle_points = np.vstack((x, y, z)).T  # Shape (num_points, 3)
    return circle_points

def model_matrix_to_extrinsic_matrix(model_matrix):
    return np.linalg.inv(model_matrix @ FromGLGamera)

def create_camera_intrinsic_from_size(width=1024, height=768, hfov=60.0, vfov=60.0):
    fx = (width / 2.0)  / np.tan(np.radians(hfov)/2)
    fy = (height / 2.0)  / np.tan(np.radians(vfov)/2)
    fx = fy # not sure why, but it looks like fx should be governed/limited by fy
    return np.array(
        [[fx, 0, width / 2.0],
         [0, fy, height / 2.0],
         [0, 0,  1]])


if __name__ == "__main__":
    main()
