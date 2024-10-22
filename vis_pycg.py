from pycg import vis, image, exp, isometry, render, video
import open3d as o3d
import numpy as np
import os


# PyCG visualizer help
# "Help",
# "Mouse Control (Arcball Mode):\n"
# " + Left button: rotate\n"
# " + Double left click: set rotation center to pointed position\n"
# " + Shift + Left: high-precision dolly.\n"
# " + (Ctrl + Shift)/(Meta) + Left: in-screen-plane rotation (RotateZ).\n"
# " + Ctrl + Left: Pan\n"
# " + Right button: Pan\n"
# " + Wheel: low-precision dolly.\n"
# " + Shift + Wheel: high-precision fov.\n"
# "Keyboard Shortcuts:\n"
# " + e(X)it: raise a KeyboardInterrupt exception to end the program directly.\n"
# " + (S)etting: show settings sidebar.\n"
# " + (A)xes: show the axes.\n"
# " + (H)istogram: show histograms if available.\n"
# " + (R)ecord: record the current camera poses (and its animation if existing).\n"
# " + (L)ight: record the sun light direction (but not saved to file).\n"
# " + (O)bject: record the object pose (but not saved to file)."
# " + (B)ackface: control whether to cull backface\n"
# " + (W)ireframe: Turn on/off the wireframe mode\n"
# " + anno(T)ation: save annotation to scene\n"
# " + F1-F12: Jump to temporary camera locations. Ctrl modifier to set.\n"
# " + +/-: increase/decrease the size of point cloud.\n"
# " + [/]: increase/decrease the fov."


if __name__ == '__main__':
    parser = exp.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='Do offline rendering')
    args = parser.parse_args()
    
    # plot 0

    # base_folder = "/home/yuepan/codes/pin-slam/bak/example_data/"
    # mesh_map = vis.from_file(os.path.join(base_folder, "mesh_poppsdolf_v1.ply"))

    # vis_map = False

    # base_folder = "/media/yuepan/Expansion/1_data/pin_slam/experiments/kitti_00_2023-11-14_23-08-45"
    # map_mesh_file = "mesh_frame_4540_24cm_pcv.ply"
    # crop_mesh_file = "crop2.ply"
    # crop_pc_file = "out_ts_crop2.ply"

    # if vis_map:
    #     traj_points = vis.from_file(os.path.join(base_folder, "slam_poses.ply"))
    #     traj_points = vis.pointcloud(traj_points, is_sphere=True, sphere_radius=0.5)
    #     mesh_map = vis.from_file(os.path.join(base_folder, "mesh", map_mesh_file))
    #     vis.show_3d([mesh_map + traj_points], [mesh_map], use_new_api=True, up_axis="+Z")
    # else:
    #     mesh_crop = vis.from_file(os.path.join(base_folder, "mesh", crop_mesh_file))
    #     pc_crop = vis.from_file(os.path.join(base_folder, "map", crop_pc_file))
    #     vis.show_3d([mesh_crop], [pc_crop], use_new_api=True, viewport_shading='NORMAL', up_axis="+Z")


    # plot 1

    # base_folder = "/media/yuepan/Expansion/1_data/pin_slam/experiments/test_ipbcar_2023-11-27_16-49-29"
    # # neural_points_feature_file = os.path.join(base_folder, "map", "out_feature_neural_points.ply")
    # # neural_points_ts_file = os.path.join(base_folder, "map", "out_ts_neural_points.ply")

    # neural_points_feature_file = os.path.join(base_folder, "map", "out_feature_cropmap_3.ply")
    # neural_points_ts_file = os.path.join(base_folder, "map", "out_ts_cropmap_3.ply")
    
    # neural_points_feature = vis.from_file(neural_points_feature_file)
    # neural_points_ts = vis.from_file(neural_points_ts_file)

    # mesh_file = os.path.join(base_folder, "mesh", "cropmap_3_mesh_25cm_rendered.ply")
    # mesh = vis.from_file(mesh_file)

    # traj_points = vis.from_file(os.path.join(base_folder, "slam_poses.ply"))
    # traj_points = vis.pointcloud(traj_points, is_sphere=True, sphere_radius=0.5)

    # vis.show_3d([neural_points_ts], [neural_points_feature], [mesh+traj_points], use_new_api=True, up_axis="+Z")

    

    base_folder = "/home/yuepan/Pictures/pin_img/example_data/dcc"

    # # base_folder = "/home/yuepan/Pictures/pin_img/example_data/forest"

    base_folder = "/media/yuepan/Expansion/1_data/pin_slam/experiments/test_ipbcar_new_2023-12-30_23-25-06/test"
    # base_folder = "/media/yuepan/Expansion/1_data/pin_slam/experiments/test_ipbcar_new_2023-12-31_12-26-25/test"

    base_folder = "/media/yuepan/Expansion/1_data/pin_slam/experiments/test_ipbcar_new_2023-12-06_23-11-16/test"

    base_folder = "/media/yuepan/Expansion/1_data/pin_slam/experiments/test_ipbcar_new_2024-01-04_15-03-52/log"

    base_folder = "/media/yuepan/Expansion/1_data/pin_slam/experiments/test_ipbcar_new_2023-12-06_23-11-16/test"

    base_folder = "/home/yuepan/Pictures/pin_img/example_data/kitti_00"

    pose_sphere_radius = 0.5

    # #scan_reg = vis.from_file(os.path.join(base_folder, "101_scan_.ply"))
    # scan_map = vis.from_file(os.path.join(base_folder, "94_scan_map.ply"))
    # #data_pool = vis.from_file(os.path.join(base_folder, "18888_training_data_pool.ply"))
    # sdf_slice = vis.from_file(os.path.join(base_folder, "94_sdf_slice.ply"))
    # cur_neural_points = vis.from_file(os.path.join(base_folder, "94_neural_points.ply"))
    # cur_sensor = vis.from_file(os.path.join(base_folder, "94_sensor_vis.ply"))
    # cur_mesh = vis.from_file(os.path.join(base_folder, "94_mesh_vis.ply"))

    # # scan_reg = vis.pointcloud(scan_reg, is_sphere=True, sphere_radius=0.15)
    # scan_map = vis.pointcloud(scan_map, is_sphere=True, sphere_radius=0.08)

    # sdf_slice = vis.pointcloud(sdf_slice, is_sphere=True, sphere_radius=0.1)

    # cur_neural_points = vis.pointcloud(cur_neural_points, is_sphere=True, sphere_radius=0.08)


    odom_pose = vis.from_file(os.path.join(base_folder, "kitti_00_odom_poses_new.ply"))
    slam_pose = vis.from_file(os.path.join(base_folder, "kitti_00_slam_poses_new.ply"))

    odom_pose = vis.pointcloud(odom_pose, is_sphere=True, sphere_radius=pose_sphere_radius)
    slam_pose = vis.pointcloud(slam_pose, is_sphere=True, sphere_radius=pose_sphere_radius)

    # odom_mesh = vis.from_file(os.path.join(base_folder, "kitti_00_mesh_24cm_odom.ply"))
    # slam_mesh = vis.from_file(os.path.join(base_folder, "kitti_00_mesh_24cm_slam.ply"))

    mesh_slam_crop1 = vis.from_file(os.path.join(base_folder, "slam_crop1_20cm_pcv.ply"))
    mesh_slam_crop2 = vis.from_file(os.path.join(base_folder, "slam_crop2_20cm_pcv.ply"))
    
    mesh_odom_crop1 = vis.from_file(os.path.join(base_folder, "odom_crop1_20cm_pcv.ply"))
    mesh_odom_crop2 = vis.from_file(os.path.join(base_folder, "odom_crop2_20cm_pcv.ply"))


    # neural_points = vis.from_file(os.path.join(base_folder, "neural_points_ipb.ply"))
    # mesh_show = vis.from_file(os.path.join(base_folder, "mesh_poppsdolf_v1.ply"))
    # sensor_show = vis.from_file(os.path.join(base_folder, "ipb_car.ply"))

    # mesh_loop = vis.from_file(os.path.join(base_folder, "forest_30cm.ply"))

    # mesh_loop = vis.from_file(os.path.join(base_folder, "mesh_30cm_pcv.ply"))

    # gt_pcd = vis.from_file(os.path.join(base_folder,"ValentineCaveGT_5_error_0_50cm.ply"))

    # mesh_loop_2 = vis.from_file(os.path.join(base_folder, "ncd_02_30cm.ply"))

    # neural_points = vis.pointcloud(neural_points, is_sphere=True, sphere_radius=0.08)

    # neural_points =  vis.from_file(os.path.join(base_folder, "crop1.ply"))
    # neural_points = vis.pointcloud(neural_points, is_sphere=True, sphere_radius=0.1)
    # neural_points = vis.transparent(neural_points, 0.1)

    # poses_slam = vis.from_file(os.path.join(base_folder, "slam_poses.ply"))
    # poses_slam = vis.pointcloud(poses_slam, is_sphere=True, sphere_radius=pose_sphere_radius)
    
    vis.show_3d([odom_pose+mesh_odom_crop1], [slam_pose+mesh_slam_crop1], use_new_api=True, up_axis="+Z")
    vis.show_3d([odom_pose+mesh_odom_crop2], [slam_pose+mesh_slam_crop2], use_new_api=True, up_axis="+Z")

    # vis.show_3d([odom_pose+odom_mesh], [slam_pose+slam_mesh], use_new_api=True, up_axis="+Z")
    # vis.show_3d([neural_points+sensor_show], [mesh_show+sensor_show], use_new_api=True, up_axis="+Z")
    
    #vis.show_3d([scan_map+cur_sensor], [sdf_slice+cur_sensor], [cur_mesh+cur_sensor], [cur_neural_points+cur_sensor], use_new_api=True, up_axis="+Z")

    # vis.show_3d([scan_map+cur_mesh+cur_sensor], use_new_api=True, up_axis="+Z")

    # vis.show_3d([scan_reg+cur_sensor], [scan_map+cur_sensor], [data_pool], [cur_mesh+cur_sensor], use_new_api=True, up_axis="+Z")

    # vis.show_3d([scan_reg], [scan_map], [data_pool], [cur_sensor], [cur_mesh], use_new_api=True, up_axis="+Z")

    # vis.show_3d([mesh_loop+poses_slam], use_new_api=True, up_axis="+Z")

    # vis.show_3d([neural_points], use_new_api=True, up_axis="+Z")
    # vis.show_3d([gt_pcd], use_new_api=True, up_axis="+Z")

    # mesh_no_loop = vis.from_file(os.path.join(base_folder, "kitti_00_mesh_24cm_no_loop.ply"))
    # poses_no_loop = vis.from_file(os.path.join(base_folder, "kitti_00_odom_poses_new.ply"))
    # poses_no_loop = vis.pointcloud(poses_no_loop, is_sphere=True, sphere_radius=pose_sphere_radius)

    # vis.show_3d([mesh_loop+poses_slam], [mesh_no_loop+poses_no_loop], use_new_api=True, up_axis="+Z")


    # neural_points = vis.from_file(os.path.join(base_folder, "neural_points_map_v4.ply"))
    # vis.show_3d([neural_points], use_new_api=True, up_axis="+Z")

    # neural_points = vis.from_file(os.path.join(base_folder, "neural_points_ipb_2.ply"))
    # mesh = vis.from_file(os.path.join(base_folder, "mesh_poppsdolf_v1.ply"))

    # mesh = vis.from_file(os.path.join(base_folder, "math_unibonn2.ply"))
    # vis.show_3d([mesh], use_new_api=True, up_axis="+Z")

    # vis.show_3d([neural_points], [mesh], use_new_api=True, up_axis="+Z")

    # mesh = vis.from_file("./bak/example_data/mesh_ipb.ply")
    # # mesh = vis.colored_mesh(mesh, ucid=0)

    # sensor_origin= np.array([29.51, 1.66, -0.21])
    # picked_point=np.array([23.7, -17.4, 0.96])
    # ray = vis.arrow(sensor_origin, picked_point)

    # # axis1 = vis.frame()
    # car_mesh = vis.from_file("./bak/example_data/ipb_car.ply")
    # # car_mesh = vis.colored_mesh(car_mesh, ucid=5)
    # neural_points = vis.pointcloud(neural_points, is_sphere=True, sphere_radius=0.05)
    # # mesh2 = vis.transparent(mesh)

    # vis.show_3d([neural_points + car_mesh + ray], [mesh + car_mesh], use_new_api=True)



    # plot 2

    # neural_points_xyz = np.random.rand(6,3)
    # neural_points_color = np.random.rand(6,3)
    # neural_points_o3d = o3d.geometry.PointCloud()
    # neural_points_o3d.points = o3d.utility.Vector3dVector(neural_points_xyz)
    # neural_points_o3d.colors = o3d.utility.Vector3dVector(neural_points_color)

    # # neural_points_example = vis.from_file("./bak/example_data/example_points.ply")
    # # neural_points_xyz = np.asarray(neural_points_example.points)
    # # print(neural_points_xyz)

    # axis_size = 0.2
    # axis1 = vis.frame(vis.Isometry(None, neural_points_xyz[0]), size = axis_size)
    # axis2 = vis.frame(vis.Isometry(None, neural_points_xyz[1]), size = axis_size)
    # axis3 = vis.frame(vis.Isometry(None, neural_points_xyz[2]), size = axis_size)
    # axis4 = vis.frame(vis.Isometry(None, neural_points_xyz[3]), size = axis_size)
    # axis5 = vis.frame(vis.Isometry(None, neural_points_xyz[4]), size = axis_size)
    # axis6 = vis.frame(vis.Isometry(None, neural_points_xyz[5]), size = axis_size)

    # query_point_xyz = np.random.rand(1,3)
    # print(query_point_xyz)
    # query_point_color = np.ones((1,3))*0.1

    # query_points_o3d = o3d.geometry.PointCloud()
    # query_points_o3d.points = o3d.utility.Vector3dVector(query_point_xyz)
    # query_points_o3d.colors = o3d.utility.Vector3dVector(query_point_color)

    # # print(query_point)
    # # query_point = vis.pointcloud(query_point)

    # neural_points_example = vis.pointcloud(neural_points_o3d, is_sphere=True, sphere_radius=0.05)
    # query_point_example = vis.pointcloud(query_points_o3d, is_sphere=True, sphere_radius=0.04)
    # vis.show_3d([neural_points_example + query_point_example + axis1 + axis2 + axis3 + axis4 + axis5 + axis6], use_new_api=True)



    # plot for zhong2024cvpr
    
    # NCD static reconstruction
    # base_folder = "/home/yuepan/codes/pin-slam/bak/example_data/cvpr_plot/ncd"
    # # mesh_4d = vis.from_file(os.path.join(base_folder, "ours_4d.ply"))
    # # mesh_nksr = vis.from_file(os.path.join(base_folder, "nksr.ply"))
    # # mesh_vdb = vis.from_file(os.path.join(base_folder, "vdb_fusion.ply"))
    # # mesh_shine = vis.from_file(os.path.join(base_folder, "shine.ply"))

    # # mesh_4d = vis.from_file(os.path.join(base_folder, "ours_4d_v2.ply"))
    # # mesh_nksr = vis.from_file(os.path.join(base_folder, "nksr_v2.ply"))
    # # mesh_vdb = vis.from_file(os.path.join(base_folder, "vdb_fusion_v2.ply"))
    # # mesh_shine = vis.from_file(os.path.join(base_folder, "shine_v2.ply"))

    # mesh_4d = vis.from_file(os.path.join(base_folder, "ours_4d_v3.ply"))
    # mesh_nksr = vis.from_file(os.path.join(base_folder, "nksr_v3.ply"))
    # mesh_vdb = vis.from_file(os.path.join(base_folder, "vdb_fusion_v3.ply"))
    # mesh_shine = vis.from_file(os.path.join(base_folder, "shine_v3.ply"))

    # pc_merged = vis.from_file(os.path.join(base_folder, "merged_point_cloud_v2.ply"))   

    # # toy car static reconstruction
    # base_folder = "/home/yuepan/codes/pin-slam/bak/example_data/cvpr_plot/toy_car"

    # mesh_4d = vis.from_file(os.path.join(base_folder, "est_ours4d.ply"))
    # mesh_nksr = vis.from_file(os.path.join(base_folder, "est_nksr_mesh.ply"))
    # mesh_vdb = vis.from_file(os.path.join(base_folder, "est_vdbfusion.ply"))
    # mesh_shine = vis.from_file(os.path.join(base_folder, "est_shine.ply"))  
    # pc_merged = vis.from_file(os.path.join(base_folder, "toycar_pc.ply"))         

    # # vis.show_3d([mesh_4d], use_new_api=True)
    # vis.show_3d([pc_merged], [mesh_4d], [mesh_nksr], [mesh_vdb], [mesh_shine], use_new_api=True)


    # dynamic removal benchmark
    # sphere_radius = 0.05

    # base_folder = "/media/yuepan/Expansion/1_data/4d_sdf/teaser" # output_00, 05, av2, teaser
    # # gt_pc = vis.from_file(os.path.join(base_folder, "gt_cloud.ply"))
    # # gt_dynamic = vis.from_file(os.path.join(base_folder, "gt_cloud_dynamic.ply"))
    # # gt_static = vis.from_file(os.path.join(base_folder, "gt_cloud_static.ply"))

    # input_01 = vis.from_file(os.path.join(base_folder, "input_points.ply"))

    # scan_02 = vis.from_file(os.path.join(base_folder, "scan2_black.ply"))
    # mesh_02 = vis.from_file(os.path.join(base_folder, "dynamic_mesh2.ply"))

    # scan_03 = vis.from_file(os.path.join(base_folder, "scan2_color.ply"))
    # mesh_03 = vis.from_file(os.path.join(base_folder, "static_mesh.ply"))
    
    # # input_01 = vis.pointcloud(input_01, is_sphere=True, sphere_radius=sphere_radius) #
    # scan_02 = vis.pointcloud(scan_02, is_sphere=True, sphere_radius=sphere_radius)
    # scan_03 = vis.pointcloud(scan_03, is_sphere=True, sphere_radius=sphere_radius)
    
    # # scan_dynamic = vis.from_file(os.path.join(base_folder, "dynamic_scan.ply"))
    # # scan_static = vis.from_file(os.path.join(base_folder, "static_scan.ply"))

    # # scan_dynamic = vis.pointcloud(scan_dynamic, is_sphere=True, sphere_radius=sphere_radius)
    # # scan_static = vis.pointcloud(scan_static, is_sphere=True, sphere_radius=sphere_radius)

    # # mesh_frame = vis.from_file(os.path.join(base_folder, "mesh_single_frame.ply"))

    # # ours = vis.from_file(os.path.join(base_folder, "4d_output.ply"))
    # # erasor = vis.from_file(os.path.join(base_folder, "erasor_output.ply"))
    # # octomap = vis.from_file(os.path.join(base_folder, "octomapfg_output.ply"))
    # # removert = vis.from_file(os.path.join(base_folder, "removert_output.ply"))

    # #vis.show_3d([removert], use_new_api=True,  up_axis="+Z")
    
    # # vis.show_3d([gt_static], [ours], [erasor], use_new_api=True,  up_axis="+Z")
    # # vis.show_3d([gt_static+gt_dynamic], use_new_api=True,  up_axis="+Z")
    # # vis.show_3d([gt_pc], use_new_api=True,  up_axis="+Z")

    # # vis.show_3d([scan_dynamic+scan_static+mesh_frame], use_new_api=True,  up_axis="+Z")
    # vis.show_3d([input_01], [scan_02+mesh_02], [scan_03], [mesh_03], use_new_api=True,  up_axis="+Z")

    # NOTE: Use F1 - F7 to switch to some preset viewpoints
    # NOTE: + - to adjust point size


    # vis.show_3d([gt_static+gt_dynamic], use_new_api=True,  up_axis="+Z") # , [octomap], [removert]
    # vis.show_3d([gt_static+gt_dynamic], [ours], [erasor], use_new_api=True,  up_axis="+Z")
    # vis.show_3d([ours], [erasor], use_new_api=True,  up_axis="+Z")
    # vis.show_3d([octomap], [removert], use_new_api=True,  up_axis="+Z")
        
    
    # scene rendering using blender
    # scene = render.Scene(up_axis='+Y').add_object(mesh_4d)
    # scene.quick_camera(w=600, h=600, plane_angle=280.0)
    # scene.preview(use_new_api=True)
    
    # scene = vis.show_3d([mesh_4d], show=False).preview(use_new_api=True)

    # # Render using NKSR style
    # render.ThemeNKSR(need_plane=True).apply_to(scene)
    # nksr_rendering = scene.render_blender()
    # nksr_rendering = image.alpha_compositing(
    #     image.gamma_transform(nksr_rendering, alpha_only=True, gamma=3.0),
    #     image.solid(nksr_rendering.shape[1], nksr_rendering.shape[0]))
    
    # image.show(nksr_rendering)


    # vis.show_3d([mesh_geom + pc_geom], use_new_api=True)

    # vis.show_3d([bunny_geom], [bunny_geom2], use_new_api=True, show=not args.offline)

    # Render bunny using different renderers.
    # s1 = vis.show_3d([bunny_geom], show=False)
    # img_filament = s1.render_filament()
    
    # if args.offline:
    #     image.write(img_filament, "out/offline.png")
    #     exit()
    
    # img_opengl = s1.render_opengl()
    # image.show(img_filament, img_opengl, subfig_size=4)