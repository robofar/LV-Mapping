


def gs_eval_offline(self, dataset, neural_points, decoders, used_poses,
                        q_main2vis=None, q_vis2main=None, 
                        eval_down_rate=0, skip_end_count: int = 0, 
                        sorrounding_map_radius = None,
                        per_cam_exposure_ab = None,
                        lpips_eval_on: bool = False,
                        pc_cd_eval_on: bool = False):
        
        # NOTE: there are some randomness of Guassian Splatting's optimization even with random seed fixed
        # This is mainly due to the randomness in GPU schedule in the differentiable rasterizer (according to the author of 3DGS)
        # For PSNR, it may have a difference of 0.1-0.2 PSNR

        # TODO: the memory bank may still have some problem

        assert self.config.use_dataloader, "Only data loader version is supported currently"

        eval_cam_name = self.dataset.cam_names # use all the cams
        # eval_cam_name = [self.dataset.loader.main_cam_name] # front cam

        with torch.no_grad():
            
            self.record_per_cam_exposure()

            if sorrounding_map_radius is not None:
                self.neural_points.sorrounding_map_radius = sorrounding_map_radius

            background = torch.tensor(self.config.bg_color, dtype=self.dtype, device=self.device)
            bg_3d = background.view(3, 1, 1)

            eval_down_scale = 2**(eval_down_rate)

            # skip_end_count means that we will skip the last n frames because the incremental mapping haven't done much mapping in such areas
            for frame_id in tqdm(range(0, self.dataset.processed_frame - skip_end_count, 1), desc="GS evaluation"):

                remove_gpu_cache()
                
                T_w_l = self.used_poses[frame_id] #
                
                if frame_id % 100 == 0:
                    self.neural_points.recreate_hash(T_w_l[:3,3], kept_points=True, with_ts=True, cur_ts=frame_id) # and at the same time reset local map
                else:
                    self.neural_points.reset_local_map(T_w_l[:3,3], cur_ts=frame_id)
                
                neural_points_data, sorrounding_neural_points_data = self.neural_points.gather_local_data()

                sorrounding_spawn_results = spawn_gaussians(sorrounding_neural_points_data, 
                    self.decoders, None, T_w_l[:3,3],
                    dist_concat_on=self.config.dist_concat_on, 
                    view_concat_on=self.config.view_concat_on, 
                    scale_filter_on=True,
                    z_far=self.config.sorrounding_map_radius,
                    learn_color_residual=self.config.learn_color_residual,
                    gs_type=self.config.gs_type)

                # load the cam datas to cur_cam_img
                self.dataset.read_frame_with_loader(frame_id, init_pose = False, use_image=True, monodepth_on=self.config.monodepth_on) # because we want to use the sky mask here

                # crop frames and possibly do LiDAR intrinsic corrections
                self.dataset.filter_and_correct()

                # deskew and reset depth map
                if self.config.deskew and frame_id > 0:
                    self.dataset.deskew_at_frame(frame_id)
                
                self.dataset.project_pointcloud_to_cams(use_only_colorized_points=True) # self.config.learn_color_residual)

                if pc_cd_eval_on:
                    cur_frame_measured_pcd_o3d = o3d.geometry.PointCloud()

                    cur_frame_measured_xyz_np = (
                        self.dataset.cur_point_cloud_torch[:,:3].detach().cpu().numpy().astype(np.float64)
                    )

                    cur_frame_measured_color_np = (
                        self.dataset.cur_point_cloud_torch[:,3:].detach().cpu().numpy().astype(np.float64)
                    )
                    cur_frame_measured_pcd_o3d.points = o3d.utility.Vector3dVector(cur_frame_measured_xyz_np)
                    cur_frame_measured_pcd_o3d.colors = o3d.utility.Vector3dVector(cur_frame_measured_color_np)

                    cur_frame_rendered_pcd_o3d = o3d.geometry.PointCloud()

                for cam_name in self.dataset.cam_names:

                    K_mat = self.dataset.K_mats[cam_name]
                    T_c_l_np = self.dataset.T_c_l_mats[cam_name]
                    T_c_l = torch.tensor(T_c_l_np, device=self.device) 
                    height = self.dataset.cam_heights[cam_name]
                    width = self.dataset.cam_widths[cam_name] 

                    cur_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()

                    cur_intrinsic_o3d.set_intrinsics(
                                    height=int(height/eval_down_scale),
                                    width=int(width/eval_down_scale),
                                    fx=K_mat[0,0]/eval_down_scale,
                                    fy=K_mat[1,1]/eval_down_scale,
                                    cx=K_mat[0,2]/eval_down_scale,
                                    cy=K_mat[1,2]/eval_down_scale)


                    T_w_c = T_w_l @ T_c_l.inverse() # need to convert to cam frame

                    # you need to also load the camera exposure coefficients here
                    cur_view_cam: CamImage = self.dataset.cur_cam_img[cam_name]
                    cur_view_cam.set_pose(T_w_c)
                    
                    cur_uid = cur_view_cam.uid
                    cur_cam_id = cur_view_cam.cam_id # cam_name
                    cur_frame_id = cur_view_cam.frame_id # frame_id

                    # find the closest train view exposure
                    if self.config.exposure_correction_on:
                        closest_train_frame_id = min(self.per_cam_exposure_ab[cur_cam_id].keys(), key=lambda k: (abs(k - cur_frame_id), k))
                        cur_exposure = (self.per_cam_exposure_ab[cur_cam_id])[closest_train_frame_id]
                        cur_view_cam.set_exposure(cur_exposure[0], cur_exposure[1])
                        
                        # print(cur_view_cam)

                    if cam_name in eval_cam_name:

                        # current values
                        render_pkg = render(cur_view_cam, None, neural_points_data, 
                            self.decoders, sorrounding_spawn_results, background, 
                            down_rate=eval_down_rate, 
                            dist_concat_on=self.config.dist_concat_on, 
                            view_concat_on=self.config.view_concat_on, 
                            correct_exposure=self.config.exposure_correction_on, 
                            learn_color_residual=self.config.learn_color_residual,
                            front_only_on=self.config.train_front_only,
                            gs_type=self.config.gs_type,
                            min_alpha=self.config.min_alpha)

                        # rendered results
                        rendered_rgb_image, rendered_depth = render_pkg["render"], render_pkg["surf_depth"] # 3, H, W / 1, H, W

                        rendered_rgb_image = torch.clamp(rendered_rgb_image, 0, 1)
                        # print(torch.max(rendered_rgb_image), torch.min(rendered_rgb_image)) # why there are value larger than 1?

                        gt_rgb_img = cur_view_cam.rgb_image_list[eval_down_rate]

                        if cur_view_cam.sky_mask_on:
                            # mask the sky part for eval
                            cur_sky_mask = cur_view_cam.sky_mask_list[eval_down_rate] # still torch
                            mask_broadcasted = cur_sky_mask.repeat(3,1,1)
                            gt_rgb_img[mask_broadcasted] = bg_3d.expand_as(gt_rgb_img)[mask_broadcasted]

                        if cam_name == "rear": # only for ipb car dataset (FIXME), use mask in the future, now it's just a ugly quick fix
                            pixel_h_used = int(910/1024*gt_rgb_img.shape[1])
                        elif cam_name == "front":
                            pixel_h_used = int(990/1024*gt_rgb_img.shape[1])
                        else:  
                            pixel_h_used = -1

                        rendered_rgb_image_for_eval = rendered_rgb_image[:,:pixel_h_used,:]
                        gt_rgb_image_for_eval = gt_rgb_img[:,:pixel_h_used,:]

                        cur_psnr = psnr(rendered_rgb_image_for_eval, gt_rgb_image_for_eval).mean().item()
                        cur_ssim = fused_ssim(rendered_rgb_image_for_eval.unsqueeze(0), gt_rgb_image_for_eval.unsqueeze(0), train=False).item()
                        # cur_ssim = ssim(rendered_rgb_image_for_eval, gt_rgb_image_for_eval).item()
                        if lpips_eval_on:
                            cur_lpips = self.lpips(rendered_rgb_image_for_eval.unsqueeze(0), gt_rgb_image_for_eval.unsqueeze(0)).item()
                        else:
                            cur_lpips = -1.0 # not available

                        if not self.silence:
                            print("Camera id: {}".format(cur_view_cam.uid))
                            print("Current view PSNR  ↑ :", f"{cur_psnr:.3f}")
                            print("Current view SSIM  ↑ :", f"{cur_ssim:.3f}")
                            print("Current view LPIPS ↓ :", f"{cur_lpips:.3f}")
                            if self.config.exposure_correction_on:
                                print("Current view exposure coefficients {:.3f}, {:.3f}".format(cur_exposure[0].item(), cur_exposure[1].item()))

                        if cur_view_cam.depth_on and rendered_depth is not None: 
                            eval_depth_max = self.config.max_range * 0.8
                            eval_depth_min = self.config.min_range
                            gt_depth_img = cur_view_cam.depth_image_list[eval_down_rate] # torch.tensor
                            depth_valid_mask = (gt_depth_img > eval_depth_min) & (rendered_depth > eval_depth_min) & (gt_depth_img < eval_depth_max) & (rendered_depth < eval_depth_max)
                            diff_depth = torch.abs(gt_depth_img - rendered_depth) # already abs
                            # diff_depth[~depth_valid_mask] = 0.0
                            diff_depth_masked = diff_depth[depth_valid_mask].detach().cpu().numpy()
                            cur_depth_l1 = np.mean(diff_depth_masked)
                            cur_depth_rmse = np.sqrt(np.mean(diff_depth_masked**2))
                            if not self.silence:
                                print("Current view Depth L1 (m) ↓ :", f"{cur_depth_l1:.3f}")
                                print("Current view Depth RMSE (m) ↓ :", f"{cur_depth_rmse:.3f}")


                            if pc_cd_eval_on: 

                                rendered_rgb_np = (rendered_rgb_image * 255).byte().permute(1, 2, 0).detach().contiguous().cpu().numpy().astype(np.uint8) 
                                rgb_img_o3d = o3d.geometry.Image(rendered_rgb_np)

                                rendered_depth_np = rendered_depth.detach().cpu().numpy().astype(np.float32) 
                                rendered_depth_np = np.transpose(rendered_depth_np, (1, 2, 0))

                                depth_img_o3d = o3d.geometry.Image(rendered_depth_np)

                                cur_rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, 
                                                                                        depth_img_o3d, 
                                                                                        depth_scale=1.0, 
                                                                                        depth_trunc=eval_depth_max, 
                                                                                        convert_rgb_to_intensity=False)

                                cur_cam_rendered_pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(
                                                                    cur_rgbd_o3d, 
                                                                    cur_intrinsic_o3d, 
                                                                    T_c_l_np)

                                cur_frame_rendered_pcd_o3d += cur_cam_rendered_pcd_o3d # already under lidar frame


                        if cur_view_cam.uid in self.train_cam_uid:
                            # as train views
                            if not self.silence:
                                print("Evalualted as a train view")
                            self.train_psnr_list.append(cur_psnr)
                            self.train_ssim_list.append(cur_ssim)
                            self.train_lpips_list.append(cur_lpips)
                            if cur_view_cam.depth_on and rendered_depth is not None: 
                                self.train_depthl1_list.append(cur_depth_l1)
                                self.train_depth_rmse_list.append(cur_depth_rmse)
                        
                        else:
                            # as test views
                            if not self.silence:
                                print("Evaluated as a test view")
                            self.test_psnr_list.append(cur_psnr)
                            self.test_ssim_list.append(cur_ssim)
                            self.test_lpips_list.append(cur_lpips)
                            if cur_view_cam.depth_on and rendered_depth is not None: 
                                self.test_depthl1_list.append(cur_depth_l1)
                                self.test_depth_rmse_list.append(cur_depth_rmse)

                # compute cd with regards to original lidar pc
                if pc_cd_eval_on: 
                    cd_metrics = eval_pair(cur_frame_rendered_pcd_o3d, cur_frame_measured_pcd_o3d, 
                        down_sample_res=0.05, threshold=0.1, 
                        truncation_acc=1.0, truncation_com=1.0) # FIXME
                    
                    cur_cd = cd_metrics['Chamfer_L1 (m)']
                    cur_f1 = cd_metrics['F-score (%)']

                    if not self.silence:
                        print("Current frame Chamfer Distance L1 (m) ↓ :", f"{cur_cd:.3f}")
                        print("Current frame F1-score (%) ↑ :", f"{cur_f1:.3f}")

                    if cur_view_cam.uid in self.train_cam_uid:
                        self.train_cd_list.append(cur_cd)
                        self.train_f1_list.append(cur_f1)
                    else:
                        self.test_cd_list.append(cur_cd)
                        self.test_f1_list.append(cur_f1)

                if q_main2vis is not None:
                    # add the eval frame to vis
                    
                    packet_to_vis= VisPacket(frame_id=frame_id,
                        current_frames=self.dataset.cur_cam_img, 
                        img_down_rate=self.config.gs_vis_down_rate)
                    
                    packet_to_vis.add_neural_points_data(self.neural_points)

                    if pc_cd_eval_on: 
                        packet_to_vis.add_scan(np.array(cur_frame_rendered_pcd_o3d.points, dtype=np.float64), np.array(cur_frame_rendered_pcd_o3d.colors, dtype=np.float64))

                    odom_poses, gt_poses, pgo_poses = self.dataset.get_poses_np_for_vis(frame_id)
                    packet_to_vis.add_traj(odom_poses, gt_poses, pgo_poses)

                    q_main2vis.put(packet_to_vis)

                if q_vis2main is not None:
                    if not q_vis2main.empty():
                        while q_vis2main.get().flag_pause:
                            continue