import open3d as o3d
import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import torchvision.utils as vutils

from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
from collections import defaultdict



class Dynamic():
    def __init__(self, cam_names, config):
        # Dict of dicts (for each camera). Then values in these dics are torch.tensors
        self.instances_pcd = {} # Contains PCD per instance ID over all frames. Later you can merge it to get full point cloud. 0 is background. >=1 are objects. (do not use -1, it is outside of FOV)
        for cam_name in cam_names:
            self.instances_pcd[cam_name] = {} # if it is empty dict for cam_name, that means I did not create foundation masks for that cam_name

        self.outside_fov_pcd = None # Contains PCD outside of the FOV of all cameras together (torch.tensor)
        
        # This is just for saving result (True or False) whether instance is dynamic or not
        self.dynamic_instance = {}
        for cam_name in cam_names:
            self.dynamic_instance[cam_name] = {}

        self.config = config

    def get_instance_pcd(self, instance_id, cam_name):
        return self.instances_pcd[cam_name][instance_id]
    
    def get_instance_pcd_o3d(self, instance_id, cam_name):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.instances_pcd[cam_name][instance_id][:,:3].detach().cpu().numpy().astype(np.float64))
        pcd_o3d.colors = o3d.utility.Vector3dVector(self.instances_pcd[cam_name][instance_id][:,3:6].detach().cpu().numpy().astype(np.float64))
        
        timestamps = self.instances_pcd[cam_name][instance_id][:, 6].unsqueeze(1)#.detach().cpu()  # [N]
        
        return pcd_o3d, timestamps
    
    def get_outside_fov_pcd(self):
        return self.outside_fov_pcd
    
    def get_outside_fov_pcd_o3d(self):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.outside_fov_pcd[:,:3].detach().cpu().numpy().astype(np.float64))
        pcd_o3d.colors = o3d.utility.Vector3dVector(self.outside_fov_pcd[:,3:6].detach().cpu().numpy().astype(np.float64))

        timestamps = self.outside_fov_pcd[:, 6].unsqueeze(1)#.detach().cpu()  # [N]

        return pcd_o3d, timestamps
    


    #############################################################
    




    def append_outside_fov_points(self, points: torch.Tensor, outside_fov_mask: torch.Tensor, points_timestamps: torch.Tensor):
        if outside_fov_mask.any():
            outside_points = points[outside_fov_mask]                  # [M, 6]
            outside_timestamps = points_timestamps[outside_fov_mask]  # [M, 1]

            # Concatenate [x, y, z, r, g, b, t] → [M, 7]
            extended_points = torch.cat([outside_points, outside_timestamps], dim=1)

            if self.outside_fov_pcd is None:
                self.outside_fov_pcd = extended_points
            else:
                self.outside_fov_pcd = torch.cat([self.outside_fov_pcd, extended_points], dim=0)






    def append_instance_points(
        self,
        points: torch.Tensor,                     # [N, 6]
        instance_ids: torch.Tensor,               # [N, 1]
        unique_instance_ids: torch.Tensor,        # [M]
        cam_name,
        ground_mask: torch.Tensor,                # [N]
        points_timestamps: torch.Tensor           # [N, 1]
    ):
        for instance_id in unique_instance_ids:
            if instance_id == -1:
                continue

            mask = instance_ids.squeeze(1) == instance_id
            matched_points = points[mask]                          # [K, 6]
            matched_timestamps = points_timestamps[mask]          # [K, 1]
            matched_points = torch.cat([matched_points, matched_timestamps], dim=1)  # [K, 7]

            if instance_id > 0:
                if self.config.use_ground_segmentation:
                    instance_ground_mask = ground_mask[mask]       
                    ground_points = matched_points[instance_ground_mask]
                    matched_points = matched_points[~instance_ground_mask]

                    ground_points_o3d = o3d.geometry.PointCloud()
                    ground_points_o3d.points = o3d.utility.Vector3dVector(ground_points[:, :3].detach().cpu().numpy())
                    ground_points_o3d.colors = o3d.utility.Vector3dVector(ground_points[:, 3:6].detach().cpu().numpy())

                    self.append_to_background(cam_name, ground_points_o3d, ground_points[:, 6].unsqueeze(1))  # Send per-point timestamps

            instance_id_key = instance_id.item()
            if instance_id_key in self.instances_pcd[cam_name]:
                self.instances_pcd[cam_name][instance_id_key] = torch.cat(
                    [self.instances_pcd[cam_name][instance_id_key], matched_points], dim=0
                )
            else:
                self.instances_pcd[cam_name][instance_id_key] = matched_points


    
    #############################################################
    


    # Debugging dynamic instances
    def estimate_eps(self, pcd, k=10, percentile=0.999, eps_scale=1.3):
        points = np.asarray(pcd.points)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(points)
        distances, _ = neigh.kneighbors(points)
        distances = np.sort(distances[:, -1])  # Sort distances to kth neighbor

        # Choose 'eps' based on the elbow point
        eps_value = distances[int(len(distances) * percentile)] * eps_scale
        #eps_value = distances[-1] # not good. largesst distance can be too large, then clusters might be too big

        '''
        # Plot the k-distance graph
        plt.plot(distances, label="Sorted k-th NN Distances")
        plt.axhline(y=eps_value, color='r', linestyle='--', label=f"{percentile}th Percentile (eps={eps_value:.3f})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {k}-th nearest neighbor")
        plt.legend()
        plt.title("K-Distance Graph with Percentile Line")
        plt.show()'
        '''
        
        

        return eps_value
    

    def get_bounding_box_properties(self, pcd, use_obb=True):
        """
        Computes bounding box properties: aspect ratio, volume, and point count.
        """
        bbox = pcd.get_oriented_bounding_box() if use_obb else pcd.get_axis_aligned_bounding_box()
        extents = bbox.extent  # Width, height, length of the bounding box

        aspect_ratio = max(extents[0], extents[1]) / min(extents[0], extents[1]) # x and y only (z not important)
        volume = np.prod(extents)  # Bounding box volume
        #point_count = len(pcd.points)
        point_count = len(pcd.voxel_down_sample(voxel_size=0.25).points)

        bbox.color = (0, 1, 0)  # RGB: Green color

        return aspect_ratio, volume, point_count, bbox

    def voxel_downsample_2d(self, points_2d, voxel_size=0.2):
        """
        Convert 2D points to 3D, voxel downsample using Open3D, then return back to 2D.
        """
        # Convert to Nx3 by padding with zeros
        points_3d = np.hstack((points_2d, np.zeros((points_2d.shape[0], 1))))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        downsampled_3d = np.asarray(pcd.points)
        return downsampled_3d[:, :2]  # Convert back to 2D

    def voxel_downsample_ordered(self, points_2d, voxel_size=0.25):
        """
        Voxel downsampling for 2D trajectory points.
        Keeps only the first point that falls into each voxel.
        Preserves the original order of points.
        """
        voxel_grid = {}
        downsampled_points = []

        for pt in points_2d:
            voxel_idx = tuple((pt // voxel_size).astype(int))
            if voxel_idx not in voxel_grid:
                voxel_grid[voxel_idx] = True
                downsampled_points.append(pt)

        return np.array(downsampled_points)

    

    def compute_curvature(self, points_2d):
        """
        Compute total bending per distance traveled across the cluster.
        """
        if points_2d.shape[0] < 3:
            return 0.0

        diffs = points_2d[1:] - points_2d[:-1]
        distances = np.linalg.norm(diffs, axis=1)
        distances = np.clip(distances, 1e-6, None)

        directions = diffs / distances[:, np.newaxis]

        dot_products = (directions[1:] * directions[:-1]).sum(axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)

        total_angle_change = np.sum(np.abs(angles))  # In radians
        total_distance = np.sum(distances)

        if total_distance < 1e-6:
            return 0.0

        curvature_strength = total_angle_change / total_distance  # radians per meter

        # Weight by total distance to prefer longer traces
        curvature_score = curvature_strength * total_distance

        return curvature_score






    def find_car_trace_cluster_centroid(self, pcd, points_timestamp, labels):
        unique_labels = np.unique(labels[labels >= 0])  # Ignore noise (-1)
        
        largest_cluster_N = -1
        largest_cluster_indices = None

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > largest_cluster_N:
                largest_cluster_N = len(cluster_indices)
                largest_cluster_indices = cluster_indices

        largest_cluster_pcd = pcd.select_by_index(largest_cluster_indices)
        largest_cluster_timestamps = points_timestamp[largest_cluster_indices]
        largest_cluster_points_np = np.asarray(largest_cluster_pcd.points)
        largest_cluster_timestamps = largest_cluster_timestamps.cpu().numpy().astype(np.int32)

        # Group points by timestamp
        
        timestamp_points = defaultdict(list)
        for pt, t in zip(largest_cluster_points_np, largest_cluster_timestamps):
            timestamp_points[int(t)].append(pt)

        centroids = []
        timestamps_sorted = sorted(timestamp_points.keys())
        for t in timestamps_sorted:
            pts = np.array(timestamp_points[t])
            centroid = np.mean(pts, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)  # (M, 3)

        # Visualize
        #o3d.visualization.draw_geometries([largest_cluster_pcd, largest_cluster_pcd.get_axis_aligned_bounding_box()])

        # poly in world frame

        # Fit polynomial
        poly_coeffs_w = np.polyfit(centroids[:,0], centroids[:,1], deg=2)
        curvature_strength_w = np.abs(poly_coeffs_w[0])

        # Uniform sampling in x-space
        x_sample_w = np.linspace(np.min(centroids[:,0]), np.max(centroids[:,0]), 100)
        y_sample_w = np.polyval(poly_coeffs_w, x_sample_w)

        # Compute distances
        sampled_points_w = np.stack([x_sample_w, y_sample_w], axis=1)
        diffs_poly_w = sampled_points_w[1:] - sampled_points_w[:-1]
        distances_poly_w = np.linalg.norm(diffs_poly_w, axis=1)
        smoothed_length_w = np.sum(distances_poly_w)

        # Print results
        print(f"Smoothed trace length (world frame): {smoothed_length_w:.3f} meters")
        print(f"Curvature strength (world frame): {curvature_strength_w:.6f}")
        ################################

        # PCA
        centroids_xy = centroids[:, :2]  # Only X, Y
        pca = PCA(n_components=2)
        centroids_xy_rot = pca.fit_transform(centroids_xy) # rotated centroids

        # x and y
        x_vals = centroids_xy_rot[:, 0]
        y_vals = centroids_xy_rot[:, 1]

        # Normalize x before fitting
        x_min = np.min(x_vals)
        x_max = np.max(x_vals)
        x_range = x_max - x_min
        x_vals_norm = (x_vals - x_min) / x_range

        eigenvalues = pca.explained_variance_
        eigen_std = np.sqrt(eigenvalues)

        print(f"Eigenvalues (variances): {eigenvalues}")
        print(f"Square roots (standard deviations): {eigen_std}")
        print(f"x range is: {x_range}")

        '''
        #################### Without normalization ######################################
        poly_coeffs = np.polyfit(x_vals, y_vals, deg=2)
        curvature_strength = np.abs(poly_coeffs[0])

        x_sample_norm = np.linspace(0, 1, 100) # from 0 to 1 because we fit polynomial on normalized data so we have to be consistent
        x_sample_real = x_sample_norm * x_range + x_min
        y_sample = np.polyval(poly_coeffs, x_sample_real)

        sampled_points = np.stack([x_sample_real, y_sample], axis=1)
        diffs_poly = sampled_points[1:] - sampled_points[:-1]
        distances_poly = np.linalg.norm(diffs_poly, axis=1)
        smoothed_length = np.sum(distances_poly)

        print(f"Smoothed trace length (unnormalized x): {smoothed_length:.3f} meters")
        print(f"Curvature strength (unnormalized x): {curvature_strength:.6f}")

        plt.figure(figsize=(10, 8))
        plt.scatter(centroids_xy_rot[:, 0], centroids_xy_rot[:, 1], c='blue', s=50, label='PCA Centroids') # Plot centroids directly in PCA frame

        # Polynomial in PCA frame
        x_fit = np.linspace(x_min, x_max, 100)
        x_fit_norm = (x_fit - x_min) / x_range
        y_fit = np.polyval(poly_coeffs, x_fit)

        plt.plot(x_fit, y_fit, c='green', linewidth=5, label='Fitted Polynomial Unnormalized')

        plt.xlabel('PCA-X (Motion Axis)', fontsize=24)
        plt.ylabel('PCA-Y (Deviation)', fontsize=24)
        plt.title('Centroids in PCA Frame', fontsize=30)

        # Ticks font size
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Legend font size
        plt.legend(fontsize=16)

        plt.grid(False)
        plt.show()
        '''



        ########################## with normalization ######################################
        poly_coeffs = np.polyfit(x_vals_norm, y_vals, deg=2)
        curvature_strength = np.abs(poly_coeffs[0])

        # Smoothed length along polynomial curve
        x_sample_norm = np.linspace(0, 1, 100) # from 0 to 1 because we fit polynomial on normalized data so we have to be consistent
        x_sample_real = x_sample_norm * x_range + x_min
        y_sample = np.polyval(poly_coeffs, x_sample_norm)

        sampled_points = np.stack([x_sample_real, y_sample], axis=1)
        diffs_poly = sampled_points[1:] - sampled_points[:-1]
        distances_poly = np.linalg.norm(diffs_poly, axis=1)
        smoothed_length = np.sum(distances_poly)

        # Print results
        print(f"Smoothed trace length (normalized x): {smoothed_length:.3f} meters")
        print(f"Curvature strength (normalized x): {curvature_strength:.6f}")

        

        #return centroids, smoothed_length, curvature_strength
        '''
        plt.figure(figsize=(10, 8))
        plt.scatter(centroids_xy_rot[:, 0], centroids_xy_rot[:, 1], c='blue', s=50, label='PCA Centroids') # Plot centroids directly in PCA frame

        # Polynomial in PCA frame
        x_fit = np.linspace(x_min, x_max, 100)
        x_fit_norm = (x_fit - x_min) / x_range
        y_fit = np.polyval(poly_coeffs, x_fit_norm)

        plt.plot(x_fit_norm, y_fit, c='green', linewidth=5, label='Fitted Polynomial Normalized')
        plt.plot(x_fit, y_fit, c='red', linewidth=5, label='Fitted Polynomial Normalized')

        plt.xlabel('PCA-X (Motion Axis)', fontsize=24)
        plt.ylabel('PCA-Y (Deviation)', fontsize=24)
        plt.title('Centroids in PCA Frame', fontsize=30)

        # Ticks font size
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Legend font size
        plt.legend(fontsize=16)

        plt.grid(False)
        plt.show()
        '''


        '''
        # ========== WORLD FRAME VISUALIZATION ==========

        # Plot in original coordinates
        plt.figure(figsize=(10, 8))
        plt.scatter(centroids_xy[:, 0], centroids_xy[:, 1], c='red', s=50, label='Centroids')

        # Inverse PCA to map back to original coordinates:
        poly_rot = np.stack([x_fit, y_fit], axis=1)
        poly_original = pca.inverse_transform(poly_rot)
        
        #plt.plot(poly_original[:, 0], poly_original[:, 1], c='blue', linewidth=2, label='Fitted Polynomial')

        plt.xlabel('X (world frame)', fontsize=24)
        plt.ylabel('Y (world frame)', fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.title('Centroids in World Frame', fontsize=30)
        plt.grid(False)
        plt.show()
        '''


        return centroids, smoothed_length, curvature_strength








    def find_car_trace_cluster(self, pcd, points_timestamp, labels, aspect_threshold=4.5, min_points=300, curvature_threshold=1000.0, min_length_threshold=1.75, max_length_threshold=6.0):
        """
        Identifies the most likely car trace cluster based on aspect ratio and size.
        """
        unique_labels = np.unique(labels[labels >= 0])  # Ignore noise (-1)
        best_trace_cluster = None
        best_trace_timestamps = None
        best_bbox = None
        best_aspect_ratio = 0
        rest_clusters = []

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_pcd = pcd.select_by_index(cluster_indices)
            cluster_timestamps = points_timestamp[cluster_indices]

            aspect_ratio, _, point_count, bbox = self.get_bounding_box_properties(cluster_pcd)
            extents = bbox.extent
            min_length = min(extents[0], extents[1])
            max_length = max(extents[0], extents[1])

            # New: Compute curvature
            cluster_np = np.asarray(cluster_pcd.points)
            cluster_xy = cluster_np[:, :2]  # Take x and y
            down_pts = self.voxel_downsample_ordered(cluster_xy, voxel_size=0.25)
            curvature = self.compute_curvature(down_pts)

            print(f"Label {label}: Aspect ratio={aspect_ratio:.2f} (max_length={max_length}, min_length={min_length}), Curvature={curvature:.4f}, Points={point_count}")

            # ((aspect_ration > aspect_threshold) and (max_length > length_threshold or min_length > length_threshold))
            if ((aspect_ratio > aspect_threshold) and (max_length > max_length_threshold and min_length > min_length_threshold) or (curvature > curvature_threshold)) and point_count >= min_points:
                if aspect_ratio > best_aspect_ratio:
                    # Store previous best in rest_clusters before replacing
                    if best_trace_cluster is not None:
                        #rest_clusters.append(best_trace_cluster)
                        rest_clusters.append((best_trace_cluster, best_trace_timestamps))
                    
                    best_trace_cluster = cluster_pcd
                    best_trace_timestamps = cluster_timestamps
                    best_bbox = bbox
                    best_aspect_ratio = aspect_ratio
                else:
                    #rest_clusters.append(cluster_pcd)
                    rest_clusters.append((cluster_pcd, cluster_timestamps))
            else:
                #rest_clusters.append(cluster_pcd)
                rest_clusters.append((cluster_pcd, cluster_timestamps))
                    


        return best_trace_cluster, best_trace_timestamps, best_bbox, best_aspect_ratio, rest_clusters
    

    #############################################################


    def append_to_background(self, cam_name, pcd, timestamps: torch.Tensor):
        background_tensor = self.instances_pcd[cam_name][0]

        points_np = np.asarray(pcd.points)
        colors_np = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points_np)

        full_np = np.concatenate([points_np, colors_np], axis=1)  # [N, 6]
        full_tensor = torch.from_numpy(full_np).to(background_tensor.device)

        timestamps = timestamps.to(full_tensor.device)
        full_tensor = torch.cat([full_tensor, timestamps], dim=1)  # [N, 7]

        self.instances_pcd[cam_name][0] = torch.cat([background_tensor, full_tensor], dim=0)


        
    
    # TODO: Whatever you append to background, you have to remove from original tensor (not important for pipeline (cuz anyways Ill be using static map), but for visualization it is important)
    def process_instance(self, cam_name, key, knn=10, min_points=15, percentile=0.999, eps_scale=1.7):
        pcd, points_timestamp = self.get_instance_pcd_o3d(key, cam_name)
        print("------")
        print(f"Processing instance {key} of camera {cam_name}")
        if (np.asarray(pcd.points).shape[0] < 500):
            print(f"Skipping instance {key} of camera {cam_name} because it has too few points ({len(pcd.points)}).")
            self.append_to_background(cam_name, pcd, points_timestamp)
            return False

        #o3d.visualization.draw_geometries([pcd])

        ################# Depth filtering
        # Convert Open3D point cloud to NumPy array
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)

        # Create mask for filtering
        threshold = np.percentile(distances, 97)
        keep_mask = distances < threshold  # [N] → True for points to keep

        # Indices to keep and remove
        keep_indices = np.where(keep_mask)[0]
        remove_indices = np.where(~keep_mask)[0]

        # Select filtered pcds and timestamps
        removed_pcd = pcd.select_by_index(remove_indices)
        removed_timestamps = points_timestamp[remove_indices]  # [M, 1]

        # Send removed points to background
        self.append_to_background(cam_name, removed_pcd, removed_timestamps)

        # Keep only inlier points in the original pcd and timestamps
        pcd = pcd.select_by_index(keep_indices)
        points_timestamp = points_timestamp[keep_indices]
        
        

        ################# Statistical outlier removal
        # Run statistical outlier removal — 'ind' are indices to keep
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        ind = np.asarray(ind)

        # Get total number of points
        num_points = len(pcd.points)

        # Compute the set of indices to remove
        all_indices = np.arange(num_points)
        remove_indices = np.setdiff1d(all_indices, ind)

        # Select removed points and corresponding timestamps
        removed_pcd = pcd.select_by_index(remove_indices)
        removed_timestamps = points_timestamp[remove_indices]

        # Append outliers to background
        self.append_to_background(cam_name, removed_pcd, removed_timestamps)

        # Keep inlier points in pcd and timestamp
        pcd = pcd.select_by_index(ind)
        points_timestamp = points_timestamp[ind]


        #o3d.visualization.draw_geometries([pcd])

        ################# Prepare for DBSCAN (scikit-learn)
        points_np = np.asarray(pcd.points)
        colors_np = np.asarray(pcd.colors)  # values in [0, 1]
        eps_value = self.estimate_eps(pcd, k=knn, percentile=percentile, eps_scale=eps_scale)
        #print(f"Estimated eps: {eps_value}")

        # DBSCAN from sklearn
        features = np.hstack([points_np, colors_np])  # shape (N, 6)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_scaled[:, 3:6] *= 0.4
        clustering = DBSCAN(eps=eps_value, min_samples=min_points).fit(points_np)
        #clustering = HDBSCAN(min_samples=20, cluster_selection_epsilon=0.6).fit(points_np)
        labels = clustering.labels_
        #max_label = labels.max()
        #print(f"Point cloud has {max_label + 1} clusters.")

        # Visualize using color (just for visualization)
        #colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        #colors[labels < 0] = 0
        #pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        #o3d.visualization.draw_geometries([pcd])

        
        ###############################
        centroids, smoothed_length, curvature_strength = self.find_car_trace_cluster_centroid(pcd, points_timestamp, labels)

        movable_car = False
        if smoothed_length >= 8.5:
            movable_car = True
            if curvature_strength >= 10.0:
                label = "Turning car"
            else:
                label = "Straight moving car"
        else:
            label = "Parked car"
        
        return movable_car
        ###############################
        '''
        # Find the best car trace cluster
        car_trace_pcd, car_trace_timestamps, car_trace_bbox, best_aspect_ratio, rest_clusters = self.find_car_trace_cluster(pcd, points_timestamp, labels, aspect_threshold=4.1)

        if car_trace_pcd is None:
            print(f"Object {key} does not contain a valid car trace. Merging whole instance into background.")
            self.append_to_background(cam_name, pcd, points_timestamp)
            return False

        # Append non-selected clusters to background
        for cluster_pcd, cluster_timestamps in rest_clusters:
            self.append_to_background(cam_name, cluster_pcd, cluster_timestamps)
            pass

        #o3d.visualization.draw_geometries([car_trace_pcd, car_trace_bbox])
        return True
        '''
    

    def process_all(self):
        for cam_name in self.instances_pcd.keys():
            for key in self.instances_pcd[cam_name].keys():
                if key not in [-1, 0]: # -1 is outside of FOV, 0 is background
                    result = self.process_instance(cam_name, key)
                    self.dynamic_instance[cam_name][key] = result
