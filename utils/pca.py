import open3d as o3d
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
from os.path import join
import math
from utils.tools import get_time


class VoxelHasherIndex(nn.Module):
    def __init__(self, points: torch.Tensor, grid_resolution: float, buffer_size=100000000, random=True) -> None:
        """Voxel Hasher for downsampling and finding neighbors in the grid. Stores in each cell the index of a point from the original point cloud

        Args:
            points (torch.Tensor): [N,3] point coordinates
            grid_resolution (float): resolution of the grid
            buffer_size (int, optional): Size of the grid buffer. The higher, the less likely voxel collisions. Defaults to 100000000.
            random (bool, optional): If True: stores random point in the cell. If False: stores the point which is closest to the voxel center.
        """
        super().__init__()
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=torch.int64, device=points.device)
        self.buffer_pt_index = torch.full(
            [buffer_size], -1, dtype=torch.int64, device=points.device)
        self.buffer_valids = torch.zeros(
            [buffer_size], dtype=torch.bool, device=points.device)
        self.grid_resolution = grid_resolution
        self.buffer_size = buffer_size

        # Fill grid
        if random:
            indices = torch.arange(
                points.shape[0], dtype=self.buffer_pt_index.dtype, device=self.buffer_pt_index.device)
            grid_coords = (
                points / self.grid_resolution).floor().to(self.primes)
        else:
            indices = gridSampling(points, resolution=self.grid_resolution)
            grid_coords = (
                points[indices] / self.grid_resolution).floor().to(self.primes)

        hash = (grid_coords * self.primes).sum(-1) % self.buffer_size
        self.buffer_pt_index[hash] = indices
        self.buffer_valids[hash] = True

    def get_indices(self):
        """returns the indices of the points that are stored in the grid. Indices are from the original point cloud

        Returns:
            indices: torch.Tensor [K]
        """
        return self.buffer_pt_index[self.buffer_valids]

    def radius_neighborhood_search(self, points: torch.Tensor, radius: float):
        """returns the indices of the potential neighbors for each point. Be aware that those might be invalid (value: -1) or just wrong due to hash collision.

        Args:
            points [N,3] (torch.Tensor): point coordinates from which to find neighbors
            radius (float): radius in which to find neighbors, be aware that the actual search radius might be higher due to rounding up to full voxel resolution

        Returns:
            indices [N,m] (torch.Tensor): for each point the m potential neighbors. m depens of radius. For 0 < m <= voxel_resolution: 3^3 = 27 neighbors, 2*voxel_resolution: 5^3 = 125...  
        """
        grid_coords = (points / self.grid_resolution).floor().to(self.primes)

        num_cells = math.ceil(radius/self.grid_resolution)
        dx = torch.arange(-num_cells, num_cells+1,
                          device=grid_coords.device, dtype=grid_coords.dtype)

        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3)

        neighbord_cells = grid_coords[..., None, :] + dx
        hash = (neighbord_cells * self.primes).sum(-1) % self.buffer_size
        return self.buffer_pt_index[hash]



class GeometricFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, points: torch.Tensor, neighborhoods: torch.Tensor,
                radius: float = 1,  normal_only: bool = True, min_valid_neigh = 4):
        """computes geometric features in a given neighborhood

        Args:
            points [N,3] (torch.Tensor): coordinates of the query points 
            neighborhoods [N,n,3] (torch.Tensor): coordinates of the neighborhood points
            radius (float): radius of the neighborhood, for normalization

        Returns:
            features [N,16] (torch.Tensor): point wise features [singular values (3) , rotation matrix of SVD (9), linearness (1), planarness (1), scatterness (1), relative density (1)]
        """
        T0 = get_time()

        valids = (points[:, None, :3] - neighborhoods).norm(dim=-1) < radius*3.4
        num_valid_neigh = valids.sum(-1)

        valid_neigh_mask = num_valid_neigh > min_valid_neigh
        neighborhoods = neighborhoods[valid_neigh_mask]
        valids = valids[valid_neigh_mask]
        num_valid_neigh = num_valid_neigh[valid_neigh_mask]
        
        points = points[valid_neigh_mask]

        p = (neighborhoods - (neighborhoods*valids[..., None]).sum(dim=-2, keepdim=True)/num_valid_neigh[:, None, None]) * valids[..., None] # / radius
           
        T1 = get_time()
        cov = p.transpose(-1, -2)@p / (num_valid_neigh[..., None, None])
        # print(cov.shape)
        # print(cov)
        T2 = get_time()
        # _, s, vh = torch.linalg.svd(cov, full_matrices=True)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov) # s in ordered in ascending order\
        lambda_1, lambda_2, lambda_3 = eigenvalues[:,2], eigenvalues[:,1], eigenvalues[:,0]

        # print(eigenvectors.shape)
        normal_vector = eigenvectors[:,0,:]
        
        if normal_only:
            return points, normal_vector, valid_neigh_mask


        T3 = get_time()
        eps = 1e-12
        linear = 1-lambda_2/(lambda_1+eps)
        planar = (lambda_2-lambda_3)/(lambda_1+eps)
        scatter = lambda_3/(lambda_1+eps) 

        # linear = 1-s[..., 1] / (s[..., 0]+1e-12)
        # planar = (s[..., 1]-s[..., 2]) / (s[..., 0]+1e-12)
        # scatter = s[..., 2] / (s[..., 0]+1e-12)
        dim_features = torch.stack([linear, planar, scatter], dim=-1)
        T4 = get_time()
        geometric_features = dim_features
        # geometric_features = torch.cat([s,  # singular values [0:3]
        #                                 # rotation matrix
        #                                 # [3:6] first, [6:9] second, [9:12]
        #                                 vh.reshape(*vh.shape[:-2], 9),
        #                                 dim_features,  # linear [12], planar [13], scatter [14]
        #                                 num_points[..., None]/neighborhoods.shape[-2]], dim=-1)  # relative density [15]

        kept_mask = (linear > 0.4) | (planar > 0.5)
        points_kept = points[kept_mask]
        normal_kept = normal_vector[kept_mask]
        valid_mask = valid_neigh_mask.clone()
        normal_valid_part = valid_mask[valid_mask > 0]
        normal_valid_part[~kept_mask] = False
        valid_mask[valid_mask > 0] = normal_valid_part

        # print("time for A (ms):", (T1-T0)*1e3)
        # print("time for B (ms):", (T2-T1)*1e3)
        # print("time for C (ms):", (T3-T2)*1e3)
        # print("time for D (ms):", (T4-T3)*1e3)
        
        return points_kept, normal_kept, valid_mask
    

def meanGridSampling(
        pcd: torch.Tensor,
        resolution_meter: float):
    """Computes the mean over all points in the grid cells

    Args:
        pcd (torch.Tensor): [...,N,3] point coordinates
        resolution_meter ([type]): grid resolution
    Returns:
        grid_coords (torch.Tensor): [...,N,3] grid coordinates
    """
    resolution = resolution_meter

    grid = torch.floor((pcd - pcd.min(dim=-2, keepdim=True)[0]) / resolution)

    v_size = grid.max().ceil()
    grid_idx = grid[..., 0] + grid[..., 1] * \
        v_size + grid[..., 2] * v_size * v_size

    unique, indices, counts = torch.unique(
        grid_idx, return_inverse=True, dim=None, return_counts=True
    )
    grid_coords = torch.zeros_like(pcd[:len(counts), :], device=pcd.device)
    indices.unsqueeze_(-1)

    grid_coords.scatter_add_(-2, indices.expand(pcd.shape), pcd)
    grid_coords /= counts.unsqueeze(-1)

    return grid_coords


def gridSampling(pcd: torch.Tensor, resolution: float):
    """Grid based downsampling. Returns the indices of the points which are closest to the grid cells. 

    Args:
        pcd (torch.Tensor): [N,3] point coordinates
        resolution (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`  
    """
    _quantization = 1000

    offset = torch.floor(pcd.min(dim=-2)[0]/resolution).long()
    grid = torch.floor(pcd / resolution)
    center = (grid + 0.5) * resolution
    dist = ((pcd - center) ** 2).sum(dim=1)**0.5
    dist = dist / dist.max() * (_quantization - 1)

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(
        grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(
        0), dtype=inverse.dtype, device=inverse.device)

    offset = 10**len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset
    idx = torch.empty(unique.shape, dtype=inverse.dtype,
                      device=inverse.device).scatter_reduce_(dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False)
    idx = idx % offset
    return idx