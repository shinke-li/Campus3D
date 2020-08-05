import open3d
import numpy as np

def read_point_cloud(file_path, use_color, return_bounds=False, *args, **kwargs):
    pcd = open3d.io.read_point_cloud(file_path, *args, **kwargs)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if use_color else np.zeros_like(points)
    if return_bounds:
        return points, colors, pcd.get_min_bound(), pcd.get_max_bound()
    else:
        return points, colors

def create_point_cloud(points, colors=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.points = open3d.utility.Vector3dVector(colors)
    return pcd

def voxel_sampling(points, voxel_size):
    pcd = create_point_cloud(points)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downpcd.points)

def kdtree(points):
    pcd = create_point_cloud(points=points)
    return open3d.geometry.KDTreeFlann(pcd)

