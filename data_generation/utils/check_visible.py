import os
import time
import json
from math import pi, ceil
from tqdm import tqdm

import cv2
import numpy as np
import open3d as o3d
import scipy
import mmengine
from scipy.spatial import ConvexHull


def get_9dof_boxes(bbox, mode, colors):
    """
    Get a list of open3d.geometry.OrientedBoundingBox objects from a (N, 9) array of bounding boxes.
    Args:
        bbox (numpy.ndarray): (N, 9) array of bounding boxes.
        mode (str): 'xyz' or 'zxy' for the rotation mode.
        colors (numpy.ndarray): (N, 3) array of RGB colors, or a single RGB color for all boxes.
    Returns:
        list: A list of open3d.geometry.OrientedBoundingBox objects.
    """
    n = bbox.shape[0]
    if isinstance(colors, tuple):
        colors = np.tile(colors, (n, 1))
    elif len(colors.shape) == 1:
        colors = np.tile(colors.reshape(1, 3), (n, 1))
    assert colors.shape[0] == n and colors.shape[1] == 3
    geo_list = []
    for i in range(n):
        center = bbox[i][:3].reshape(3, 1)
        scale = bbox[i][3:6].reshape(3, 1)
        rot = bbox[i][6:].reshape(3, 1)
        if mode == "xyz":
            rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(rot)
        elif mode == "zxy":
            rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
        else:
            raise NotImplementedError
        geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)
        color = colors[i]
        geo.color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        geo_list.append(geo)
    return geo_list


def interpolate_bbox_points(bbox, granularity=0.2, return_size=False):
    """
    Get the surface points of a 3D bounding box.
    Args:
        bbox: an open3d.geometry.OrientedBoundingBox object.
        granularity: the roughly desired distance between two adjacent surface points.
        return_size: if True, return m1, m2, m3 as well.
    Returns:
        M x 3 numpy array of Surface points of the bounding box
        (m1, m2, m3): if return_size is True, return the number for each dimension.)
    """
    corners = np.array(bbox.get_box_points())
    v1, v2, v3 = (
        corners[1] - corners[0],
        corners[2] - corners[0],
        corners[3] - corners[0],
    )
    l1, l2, l3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
    assert (
        np.allclose(v1.dot(v2), 0)
        and np.allclose(v2.dot(v3), 0)
        and np.allclose(v3.dot(v1), 0)
    )
    transformation_matrix = np.column_stack((v1, v2, v3))
    m1, m2, m3 = l1 / granularity, l2 / granularity, l3 / granularity
    print("Here is m1,m2,m3",m1, m2, m3)
    m1, m2, m3 = int(np.ceil(m1)), int(np.ceil(m2)), int(np.ceil(m3))
    print("Here is l1,l2,l3",l1, l2, l3)
    coords = np.array(
        np.meshgrid(np.arange(m1 + 1), np.arange(m2 + 1), np.arange(m3 + 1))
    ).T.reshape(-1, 3)
    condition = (
        (coords[:, 0] == 0)
        | (coords[:, 0] == m1 - 1)
        | (coords[:, 1] == 0)
        | (coords[:, 1] == m2 - 1)
        | (coords[:, 2] == 0)
        | (coords[:, 2] == m3 - 1)
    )
    surface_points = coords[condition].astype(
        "float32"
    )  # keep only the points on the surface
    surface_points /= np.array([m1, m2, m3])
    mapped_coords = surface_points @ transformation_matrix
    mapped_coords = mapped_coords.reshape(-1, 3) + corners[0]
    if return_size:
        return mapped_coords, (m1, m2, m3)
    return mapped_coords

def _compute_area(points):
    """
    Computes the area of a set of points.
    """
    if len(points)<3:
        return 0
    try:
        hull = ConvexHull(points)
        area = hull.volume
        return area
    except scipy.spatial.qhull.QhullError as e:
        if "QH6154" in str(e):  # in the same line
            return 0
        if "QH6013" in str(e):  # same x coordinate
            return 0
        else:
            print(points)
            raise e



def update_vertex_visibility(
    img,
    o_box,
    extrinsic_c2w,
    intrinsic,
    vertex_visible_list
):
    
    
    
    extrinsic = np.linalg.inv(extrinsic_c2w)
    h,w  = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    
    box = get_9dof_boxes(np.array([o_box]), mode="zxy", colors=(0, 0, 192))[0]
    center = box.get_center()
    corners = np.array(box.get_box_points())
   
    center_2d = (
        intrinsic
        @ extrinsic
        @ np.array([center[0], center[1], center[2], 1]).reshape(4, 1)
    )
    center_2d = center_2d[:2] / center_2d[2]
    center_in_img = ((0<int(center_2d[0])<w) & (0<int(center_2d[1])<h))

    for i in range(len(corners)):
        corner = corners[i]
        corner_2d = (
        intrinsic
        @ extrinsic
        @ np.array([corner[0], corner[1], corner[2], 1]).reshape(4, 1)
        )
        corner_2d = corner_2d[:2] / corner_2d[2]
        if ((0<int(corner_2d[0])<w) & (0<int(corner_2d[1])<h)):
            vertex_visible_list[i] = True

    return center_in_img, vertex_visible_list
def generate_points_on_cube(vertices, granularity=0.02):
    """
    在长方体的表面均匀生成点列。

    参数:
        vertices (np.array): 长方体的 8 个顶点坐标，形状为 (8, 3)。
        num_points_per_face (int): 每个面上生成的点数。

    返回:
        np.array: 生成的表面点列，形状为 (6 * num_points_per_face, 3)。
    """
    # 确保输入顶点形状正确
    if vertices.shape != (8, 3):
        raise ValueError("顶点坐标的形状必须为 (8, 3)。")

    # 定义长方体的 6 个面（每个面由 4 个顶点组成）
    faces = [
        [1,7,4,6],  # 前面
        [0,2,5,3],  # 后面
        [1,7,2,0],  # 底面
        [6,4,5,3],  # 顶面
        [1,0,3,6],  # 左面
        [4,7,2,5]   # 右面
    ]

    # 存储生成的点
    surface_points = []

    # 对每个面生成均匀分布的点
    for face in faces:
        # 获取当前面的 4 个顶点
        v0, v1, v2, v3 = vertices[face]
        v_changed = np.linalg.norm(v3-v0)
        u_changed = np.linalg.norm(v1-v0)
        
        # 生成均匀分布的参数 u 和 v
        u = np.linspace(0, 1, int(u_changed/granularity))
        v = np.linspace(0, 1, int(v_changed/granularity))
        if len(u)<2:
            u = np.array([0.0,1.0])
        if len(v)<2:
            v = np.array([0.0,1.0])
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        points = []
        for u_,v_ in zip(u,v):
            points.append(
                (1 - u_) * (1 - v_) * v0 +
                u_ * (1 - v_) * v1 +
                u_ * v_ * v2 +
                (1 - u_) * v_ * v3
            )
        surface_points.append(points)

    surface_points = np.vstack(surface_points)
    return surface_points
def count_inside_image_final(depth_map,
    o_box,
    extrinsic_c2w,
    depth_intrinsic,
):
    """Counts visible points of a 3D box in a depth image.

    Generates points on a 3D box, projects them into depth image coordinates,
    and counts how many are both within image bounds and not occluded by depth.

    Args:
        depth_map (numpy.ndarray): Depth image (H,W) for occlusion checks.
        o_box: 3D bounding box object (must have get_box_points() method).
        extrinsic_c2w (numpy.ndarray): 4x4 extrinsic matrix (camera-to-world).
        depth_intrinsic (numpy.ndarray): 3x3 depth camera intrinsic matrix.

    Returns:
        tuple: (total_points, visible_points, ratio) where:
            - total_points: Area of all projected points (in pixels).
            - visible_points: Area of non-occluded visible points.
            - ratio: Always returns 1 (legacy/compatibility).

    Note:
        Uses helper functions:
        - generate_points_on_cube(): Creates point cloud on box surface
        - _compute_area(): Computes area of projected points

    Example:
        >>> depth = np.random.uniform(0,10,(480,640))
        >>> total, visible, _ = count_inside_image_final(
                depth, box, extrinsics, K_depth)
    """
    extrinsic = np.linalg.inv(extrinsic_c2w)
    h,w  = depth_map.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    
    box = get_9dof_boxes(np.array([o_box]), mode="zxy", colors=(0, 0, 192))[0]
    center = box.get_center()
    corners = np.array(box.get_box_points())
    points = generate_points_on_cube(np.array(box.get_box_points()))
  
    points = np.concatenate(
        [points, np.ones_like(points[..., :1])], axis=-1
    )
    pts = depth_intrinsic @ np.linalg.inv(extrinsic_c2w) @ points.T
    xs, ys, zs = pts[0, :], pts[1, :], pts[2, :]
    
    xs, ys = xs / zs, ys / zs
    
    pts = np.stack([xs.astype(int), ys.astype(int)], axis=-1)
    pts = np.unique(pts, axis=0)
    
    width =w
    height = h
    visible_indices = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys, zs = xs[visible_indices], ys[visible_indices], zs[visible_indices]
    xs, ys = xs.astype(int), ys.astype(int)
    
    
    
    visible_indices = depth_map[ys, xs] >= zs
    xs_filter, ys_filter = xs[visible_indices], ys[visible_indices]
    
    pts_filter = np.stack([xs_filter, ys_filter], axis=-1)
    pts_filter = np.unique(pts_filter, axis=0)
    
    return _compute_area(pts),_compute_area(pts_filter),1
