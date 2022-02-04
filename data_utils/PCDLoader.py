import open3d as o3d
import numpy as np
import os
import random
import math
from pathlib import Path
from scipy.spatial.transform import Rotation as R


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

"""
def load_pcd(file_name):
    # pc = pypcd.PointCloud.from_path(file_name)
    pc = pcl.load(file_name)
    return pc

def add_normal(np_pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np_pts)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16))
    return pc


def read_pcd(pcd_path):
    with open(pcd_path) as f:
        while True:
            ln = f.readline().strip()
            if ln.startswith('DATA'):
                break

        points= np.loadtxt(f)
        points = points[:, 0:4]

        return points

def normalizePCD(np_pts):
    norm_pointcloud = np_pts - np.mean(np_pts, axis=0) # translate to origin
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1)) # normalize

    return norm_pointcloud

"""

def find_classes(root_dir):
    root_dir = Path(root_dir)
    folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    return classes

def rand_rotation(pointcloud, with_normal=True, SO3=False):
    assert len(pointcloud.shape) == 2
    roll, pitch, yaw = np.random.rand(3)*np.pi*2
    if SO3 is False:
        pitch, roll = 0.0, 0.0

    rot_matrix = R.from_euler('XZY', (roll, yaw, pitch)).as_matrix()
    # Transform the rotation matrix for points with normals. Shape (6,6)
    zero_matrix = np.zeros((3,3))
    tmp_matrix = np.concatenate((rot_matrix,zero_matrix),axis=1) # [R,0]
    tmp_matrix_2 = np.concatenate((zero_matrix, rot_matrix), axis=1) # [0,R]
    # [[R,0],[0,R]]
    rot_matrix_with_normal = np.concatenate((tmp_matrix, tmp_matrix_2), axis=0)
    if with_normal is True:
        rot_pointcloud = rot_matrix_with_normal.dot(pointcloud.T).T
    else:                                          # up=[-0.0694, -0.9768, 0.2024])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    return  rot_pointcloud


def normalize_pointcloud(pointcloud):
    assert len(pointcloud.shape)==2
    if pointcloud.shape[1] == 3: # without normals
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) # translate to origin
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1)) # normalize
        return norm_pointcloud

    else: # with normals
        pointcloud_tmp, pointcloud_norm = np.split(pointcloud, 2, axis=1)
        # translate points to origin
        norm_pointcloud_tmp = pointcloud_tmp - np.mean(pointcloud_tmp, axis=0)
        # normalize points
        norm_pointcloud_tmp /= np.max(np.linalg.norm(norm_pointcloud_tmp, axis=1))
        norm_pointcloud = np.concatenate((norm_pointcloud_tmp, pointcloud_norm), axis=1)
        return  norm_pointcloud


class PointCloudData(Dataset):
    def __init__(self, root_dir, num_point=1024, sample_method='random', rotation='z'):
        self.root_dir = root_dir
        self.num_point = num_point
        self.classes = find_classes(Path(root_dir))
        self.files = []

        for category in self.classes.keys():
            new_dir = self.root_dir/Path(category)
            for file in os.listdir(new_dir):
                if file.endswith('.pcd'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        print(pcd_path)

        point_cloud = o3d.io.read_point_cloud(filename=str(pcd_path))
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                  radius=0.1, max_nn=16))
        point_cloud.normalize_normals()

        # align the normal vectors to z axis
        o3d.geometry.PointCloud.orient_normals_to_align_with_direction(
            point_cloud,
            orientation_reference=np.array([0., 0., 1.])
        )

        # draw point cloud
        # o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)

        # convert to numpy
        points = np.asarray(point_cloud.points).astype(np.float32)
        norms = np.asarray(point_cloud.normals).astype(np.float32)
        pointcloud_np = np.concatenate((points, norms), axis=1)

        # centralize and normalize point cloud
        pointcloud_np = normalize_pointcloud(pointcloud_np)
        pointcloud_np = rand_rotation(pointcloud_np)
        # print(pointcloud_np.shape)

        # random select points
        sel_pts_idx = np.random.choice(pointcloud_np.shape[0],
                                       size=self.num_point,
                                       replace=False).reshape(-1)
        pointcloud_np = pointcloud_np[sel_pts_idx]
        # print(self.classes[category])

        return pointcloud_np, self.classes[category]





















if __name__ == '__main__':
    # test_pcd = load_pcd("/home/airocs/Desktop/visual_data_pcd/eraser/1.pcd")
    """
    test_pcd = read_pcd("/home/airocs/Desktop/visual_data_pcd/eraser/1.pcd")
    print(type(test_pcd))
    print(test_pcd.shape)

    pc_normalized = normalizePCD(test_pcd)
    print(pc_normalized.shape)

    pointcloud_withNormal = add_normal(pc_normalized)
    print(type(pointcloud_withNormal))
    print(pointcloud_withNormal.has_normals())
    # o3d.visualization.draw(pointcloud_withNormal,show_ui=True)
    """

    # Open3D dataset
    """
    test_pcd = o3d.io.read_point_cloud("/home/airocs/Desktop/visual_data_pcd/eraser/1.pcd")
    test_pcd.normalize_normals()
    test_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                              radius=0.1, max_nn=16))

    o3d.geometry.PointCloud.orient_normals_to_align_with_direction(
        # pointcloud_withNormal,
        test_pcd,
        orientation_reference=np.array([0., 0., 1.])
    )

    # o3d.geometry.orient_normals_towards_camera_location(
    #     pointcloud_withNormal,
    #     orientation_reference=np.array([0., 0., 1.])
    # )
    # o3d.visualization.draw(pointcloud_withNormal, show_ui=True)
    o3d.visualization.draw_geometries([test_pcd],
                                      point_show_normal=True)
                                      # show_ui=True,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024])
    points = np.asarray(test_pcd.points).astype(np.float32)
    norms = np.asarray(test_pcd.normals).astype(np.float32)

    print(points.shape)
    print(norms.shape)
    """
    path_dir = "/home/airocs/Desktop/visual_data_pcd/"
    # a = find_classes(path_dir)
    # print(type(a))
    # print(len(a))
    # print(a[0])
    pointcloud_data = PointCloudData(path_dir)
    # pointcloud_data.testFunc(100)



