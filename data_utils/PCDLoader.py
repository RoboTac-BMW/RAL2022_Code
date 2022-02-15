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
    else:

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


class PCDPointCloudData(Dataset):
    def __init__(self, root_dir,
                 folder='Train',
                 num_point=1024,
                 est_normal=True,
                 random_num=False,
                 list_num_point=[1024],
                 rotation='z'):

        self.root_dir = root_dir
        self.folder = folder
        self.num_point = num_point
        self.est_normal = est_normal
        self.random_num = random_num
        self.list_num_point = list_num_point
        self.classes = find_classes(Path(root_dir))
        self.files = []

        for category in self.classes.keys():
            new_dir = self.root_dir/Path(category)/folder
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
        point_cloud = o3d.io.read_point_cloud(filename=str(pcd_path))

        if self.est_normal is True:
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

        else:
            points = np.asarray(point_cloud.points).astype(np.float32)
            pointcloud_np = points

        # centralize and normalize point cloud
        pointcloud_np = normalize_pointcloud(pointcloud_np)
        pointcloud_np = rand_rotation(pointcloud_np)
        # print(pointcloud_np.shape)

        # random select points
        # TODO
        if self.random_num is False:
            sample_size = self.num_point
        else:
            sample_size = random.choice(self.list_num_point)

        sel_pts_idx = np.random.choice(pointcloud_np.shape[0],
                                       size=sample_size,
                                       replace=False).reshape(-1)
        pointcloud_np = pointcloud_np[sel_pts_idx]
        # print(self.classes[category])

        # return pointcloud_np, self.classes[category]
        return {'pointcloud': pointcloud_np,
                'category': self.classes[category]}

class PCDTest(Dataset):
    def __init__(self, pcd_dir, sub_sample=False, sample_num=None, est_normal=True,
                 radius=0.1, max_nn=16):

        self.pcd_dir = pcd_dir
        self.sub_sample = sub_sample
        self.sample_num = sample_num
        self.est_normal = est_normal
        self.radius = radius
        self.max_nn = max_nn
        self.files = []

        for file in os.listdir(pcd_dir):
            if file.endswith('.pcd'):
                sample = {}
                sample['pcd_path'] = Path(pcd_dir)/file
                self.files.append(sample)

    def __len__(self):
        return len(self.files)


    def __getitem__(self,idx):
        pcd_path = self.files[idx]['pcd_path']
        point_cloud = o3d.io.read_point_cloud(filename=str(pcd_path))
        if self.est_normal is True:
            point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                      radius=self.radius, max_nn=self.max_nn))
            point_cloud.normalize_normals()

        o3d.geometry.PointCloud.orient_normals_to_align_with_direction(
            point_cloud,
            orientation_reference=np.array([0., 0., 1.])
        )

        points = np.asarray(point_cloud.points).astype(np.float32)
        norms = np.asarray(point_cloud.normals).astype(np.float32)
        pointcloud_np = np.concatenate((points, norms), axis=1)

        pointcloud_np = normalize_pointcloud(pointcloud_np)

        if self.sub_sample is True:
            sel_pts_idx = np.random.choice(pointcloud_np.shape[0],
                                           size=self.sample_num,
                                           replace=False).reshape(-1)
            pointcloud_np = pointcloud_np[sel_pts_idx]

        return pointcloud_np



if __name__ == '__main__':

    path_dir = "/home/airocs/cong_workspace/tools/Pointnet_Pointnet2_pytorch/data/visual_data_pcd/can/Test"
    # a = find_classes(path_dir)
    # print(type(a))
    # print(len(a))
    # print(a[0])
    # pointcloud_data = PCDPointCloudData(path_dir)
    pointcloud_data = PCDTest(path_dir)
    # pointcloud_data.testFunc(100)



