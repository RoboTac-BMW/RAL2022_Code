import open3d as o3d
import numpy as np
import os
import random
import math
from pathlib import Path
from scipy.spatial.transform import Rotation as R


import torch
import json
import ast
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
        ###
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(norm_pointcloud)
        # o3d.io.write_point_cloud("/home/airocs/Desktop/test_MCDrop/test.pcd", pcd)
        return norm_pointcloud

    elif pointcloud.shape[1] == 6: # with normals
        pointcloud_tmp, pointcloud_norm = np.split(pointcloud, 2, axis=1)
        # translate points to origin
        norm_pointcloud_tmp = pointcloud_tmp - np.mean(pointcloud_tmp, axis=0)
        # normalize points
        norm_pointcloud_tmp /= np.max(np.linalg.norm(norm_pointcloud_tmp, axis=1))
        norm_pointcloud = np.concatenate((norm_pointcloud_tmp, pointcloud_norm), axis=1)
        return  norm_pointcloud

    else:
        raise ValueError("Wrong PointCloud Input")


def sub_and_downSample(pointcloud, sample_num):
    assert len(pointcloud.shape)==2
    # print("Old shape", pointcloud.shape)
    if pointcloud.shape[1] == 3:
        num_point = pointcloud.shape[0]

        while(num_point < int(sample_num)):
            # print(pointcloud[-1].shape)
            # print("!!!!!!!!!!!!!!!!!!!!!!!")
            # print(num_point)
            # pointcloud = np.concatenate((pointcloud, pointcloud[-1]), axis=0)
            pointcloud = np.insert(pointcloud,-1, pointcloud[-1], axis=0)
            num_point = pointcloud.shape[0]

        if(num_point>sample_num):
            sel_pts_idx = np.random.choice(pointcloud.shape[0],
                                           size=sample_num,
                                           replace=False).reshape(-1)
            pointcloud= pointcloud[sel_pts_idx]
        # print(pointcloud.shape)


        return pointcloud

    else:
        raise NotImplementedError("Point Cloud shape is not correct! Should be (n*3)")



class PCDPointCloudData(Dataset):
    def __init__(self, root_dir,
                 folder='Train',
                 num_point=1024, # numble of point to sample
                 sample=True, # sample the pc or not
                 sample_method='Voxel', # Random or Voxel
                 est_normal=False, # estimate normals or not
                 random_num=False, # Not Implemented TODO
                 list_num_point=[1024], # Not Implemented TODO
                 rotation='z'): # rotation method, False or 'z'

        self.root_dir = root_dir
        self.folder = folder
        self.sample = sample
        self.sample_method = sample_method
        self.num_point = num_point
        self.est_normal = est_normal
        self.random_num = random_num
        self.list_num_point = list_num_point
        self.rotation = rotation
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
        if self.sample_method == 'Voxel':
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.004)
            # To test
            # o3d.io.write_point_cloud("/home/airocs/Desktop/test_MCDrop/down_sampled" + str(idx) + ".pcd", point_cloud)


        if self.est_normal is True:
            # TODO Add estimate normals before down_sample
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
        if self.rotation == 'z':
            pointcloud_np = rand_rotation(pointcloud_np, with_normal=self.est_normal)
        elif self.rotation is False:
            pointcloud_np = pointcloud_np
        else:
            raise ValueError("Invalid Rotation input")
        # print(pointcloud_np.shape)

        # random select points
        # TODO
        if self.random_num is False:
            sample_size = self.num_point
        else:
            raise NotImplementedError()
            # sample_size = random.choice(self.list_num_point)

        if self.sample is True:
            pointcloud_np_sampled = sub_and_downSample(pointcloud_np, self.num_point)
        # print(self.classes[category])

        # return pointcloud_np, self.classes[category]
            return {'pointcloud': pointcloud_np_sampled,
                    'category': self.classes[category]}
        else:
            return {'pointcloud': pointcloud_np,
                    'category': self.classes[category]}

class PCDTest(Dataset):
    def __init__(self, pcd_path, sub_sample=True,
                 sample_num=None, est_normal=False,
                 sample_method='Voxel'):

        # self.pcd_dir = pcd_dir
        self.pcd_path = pcd_path
        self.sub_sample = sub_sample
        self.sample_num = sample_num
        self.est_normal = est_normal
        self.sample_method = sample_method
        self.files = []

        # for file in os.listdir(pcd_dir):
        #     if file.endswith('.pcd'):
        #         sample = {}
        #         sample['pcd_path'] = Path(pcd_dir)/file
        #         self.files.append(sample)
        sample={}
        sample['pcd_path'] = self.pcd_path
        self.files.append(sample)


    def __len__(self):
        return len(self.files)


    def __getitem__(self,idx):
        pcd_path = self.files[idx]['pcd_path']
        point_cloud = o3d.io.read_point_cloud(filename=str(pcd_path))

        if self.est_normal is True:
            raise NotImplementedError("Not implemented with normals")

        """
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
        """

        if self.sample_method is 'Voxel':
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.004)
        else:
            raise NotImplementedError("Other sample methods not implemented")

        points = np.asarray(point_cloud.points).astype(np.float32)
        pointcloud_np = points
        pointcloud_np = normalize_pointcloud(pointcloud_np)

        if self.sub_sample is True:
            pointcloud_np = sub_and_downSample(pointcloud_np, self.sample_num)

        return pointcloud_np



class PCDActiveVision(PCDPointCloudData):
    def __init__(self,
                 root_dir,
                 active_path,
                 active_sample_num=1500,
                 folder='Train',
                 num_point=1024,
                 sample=True,
                 sample_method='Voxel',
                 est_normal = False,
                 random_num = False,
                 list_num_point = [1024],
                 rotation='z',
                 random_shuffle=False):

        super(PCDActiveVision, self).__init__(root_dir, folder, num_point, sample, sample_method,
                         est_normal, random_num, list_num_point, rotation)
        self.active_path = active_path
        self.active_sample_num = active_sample_num

        with open(self.active_path) as file:
            lines = [line.rstrip() for line in file]
            if random_shuffle is True:
                random.shuffle(lines)

            for i in range(self.active_sample_num):
                # print(lines[i])
                # print(type(lines[i]))
                converted_string=json.loads(lines[i])
                # print(converted_string)
                # print(type(converted_string))
                self.files.append(converted_string)

        print(len(self.files))


if __name__ == '__main__':

    path_dir = "/home/airocs/cong_workspace/tools/Pointnet_Pointnet2_pytorch/data/active_vision_pcd_1500/"
    active_path = "/home/airocs/Desktop/active_entropy_files.txt"
    PCDActiveVision(root_dir=path_dir, active_path='/home/airocs/Desktop/active_entropy_files.txt')
    # a = find_classes(path_dir)
    # print(type(a))
    # print(len(a))
    # print(a[0])
    # pointcloud_data = PCDPointCloudData(path_dir)
    # pointcloud_data = PCDTest(path_dir)
    # pointcloud_data.testFunc(100)



