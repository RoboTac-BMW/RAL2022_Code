import open3d as o3d
import numpy as np
import os
import random
from pathlib import Path
import math
from scipy.spatial.transform import Rotation as R


def find_classes(root_dir):
    root_dir = Path(root_dir)
    folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    return classes


def rand_rotation(pointcloud, with_normal=False, SO3=False):
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


def generate_rotated_PC(root_dir):
    root_dir = Path(root_dir)
    classes = find_classes(root_dir)

    for category in classes.keys():
        print(category)
        new_dir = root_dir/Path(category)
        for file in os.listdir(new_dir):
            if file.endswith('pcd'):
                pc_o3d = o3d.io.read_point_cloud(filename=str(new_dir/file))
                pc_np = np.asarray(pc_o3d.points).astype(np.float32)
                for i in range(4):
                    pc_rotated = rand_rotation(pc_np)
                    new_file_name = str(new_dir/file)[:-4] + '_' + str(i) + '.pcd'
                    print(new_file_name)
                    new_pcd = o3d.geometry.PointCloud()
                    new_pcd.points = o3d.utility.Vector3dVector(pc_rotated)
                    o3d.io.write_point_cloud(new_file_name, new_pcd)

                break
        break


if __name__ == "__main__":
    generate_rotated_PC("/home/airocs/Desktop/rotate_visual_data_pcd")



