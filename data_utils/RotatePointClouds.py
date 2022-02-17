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


def rand_rotation_matrix(with_normal=False, SO3=False):
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
        return rot_matrix_with_normal
    else:
        return rot_matrix


def generate_normals(root_dir):
    root_dir = Path(root_dir)
    classes = find_classes(root_dir)

    for category in classes.keys():
        print(category)
        new_dir = root_dir/Path(category)
        for file in os.listdir(new_dir):
            if file.endswith('pcd'):
                point_cloud = o3d.io.read_point_cloud(filename=str(new_dir/file))
                point_np = np.asarray(point_cloud.points).astype(np.float32)
                point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                          radius=0.1, max_nn=16))
                point_cloud.normalize_normals()


                # o3d.geometry.PointCloud.orient_normals_to_align_with_direction(
                #     point_cloud,
                #     orientation_reference=np.array([0., 0., 1.])
                # )
                o3d.geometry.PointCloud.orient_normals_towards_camera_location(
                    point_cloud,
                    # camera_location= np.mean(point_np, axis=0)
                    camera_location= np.array([0., 0., 0.])
                )

                o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
                # o3d.io.write_point_cloud(file, point_cloud, write_ascii=True)




def generate_rotated_PC(root_dir, folder=None, times=4):
    root_dir = Path(root_dir)
    classes = find_classes(root_dir)

    for category in classes.keys():
        print(category)
        if folder is None:
            new_dir = root_dir/Path(category)
        else:
            new_dir= root_dir/Path(category)/folder

        for file in os.listdir(new_dir):
            if file.endswith('pcd'):
                pc_o3d = o3d.io.read_point_cloud(filename=str(new_dir/file))
                # pc_np = np.asarray(pc_o3d.points).astype(np.float32)
                for i in range(times):
                    # pc_rotated = rand_rotation(pc_np)
                    new_file_name = str(new_dir/file)[:-4] + '_' + str(i+1) + '.pcd'
                    print(new_file_name)
                    rand_R = rand_rotation_matrix()
                    # print(rand_R)
                    # print(type(rand_R))
                    pc_o3d.rotate(rand_R)
                    o3d.io.write_point_cloud(new_file_name, pc_o3d)



def generate_subsampled_PC(root_dir, folder=None, min_num=0.8, times=4, sample='Random', replace=False):
    root_dir = Path(root_dir)
    classes = find_classes(root_dir)

    for category in classes.keys():
        print(category)
        if folder is None:
            new_dir = root_dir/Path(category)
        else:
            new_dir= root_dir/Path(category)/folder

        for file in os.listdir(new_dir):
            if file.endswith('pcd'):
                pc_o3d = o3d.io.read_point_cloud(filename=str(new_dir/file))
                point_np = np.asarray(pc_o3d.points).astype(np.float32)
                point_num = point_np.shape[0]
                # print(point_np.shape)
                if(type(min_num) is float):
                    min_num = int(min_num * point_num)
                    if min_num < 50:
                        min_num = 50

                elif (type(min_num) is int):
                    min_num = min_num

                else:
                    raise TypeError("Wrong input for min_num")

                for i in range(times):
                    if replace is False:
                        new_file_name = str(new_dir/file)[:-4] + '_' + str(i+1) + '.pcd'
                    else:
                        new_file_name = str(new_dir/file)

                    print(new_file_name)
                    if sample is 'Random':
                        sample_pt_num = random.randint(min_num, point_num)
                    elif sample is 'Uniform':
                        sample_pt_num = min_num
                    else:
                        raise ValueError("Wrong input for sample")

                    sel_ptx_idx = np.random.choice(point_np.shape[0],
                                                   size=sample_pt_num,
                                                   replace=False).reshape(-1)
                    sampled_pointcloud = point_np[sel_ptx_idx]
                    print(sampled_pointcloud.shape)
                    new_pcd = o3d.geometry.PointCloud()
                    new_pcd.points = o3d.utility.Vector3dVector(sampled_pointcloud)
                    o3d.io.write_point_cloud(new_file_name, new_pcd)



if __name__ == "__main__":
    # generate_rotated_PC("/home/airocs/Desktop/Rotated_visual_data_pcd")
    # generate_subsampled_PC("/home/airocs/Desktop/sampled_tactile_data_set")
    # generate_rotated_PC("/home/airocs/Desktop/sampled_tactile_data_set")
    # generate_normals("/home/airocs/Desktop/sampled_tactile_data_set")
    # generate_subsampled_PC("/home/airocs/Desktop/sampled_tactile_data_set", folder="Train",
    #                        min_num=50, times=1, sample='Uniform', replace=True)
    generate_subsampled_PC("/home/airocs/Desktop/sampled_tactile_data_set", folder="Test", min_num=30, times=5)




