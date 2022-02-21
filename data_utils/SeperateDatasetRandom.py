import open3d as o3d
import numpy as np
import os
import random
import shutil
from pathlib import Path
import math
from scipy.spatial.transform import Rotation as R


def find_classes(root_dir):
    root_dir = Path(root_dir)
    folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    return classes


def create_dir(parent_dir, child_dir):
    path = os.path.join(parent_dir, child_dir)
    print(str(path))
    try:
        os.mkdir(path)
        print("Create new folder")
    except FileExistsError:
        print("Folder already exists")

def move_file(parent_dir, child_dir, file_name, move=True):
    if move is True:
        try:
            os.rename(str(parent_dir) + '/' + str(file_name),
                      str(parent_dir) + '/' + str(child_dir) + '/' +  str(file_name))
            print("Moved " + str(file_name) + " into folder: " + str(child_dir))
        except FileExistsError:
            print("File already exists")

    else:
        try:
            shutil.copyfile(str(parent_dir) + '/' + str(file_name),
                            str(parent_dir) + '/' + str(child_dir) + '/' +  str(file_name))
            print("Copied " + str(file_name) + " into folder: " + str(child_dir))
        except FileExistsError:
            print("File already exists")


def seperate_dataset(root_dir, num_test, move=True):
    root_dir = Path(root_dir)
    classes = find_classes(root_dir)

    for category in classes.keys():
        print(category)
        new_dir = root_dir/Path(category)
        tmp_dir = root_dir/Path(category)/"Train"
        create_dir(new_dir, "Train")
        create_dir(new_dir, "Test")
        create_dir(tmp_dir, "No_use")

        for i in range(num_test):
            fileName = random.choice(os.listdir(tmp_dir))
            if fileName.endswith('pcd'):
                move_file(tmp_dir, "No_use", fileName, move)

        """
        for i in range(num_test):
            fileName = random.choice(os.listdir(new_dir))
            if fileName.endswith('pcd'):
                move_file(new_dir, "Test", fileName, move)

        for other_files in os.listdir(new_dir):
            if other_files.endswith('pcd'):
                move_file(new_dir, "Train", other_files)
        """


if __name__ == "__main__":
    # seperate_dataset("/home/airocs/Desktop/Dataset/Rotated_visual_data_pcd/", 25)
    # seperate_dataset("/home/airocs/Desktop/sampled_tactile_data_set", 20)
    seperate_dataset("/home/prajval/cong_workspace/Pointnet_Pointnet2_pytorch/data/active_vision_pcd_4000", 100, True)



