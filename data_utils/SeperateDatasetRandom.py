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


def create_dir(parent_dir, child_dir):
    path = os.path.join(parent_dir, child_dir)
    print(str(path))
    try:
        os.mkdir(path)
        print("Create new folder")
    except FileExistsError:
        print("Folder already exists")

def move_file(parent_dir, child_dir, file_name):
    try:
        os.rename(str(parent_dir) + '/' + str(file_name),
                  str(parent_dir) + '/' + str(child_dir) + '/' +  str(file_name))
        print("Moved " + str(file_name) + " into folder: " + str(child_dir))
    except FileExistsError:
        print("File already exists")

def seperate_dataset(root_dir, num_test):
    root_dir = Path(root_dir)
    classes = find_classes(root_dir)

    for category in classes.keys():
        print(category)
        new_dir = root_dir/Path(category)
        create_dir(new_dir, "Train")
        create_dir(new_dir, "Test")

        for i in range(num_test):
            fileName = random.choice(os.listdir(new_dir))
            if fileName.endswith('pcd'):
                move_file(new_dir, "Test", fileName)

        for other_files in os.listdir(new_dir):
            if other_files.endswith('pcd'):
                move_file(new_dir, "Train", other_files)


if __name__ == "__main__":
    # seperate_dataset("/home/airocs/Desktop/Dataset/Rotated_visual_data_pcd/", 25)
    seperate_dataset("/home/airocs/Desktop/Dataset/Sampled_tactile_data_set", 20)



