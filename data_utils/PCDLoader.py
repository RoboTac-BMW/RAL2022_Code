import open3d as o3d
import numpy as np
# import pcl
# from pypcd import pypcd

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




if __name__ == '__main__':
    # test_pcd = load_pcd("/home/airocs/Desktop/visual_data_pcd/eraser/1.pcd")
    test_pcd = read_pcd("/home/airocs/Desktop/visual_data_pcd/eraser/1.pcd")
    print(type(test_pcd))
    print(test_pcd.shape)

    pc_normalized = normalizePCD(test_pcd)
    print(pc_normalized.shape)

    pointcloud_withNormal = add_normal(pc_normalized)
    print(type(pointcloud_withNormal))
    print(pointcloud_withNormal.has_normals())
    # o3d.visualization.draw(pointcloud_withNormal,show_ui=True)
    o3d.geometry.PointCloud.orient_normals_to_align_with_direction(
        pointcloud_withNormal,
        orientation_reference=np.array([0., 0., 1.])
    )

    # o3d.geometry.orient_normals_towards_camera_location(
    #     pointcloud_withNormal,
    #     orientation_reference=np.array([0., 0., 1.])
    # )
    o3d.visualization.draw(pointcloud_withNormal, show_ui=True)
    o3d.visualization.draw_geometries([pointcloud_withNormal],
                                      point_show_normal=True)
                                      # show_ui=True,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024])
    # o3d.visualization.draw

    # print(test_pcd.data)
    # print(test_pcd.fields)
    # print(test_pcd.count)
    # print(test_pcd.width)
    # print(test_pcd.height)
    # print(type(test_pcd.pc_data[0]))
    # print(test_pcd.pc_data.shape)
