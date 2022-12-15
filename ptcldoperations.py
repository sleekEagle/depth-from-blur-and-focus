#record mkv video and then extract depth and RGB images from the mkv video with open3D provided Azure Kinect code
#http://www.open3d.org/docs/latest/tutorial/Basic/azure_kinect.html

import open3d as o3d
import numpy as np

color_raw = o3d.io.read_image("C:\\Users\\lahir\\anaconda3\\envs\\defocus\\Lib\\site-packages\\open3d\\frames\\color\\00085.jpg")
depth_raw = o3d.io.read_image("C:\\Users\\lahir\\anaconda3\\envs\\defocus\\Lib\\site-packages\\open3d\\frames\\depth\\00085.png")

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
print(rgbd_image)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3))) 
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)

o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])

