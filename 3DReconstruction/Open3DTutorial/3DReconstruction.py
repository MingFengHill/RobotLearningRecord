import os
import sys
import math
import time
import random
import string
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
import scipy.optimize as opt
import open3d_tutorial as o3dtut


def alpha_shapes():
    mesh = o3dtut.get_bunny_mesh()
    pcd = mesh.sample_points_poisson_disk(750)
    o3d.visualization.draw_geometries([pcd])
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


def ball_pivoting():
    gt_mesh = o3dtut.get_bunny_mesh()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(3000)
    o3d.visualization.draw_geometries([pcd])
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd, rec_mesh])


def poisson_surface_reconstruction():
    pcd = o3dtut.get_eagle_pcd()
    print(pcd)
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])


def main():
    poisson_surface_reconstruction()


if __name__ == "__main__":
    main()