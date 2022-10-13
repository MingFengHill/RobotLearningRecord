import numpy as np
import open3d as o3d


# urdfpy
def matrix_to_rpy(R, solution=1):
    """Convert a 3x3 transform matrix to roll-pitch-yaw coordinates.

    The roll-pitchRyaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    There are typically two possible roll-pitch-yaw coordinates that could have
    created a given rotation matrix. Specify ``solution=1`` for the first one
    and ``solution=2`` for the second one.

    Parameters
    ----------
    R : (3,3) float
        A 3x3 homogenous rotation matrix.
    solution : int
        Either 1 or 2, indicating which solution to return.

    Returns
    -------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    """
    R = np.asanyarray(R, dtype=np.float64)
    r = 0.0
    p = 0.0
    y = 0.0

    if np.abs(R[2,0]) >= 1.0 - 1e-12:
        y = 0.0
        if R[2,0] < 0:
            p = np.pi / 2
            r = np.arctan2(R[0,1], R[0,2])
        else:
            p = -np.pi / 2
            r = np.arctan2(-R[0,1], -R[0,2])
    else:
        if solution == 1:
            p = -np.arcsin(R[2,0])
        else:
            p = np.pi + np.arcsin(R[2,0])
        r = np.arctan2(R[2,1] / np.cos(p), R[2,2] / np.cos(p))
        y = np.arctan2(R[1,0] / np.cos(p), R[0,0] / np.cos(p))

    return np.array([r, p, y], dtype=np.float64)


def rotation_matrix_2_rpy():
    rotation_matrix = []
    with open('rotation_matrix.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            rotation_matrix.append(list(map(float, line.split(' '))))
    if len(rotation_matrix) != 3 or len(rotation_matrix[0]) != 3:
        print("[ERROR] 矩阵不是3x3")
        return
    rotation_matrix = np.asarray(rotation_matrix)
    angles = matrix_to_rpy(rotation_matrix)
    print("R P Y: {}".format(angles))


def quaternion_to_rotation_matrix(q):
    """ 四元数转旋转矩阵 """
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    # 检查四元数是否单位化
    if (x ** 2 + y ** 2 + z ** 2 + w ** 2) != 1:
        print("[WARR] Not a unit quaternion: {}".format(x ** 2 + y ** 2 + z ** 2 + w ** 2))
    # 四元数转旋转矩阵
    # https://zhuanlan.zhihu.com/p/45404840
    r11 = 1 - 2 * y * y - 2 * z * z
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * x * z + 2 * w * y
    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x * x - 2 * z * z
    r23 = 2 * y * z - 2 * w * x
    r31 = 2 * x * z - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x * x - 2 * y * y
    return [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]


def transform_matrix_inverse(base2end):
    base2end.append([0, 0, 0, 1])
    base2end = np.matrix(base2end)
    end2base_matrix = np.linalg.inv(base2end)
    print(end2base_matrix)
    end2base = ([[end2base_matrix[0][0], end2base_matrix[0][1], end2base_matrix[0][2], end2base_matrix[0][3]],
                 [end2base_matrix[1][0], end2base_matrix[1][1], end2base_matrix[1][2], end2base_matrix[0][3]],
                 [end2base_matrix[2][0], end2base_matrix[2][1], end2base_matrix[2][2], end2base_matrix[0][3]]])
    return end2base


def read_image(path):
    # 支持选中点
    pcd = o3d.io.read_point_cloud(path)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()

    # 画出选中点，验证选中点的位置符合预期
    selected_points = vis.get_picked_points()
    if len(selected_points) > 0:
        draw = [pcd]
        point_array = np.asarray(pcd.points)
        for i in selected_points:
            mesh_circle = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mesh_circle.compute_vertex_normals()
            mesh_circle.paint_uniform_color([0.9, 0.1, 0.1])
            mesh_circle = mesh_circle.translate((point_array[i][0], point_array[i][1], point_array[i][2]))
            draw.append(mesh_circle)
        o3d.visualization.draw_geometries(draw)

def main():
    pass


if __name__ == '__main__':
    main()
