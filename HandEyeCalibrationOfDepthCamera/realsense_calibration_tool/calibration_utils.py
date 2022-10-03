import numpy as np
import open3d as o3d


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


def read_iamge(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])


def main():
    pass


if __name__ == '__main__':
    main()
