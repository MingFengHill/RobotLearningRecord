from scipy.spatial.transform import Rotation as R

# 旋转举证 -> 欧拉角
r = R.from_matrix([[-0.0003, 1.0000, 0.0006],
                    [-1.0000, -0.0003, 0.0006],
                    [0.0006, -0.0006, 1.0000]])
print(r.as_euler('xyz', degrees=False))