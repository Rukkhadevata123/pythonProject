import numpy as np
import matplotlib.pyplot as plt


def translation_matrix(tx, ty, tz):
    """创建平移变换矩阵"""
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T


def scale_matrix(sx, sy, sz):
    """创建缩放变换矩阵"""
    S = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])
    return S


def rotation_z_matrix(angle_degrees):
    """创建绕Z轴旋转变换矩阵"""
    angle_rad = np.radians(angle_degrees)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    R = np.array(
        [
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return R


def scale_around_center_matrix(sx, sy, sz, center):
    """创建以指定中心进行缩放的变换矩阵"""
    cx, cy, cz = center

    # 先平移到原点，缩放，再平移回去
    T1 = translation_matrix(-cx, -cy, -cz)
    S = scale_matrix(sx, sy, sz)
    T2 = translation_matrix(cx, cy, cz)

    # 组合变换矩阵
    return T2 @ S @ T1


def to_homogeneous(point):
    """将3D点转换为齐次坐标"""
    return np.array([point[0], point[1], point[2], 1])


def from_homogeneous(point):
    """将齐次坐标转换为3D点"""
    return np.array([point[0], point[1], point[2]])


def main():
    A = np.array([1, 1, 1])
    B = np.array([3, 1, 1])
    C = np.array([2, 3, 1])

    print("原始三角形顶点：")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")

    centroid = (A + B + C) / 3
    print(f"\n三角形重心：{centroid}")

    A_h = to_homogeneous(A)
    B_h = to_homogeneous(B)
    C_h = to_homogeneous(C)

    T = translation_matrix(-2, 4, 1)
    print("\n1. 平移变换矩阵：")
    print(T)

    A1_h = T @ A_h
    B1_h = T @ B_h
    C1_h = T @ C_h

    A1 = from_homogeneous(A1_h)
    B1 = from_homogeneous(B1_h)
    C1 = from_homogeneous(C1_h)

    print("\n平移后的顶点：")
    print(f"A1 = {A1}")
    print(f"B1 = {B1}")
    print(f"C1 = {C1}")

    # 重心平移
    new_centroid = centroid + np.array([-2, 4, 1])

    # 2. 缩放
    S = scale_around_center_matrix(1.5, 0.5, 1.0, new_centroid)
    print("\n2. 以重心为中心的缩放变换矩阵：")
    print(S)

    A2_h = S @ A1_h
    B2_h = S @ B1_h
    C2_h = S @ C1_h

    A2 = from_homogeneous(A2_h)
    B2 = from_homogeneous(B2_h)
    C2 = from_homogeneous(C2_h)

    print("\n缩放后的顶点：")
    print(f"A2 = {A2}")
    print(f"B2 = {B2}")
    print(f"C2 = {C2}")

    # 3. 旋转
    R = rotation_z_matrix(60)
    print("\n3. 绕Z轴旋转60度的变换矩阵：")
    print(R)

    A3_h = R @ A2_h
    B3_h = R @ B2_h
    C3_h = R @ C2_h

    A_prime = from_homogeneous(A3_h)
    B_prime = from_homogeneous(B3_h)
    C_prime = from_homogeneous(C3_h)

    print("\n最终变换后的顶点：")
    print(f"A' = {A_prime}")
    print(f"B' = {B_prime}")
    print(f"C' = {C_prime}")

    # 组合变换
    M_combined = R @ S @ T
    print("\n组合变换矩阵 M = R × S × T：")
    print(M_combined)

    A_prime_combined = from_homogeneous(M_combined @ A_h)
    B_prime_combined = from_homogeneous(M_combined @ B_h)
    C_prime_combined = from_homogeneous(M_combined @ C_h)

    print("\n组合变换结果：")
    print(f"A' = {A_prime_combined}")
    print(f"B' = {B_prime_combined}")
    print(f"C' = {C_prime_combined}")

    # 下面开始绘图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    original_vertices = np.array([A, B, C, A])  # 闭合
    ax.plot(
        original_vertices[:, 0],
        original_vertices[:, 1],
        original_vertices[:, 2],
        "b-",
        linewidth=2,
        label="Original Triangle",
    )
    ax.scatter(
        [A[0], B[0], C[0]],
        [A[1], B[1], C[1]],
        [A[2], B[2], C[2]],
        color="blue",
        s=100,
        alpha=0.8,
    )

    # 绘制变换后的三角形
    transformed_vertices = np.array([A_prime, B_prime, C_prime, A_prime])  # 闭合
    ax.plot(
        transformed_vertices[:, 0],
        transformed_vertices[:, 1],
        transformed_vertices[:, 2],
        "r-",
        linewidth=2,
        label="Transformed Triangle",
    )
    ax.scatter(
        [A_prime[0], B_prime[0], C_prime[0]],
        [A_prime[1], B_prime[1], C_prime[1]],
        [A_prime[2], B_prime[2], C_prime[2]],
        color="red",
        s=100,
        alpha=0.8,
    )

    ax.text(A[0], A[1], A[2], "A", fontsize=12, color="blue")
    ax.text(B[0], B[1], B[2], "B", fontsize=12, color="blue")
    ax.text(C[0], C[1], C[2], "C", fontsize=12, color="blue")

    ax.text(A_prime[0], A_prime[1], A_prime[2], "A'", fontsize=12, color="red")
    ax.text(B_prime[0], B_prime[1], B_prime[2], "B'", fontsize=12, color="red")
    ax.text(C_prime[0], C_prime[1], C_prime[2], "C'", fontsize=12, color="red")

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Triangle Affine Transformation\n(Translation → Scaling → Rotation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 坐标轴等比例显示
    all_points = np.vstack([original_vertices[:-1], transformed_vertices[:-1]])
    max_range = np.array(
        [all_points[:, 0].max(), all_points[:, 1].max(), all_points[:, 2].max()]
    ).max()
    min_range = np.array(
        [all_points[:, 0].min(), all_points[:, 1].min(), all_points[:, 2].min()]
    ).min()

    ax.set_xlim(min_range - 1, max_range + 1)
    ax.set_ylim(min_range - 1, max_range + 1)
    ax.set_zlim(min_range - 1, max_range + 1)

    plt.savefig("triangle_affine_transformation.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
