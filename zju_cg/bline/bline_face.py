import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time


def bspline_basis_iterative(i, k, u, knots):
    """
    计算B样条基函数值（迭代方法）
    i: 基函数索引
    k: 阶数(degree+1)
    u: 参数值
    knots: 节点矢量
    """
    # 输入验证
    if not (0 <= i < len(knots) - k):
        return 0.0

    N = np.zeros((len(knots), k))

    # 初始化1阶基函数
    for j in range(len(knots) - 1):
        if knots[j] <= u < knots[j + 1]:
            N[j, 0] = 1.0

    # 处理末端节点特殊情况
    if u == knots[-1] and knots[-2] < knots[-1]:
        N[len(knots) - 2, 0] = 1.0

    # 递推计算高阶基函数
    for p in range(1, k):
        for j in range(len(knots) - p - 1):
            denom1 = knots[j + p] - knots[j]
            left = (u - knots[j]) / denom1 if denom1 != 0 else 0

            denom2 = knots[j + p + 1] - knots[j + 1]
            right = (knots[j + p + 1] - u) / denom2 if denom2 != 0 else 0

            N[j, p] = left * N[j, p - 1] + right * N[j + 1, p - 1]

    return N[i, k - 1]


def calculate_bspline_surface_point(control_points, u, v, u_knots, v_knots, p, q):
    """
    计算B样条曲面上的一个点
    control_points: 控制点网格 [m×n×3]
    u, v: 参数值
    u_knots, v_knots: u和v方向的节点矢量
    p, q: u和v方向的次数
    """
    m, n = control_points.shape[0], control_points.shape[1]
    k_u, k_v = p + 1, q + 1  # 阶数

    # 初始化点坐标
    point = np.zeros(3)

    # 双重循环计算曲面点
    for i in range(m):
        for j in range(n):
            # 计算两个方向的基函数
            N_i_p = bspline_basis_iterative(i, k_u, u, u_knots)
            N_j_q = bspline_basis_iterative(j, k_v, v, v_knots)

            # 累加权重控制点
            point += N_i_p * N_j_q * control_points[i, j]

    return point


def generate_bspline_surface(
    control_points, p=3, q=3, u_samples=30, v_samples=30, clamped=True
):
    """
    生成双三次B样条曲面
    control_points: 控制点网格 [m×n×3]
    p, q: u和v方向的次数（均为3表示双三次）
    u_samples, v_samples: u和v方向的采样点数
    clamped: 是否使用夹紧B样条
    """
    start_time = time.time()

    # 控制点网格维度
    m, n = control_points.shape[0], control_points.shape[1]
    k_u, k_v = p + 1, q + 1  # 阶数

    # 生成节点矢量
    if clamped:
        # 夹紧B样条节点矢量
        inner_u_knots = list(range(1, m - p)) if m > p else []
        inner_v_knots = list(range(1, n - q)) if n > q else []

        u_knots = [0] * k_u + inner_u_knots + [m - p] * k_u
        v_knots = [0] * k_v + inner_v_knots + [n - q] * k_v

        u_min, u_max = 0, m - p
        v_min, v_max = 0, n - q
    else:
        # 均匀B样条节点矢量
        u_knots = list(range(m + k_u))
        v_knots = list(range(n + k_v))

        u_min, u_max = p, m
        v_min, v_max = q, n

    # 参数采样
    u_values = np.linspace(u_min, u_max, u_samples)
    v_values = np.linspace(v_min, v_max, v_samples)

    # 创建结构化网格存储曲面点
    surface_points = np.zeros((u_samples, v_samples, 3))

    print("计算B样条曲面点...")
    # 计算曲面上每个点的坐标
    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            surface_points[i, j] = calculate_bspline_surface_point(
                control_points, u, v, u_knots, v_knots, p, q
            )

            # 打印进度
            if (i * v_samples + j) % (u_samples * v_samples // 10) == 0:
                progress = (i * v_samples + j) / (u_samples * v_samples) * 100
                print(f"进度: {progress:.1f}%")

    end_time = time.time()
    print(f"计算完成，用时: {end_time - start_time:.3f}秒")

    # 分离坐标分量
    surface_x = surface_points[:, :, 0]
    surface_y = surface_points[:, :, 1]
    surface_z = surface_points[:, :, 2]

    return surface_x, surface_y, surface_z, u_values, v_values


def plot_bspline_surface(control_points, surface_x, surface_y, surface_z, p=3, q=3):
    """绘制B样条曲面和控制点网格"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制B样条曲面
    surf = ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        cmap=cm.viridis,
        linewidth=0.3,
        antialiased=True,
        edgecolor="lightgrey",
        alpha=0.8,
    )

    # 绘制控制点网格
    m, n = control_points.shape[0], control_points.shape[1]

    # 绘制控制点
    control_x = control_points[:, :, 0]
    control_y = control_points[:, :, 1]
    control_z = control_points[:, :, 2]

    ax.scatter(
        control_points[:, :, 0].flatten(),
        control_points[:, :, 1].flatten(),
        control_points[:, :, 2].flatten(),
        color="red",
        s=30,
        label="Control Points",
    )

    # 绘制控制网格
    for i in range(m):
        ax.plot(
            control_x[i, :],
            control_y[i, :],
            control_z[i, :],
            "r--",
            linewidth=1.0,
            alpha=0.6,
        )

    for j in range(n):
        ax.plot(
            control_x[:, j],
            control_y[:, j],
            control_z[:, j],
            "r--",
            linewidth=1.0,
            alpha=0.6,
        )

    # 设置图表
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(f"Bicubic B-spline Surface ({p}×{q} degree)", fontsize=16)

    # 添加颜色条
    fig.colorbar(surf, shrink=0.6, aspect=10).set_label("Z-Value")

    # 调整视角
    ax.view_init(elev=30, azim=-45)

    plt.tight_layout()
    plt.savefig(f"bspline_surface_deg_{p}x{q}.png", dpi=300)
    plt.show()


def main():
    """主函数"""
    # 定义控制点网格 (4x4)
    control_points = np.array(
        [
            [[0, 0, 0], [1, 0, 1], [2, 0, 1], [3, 0, 0]],
            [[0, 1, 1], [1, 1, 2], [2, 1, 2], [3, 1, 1]],
            [[0, 2, 1], [1, 2, 2], [2, 2, 2], [3, 2, 1]],
            [[0, 3, 0], [1, 3, 1], [2, 3, 1], [3, 3, 0]],
        ]
    )

    # B样条曲面次数
    p = 3  # u 方向次数
    q = 3  # v 方向次数

    print("开始生成双三次B样条曲面...")
    print(f"控制点网格尺寸: {control_points.shape}")

    # 生成B样条曲面
    surface_x, surface_y, surface_z, u_values, v_values = generate_bspline_surface(
        control_points,
        p=p,
        q=q,
        u_samples=50,
        v_samples=50,
        clamped=True,  # 使用夹紧B样条使曲面通过角点
    )

    # 绘制曲面
    plot_bspline_surface(control_points, surface_x, surface_y, surface_z, p, q)


if __name__ == "__main__":
    main()
