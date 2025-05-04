import numpy as np
import matplotlib.pyplot as plt
import time

# 定义控制点网格 (4x4)
CONTROL_POINTS = np.array(
    [
        [[0, 0, 0], [1, 0, 1], [2, 0, 1], [3, 0, 0]],
        [[0, 1, 1], [1, 1, 2], [2, 1, 2], [3, 1, 1]],
        [[0, 2, 1], [1, 2, 2], [2, 2, 2], [3, 2, 1]],
        [[0, 3, 0], [1, 3, 1], [2, 3, 1], [3, 3, 0]],
    ]
)


# 计算组合数
def comb(n, k):
    """计算组合数 C(n,k)"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    # 计算组合数 C(n,k)
    result = 1
    for i in range(k):
        result *= n - i
        result //= i + 1
    return result


# 计算伯恩斯坦多项式
def bernstein(n, i, t):
    """计算伯恩斯坦多项式 B_{i,n}(t)"""
    return comb(n, i) * (t**i) * ((1 - t) ** (n - i))


# 方法一：直接使用伯恩斯坦多项式计算法
def bezier_surface_bernstein(control_points, sample_u=20, sample_v=20):
    """使用伯恩斯坦多项式直接计算贝塞尔曲面点"""
    print("\n===== 使用伯恩斯坦多项式计算法 =====")
    start_time = time.time()

    # n = m = 4, coord_dim = 3
    n_degree, m_degree, coord_dim = control_points.shape
    u_values = np.linspace(0, 1, sample_u)
    v_values = np.linspace(0, 1, sample_v)

    # 创建结构化网格以用于绘制曲面
    surface_points = np.zeros((sample_u, sample_v, coord_dim))

    print(f"控制点形状: {control_points.shape}")
    print(f"左下角控制点: {control_points[0, 0]}")
    print(f"右上角控制点: {control_points[n_degree-1, m_degree-1]}")

    # 记录最大和最小值以便检查
    min_z = float("inf")
    max_z = float("-inf")

    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            point = np.zeros(coord_dim)  # 初始化为三维点

            # 双重循环计算双三次贝塞尔曲面点
            for ki in range(n_degree):
                for kj in range(m_degree):
                    # 计算伯恩斯坦多项式系数
                    coef_u = bernstein(n_degree - 1, ki, u)
                    coef_v = bernstein(m_degree - 1, kj, v)

                    # 累加各控制点的贡献
                    point += coef_u * coef_v * control_points[ki, kj]

            surface_points[i, j] = point

            # 更新z值范围
            min_z = min(min_z, point[2])
            max_z = max(max_z, point[2])

            # 打印一些特定点的值以便检查
            if (
                (i == 0 and j == 0)
                or (i == sample_u - 1 and j == sample_v - 1)
                or (i == sample_u // 2 and j == sample_v // 2)
            ):
                print(f"曲面点 u={u:.2f}, v={v:.2f}: {point}")

    end_time = time.time()
    print(f"计算时间: {end_time - start_time:.4f}秒")
    print(f"Z值范围: {min_z} 到 {max_z}")

    # 分离坐标
    surface_x = surface_points[:, :, 0]
    surface_y = surface_points[:, :, 1]
    surface_z = surface_points[:, :, 2]

    return surface_x, surface_y, surface_z


# 方法二：使用de Casteljau递推算法
def bezier_surface_casteljau(control_points, sample_u=20, sample_v=20):
    """使用de Casteljau算法递推计算贝塞尔曲面点"""
    print("\n===== 使用de Casteljau递推算法 =====")
    start_time = time.time()

    n_degree, m_degree, _ = control_points.shape  # n = m = 4
    u_values = np.linspace(0, 1, sample_u)
    v_values = np.linspace(0, 1, sample_v)

    # 创建结构化网格以用于绘制曲面
    surface_points = np.zeros((sample_u, sample_v, 3))

    print(f"控制点形状: {control_points.shape}")
    print(f"左下角控制点: {control_points[0, 0]}")
    print(f"右上角控制点: {control_points[n_degree-1, m_degree-1]}")

    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            # 沿 u 方向递推
            q_points = control_points.copy().astype(np.float64)
            for k in range(1, n_degree):
                temp_q = np.zeros((n_degree - k, m_degree, 3))
                for i_inner in range(n_degree - k):
                    for j_inner in range(m_degree):
                        temp_q[i_inner, j_inner] = (1 - u) * q_points[
                            i_inner, j_inner
                        ] + u * q_points[i_inner + 1, j_inner]
                q_points = temp_q

            # 沿 v 方向递推
            r_points = q_points[0, :, :].copy()  # 沿 u 方向递推后的第一行
            for k in range(1, m_degree):
                temp_r = np.zeros((m_degree - k, 3))
                for j_inner in range(m_degree - k):
                    temp_r[j_inner] = (1 - v) * r_points[j_inner] + v * r_points[
                        j_inner + 1
                    ]
                r_points = temp_r

            surface_points[i, j] = r_points[0]

            # 打印一些特定点的值以便检查
            if (
                (i == 0 and j == 0)
                or (i == sample_u - 1 and j == sample_v - 1)
                or (i == sample_u // 2 and j == sample_v // 2)
            ):
                print(f"曲面点 u={u:.2f}, v={v:.2f}: {r_points[0]}")

    end_time = time.time()
    print(f"计算时间: {end_time - start_time:.4f}秒")

    min_z = np.min(surface_points[:, :, 2])
    max_z = np.max(surface_points[:, :, 2])
    print(f"Z值范围: {min_z} 到 {max_z}")

    # 分离坐标
    surface_x = surface_points[:, :, 0]
    surface_y = surface_points[:, :, 1]
    surface_z = surface_points[:, :, 2]

    return surface_x, surface_y, surface_z


def plot_surface_3d(
    ax,
    surface_x,
    surface_y,
    surface_z,
    control_points,
    colormap="viridis",
    title="Bezier Surface",
):
    """在给定的子图上绘制3D曲面及控制点"""
    # 绘制曲面
    surf = ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        cmap=colormap,
        edgecolor="grey",
        linewidth=0.3,
        alpha=0.8,
        antialiased=True,
    )

    # 绘制控制点
    control_x = control_points[:, :, 0].flatten()
    control_y = control_points[:, :, 1].flatten()
    control_z = control_points[:, :, 2].flatten()
    ax.scatter(control_x, control_y, control_z, color="red", s=30)

    # 绘制控制多边形
    n_points, m_points = control_points.shape[:2]
    for i in range(n_points):
        ax.plot(
            control_points[i, :, 0],
            control_points[i, :, 1],
            control_points[i, :, 2],
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )
    for j in range(m_points):
        ax.plot(
            control_points[:, j, 0],
            control_points[:, j, 1],
            control_points[:, j, 2],
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    # 设置坐标轴和视角
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.view_init(elev=30, azim=-45)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 2.5)
    ax.set_title(title, fontsize=14)

    return surf


# 绘制曲面和控制点多边形 - 比较两种方法
def plot_combined_comparison(
    control_points, surf_x1, surf_y1, surf_z1, surf_x2, surf_y2, surf_z2
):
    """比较两种方法的结果"""
    fig = plt.figure(figsize=(18, 8))

    # 左侧：伯恩斯坦方法
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = plot_surface_3d(
        ax1,
        surf_x1,
        surf_y1,
        surf_z1,
        control_points,
        colormap="viridis",
        title="Bernstein Polynomial Method",
    )

    # 右侧：de Casteljau方法
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = plot_surface_3d(
        ax2,
        surf_x2,
        surf_y2,
        surf_z2,
        control_points,
        colormap="plasma",
        title="de Casteljau Algorithm",
    )

    # 添加颜色条
    fig.colorbar(surf1, ax=ax1, shrink=0.6).set_label("Z Value")
    fig.colorbar(surf2, ax=ax2, shrink=0.6).set_label("Z Value")

    plt.tight_layout()
    plt.savefig("bezier_surface_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


# 绘制详细视图 - 单一方法
def plot_detailed_view(
    control_points,
    surf_x, surf_y, surf_z,
    method_name="Bicubic Bezier Surface"
):
    """详细绘制单个贝塞尔曲面"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制曲面和控制点
    surf = plot_surface_3d(
        ax,
        surf_x,
        surf_y,
        surf_z,
        control_points,
        colormap="viridis",
        title=method_name,
    )

    # 调整额外参数
    ax.get_figure().colorbar(surf, ax=ax, shrink=0.7, aspect=10).set_label(
        "Z Value", fontsize=12
    )
    ax.legend(["Control Points"], fontsize=12, loc="upper right")

    plt.tight_layout()
    filename = f"bezier_surface_{method_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """主函数"""
    # 设置共享参数
    sample_u = 600
    sample_v = 600

    # 使用两种方法计算贝塞尔曲面
    surf_x1, surf_y1, surf_z1 = bezier_surface_bernstein(
        CONTROL_POINTS, sample_u, sample_v
    )
    surf_x2, surf_y2, surf_z2 = bezier_surface_casteljau(
        CONTROL_POINTS, sample_u, sample_v
    )

    # 比较两种方法结果是否一致
    bernstein_points = np.stack([surf_x1, surf_y1, surf_z1], axis=2)
    casteljau_points = np.stack([surf_x2, surf_y2, surf_z2], axis=2)
    diff = np.abs(bernstein_points - casteljau_points).max()
    print(f"\n两种方法计算结果的最大差异: {diff}")

    # 绘制比较视图
    plot_combined_comparison(
        CONTROL_POINTS, surf_x1, surf_y1, surf_z1, surf_x2, surf_y2, surf_z2
    )

    # 绘制各自的详细视图
    plot_detailed_view(CONTROL_POINTS,
                       surf_x1, surf_y1, surf_z1,
                       "Bernstein Method")
    plot_detailed_view(CONTROL_POINTS,
                       surf_x2, surf_y2, surf_z2,
                       "de Casteljau Method")


if __name__ == "__main__":
    main()
