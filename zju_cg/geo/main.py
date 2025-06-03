import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# 参数设置
radius = 1.0  # 圆的半径
height = 5.0  # 运动总距离
num_steps = 50  # 运动步数
num_points = 40  # 圆周上的点数

# 创建圆周上的点坐标
theta = np.linspace(0, 2 * np.pi, num_points)
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)

# 初始化图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 设置图形属性
ax.set_title("Circle Moving Along Line to Form Solid Cylinder", fontsize=14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-radius * 1.5, radius * 1.5)
ax.set_ylim(-radius * 1.5, radius * 1.5)
ax.set_zlim(0, height)

# 存储所有圆的轨迹，用于构建圆柱体
cylinder_data = []

# 初始化当前运动的圆
(current_circle,) = ax.plot([], [], [], "r-", lw=2)
# 初始化实心圆填充
circle_fill = None


def init():
    """初始化动画"""
    current_circle.set_data([], [])
    current_circle.set_3d_properties([])
    return [current_circle]


def update(step):
    """更新圆的位置并构建圆柱体"""
    z = step / num_steps * height

    # 清除之前的图形
    ax.clear()

    # 重新设置坐标轴
    ax.set_title("Solid Cylinder Generation by Moving Circle", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-radius * 1.5, radius * 1.5)
    ax.set_ylim(-radius * 1.5, radius * 1.5)
    ax.set_zlim(0, height)

    # 存储当前圆的信息
    cylinder_data.append((z, np.copy(x_circle), np.copy(y_circle)))

    # 绘制当前实心圆
    # 使用三角形填充实现实心圆
    r = np.linspace(0, radius, 10)  # 从中心到边缘的半径采样
    for ri in r:
        xi = ri * np.cos(theta)
        yi = ri * np.sin(theta)
        ax.plot(xi, yi, z * np.ones_like(xi), "r-", alpha=0.5, linewidth=0.5)

    # 绘制圆的边缘
    ax.plot(x_circle, y_circle, z * np.ones_like(x_circle), "r-", lw=2)

    # 绘制圆柱体
    if step > 0:
        # 绘制底面（实心圆）
        for ri in np.linspace(0, radius, 100):
            xi = ri * np.cos(theta)
            yi = ri * np.sin(theta)
            ax.plot(xi, yi, 0 * np.ones_like(xi), "b-", alpha=0.3, linewidth=0.5)

        # 获取所有z值
        z_values = [data[0] for data in cylinder_data]

        # 绘制圆柱侧面
        for t in np.linspace(0, 1, 20):  # 在圆周上取点
            angle = 2 * np.pi * t
            x_line = radius * np.cos(angle)
            y_line = radius * np.sin(angle)
            z_line = z_values
            ax.plot(
                [x_line] * len(z_line), [y_line] * len(z_line), z_line, "b-", alpha=0.3
            )

        # 创建表面网格
        r_grid, z_grid = np.meshgrid(np.linspace(0, radius, 10), z_values)

        # 在几个角度上创建实心圆柱体切片
        for angle in np.linspace(0, 2 * np.pi, 100):
            x_surface = r_grid * np.cos(angle)
            y_surface = r_grid * np.sin(angle)
            ax.plot_surface(
                x_surface,
                y_surface,
                z_grid,
                color="blue",
                alpha=0.1,
                linewidth=0,
                antialiased=True,
            )

    # 添加文本说明
    ax.text2D(
        0.05, 0.95, f"Height: {z:.1f}/{height}", transform=ax.transAxes, fontsize=12
    )

    return [ax]


# 创建动画
ani = FuncAnimation(
    fig,
    update,
    frames=np.arange(0, num_steps + 1),
    init_func=init,
    blit=False,
    interval=100,
)

# 显示结果
plt.tight_layout()
plt.show()

# 保存动画（可选）
# ani.save('solid_cylinder_generation.mp4', writer='ffmpeg', fps=10)
