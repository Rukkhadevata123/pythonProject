import numpy as np
import matplotlib.pyplot as plt

# 画布尺寸
width, height = 700, 700


# 绘制像素的工具函数
def draw_pixel(img, x, y, color):
    """在图像上绘制一个像素"""
    if 0 <= x < width and 0 <= y < height:
        img[y, x] = color


def get_pixel(img, x, y):
    """获取图像上一个像素的颜色"""
    if 0 <= x < width and 0 <= y < height:
        return img[y, x]
    return None


# Bresenham直线算法实现
def bresenham_line(x0, y0, x1, y1):
    """
    使用Bresenham算法计算直线上的点
    返回直线上所有点的列表
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


# 种子填充算法 - 扫描线优化版
def scanline_seed_fill(img, x, y, fill_color, boundary_color):
    """
    扫描线种子填充算法

    参数:
    img: 图像数组
    x, y: 种子点坐标
    fill_color: 填充颜色
    boundary_color: 边界颜色
    """
    # 判断种子点是否有效（不在边界上且未被填充）
    current_color = get_pixel(img, x, y)
    if (
        current_color is None
        or np.array_equal(current_color, boundary_color)
        or np.array_equal(current_color, fill_color)
    ):
        return

    # 创建一个栈来存储种子点
    stack = []
    stack.append((x, y))

    # 主循环
    while stack:
        # 取出栈顶种子点
        x, y = stack.pop()

        # 保存当前扫描线的y坐标
        current_y = y

        # 向左扫描，找到左端点
        xl = x
        while xl > 0:
            left_color = get_pixel(img, xl - 1, current_y)
            if (
                left_color is None
                or np.array_equal(left_color, boundary_color)
                or np.array_equal(left_color, fill_color)
            ):
                break
            xl -= 1

        # 向右扫描，找到右端点
        xr = x
        while xr < width - 1:
            right_color = get_pixel(img, xr + 1, current_y)
            if (
                right_color is None
                or np.array_equal(right_color, boundary_color)
                or np.array_equal(right_color, fill_color)
            ):
                break
            xr += 1

        # 填充当前扫描线的区段[xl, xr]
        for i in range(xl, xr + 1):
            draw_pixel(img, i, current_y, fill_color)

        # 检查上一条扫描线（y+1）
        if current_y < height - 1:
            find_and_add_seeds(
                img, xl, xr, current_y + 1, fill_color, boundary_color, stack
            )

        # 检查下一条扫描线（y-1）
        if current_y > 0:
            find_and_add_seeds(
                img, xl, xr, current_y - 1, fill_color, boundary_color, stack
            )


def find_and_add_seeds(img, xl, xr, y, fill_color, boundary_color, stack):
    """
    在给定扫描线上查找并添加新的种子点

    参数:
    img: 图像数组
    xl, xr: 当前区段的左右端点
    y: 待检查的扫描线y坐标
    fill_color, boundary_color: 填充颜色和边界颜色
    stack: 存储种子点的栈
    """
    span_added = False  # 标记是否已添加种子点

    # 在区间[xl, xr]范围内查找连续的可填充段
    for i in range(xl, xr + 1):
        current_color = get_pixel(img, i, y)
        if (
            current_color is not None
            and not np.array_equal(current_color, boundary_color)
            and not np.array_equal(current_color, fill_color)
        ):
            if not span_added:
                # 找到一段的起始点，添加一个种子点（直接添加当前点）
                stack.append((i, y))
                span_added = True
        else:
            # 遇到不可填充点，重置标记
            span_added = False


# 创建一个不规则凹多边形
def create_concave_polygon():
    """创建一个不规则凹多边形"""
    # 创建一个星形多边形（凹多边形的典型例子）
    n_points = 7  # 星形的点数
    inner_radius = 200
    outer_radius = 300
    center_x, center_y = width // 2, height // 2

    # 创建顶点
    vertices = []
    for i in range(2 * n_points):
        # 交替使用内半径和外半径
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = np.pi * i / n_points
        x = center_x + int(radius * np.cos(angle))
        y = center_y + int(radius * np.sin(angle))
        vertices.append((x, y))

    return vertices


# 演示种子填充算法
def demo_seed_fill():
    # 创建一个凹多边形
    polygon = create_concave_polygon()

    # 定义颜色
    background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
    boundary_color = np.array([0.0, 0.0, 0.0])  # 黑色边界
    fill_color = np.array([0.6, 0.2, 0.8])  # 紫色填充

    # 定义种子点（多边形内部的一点）
    seed_x, seed_y = width // 2, height // 2

    # 创建图像
    img = np.ones((height, width, 3)) * background_color

    # 绘制多边形边界
    for i in range(len(polygon)):
        x1, y1 = int(polygon[i][0]), int(polygon[i][1])
        x2, y2 = int(polygon[(i + 1) % len(polygon)][0]), int(
            polygon[(i + 1) % len(polygon)][1]
        )
        for x, y in bresenham_line(x1, y1, x2, y2):
            if 0 <= x < width and 0 <= y < height:
                img[y, x] = boundary_color

    # 应用扫描线种子填充算法
    scanline_seed_fill(img, seed_x, seed_y, fill_color, boundary_color)

    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title("scanline_seed_fill")
    plt.axis("off")
    plt.savefig("seed_fill_demo.png", dpi=300)


if __name__ == "__main__":
    demo_seed_fill()
