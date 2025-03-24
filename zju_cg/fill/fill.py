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


# 多边形填充算法 - 活性边表法（优化版）
def poly_fill_aet(img, polygon, fill_color):
    """
    基于活性边表(AET)和新边表(NET)的多边形填充算法

    参数:
    img: 图像数组
    polygon: 多边形顶点列表 [(x1,y1), (x2,y2), ..., (xn,yn)]
    fill_color: 填充颜色
    """
    # 确定扫描线的范围（y坐标的最小值和最大值）
    y_min = max(0, min(int(vertex[1]) for vertex in polygon))
    y_max = min(height - 1, max(int(vertex[1]) for vertex in polygon))

    # 初始化新边表NET，为每条可能的扫描线创建一个空列表
    NET = [[] for _ in range(y_max + 1)]

    # 构建新边表，处理多边形的所有边
    edges_count = len(polygon)
    for i in range(edges_count):
        # 获取边的两个端点
        x1, y1 = int(polygon[i][0]), int(polygon[i][1])
        x2, y2 = int(polygon[(i + 1) % edges_count][0]), int(
            polygon[(i + 1) % edges_count][1]
        )

        # 确保y1 <= y2，如果不是则交换端点
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        # 计算边的x的增量（斜率的倒数）
        if y2 != y1:  # 忽略水平边
            delta_x = (x2 - x1) / (y2 - y1)
            # 将边的信息添加到对应扫描线的新边表中
            NET[y1].append([x1, y2, delta_x])

    # 初始化活性边表AET为空
    AET = []

    # 按扫描线从下到上处理
    for y in range(y_min, y_max + 1):
        # 将新边表中y_min等于当前扫描线的边加入AET
        AET.extend(NET[y])

        # 按x坐标从小到大排序AET
        AET.sort(key=lambda edge: edge[0])

        # 对配对的交点进行填充
        for i in range(0, len(AET), 2):
            if i + 1 < len(AET):
                # 左闭右开区间
                x_start = max(0, int(AET[i][0]))
                x_end = min(width - 1, int(AET[i + 1][0]))

                # 在扫描线上填充颜色
                for x in range(x_start, x_end + 1):  # 注意是闭区间，避免缝隙
                    draw_pixel(img, x, y, fill_color)

        # 更新AET，移除已完成的边（y_max = 当前扫描线）
        AET = [edge for edge in AET if edge[1] > y]

        # 更新剩余边的x值
        for edge in AET:
            edge[0] += edge[2]  # x += delta_x


# 多边形填充算法 - 边界标志法（修复版）
def poly_fill_edge_flag(img, polygon, fill_color, background_color=np.array([1, 1, 1])):
    """
    基于边界标志的多边形填充算法（修复版）

    参数:
    img: 图像数组
    polygon: 多边形顶点列表 [(x1,y1), (x2,y2), ..., (xn,yn)]
    fill_color: 填充颜色
    background_color: 背景颜色
    """
    # 第一步：找出多边形的边界框
    x_min = max(0, min(int(vertex[0]) for vertex in polygon))
    x_max = min(width - 1, max(int(vertex[0]) for vertex in polygon))
    y_min = max(0, min(int(vertex[1]) for vertex in polygon))
    y_max = min(height - 1, max(int(vertex[1]) for vertex in polygon))

    # 创建一个边界标志矩阵，初始化为False
    edge_marks = np.zeros((height, width), dtype=bool)  # 注意shape是(height, width)

    # 第二步：标记多边形的所有边
    edges_count = len(polygon)
    for i in range(edges_count):
        # 获取边的两个端点
        x1, y1 = int(polygon[i][0]), int(polygon[i][1])
        x2, y2 = int(polygon[(i + 1) % edges_count][0]), int(
            polygon[(i + 1) % edges_count][1]
        )

        # 使用Bresenham算法对每条边进行直线扫描转换
        line_points = bresenham_line(x1, y1, x2, y2)

        # 标记边界点
        for x, y in line_points:
            if 0 <= x < width and 0 <= y < height:
                edge_marks[y, x] = True  # 注意是edge_marks[y, x]而非edge_marks[x, y]

    # 第三步：根据边界标志填充多边形（优化版）
    for y in range(y_min, y_max + 1):
        # 收集当前扫描线上的所有边界点
        intersections = []
        for x in range(x_min, x_max + 1):
            if edge_marks[y, x]:
                intersections.append(x)

        # 确保交点数量为偶数（处理特殊情况）
        if len(intersections) % 2 == 1:
            intersections.append(x_max)  # 添加一个边界点保证偶数

        # 按交点排序
        intersections.sort()

        # 成对处理交点，在每对交点之间填充颜色
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                start_x = intersections[i]
                end_x = intersections[i + 1]

                # 填充当前区间
                for x in range(start_x, end_x + 1):
                    draw_pixel(img, x, y, fill_color)


# 种子填充算法 - 递归版
def boundary_fill4_recursive(
    img, x, y, boundary_color, fill_color, depth=0, max_depth=3000
):
    """
    边界表示的4连通区域的递归填充算法（增加深度限制）

    参数:
    img: 图像数组
    x, y: 种子点坐标
    boundary_color: 边界颜色
    fill_color: 填充颜色
    depth: 当前递归深度
    max_depth: 最大递归深度
    """
    # 添加深度限制，防止栈溢出
    if depth > max_depth:
        return

    current_color = get_pixel(img, x, y)
    if (
        current_color is not None
        and not np.array_equal(current_color, boundary_color)
        and not np.array_equal(current_color, fill_color)
    ):

        draw_pixel(img, x, y, fill_color)

        # 递归处理4个相邻点，增加深度计数
        boundary_fill4_recursive(
            img, x + 1, y, boundary_color, fill_color, depth + 1, max_depth
        )
        boundary_fill4_recursive(
            img, x - 1, y, boundary_color, fill_color, depth + 1, max_depth
        )
        boundary_fill4_recursive(
            img, x, y + 1, boundary_color, fill_color, depth + 1, max_depth
        )
        boundary_fill4_recursive(
            img, x, y - 1, boundary_color, fill_color, depth + 1, max_depth
        )


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
    inner_radius = 100
    outer_radius = 280
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


# 主函数：展示不同的填充算法
def main():
    # 创建一个凹多边形
    polygon = create_concave_polygon()

    # 定义颜色
    background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
    boundary_color = np.array([0.0, 0.0, 0.0])  # 黑色边界
    fill_color = np.array([0.6, 0.2, 0.8])  # 紫色填充

    # 定义种子点（多边形内部的一点）
    seed_x, seed_y = width // 2, height // 2

    # 创建图像并测试不同的填充算法
    plt.figure(figsize=(15, 10))

    # 1. 活性边表法
    img1 = np.ones((height, width, 3)) * background_color
    poly_fill_aet(img1, polygon, fill_color)
    plt.subplot(221)
    plt.imshow(img1)
    plt.title("poly_fill_aet")
    plt.axis("off")

    # 2. 边界标志法
    img2 = np.ones((height, width, 3)) * background_color
    poly_fill_edge_flag(img2, polygon, fill_color)
    plt.subplot(222)
    plt.imshow(img2)
    plt.title("poly_fill_edge_flag")
    plt.axis("off")

    # 3. 绘制多边形边界用于种子填充算法
    img3 = np.ones((height, width, 3)) * background_color
    img4 = np.ones((height, width, 3)) * background_color

    # 绘制多边形边界
    for i in range(len(polygon)):
        x1, y1 = int(polygon[i][0]), int(polygon[i][1])
        x2, y2 = int(polygon[(i + 1) % len(polygon)][0]), int(
            polygon[(i + 1) % len(polygon)][1]
        )
        for x, y in bresenham_line(x1, y1, x2, y2):
            if 0 <= x < width and 0 <= y < height:
                img3[y, x] = boundary_color
                img4[y, x] = boundary_color

    # 3. 递归种子填充
    try:
        boundary_fill4_recursive(img3, seed_x, seed_y, boundary_color, fill_color)
    except RecursionError:
        # 如果发生递归错误，填充部分区域
        img3[10:30, 10:200] = [1, 0, 0]  # 添加红色警告区域

    plt.subplot(223)
    plt.imshow(img3)
    plt.title("boundary_fill4_recursive")
    plt.axis("off")

    # 4. 扫描线种子填充
    scanline_seed_fill(img4, seed_x, seed_y, fill_color, boundary_color)
    plt.subplot(224)
    plt.imshow(img4)
    plt.title("scanline_seed_fill")
    plt.axis("off")

    # 保存图像
    plt.tight_layout()
    plt.savefig("/home/yoimiya/zju_cg/fill/polygon_fill_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
