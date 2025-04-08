import matplotlib.pyplot as plt

# 这里定义控制点
P0 = (0, 0)
P1 = (1, 2.2)
P2 = (3, 2)
P3 = (4, 0.32)

def bezier_curve(control_points, num_points=1000):
    n = len(control_points) - 1  # 阶数
    t_values = [i / num_points for i in range(num_points + 1)]
    points = []

    for t in t_values:
        Q = control_points[:]  # 初始化当前层的控制点
        for k in range(1, n + 1):  # 递推更新控制点
            for i in range(n - k + 1):
                Q[i] = (
                    (1 - t) * Q[i][0] + t * Q[i + 1][0],
                    (1 - t) * Q[i][1] + t * Q[i + 1][1],
                )
        points.append(Q[0])  # 最终点就是曲线上的点
    return points


def bresenham_line(x0, y0, x1, y1):
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


def draw_control_polygon(control_points):
    for i in range(len(control_points) - 1):
        line_points = bresenham_line(
            int(control_points[i][0] * 100),
            int(control_points[i][1] * 100),
            int(control_points[i + 1][0] * 100),
            int(control_points[i + 1][1] * 100),
        )
        for x, y in line_points:
            plt.scatter(x / 100, y / 100, color="gray", s=1)


def main():
    control_points = [P0, P1, P2, P3]
    curve_points = bezier_curve(control_points, num_points=1000)
    for i in range(len(curve_points) - 1):
        line_points = bresenham_line(
            int(curve_points[i][0] * 100),
            int(curve_points[i][1] * 100),
            int(curve_points[i + 1][0] * 100),
            int(curve_points[i + 1][1] * 100),
        )
        for x, y in line_points:
            plt.scatter(x / 100, y / 100, color="blue", s=1)
    draw_control_polygon(control_points)
    for point in control_points:
        plt.scatter(point[0], point[1], color="red")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("bezier_curve.png", dpi=300)


if __name__ == "__main__":
    main()
