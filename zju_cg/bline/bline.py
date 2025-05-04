import numpy as np
import matplotlib.pyplot as plt


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
        raise ValueError(f"基函数索引i必须在[0, {len(knots)-k-1}]范围内")

    N = np.zeros((len(knots), k))

    # 初始化0阶基函数
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


def calculate_bspline_point(control_points, u, knots, k):
    """
    计算B样条曲线上的点
    control_points: 控制点列表
    u: 参数值
    knots: 节点矢量
    k: 阶数
    """
    # 输入验证
    if len(control_points) < k:
        raise ValueError(f"至少需要{k}个控制点")

    point = np.zeros(2)
    for i in range(len(control_points)):
        basis = bspline_basis_iterative(i, k, u, knots)
        point += basis * np.array(control_points[i])
    return point


def generate_uniform_bspline(control_points, degree=3, num_points=500, clamped=True):
    """
    生成均匀B样条曲线
    control_points: 控制点列表
    degree: 曲线次数
    num_points: 采样点数量
    clamped: 是否使用夹紧B样条
    """
    n = len(control_points)
    k = degree + 1

    # 节点向量生成
    if clamped:
        if n < k:
            raise ValueError(f"夹紧B样条需要至少{k}个控制点")
        inner_knots = list(range(1, n - k + 2)) if n > k else []
        knots = [0] * k + inner_knots + [n - k + 1] * k
        u_min, u_max = 0, n - k + 1
    else:
        knots = list(range(n + k))
        u_min, u_max = k - 1, n

    # 参数采样
    u_values = np.linspace(u_min, u_max, num_points)

    # 计算曲线点
    curve_points = []
    for u in u_values:
        try:
            point = calculate_bspline_point(control_points, u, knots, k)
            curve_points.append(point)
        except ValueError as e:
            print(f"参数u={u}计算失败: {str(e)}")
            continue

    return np.array(curve_points), u_values


def plot_basic_bspline(control_points, curve_points, degree):
    """绘制B样条曲线基本视图"""
    plt.figure(figsize=(10, 6))

    # 控制多边形
    control_points = np.array(control_points)
    plt.plot(control_points[:, 0], control_points[:, 1], "r--", label="Control Polygon")
    plt.scatter(
        control_points[:, 0], control_points[:, 1], c="red", label="Control Points"
    )

    # B样条曲线
    plt.plot(
        curve_points[:, 0],
        curve_points[:, 1],
        "b-",
        linewidth=2,
        label=f"Degree {degree} B-spline",
    )

    # 图表设置
    plt.title(f"B-spline Curve (Degree {degree})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_basis_functions(degree, num_control_points, clamped=True):
    """绘制B样条基函数"""
    k = degree + 1
    n = num_control_points

    # 节点向量生成
    if clamped:
        if n < k:
            raise ValueError(f"需要至少{k}个控制点")
        inner_knots = list(range(1, n - k + 2)) if n > k else []
        knots = [0] * k + inner_knots + [n - k + 1] * k
        u_min, u_max = 0, n - k + 1
    else:
        knots = list(range(n + k))
        u_min, u_max = k - 1, n

    # 参数采样
    u_values = np.linspace(u_min, u_max, 1000)

    plt.figure(figsize=(12, 6))

    # 计算并绘制基函数
    for i in range(n):
        basis = [
            (
                bspline_basis_iterative(i, k, u, knots)
                if (u >= knots[i] and u <= knots[i + k])
                else 0
            )
            for u in u_values
        ]
        plt.plot(u_values, basis, label=f"N_{i}")

    # 标记有效范围
    plt.axvline(x=u_min, color="r", linestyle="--", alpha=0.5)
    plt.axvline(x=u_max, color="r", linestyle="--", alpha=0.5)

    plt.title(
        f"B-spline Basis Functions (Degree {degree}, {'Clamped' if clamped else 'Uniform'})"
    )
    plt.xlabel("Parameter u")
    plt.ylabel("Basis Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # 控制点定义
    control_points = [(0, 0), (1, 2), (2, 1), (3, 2), (4, 0), (5, 1), (6, 0.5)]

    degree = 3  # 三次B样条

    try:
        # 生成曲线
        curve_points, u_values = generate_uniform_bspline(
            control_points, degree=degree, clamped=True
        )

        # 绘制曲线
        plot_basic_bspline(control_points, curve_points, degree)

        # 绘制基函数
        plot_basis_functions(degree, len(control_points), clamped=True)

        # 输出信息
        print(f"控制点数量: {len(control_points)}")
        print(f"B样条次数: {degree}")
        print(f"生成曲线点数量: {len(curve_points)}")

    except ValueError as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
