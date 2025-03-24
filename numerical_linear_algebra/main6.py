import numpy as np
import time
from error import estimate_relative_error, iterative_refinement
from lu_decomposition import solve_with_partial_pivoting
from cond import condition_number, normInf


def create_special_matrix(n):
    """
    创建一个特殊的n阶方阵，其形状如下：
    第一行: [1, 0, 0, ..., 0, 1]
    第二行: [-1, 1, 0, ..., 0, 1]
    第三行: [-1, -1, 1, ..., 0, 1]
    ...
    倒数第二行: [-1, -1, -1, ..., 1, 1]
    倒数第一行: [-1, -1, -1, ..., -1, 1]

    参数:
    n -- 矩阵的阶数

    返回:
    A -- 特殊形状的n阶方阵
    """
    A = np.zeros((n, n))

    # 设置对角线和最后一列
    for i in range(n):
        # 对角线元素为1
        A[i, i] = 1

        # 最后一列全为1
        A[i, n - 1] = 1

        # 对角线左侧元素为-1
        for j in range(i):
            A[i, j] = -1

    return A


def test_error_estimation():
    """
    测试特殊矩阵的求解误差估计
    1. 创建n阶特殊矩阵A_n
    2. 随机生成真实解x_real
    3. 计算b = A_n * x_real
    4. 使用列主元消去法求解方程组得到x_est
    5. 估计相对误差并与真实相对误差比较
    6. 使用迭代优化改进解并比较误差
    """
    # 设置随机种子以保证结果可重现
    np.random.seed(42)

    # 要测试的矩阵阶数范围
    n_range = range(5, 31, 1)

    # 结果表头
    print(
        f"{'阶数':<5}{'条件数':<15}{'真实相对误差':<15}{'估计相对误差':<20}{'估计/真实':<10}{'迭代后误差':<15}{'迭代次数':<10}"
    )
    print("-" * 85)

    for n in n_range:
        # 1. 创建特殊矩阵A_n
        A = create_special_matrix(n)

        # 2. 随机生成真实解（范围从-10到10）
        x_real = np.random.uniform(-10, 10, n)

        # 3. 计算右端向量b
        b = A @ x_real

        # 4. 使用列主元消去法求解方程组
        x_est = solve_with_partial_pivoting(A, b)

        # 计算真实相对误差
        true_error = normInf(x_real - x_est) / normInf(x_real)

        # 5. 估计相对误差
        est_error = estimate_relative_error(A, x_est, b)

        # 计算估计误差与真实误差的比值
        ratio = est_error / true_error if true_error > 0 else float("inf")

        # 6. 使用迭代优化改进解
        x_refined, iterations = iterative_refinement(A, b, x_est)
        refined_error = normInf(x_real - x_refined) / normInf(x_real)

        # 计算矩阵条件数
        cond = condition_number(A)

        # 打印结果
        print(
            f"{n:<5}{cond:<15.3e}{true_error:<15.3e}{est_error:<20.3e}{ratio:<10.3f}{refined_error:<15.3e}{iterations:<10}"
        )

    # 分析结果
    print("\n结果分析:")
    print("1. 随着矩阵阶数增加，条件数的变化情况")
    print("2. 估计误差上界与实际误差的关系")
    print("3. 迭代优化对解的改进程度")


def analyze_specific_case(n=30):
    """分析特定阶数矩阵的具体求解过程"""
    print(f"\n===== {n}阶特殊矩阵的详细分析 =====")

    # 创建矩阵和解向量
    A = create_special_matrix(n)
    x_real = np.random.uniform(-10, 10, n)
    b = A @ x_real

    print(f"矩阵条件数: {condition_number(A):.3e}")

    # 求解和初始误差
    start_time = time.time()
    x_est = solve_with_partial_pivoting(A, b)
    solve_time = time.time() - start_time
    true_error = normInf(x_real - x_est) / normInf(x_real)

    print(f"列主元消去法求解时间: {solve_time:.6f}秒")
    print(f"初始真实相对误差: {true_error:.3e}")

    # 估计误差
    est_error = estimate_relative_error(A, x_est, b)
    print(f"估计相对误差上界: {est_error:.3e}")
    print(f"估计/真实比率: {est_error/true_error:.3f}")

    # 迭代优化
    start_time = time.time()
    x_refined, iterations = iterative_refinement(A, b, x_est)
    refine_time = time.time() - start_time
    refined_error = normInf(x_real - x_refined) / normInf(x_real)

    print(f"迭代优化时间: {refine_time:.6f}秒")
    print(f"迭代次数: {iterations}")
    print(f"优化后真实相对误差: {refined_error:.3e}")
    print(f"误差改进比例: {true_error/refined_error:.3f}倍")

    # 对比向量的前几个元素
    print("\n解向量的前5个元素比较:")
    print(
        f"{'元素':<5}{'真实解':<15}{'初始估计解':<15}{'相对误差':<15}{'优化后解':<15}{'优化后误差':<15}"
    )
    for i in range(min(5, n)):
        rel_err = abs(x_real[i] - x_est[i]) / abs(x_real[i]) if x_real[i] != 0 else 0
        refined_rel_err = (
            abs(x_real[i] - x_refined[i]) / abs(x_real[i]) if x_real[i] != 0 else 0
        )
        print(
            f"{i:<5}{x_real[i]:<15.6f}{x_est[i]:<15.6f}{rel_err:<15.3e}{x_refined[i]:<15.6f}{refined_rel_err:<15.3e}"
        )


if __name__ == "__main__":
    print("测试特殊矩阵求解的误差估计和优化\n")
    test_error_estimation()
    analyze_specific_case(30)
