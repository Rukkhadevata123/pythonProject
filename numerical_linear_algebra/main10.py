# main10.py
import numpy as np
import matplotlib.pyplot as plt
from iterators import (
    jacobi_iterator,
    gauss_seidel_iterator,
    sor_iterator,
    auto_select_omega,
)


def exact_solution(x, epsilon, a):
    """计算精确解"""
    return (1 - a) / (1 - np.exp(-1 / epsilon)) * (1 - np.exp(-x / epsilon)) + a * x


def construct_matrix(epsilon, n):
    """构造差分方程的系数矩阵"""
    h = 1 / n
    A = np.zeros((n - 1, n - 1))

    # 填充主对角线
    np.fill_diagonal(A, -(2 * epsilon + h))

    # 填充上对角线
    for i in range(n - 2):
        A[i, i + 1] = epsilon + h

    # 填充下对角线
    for i in range(1, n - 1):
        A[i, i - 1] = epsilon

    return A


def construct_rhs(a, n):
    """构造右侧向量
    注意：右侧向量需要考虑边界条件的影响
    """
    h = 1 / n
    b = np.full(n - 1, a * h**2)
    # 处理右边界条件 y(1) = 1
    b[n - 2] -= (epsilon + h) * 1
    return b


def solve_and_compare(epsilon, a, n):
    """求解并比较不同迭代方法的结果"""
    # 构造矩阵和右侧向量
    A = construct_matrix(epsilon, n)
    b = construct_rhs(a, n)

    # 生成网格点
    x_points = np.linspace(0, 1, n + 1)[1:-1]

    # 计算精确解
    exact = exact_solution(x_points, epsilon, a)

    # 初始猜测
    x0 = np.zeros_like(b)

    # 提高收敛精度以达到4位有效数字
    tol = 1e-8

    # Jacobi迭代
    x_jacobi, iter_jacobi, res_jacobi = jacobi_iterator(
        A, b, x0.copy(), max_iter=10000, tol=tol
    )
    error_jacobi = np.linalg.norm(x_jacobi - exact) / np.linalg.norm(exact)

    # Gauss-Seidel迭代
    x_gs, iter_gs, res_gs = gauss_seidel_iterator(
        A, b, x0.copy(), max_iter=10000, tol=tol
    )
    error_gs = np.linalg.norm(x_gs - exact) / np.linalg.norm(exact)

    # SOR迭代 (自动选择omega)
    omega = auto_select_omega(A)
    x_sor, iter_sor, res_sor = sor_iterator(
        A, b, omega, x0.copy(), max_iter=10000, tol=tol
    )
    error_sor = np.linalg.norm(x_sor - exact) / np.linalg.norm(exact)

    # 打印结果
    print(f"\nε = {epsilon}, a = {a}, n = {n}")
    print(f"Jacobi: {iter_jacobi} iterations, relative error = {error_jacobi:.4e}")
    print(f"Gauss-Seidel: {iter_gs} iterations, relative error = {error_gs:.4e}")
    print(
        f"SOR (ω={omega:.3f}): {iter_sor} iterations, relative error = {error_sor:.4e}"
    )

    # 绘制结果图
    plt.figure(figsize=(12, 8))

    # 1. 解的比较
    plt.subplot(2, 1, 1)
    plt.plot(x_points, exact, "k-", linewidth=2, label="Exact")
    plt.plot(x_points, x_jacobi, "b--", label=f"Jacobi ({iter_jacobi} iter)")
    plt.plot(x_points, x_gs, "r:", label=f"G-S ({iter_gs} iter)")
    plt.plot(x_points, x_sor, "g-.", label=f"SOR ({iter_sor} iter)")
    plt.title(f"Solution Comparison for ε = {epsilon}, a = {a}, n = {n}")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.grid(True)

    # 2. 收敛历史
    plt.subplot(2, 1, 2)
    plt.semilogy(range(len(res_jacobi)), res_jacobi, "b-", label="Jacobi")
    plt.semilogy(range(len(res_gs)), res_gs, "r-", label="Gauss-Seidel")
    plt.semilogy(range(len(res_sor)), res_sor, "g-", label=f"SOR (ω={omega:.3f})")
    plt.xlabel("Iteration")
    plt.ylabel("Relative Residual (log scale)")
    plt.title("Convergence History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"bvp_epsilon_{epsilon}.png")
    plt.show()


if __name__ == "__main__":
    # 参数设置
    n = 100
    a = 0.5

    # 不同的epsilon值
    epsilons = [1, 0.1, 0.01, 0.0001]

    # 对每个epsilon求解
    for epsilon in epsilons:
        solve_and_compare(epsilon, a, n)
