"""
线性方程组迭代求解方法比较
1. Hilbert矩阵测试共轭梯度法
2. Jacobi、Gauss-Seidel、共轭梯度法求解给定线性方程组
"""

import numpy as np
import matplotlib.pyplot as plt
from iterators import jacobi_iterator, gauss_seidel_iterator
from cg import conjugate_gradient


def create_hilbert_matrix(n):
    """创建n阶Hilbert矩阵"""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H


def test_hilbert_matrix_cg():
    """
    使用Hilbert矩阵测试共轭梯度法
    Hilbert矩阵是经典的病态矩阵，条件数随着阶数增加而急剧增大
    """
    print("\n=== 题目1: Hilbert矩阵测试共轭梯度法 ===")

    # 测试不同阶数的Hilbert矩阵
    sizes = [5, 10, 15, 20]

    # 存储各个矩阵规模的结果
    iterations_list = []
    condition_numbers = []
    relative_errors = []

    plt.figure(figsize=(12, 8))

    for n in sizes:
        print(f"\n测试 {n}×{n} Hilbert矩阵")

        # 创建Hilbert矩阵
        H = create_hilbert_matrix(n)

        # 计算条件数
        cond_num = np.linalg.cond(H)
        condition_numbers.append(cond_num)
        print(f"条件数: {cond_num:.3e}")

        # 创建右侧向量: b_i = (1/3) * sum(H_{i,j})
        b = np.sum(H, axis=1) / 3

        # 精确解
        x_exact = np.linalg.solve(H, b)

        # 共轭梯度法求解
        tol = 1e-10  # 设置较高精度以观察收敛行为
        x_cg, iter_cg, res_history = conjugate_gradient(
            H, b, tol=tol, max_iter=1000, return_history=True
        )

        # 相对误差
        rel_error = np.linalg.norm(x_cg - x_exact) / np.linalg.norm(x_exact)

        iterations_list.append(iter_cg)
        relative_errors.append(rel_error)

        print(f"迭代次数: {iter_cg}")
        print(f"相对误差: {rel_error:.6e}")
        print(f"最终残差: {res_history[-1]:.6e}")

        # 绘制收敛历史
        plt.subplot(2, 2, sizes.index(n) + 1)
        plt.semilogy(res_history, "b-", linewidth=1.5)
        plt.xlabel("Iterations")
        plt.ylabel("Relative Residual (log scale)")
        plt.title(f"{n}×{n} Hilbert Matrix (Condition Number:{cond_num:.1e})")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("hilbert_cg_convergence.png")

    # 结果总结
    print("\nHilbert矩阵CG求解结果总结:")
    print("矩阵大小  条件数      迭代次数  相对误差")
    for i, n in enumerate(sizes):
        print(
            f"{n:5d}×{n:<5d} {condition_numbers[i]:.3e}  {iterations_list[i]:5d}    {relative_errors[i]:.3e}"
        )

    # 绘制条件数与迭代次数/误差的关系
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, iterations_list, "bo-", linewidth=2)
    plt.xlabel("Matrix Size")
    plt.ylabel("Iterations")
    plt.title("Matrix Size vs. Iteration Count")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(condition_numbers, relative_errors, "ro-", linewidth=2)
    plt.xlabel("Condition Number (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.title("Condition Number vs. Solution Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("hilbert_cg_analysis.png")


def test_iterative_methods_comparison():
    """
    比较Jacobi、Gauss-Seidel和共轭梯度法求解给定线性方程组
    """
    print("\n=== 题目2: 三种迭代方法求解线性方程组比较 ===")

    # 给定的系数矩阵
    A = np.array(
        [
            [10, 1, 2, 3, 4],
            [1, 9, -1, 2, -3],
            [2, -1, 7, 3, -5],
            [3, 2, 3, 12, -1],
            [4, -3, -5, -1, 15],
        ]
    )

    # 常数项向量
    b = np.array([12, -27, 14, -17, 12])

    # 检查矩阵性质
    print("\n矩阵分析:")
    eigenvalues = np.linalg.eigvals(A)
    is_symmetric = np.allclose(A, A.T)
    is_pos_def = is_symmetric and np.all(eigenvalues > 0)
    is_diag_dominant = all(
        abs(A[i, i]) > sum(abs(A[i, j]) for j in range(5) if j != i) for i in range(5)
    )

    print(f"矩阵是否对称: {is_symmetric}")
    print(f"矩阵是否正定: {is_pos_def}")
    print(f"矩阵是否对角占优: {is_diag_dominant}")
    print(f"特征值: {eigenvalues}")
    print(f"条件数: {np.linalg.cond(A):.6f}")

    # 计算精确解作为参考
    x_exact = np.linalg.solve(A, b)
    print("\n精确解:", x_exact)

    # 设置共同的迭代参数
    tol = 1e-10
    max_iter = 1000

    # 1. Jacobi迭代法
    x_jacobi, iter_jacobi, res_jacobi = jacobi_iterator(
        A, b, max_iter=max_iter, tol=tol
    )
    error_jacobi = np.linalg.norm(x_jacobi - x_exact) / np.linalg.norm(x_exact)

    # 2. Gauss-Seidel迭代法
    x_gs, iter_gs, res_gs = gauss_seidel_iterator(A, b, max_iter=max_iter, tol=tol)
    error_gs = np.linalg.norm(x_gs - x_exact) / np.linalg.norm(x_exact)

    # 3. 共轭梯度法
    x_cg, iter_cg, res_cg = conjugate_gradient(
        A, b, tol=tol, max_iter=max_iter, return_history=True
    )
    error_cg = np.linalg.norm(x_cg - x_exact) / np.linalg.norm(x_exact)

    # 结果比较
    print("\n迭代法求解结果比较:")
    print(f"{'方法':<15} {'迭代次数':<10} {'相对误差':<15} {'最终残差':<15}")
    print("-" * 55)
    print(
        f"{'Jacobi':<15} {iter_jacobi:<10d} {error_jacobi:<15.6e} {res_jacobi[-1]:<15.6e}"
    )
    print(f"{'Gauss-Seidel':<15} {iter_gs:<10d} {error_gs:<15.6e} {res_gs[-1]:<15.6e}")
    print(f"{'共轭梯度':<15} {iter_cg:<10d} {error_cg:<15.6e} {res_cg[-1]:<15.6e}")

    # 解的比较
    print("\n各方法求解结果:")
    result_table = np.column_stack((x_exact, x_jacobi, x_gs, x_cg))
    headers = ["精确解", "Jacobi", "Gauss-Seidel", "共轭梯度"]

    for i, header in enumerate(headers):
        print(f"{header:<12}", end="")
    print()
    for i in range(5):
        for j in range(4):
            print(f"{result_table[i, j]:<12.6f}", end="")
        print()

    # 收敛性分析图
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(res_jacobi)), res_jacobi, "b-", label="Jacobi")
    plt.semilogy(range(len(res_gs)), res_gs, "r-", label="Gauss-Seidel")
    plt.semilogy(range(len(res_cg)), res_cg, "g-", label="Conjugate Gradient")
    plt.xlabel("Iterations")
    plt.ylabel("Relative Residual (log scale)")
    plt.title("Convergence Comparison of Three Iterative Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig("iterative_methods_comparison.png")

    # 收敛前50次迭代的细节比较
    plt.figure(figsize=(10, 6))
    max_iter_to_show = min(50, min(len(res_jacobi), len(res_gs), len(res_cg)))
    plt.semilogy(
        range(max_iter_to_show), res_jacobi[:max_iter_to_show], "b-", label="Jacobi"
    )
    plt.semilogy(
        range(max_iter_to_show), res_gs[:max_iter_to_show], "r-", label="Gauss-Seidel"
    )
    plt.semilogy(
        range(max_iter_to_show),
        res_cg[:max_iter_to_show],
        "g-",
        label="Conjugate Gradient",
    )
    plt.xlabel("Iterations")
    plt.ylabel("Relative Residual (log scale)")
    plt.title("Convergence Detail of First 50 Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("iterative_methods_first50.png")


def main():
    # 移除中文字体设置，使用默认英文字体
    # plt.rcParams["font.sans-serif"] = ["SimHei"]  # 注释掉这行
    # plt.rcParams["axes.unicode_minus"] = False    # 注释掉这行

    # 运行题目1：Hilbert矩阵测试
    test_hilbert_matrix_cg()

    # 运行题目2：迭代法比较
    test_iterative_methods_comparison()

    plt.show()


if __name__ == "__main__":
    main()
