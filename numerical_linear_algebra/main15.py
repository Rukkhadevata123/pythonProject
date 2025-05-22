"""
两个三对角矩阵的全部特征值和特征向量
矩阵A: 主对角线全为4，次对角线全为1
矩阵B: 主对角线全为2，次对角线全为-1
"""

import numpy as np
from symmetry_qr import eigensolver_symmetric


def create_tridiagonal_matrix_A(n=100):
    """
    创建第一种三对角矩阵，主对角线全为4，次对角线全为1

    参数:
        n: 矩阵阶数

    返回:
        A: 三对角矩阵
    """
    A = np.zeros((n, n))

    # 设置主对角线元素为4
    for i in range(n):
        A[i, i] = 4.0

    # 设置次对角线元素为1
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    return A


def create_tridiagonal_matrix_B(n=100):
    """
    创建第二种三对角矩阵，主对角线全为2，次对角线全为-1

    参数:
        n: 矩阵阶数

    返回:
        B: 三对角矩阵
    """
    B = np.zeros((n, n))

    # 设置主对角线元素为2
    for i in range(n):
        B[i, i] = 2.0

    # 设置次对角线元素为-1
    for i in range(n - 1):
        B[i, i + 1] = -1.0
        B[i + 1, i] = -1.0

    return B


def compare_eigenvalues(matrix, matrix_name, n, top_k=5):
    """
    比较自实现算法与NumPy计算的特征值

    参数:
        matrix: 输入矩阵
        matrix_name: 矩阵名称（用于显示）
        n: 矩阵阶数
        top_k: 显示前k个特征值

    返回:
        None，直接打印结果
    """
    print(f"\n{'-'*50}")
    print(f"矩阵 {matrix_name} ({n}×{n}) 的特征值分析")
    print(f"{'-'*50}")

    # 使用自实现的eigensolver_symmetric算法（包含反迭代法）
    our_eigenvalues, our_eigenvectors = eigensolver_symmetric(matrix, tol=1e-10)

    # 使用NumPy计算
    numpy_eigenvalues, numpy_eigenvectors = np.linalg.eigh(matrix)

    # 排序NumPy特征值（从大到小，与我们的算法保持一致）
    numpy_eigenvalues = np.flip(numpy_eigenvalues)
    numpy_eigenvectors = np.flip(numpy_eigenvectors, axis=1)

    # 打印前top_k个特征值
    print(f"\n前{top_k}个特征值对比:")
    print(
        f"{'索引':^5} | {'自实现算法':^20} | {'NumPy':^20} | {'绝对误差':^15} | {'相对误差':^15}"
    )
    print(f"{'-'*75}")

    for i in range(min(top_k, n)):
        our_val = our_eigenvalues[i]
        numpy_val = numpy_eigenvalues[i]
        abs_error = abs(our_val - numpy_val)
        rel_error = abs_error / abs(numpy_val) if abs(numpy_val) > 1e-10 else 0

        print(
            f"{i:^5} | {our_val:^20.12f} | {numpy_val:^20.12f} | {abs_error:^15.6e} | {rel_error:^15.6e}"
        )

    # 计算所有特征值的误差统计
    all_abs_errors = [abs(our_eigenvalues[i] - numpy_eigenvalues[i]) for i in range(n)]
    all_rel_errors = [
        (
            abs(our_eigenvalues[i] - numpy_eigenvalues[i]) / abs(numpy_eigenvalues[i])
            if abs(numpy_eigenvalues[i]) > 1e-10
            else 0
        )
        for i in range(n)
    ]

    print(f"\n所有特征值误差统计:")
    print(f"最大绝对误差: {max(all_abs_errors):.6e}")
    print(f"平均绝对误差: {np.mean(all_abs_errors):.6e}")
    print(f"最大相对误差: {max(all_rel_errors):.6e}")
    print(f"平均相对误差: {np.mean(all_rel_errors):.6e}")

    # 检验特征向量 - 修改为验证所有（或top_k个）特征向量
    num_vectors_to_check = min(top_k, n)
    print(f"\n特征向量验证 (验证前{num_vectors_to_check}个):")
    print(f"{'索引':^5} | {'残差 ||Av-λv||':^20}")
    print(f"{'-'*30}")

    # 计算所有特征向量的残差
    all_residuals = []
    for i in range(num_vectors_to_check):
        v = np.array(our_eigenvectors[:, i])
        lambda_val = our_eigenvalues[i]
        residual = np.linalg.norm(matrix @ v - lambda_val * v)
        all_residuals.append(residual)
        print(f"{i:^5} | {residual:^20.12e}")

    # 打印特征向量残差统计
    print(f"\n特征向量残差统计:")
    print(f"最大残差: {max(all_residuals):.6e}")
    print(f"平均残差: {np.mean(all_residuals):.6e}")
    print(f"最小残差: {min(all_residuals):.6e}")


def main():
    # 设置矩阵阶数
    n = 100

    # 创建两种三对角矩阵
    A = create_tridiagonal_matrix_A(n)
    B = create_tridiagonal_matrix_B(n)

    # 分析矩阵A - 显示所有特征值
    compare_eigenvalues(A, "A (对角线为4，次对角线为1)", n, top_k=n)

    # 分析矩阵B - 显示所有特征值
    compare_eigenvalues(B, "B (对角线为2，次对角线为-1)", n, top_k=n)


if __name__ == "__main__":
    main()
