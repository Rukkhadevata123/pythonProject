import numpy as np
import time
from symmetry_qr import implicit_symmetric_qr


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

    # 使用自实现的QR算法
    start_time = time.time()
    our_eigenvalues, our_eigenvectors = implicit_symmetric_qr(
        matrix, tol=1e-10, max_iter=200
    )
    our_time = time.time() - start_time

    # 使用NumPy计算
    start_time = time.time()
    numpy_eigenvalues, numpy_eigenvectors = np.linalg.eigh(matrix)
    numpy_time = time.time() - start_time

    # 排序NumPy特征值（从大到小，与我们的算法保持一致）
    numpy_eigenvalues = np.flip(numpy_eigenvalues)
    numpy_eigenvectors = np.flip(numpy_eigenvectors, axis=1)

    # 打印计算时间对比
    print(f"计算时间对比:")
    print(f"自实现QR算法: {our_time:.6f}秒")
    print(f"NumPy算法: {numpy_time:.6f}秒")
    print(f"时间比: {our_time/numpy_time:.2f}倍")

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

    # 检验特征向量
    print(f"\n特征向量验证 (随机抽取前5个):")
    print(f"{'索引':^5} | {'残差 ||Av-λv||':^20}")
    print(f"{'-'*30}")

    for i in range(min(5, n)):
        v = np.array(our_eigenvectors[:, i])
        lambda_val = our_eigenvalues[i]
        residual = np.linalg.norm(matrix @ v - lambda_val * v)
        print(f"{i:^5} | {residual:^20.12e}")


def main():
    # 设置矩阵阶数
    n = 100

    # 创建两种三对角矩阵
    A = create_tridiagonal_matrix_A(n)
    B = create_tridiagonal_matrix_B(n)

    # 分析矩阵A
    compare_eigenvalues(A, "A (对角线为4，次对角线为1)", n)

    # 分析矩阵B
    compare_eigenvalues(B, "B (对角线为2，次对角线为-1)", n)

    # 理论分析
    print(f"\n{'-'*50}")
    print(f"理论分析")
    print(f"{'-'*50}")
    print("矩阵A的特征值理论公式: λ_k = 4 + 2*cos(kπ/(n+1)), k=1,2,...,n")
    print("矩阵B的特征值理论公式: λ_k = 2 - 2*cos(kπ/(n+1)), k=1,2,...,n")

    # 计算几个理论特征值进行验证
    k_values = [1, 2, 3, n - 1, n]
    print(f"\n验证几个特征值计算结果与理论公式:")
    print(
        f"{'k':^5} | {'理论λA_k':^15} | {'计算λA_k':^15} | {'理论λB_k':^15} | {'计算λB_k':^15}"
    )
    print(f"{'-'*70}")

    our_eigenvalues_A, _ = implicit_symmetric_qr(A)
    our_eigenvalues_B, _ = implicit_symmetric_qr(B)

    for k_idx in k_values:
        k = k_idx
        lambda_A_theory = 4 + 2 * np.cos(k * np.pi / (n + 1))
        lambda_B_theory = 2 - 2 * np.cos(k * np.pi / (n + 1))

        idx_A = n - k  # 由于我们的特征值是从大到小排序的
        idx_B = k - 1  # 矩阵B的特征值顺序正好符合公式

        lambda_A_computed = our_eigenvalues_A[idx_A]
        lambda_B_computed = our_eigenvalues_B[idx_B]

        print(
            f"{k:^5} | {lambda_A_theory:^15.8f} | {lambda_A_computed:^15.8f} | {lambda_B_theory:^15.8f} | {lambda_B_computed:^15.8f}"
        )

    # 测试不同规模的矩阵
    sizes = [10, 50, 200]
    print(f"\n{'-'*50}")
    print(f"不同规模矩阵的性能对比")
    print(f"{'-'*50}")

    for size in sizes:
        A_small = create_tridiagonal_matrix_A(size)

        # 仅测量计算时间
        start_time = time.time()
        our_eigenvalues, _ = implicit_symmetric_qr(A_small)
        our_time = time.time() - start_time

        start_time = time.time()
        numpy_eigenvalues, _ = np.linalg.eigh(A_small)
        numpy_time = time.time() - start_time

        print(f"矩阵规模 {size}×{size}:")
        print(f"自实现QR算法: {our_time:.6f}秒")
        print(f"NumPy算法: {numpy_time:.6f}秒")
        print(f"时间比: {our_time/numpy_time:.2f}倍\n")


if __name__ == "__main__":
    main()
