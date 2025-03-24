import numpy as np
import time
from cholesky import solve_with_cholesky, is_positive_definite
from better_square import solve_with_ldl
from lu_decomposition import solve_linear_system, solve_with_partial_pivoting


def create_hilbert_matrix(n):
    """创建n阶Hilbert矩阵"""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H


def create_b_vector(n):
    """创建b向量，b_i = sum_{j=1}^{n} 1/(i+j-1)"""
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += 1.0 / (i + j + 1)
    return b


def main():
    n = 40  # Hilbert矩阵阶数

    # 创建Hilbert矩阵
    print(f"创建{n}阶Hilbert矩阵...")
    A = create_hilbert_matrix(n)

    # 创建b向量
    b = create_b_vector(n)

    # 预期解为全1向量
    x_expected = np.ones(n)

    print(f"=== 求解{n}阶Hilbert矩阵线性方程组 ===")
    print(f"矩阵条件数: {np.linalg.cond(A):.2e}")

    # 检查是否是正定矩阵
    if not is_positive_definite(A):
        print("警告: Hilbert矩阵数值上可能不是正定的，可能影响结果")

    # 使用普通高斯消元法求解
    print("\n使用普通高斯消元法求解中...")
    try:
        start_time = time.time()
        x_gauss = solve_linear_system(A, b)
        gauss_time = time.time() - start_time
        gauss_error = np.linalg.norm(x_gauss - x_expected)
        gauss_residual = np.linalg.norm(A @ x_gauss - b)
    except Exception as e:
        print(f"普通高斯消元法失败: {e}")
        gauss_time = float("nan")
        gauss_error = float("nan")
        gauss_residual = float("nan")
        x_gauss = np.full(n, float("nan"))

    # 使用列主元高斯消元法求解
    print("使用列主元高斯消元法求解中...")
    try:
        start_time = time.time()
        x_partial = solve_with_partial_pivoting(A, b)
        partial_time = time.time() - start_time
        partial_error = np.linalg.norm(x_partial - x_expected)
        partial_residual = np.linalg.norm(A @ x_partial - b)
    except Exception as e:
        print(f"列主元高斯消元法失败: {e}")
        partial_time = float("nan")
        partial_error = float("nan")
        partial_residual = float("nan")
        x_partial = np.full(n, float("nan"))

    # 使用Cholesky分解(LLT)求解
    print("使用Cholesky分解求解中...")
    try:
        start_time = time.time()
        x_cholesky = solve_with_cholesky(A, b)
        cholesky_time = time.time() - start_time
        cholesky_error = np.linalg.norm(x_cholesky - x_expected)
        cholesky_residual = np.linalg.norm(A @ x_cholesky - b)
    except Exception as e:
        print(f"Cholesky分解失败: {e}")
        cholesky_time = float("nan")
        cholesky_error = float("nan")
        cholesky_residual = float("nan")
        x_cholesky = np.full(n, float("nan"))

    # 使用改进平方根法(LDLT)求解
    print("使用LDLT分解求解中...")
    try:
        start_time = time.time()
        x_ldlt = solve_with_ldl(A, b)
        ldlt_time = time.time() - start_time
        ldlt_error = np.linalg.norm(x_ldlt - x_expected)
        ldlt_residual = np.linalg.norm(A @ x_ldlt - b)
    except Exception as e:
        print(f"LDLT分解失败: {e}")
        ldlt_time = float("nan")
        ldlt_error = float("nan")
        ldlt_residual = float("nan")
        x_ldlt = np.full(n, float("nan"))

    # 打印结果
    print("\n=== 结果比较 ===")
    print(f"{'方法':<20}{'用时(秒)':<15}{'误差':<20}{'残差':<20}")
    print(
        f"{'普通高斯消元':<20}{gauss_time:<15.6f}{gauss_error:<20.6e}{gauss_residual:<20.6e}"
    )
    print(
        f"{'列主元高斯消元':<20}{partial_time:<15.6f}{partial_error:<20.6e}{partial_residual:<20.6e}"
    )
    print(
        f"{'Cholesky (LLT)':<20}{cholesky_time:<15.6f}{cholesky_error:<20.6e}{cholesky_residual:<20.6e}"
    )
    print(
        f"{'LDLT分解':<20}{ldlt_time:<15.6f}{ldlt_error:<20.6e}{ldlt_residual:<20.6e}"
    )

    # 比较前5个元素
    print("\n=== 解向量的前5个元素 ===")
    print("预期解:")
    print(f"前5个: {x_expected[:5]}")

    print("\n普通高斯消元:")
    print(f"前5个: {x_gauss[:5]}")

    print("\n列主元高斯消元:")
    print(f"前5个: {x_partial[:5]}")

    print("\nCholesky分解:")
    print(f"前5个: {x_cholesky[:5]}")

    print("\nLDLT分解:")
    print(f"前5个: {x_ldlt[:5]}")


if __name__ == "__main__":
    main()
