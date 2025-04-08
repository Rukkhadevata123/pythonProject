import numpy as np
import time
from better_square import solve_with_ldl
from cholesky import solve_with_cholesky
from lu_decomposition import solve_linear_system, solve_with_partial_pivoting
from qr import qr_solve


def create_tridiagonal_matrix(n):
    """创建特殊的三对角矩阵，对角线为10，次对角线为1"""
    A = np.zeros((n, n))

    # 设置主对角线为10
    np.fill_diagonal(A, 10)

    # 设置上下次对角线为1
    np.fill_diagonal(A[1:, :-1], 1)  # 下次对角线
    np.fill_diagonal(A[:-1, 1:], 1)  # 上次对角线

    return A


def main():
    n = 100  # 矩阵阶数

    # 创建特殊三对角矩阵
    A = create_tridiagonal_matrix(n)

    # 创建右端向量b，使解为全1向量
    x_expected = np.ones(n)
    b = A @ x_expected

    print(f"=== 求解{n}阶三对角线性方程组 ===")
    print("矩阵A的形式：")
    print("[ 10  1  0  0 ...]")
    print("[  1 10  1  0 ...]")
    print("[  0  1 10  1 ...]")
    print("[  .  .  .  . ...]")

    # 使用普通高斯消元法求解
    start_time = time.time()
    x_gauss = solve_linear_system(A, b)
    gauss_time = time.time() - start_time
    gauss_error = np.linalg.norm(x_gauss - x_expected)
    gauss_residual = np.linalg.norm(A @ x_gauss - b)

    # 使用列主元高斯消元法求解
    start_time = time.time()
    x_partial = solve_with_partial_pivoting(A, b)
    partial_time = time.time() - start_time
    partial_error = np.linalg.norm(x_partial - x_expected)
    partial_residual = np.linalg.norm(A @ x_partial - b)

    # 使用Cholesky分解(LLT)求解
    start_time = time.time()
    x_cholesky = solve_with_cholesky(A, b)
    cholesky_time = time.time() - start_time
    cholesky_error = np.linalg.norm(x_cholesky - x_expected)
    cholesky_residual = np.linalg.norm(A @ x_cholesky - b)

    # 使用改进平方根法(LDLT)求解
    start_time = time.time()
    x_ldlt = solve_with_ldl(A, b)
    ldlt_time = time.time() - start_time
    ldlt_error = np.linalg.norm(x_ldlt - x_expected)
    ldlt_residual = np.linalg.norm(A @ x_ldlt - b)

    start_time = time.time()
    x_qr = qr_solve(A, b)
    qr_time = time.time() - start_time
    qr_error = np.linalg.norm(x_qr - x_expected)
    qr_residual = np.linalg.norm(A @ x_qr - b)

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
    print(f"{'QR分解':<20}{qr_time:<15.6f}{qr_error:<20.6e}{qr_residual:<20.6e}")

    # 比较前5个和后5个元素
    print("\n=== 解向量的前5个和后5个元素 ===")
    print("预期解:")
    print(f"前5个: {x_expected[:5]}")
    print(f"后5个: {x_expected[-5:]}")

    print("\n普通高斯消元:")
    print(f"前5个: {x_gauss[:5]}")
    print(f"后5个: {x_gauss[-5:]}")

    print("\n列主元高斯消元:")
    print(f"前5个: {x_partial[:5]}")
    print(f"后5个: {x_partial[-5:]}")

    print("\nCholesky分解:")
    print(f"前5个: {x_cholesky[:5]}")
    print(f"后5个: {x_cholesky[-5:]}")

    print("\nLDLT分解:")
    print(f"前5个: {x_ldlt[:5]}")
    print(f"后5个: {x_ldlt[-5:]}")

    # 在比较元素部分添加
    print("\nQR分解:")
    print(f"前5个: {x_qr[:5]}")
    print(f"后5个: {x_qr[-5:]}")


if __name__ == "__main__":
    main()
