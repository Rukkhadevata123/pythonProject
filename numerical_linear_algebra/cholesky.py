import numpy as np
from substitution import forward_substitution, back_substitution


def is_symmetric(A, tol=1e-8):
    # tol 为容差
    return np.allclose(A, A.T, rtol=tol)


def is_positive_definite(A, tol=1e-8):
    # eigvals()返回矩阵的特征值
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > tol)


def cholesky_decomposition(A):
    """
    对正定对称矩阵A进行Cholesky分解，返回下三角矩阵L，使得A = L*L^T
    返回:下三角矩阵L
    """
    # 检查矩阵是否为正定对称矩阵
    if not is_symmetric(A):
        raise ValueError("输入矩阵不是对称矩阵")
    if not is_positive_definite(A):
        raise ValueError("输入矩阵不是正定矩阵")

    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            if i == j:  # 对角线元素
                s = sum(L[i, k] ** 2 for k in range(j))
                L[i, j] = np.sqrt(A[i, i] - s)
            else:  # 非对角线元素
                s = sum(L[i, k] * L[j, k] for k in range(j))
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L


def solve_with_cholesky(A, b):
    # 1.Cholesky分解 A = L*L^T
    L = cholesky_decomposition(A)

    # 2.前代法求解 Ly = b
    y = forward_substitution(L, b)

    # 3.回代法求解 L^T x = y
    x = back_substitution(L.T, y)

    return x


def test_cholesky():
    """测试Cholesky分解与求解"""
    # 创建一个正定对称矩阵
    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]], dtype=float)

    b = np.array([1, 2, 3], dtype=float)

    print("=== Cholesky分解测试 ===")
    L = cholesky_decomposition(A)
    print("原矩阵A:")
    print(A)
    print("\nCholesky分解得到的L:")
    print(L)
    print("\n验证 L*L^T == A:")
    print(np.allclose(L @ L.T, A))

    print("\n=== 使用Cholesky分解求解线性方程组 ===")
    x = solve_with_cholesky(A, b)
    print(f"解向量 x = {x}")
    print(f"残差 ||Ax-b|| = {np.linalg.norm(A @ x - b)}")


if __name__ == '__main__':
    test_cholesky()