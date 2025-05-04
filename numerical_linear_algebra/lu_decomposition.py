import numpy as np
from substitution import forward_substitution, back_substitution


def lu_decomposition(A):
    """
    对矩阵A进行LU分解，返回下三角矩阵L和上三角矩阵U

    参数:
        A: 输入矩阵 (n x n)

    返回:
        L: 单位下三角矩阵 (对角线元素为1)
        U: 上三角矩阵

    算法步骤:
        1. 初始化L为单位矩阵，U为A的副本
        2. 对每个主元位置k:
           a. 对i > k的行，计算消元因子factor = U[i,k]/U[k,k]
           b. 存储factor到L[i,k]
           c. 更新U的第i行: U[i,k:] -= factor * U[k,k:]
    """
    n = len(A)
    L = np.eye(n)  # 单位下三角矩阵
    U = A.copy().astype(float)  # 确保浮点运算

    for k in range(n):
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]  # 向量化更新

    return L, U


def solve_linear_system(A, b):
    """
    使用LU分解求解线性方程组 Ax = b

    参数:
        A: 系数矩阵 (n x n)
        b: 右侧向量 (n x 1)

    返回:
        x: 解向量 (n x 1)

    求解步骤:
        1. LU分解得到L和U
        2. 前代法解Ly = b
        3. 回代法解Ux = y
    """
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x


def lu_decomposition_with_partial_pivoting(A):
    """
    带列主元的LU分解，计算PA = LU

    参数:
        A: 输入矩阵 (n x n)

    返回:
        L: 单位下三角矩阵
        U: 上三角矩阵
        P: 置换矩阵

    改进点:
        1. 每步消元前选择列主元
        2. 记录行交换到置换矩阵P
    """
    n = len(A)
    L = np.eye(n)
    U = A.copy().astype(float)
    P = np.eye(n)

    for k in range(n):
        # 列主元选择
        pivot = k + np.argmax(np.abs(U[k:, k]))

        if pivot != k:
            # 行交换
            U[[k, pivot]] = U[[pivot, k]]
            L[[k, pivot], :k] = L[[pivot, k], :k]
            P[[k, pivot]] = P[[pivot, k]]

        # 高斯消元
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U, P


def solve_with_partial_pivoting(A, b):
    """
    使用列主元LU分解求解线性方程组

    参数:
        A: 系数矩阵 (n x n)
        b: 右侧向量 (n x 1)

    返回:
        x: 解向量 (n x 1)

    注意:
        需要先对b应用置换矩阵P
    """
    L, U, P = lu_decomposition_with_partial_pivoting(A)
    pb = P @ b  # 应用置换
    y = forward_substitution(L, pb)
    x = back_substitution(U, y)
    return x


def test_linear_system():
    """测试LU分解求解线性方程组"""
    A = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]], dtype=float)
    b = np.array([1, -2, 0], dtype=float)

    print("基本LU分解测试:")
    x = solve_linear_system(A, b)
    print(f"解x = {x}")
    print(f"残差||Ax-b|| = {np.linalg.norm(A @ x - b):.2e}")

    print("\n列主元LU分解测试:")
    x_pp = solve_with_partial_pivoting(A, b)
    print(f"解x = {x_pp}")
    print(f"残差||Ax-b|| = {np.linalg.norm(A @ x_pp - b):.2e}")


if __name__ == "__main__":
    test_linear_system()
