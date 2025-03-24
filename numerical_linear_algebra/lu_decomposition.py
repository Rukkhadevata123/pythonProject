import numpy as np
from substitution import forward_substitution, back_substitution


def lu_decomposition(A):
    """
    对矩阵A进行LU分解，返回下三角矩阵L和上三角矩阵U。

    参数:
    A -- 输入矩阵 (n x n)

    返回:
    L -- 下三角矩阵，对角线元素为1
    U -- 上三角矩阵
    """
    n = len(A)
    L = np.zeros((n, n))
    U = A.copy()

    np.fill_diagonal(L, 1)

    for k in range(n):
        for i in range(k+1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            for j in range(k, n):
                U[i, j] = U[i, j] - factor * U[k, j]

    return L, U


def solve_linear_system(A, b):
    """
    使用LU分解和前代法、回代法求解线性方程组 Ax = b

    参数:
    A -- 系数矩阵 (n x n)
    b -- 右侧向量 (n x 1)

    返回:
    x -- 解向量 (n x 1)
    """
    # 1.LU分解
    L, U = lu_decomposition(A)

    # 2.前代法求解Ly = b
    y = forward_substitution(L, b)

    # 3.回代法求解Ux = y
    x = back_substitution(U, y)

    return x


def test_linear_system():
    A = np.array([[3, 2, -1],
                  [2, -2, 4],
                  [-1, 0.5, -1]], dtype=float)
    b = np.array([1, -2, 0], dtype=float)

    x = solve_linear_system(A, b)

    print("线性方程组的解:")
    print(f"x = {x}")
    print("\n验证 Ax = b:")
    print(f"Ax = {np.dot(A, x)}")
    print(f"b  = {b}")
    print(f"误差: {np.linalg.norm(np.dot(A, x) - b)}")


if __name__ == '__main__':
    test_linear_system()


def lu_decomposition_with_partial_pivoting(A):
    """
    带列主元的LU分解，计算PA=LU

    参数:
    A -- 输入矩阵 (n x n)

    返回:
    L -- 下三角矩阵
    U -- 上三角矩阵
    P -- 置换矩阵
    """
    n = len(A)
    L = np.zeros((n, n))
    U = A.copy()
    P = np.eye(n)

    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k  # 找max

        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]  # np语法糖
            P[[k, pivot]] = P[[pivot, k]]

        L[k, k] = 1.0

        for i in range(k+1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] = U[i, k:] - factor * U[k, k:]

    return L, U, P


def lu_decomposition_with_complete_pivoting(A):
    """
    全主元LU分解，计算PAQ=LU

    参数:
    A -- 输入矩阵 (n x n)

    返回:
    L -- 下三角矩阵
    U -- 上三角矩阵
    P -- 行置换矩阵
    Q -- 列置换矩阵
    """
    n = len(A)
    L = np.zeros((n, n))
    U = A.copy()
    P = np.eye(n)
    Q = np.eye(n)

    for k in range(n):
        sub_matrix = np.abs(U[k:, k:])
        i_max, j_max = np.unravel_index(np.argmax(sub_matrix), sub_matrix.shape)
        i_max += k
        j_max += k  # 更改为绝对索引

        if i_max != k:
            U[[k, i_max]] = U[[i_max, k]]
            if k > 0:
                L[[k, i_max], :k] = L[[i_max, k], :k]
            P[[k, i_max]] = P[[i_max, k]]

        if j_max != k:
            U[:, [k, j_max]] = U[:, [j_max, k]]
            Q[:, [k, j_max]] = Q[:, [j_max, k]]

        L[k, k] = 1.0

        for i in range(k+1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] = U[i, k:] - factor * U[k, k:]

    return L, U, P, Q


def solve_with_partial_pivoting(A, b):
    """使用列主元LU分解求解线性方程组"""
    L, U, P = lu_decomposition_with_partial_pivoting(A)
    # 先应用置换矩阵到b
    pb = P @ b
    # 解Ly = Pb
    y = forward_substitution(L, pb)
    # 解Ux = y
    x = back_substitution(U, y)
    return x


def solve_with_complete_pivoting(A, b):
    """使用全主元LU分解求解线性方程组"""
    L, U, P, Q = lu_decomposition_with_complete_pivoting(A)
    # 先应用行置换矩阵到b
    pb = P @ b
    # 解Ly = Pb
    y = forward_substitution(L, pb)
    # 解Uz = y
    z = back_substitution(U, y)
    # 根据推导，需要把X的解还原到原始的顺序
    x = Q @ z
    return x


def test_pivoting():
    A = np.array([[0.1, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]], dtype=float)
    b = np.array([13.1, 32, 50], dtype=float)

    print("列主元LU分解测试")
    x_partial = solve_with_partial_pivoting(A, b)
    print(f"解向量 x = {x_partial}")
    print(f"残差 ||Ax-b|| = {np.linalg.norm(A @ x_partial - b)}")

    print("\n全主元LU分解测试")
    x_complete = solve_with_complete_pivoting(A, b)
    print(f"解向量 x = {x_complete}")
    print(f"残差 ||Ax-b|| = {np.linalg.norm(A @ x_complete - b)}")


if __name__ == '__main__':
    print("\n")
    test_pivoting()
