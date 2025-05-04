import numpy as np
from substitution import forward_substitution, back_substitution


def is_symmetric(A, tol=1e-8):
    """
    检查矩阵是否为对称矩阵

    参数:
        A: 待检查的矩阵
        tol: 容差阈值，用于判断浮点数是否相等

    返回:
        bool: 如果矩阵是对称的返回True，否则返回False

    说明:
        - 对称矩阵满足A = A^T
        - 使用numpy.allclose函数比较A和A^T，考虑浮点误差
    """
    return np.allclose(A, A.T, rtol=tol)


def ldl_decomposition(A):
    """
    对对称矩阵A进行LDL^T分解，将矩阵分解为单位下三角矩阵L和对角矩阵D的乘积
    使得 A = L*D*L^T

    参数:
        A: 输入矩阵 (n x n)，必须是对称矩阵

    返回:
        L: 单位下三角矩阵（对角线元素均为1）
        D: 对角矩阵的对角元素，存储为一维数组

    算法原理:
        LDL^T分解是Cholesky分解的变种，不要求矩阵正定
        如果将A的元素表示为a_ij，L的元素为l_ij，D的对角元素为d_j，则：
        1. d_j = a_jj - ∑(k=1 to j-1) l_jk^2 * d_k
        2. l_ij = (a_ij - ∑(k=1 to j-1) l_ik * l_jk * d_k) / d_j，其中i > j

    注意:
        - 如果矩阵不是对称的，函数会抛出ValueError
        - 如果遇到d_j = 0，分解将失败（矩阵奇异）
    """
    # 检查矩阵是否为对称矩阵
    if not is_symmetric(A):
        raise ValueError("输入矩阵不是对称矩阵")

    n = len(A)
    L = np.eye(n)  # 初始化为单位矩阵，保证对角线元素为1
    D = np.zeros(n)  # 只存储对角线元素

    for j in range(n):
        # 计算D的对角线元素 d_j = a_jj - ∑(k=0 to j-1) l_jk^2 * d_k
        D[j] = A[j, j]
        for k in range(j):
            D[j] -= L[j, k] ** 2 * D[k]

        # 如果对角元素接近于零，可能是数值不稳定或矩阵接近奇异
        if abs(D[j]) < 1e-14:
            print(f"警告: D[{j}]接近零，可能导致数值不稳定")

        # 计算L的第j列元素 l_ij = (a_ij - ∑(k=0 to j-1) l_ik * l_jk * d_k) / d_j
        for i in range(j + 1, n):
            L[i, j] = A[i, j]  # 首先设置为a_ij
            # 减去之前列的贡献
            for k in range(j):
                L[i, j] -= L[i, k] * L[j, k] * D[k]
            # 除以当前的对角元素
            L[i, j] /= D[j]

    return L, D


def solve_with_ldl(A, b):
    """
    使用LDL^T分解求解线性方程组 Ax = b

    参数:
        A: 系数矩阵，必须是对称矩阵
        b: 右侧向量

    返回:
        x: 方程组的解

    求解步骤:
        1. 对A进行LDL^T分解，得到L和D
        2. 将原方程Ax = b转化为LDL^Tx = b
        3. 分三步求解:
           a. 前代法求解Ly = b
           b. 求解Dz = y
           c. 回代法求解L^Tx = z

    复杂度:
        - 时间复杂度: O(n^3)用于分解，O(n^2)用于求解
        - 空间复杂度: O(n^2)

    优势:
        - 比一般的高斯消元法更高效，当A是对称矩阵时
        - 比Cholesky分解更稳定，不要求矩阵正定
    """
    # 1. LDL^T分解
    L, d = ldl_decomposition(A)

    # 2. 前代法求解 Ly = b（L是单位下三角矩阵）
    y = forward_substitution(L, b)

    # 3. 对角系统求解 Dz = y（D是对角矩阵）
    z = np.zeros_like(y)
    for i in range(len(y)):
        if abs(d[i]) < 1e-14:
            raise ValueError(f"对角元素D[{i}]接近零，方程组可能无解或有无穷多解")
        z[i] = y[i] / d[i]

    # 4. 回代法求解 L^T x = z（L^T是单位上三角矩阵）
    x = back_substitution(L.T, z)

    return x
