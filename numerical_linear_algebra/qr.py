"""
householder qr
for j = 1:n
    if j<m
        [v,beta] = house(A(j:m,j))
        A(j:m,j:n) = (I_(m-j+1) - beta*v*v')*A(j:m,j:n) 节省空间
        d(j) = beta
        A(j+1:m,j) = v(2:m-j+1)
    end
end
"""

import numpy as np
from householder import householder


def householder_qr(A):
    """
    使用 Householder 变换计算 QR 分解

    参数:
        A: 输入矩阵 (m x n)

    返回:
        Q: 正交矩阵 (m x m)
        R: 上三角矩阵 (m x n)
    """
    m, n = A.shape
    A_copy = A.copy()
    d = np.zeros(min(m, n))

    # 存储 Householder 向量
    vs = []

    # 使用 Householder 变换将 A 转换为上三角形式
    for j in range(min(m - 1, n)):
        # 对当前列的子向量应用 Householder 变换
        v, beta = householder(A_copy[j:m, j])

        # 存储用于构造 Q 矩阵的信息
        vs.append(v)
        d[j] = beta

        # 应用 Householder 变换到 A 的子矩阵
        # (I - beta * v * v') * A[j:m, j:n]
        for k in range(j, n):
            # 计算 v' * A[j:m, k]
            v_dot_A = np.dot(v, A_copy[j:m, k])

            # 更新 A[j:m, k]
            A_copy[j:m, k] = A_copy[j:m, k] - beta * v_dot_A * v

        # 存储 Householder 向量（除了第一个元素）
        if j + 1 < m:
            A_copy[j + 1 : m, j] = v[1:]

    # 提取上三角矩阵 R
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i <= j:
                R[i, j] = A_copy[i, j]

    # 构造正交矩阵 Q
    Q = np.eye(m)

    # 从右到左应用 Householder 变换以构造 Q
    for j in range(min(m - 1, n) - 1, -1, -1):
        v = np.zeros(m)
        v[j] = 1
        v[j + 1 : m] = A_copy[j + 1 : m, j]

        beta = d[j]

        # Q = Q * (I - beta * v * v')
        v_dot_Q = np.zeros(m)
        for i in range(m):
            v_dot_Q[i] = np.dot(v, Q[:, i])

        for i in range(m):
            Q[:, i] = Q[:, i] - beta * v_dot_Q[i] * v

    return Q, R


def compact_householder_qr(A):
    """
    使用 Householder 变换计算 QR 分解（紧凑形式实现）

    参数:
        A: 输入矩阵 (m x n)

    返回:
        Q: 正交矩阵 (m x m)
        R: 上三角矩阵 (m x n)
    """
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for j in range(min(m - 1, n)):
        # 对当前列的子向量应用 Householder 变换
        v, beta = householder(R[j:m, j])

        # 应用 Householder 变换到 R 的子矩阵
        H_sub = np.eye(m - j) - beta * np.outer(v, v)
        R[j:m, j:n] = H_sub @ R[j:m, j:n]

        # 应用 Householder 变换到 Q
        H = np.eye(m)
        H[j:m, j:m] = H_sub
        Q = Q @ H

    return Q, R


def givens_qr(A):
    """
    使用 Givens 旋转计算 QR 分解

    参数:
        A: 输入矩阵 (m x n)

    返回:
        Q: 正交矩阵 (m x m)
        R: 上三角矩阵 (m x n)
    """
    from givens import givens, givens_matrix

    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for j in range(n):  # 对每一列
        for i in range(m - 1, j, -1):  # 从下往上消元
            if R[i, j] != 0:  # 如果元素不为零
                # 计算 Givens 旋转参数
                c, s = givens(R[i - 1, j], R[i, j])

                # 构造 Givens 矩阵
                G = givens_matrix(m, i - 1, i, c, s)

                # 应用 Givens 旋转到 R
                R = G @ R

                # 更新 Q (注意这里是 G.T 而不是 G)
                Q = Q @ G.T

    return Q, R


def qr_solve(A, b):
    """
    使用 QR 分解求解线性方程组 Ax = b

    参数:
        A: 系数矩阵 (m x n)
        b: 右侧向量 (m)

    返回:
        x: 解向量 (n)
    """
    Q, R = householder_qr(A)

    # 计算 Q^T * b
    y = Q.T @ b

    # 求解上三角系统 Rx = y
    m, n = A.shape
    if m < n:
        raise ValueError("方程组没有唯一解")

    # 仅取上三角部分
    R_square = R[:n, :n]
    y_square = y[:n]

    # 使用导入的回代法求解上三角系统
    from substitution import back_substitution

    x = back_substitution(R_square, y_square)

    return x


def qr_least_squares(A, b):
    """
    使用 QR 分解求解最小二乘问题 min ||Ax - b||_2

    参数:
        A: 系数矩阵 (m x n), 通常 m > n
        b: 右侧向量 (m)

    返回:
        x: 最小二乘解 (n)
        residual: 残差 ||Ax - b||_2
    """
    Q, R = householder_qr(A)

    # 计算 Q^T * b
    y = Q.T @ b

    # 求解上三角系统 Rx = y (仅前 n 行)
    n = A.shape[1]
    R_square = R[:n, :n]
    y_square = y[:n]

    # 使用导入的回代法求解上三角系统
    from substitution import back_substitution

    x = back_substitution(R_square, y_square)

    # 计算残差
    residual = np.linalg.norm(A @ x - b)

    return x, residual
