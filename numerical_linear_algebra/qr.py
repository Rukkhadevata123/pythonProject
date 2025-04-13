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
from givens import givens, givens_matrix
from substitution import back_substitution


def householder_qr(A):
    # A: 输入矩阵 (m x n)
    # Q: 正交矩阵 (m x m)
    # R: 上三角矩阵 (m x n)
    # 这里就是课本上的空间节约版
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
    # 使用 Givens 旋转计算 QR 分解
    # A: 输入矩阵 (m x n)
    # Q: 正交矩阵 (m x m)
    # R: 上三角矩阵 (m x n)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    # 为什么这里是两个for循环？因为每一列消元过程，givens变换只能对两行旋转操作
    # 一层循环是列，另一层循环是行
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
    # 这是用来求解方程组的方法
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
    x = back_substitution(R_square, y_square)

    return x
