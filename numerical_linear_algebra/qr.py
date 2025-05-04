import numpy as np
from householder import householder
from givens import givens, givens_matrix
from substitution import back_substitution


def householder_qr(A):
    """
    使用Householder变换进行QR分解（空间优化版本）

    参数:
        A: 输入矩阵 (m x n), m >= n

    返回:
        Q: 正交矩阵 (m x m)
        R: 上三角矩阵 (m x n)

    数学原理:
        1. 对每一列j，计算Householder向量v和系数beta，使得:
           (I - beta*v*v') * A[j:m,j] = [r_jj, 0, ..., 0]^T
        2. 将变换应用到子矩阵A[j:m,j:n]
        3. 隐式存储v在下三角部分，显式计算Q和R
    """
    m, n = A.shape
    R = A.copy().astype(float)  # 确保浮点运算
    Q = np.eye(m)  # 初始化正交矩阵

    for j in range(min(m - 1, n)):  # 每列处理
        # 计算当前列的Householder变换
        v, beta = householder(R[j:m, j])

        # 应用变换到子矩阵 (高效实现)
        # 数学等价于: R[j:m,j:n] = (I - beta*v*v') @ R[j:m,j:n]
        w = beta * np.dot(v, R[j:m, j:n])
        R[j:m, j:n] -= np.outer(v, w)

        # 累积变换到Q (注意是左乘)
        # 数学等价于: Q = Q @ H，其中H = [I 0; 0 (I-beta*v*v')]
        Q_j = np.eye(m)
        Q_j[j:m, j:m] -= beta * np.outer(v, v)
        Q = Q @ Q_j

    return Q, R


def givens_qr(A):
    """
    使用Givens旋转进行QR分解

    参数:
        A: 输入矩阵 (m x n)

    返回:
        Q: 正交矩阵 (m x m)
        R: 上三角矩阵 (m x n)

    算法特点:
        1. 通过平面旋转逐步消元
        2. 适合稀疏矩阵或特定结构矩阵
        3. 数值稳定性与Householder相当
    """
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)

    for j in range(n):  # 列优先处理
        for i in range(m - 1, j, -1):  # 从底部向上消元
            if abs(R[i, j]) > 1e-12:  # 避免数值误差
                # 计算Givens旋转参数
                c, s = givens(R[i - 1, j], R[i, j])

                # 构造旋转矩阵
                G = givens_matrix(m, i - 1, i, c, s)

                # 应用旋转
                R = G @ R
                Q = Q @ G.T  # 累积正交变换

    return Q, R


def qr_solve(A, b):
    """
    基于QR分解的线性方程组求解器

    参数:
        A: 系数矩阵 (m x n)
        b: 右侧向量 (m)

    返回:
        x: 最小二乘解 (n)

    异常处理:
        当m < n时抛出ValueError

    数学保证:
        返回的解x最小化||Ax - b||_2
    """
    m, n = A.shape
    if m < n:
        raise ValueError("欠定系统: m < n")

    Q, R = householder_qr(A)
    y = Q.T @ b  # 计算投影

    # 解上三角系统
    x = back_substitution(R[:n, :n], y[:n])

    return x
