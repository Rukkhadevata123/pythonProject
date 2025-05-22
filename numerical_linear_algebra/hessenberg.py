"""
上Hessenberg分解
=============

理论基础:
--------
上Hessenberg矩阵是一种次对角线以下元素全为零的矩阵，即对所有i > j+1，有h_ij = 0。
将一般矩阵变换为Hessenberg形式是计算特征值的重要预处理步骤，可大幅提高后续算法效率。

数学表示:
--------
对于矩阵A，其上Hessenberg分解为：
A = QHQ^T
其中Q为正交矩阵，H为上Hessenberg矩阵。

算法原理:
--------
类似于QR分解，但不是将整个下三角置零，而是只将主次对角线以下的元素置零，
即每列只消去该列对角线下面第二个元素及以下的元素。

应用场景:
--------
1. 特征值计算的预处理步骤
2. QR算法的效率优化
3. 大型稀疏矩阵的特征值分析
"""

from householder import householder, apply_householder
import numpy as np


def hessenberg_decomposition(A):
    """
    使用Householder变换将矩阵A约化为上Hessenberg形式

    参数:
        A: 输入方阵 (n x n)

    返回:
        H: 上Hessenberg矩阵
        Q: 正交矩阵，满足A = QHQ^T

    算法步骤:
        1. 逐列应用Householder变换，但只消除主次对角线以下的元素
        2. 对每列k，消除A(k+1:n, k)中的元素，保留A(k,k)和A(k+1,k)
        3. 变换应用于矩阵的两侧，以保持相似性

    特点:
        - 保持矩阵的特征值不变
        - 计算复杂度为O(10/3 * n^3)，远低于直接特征值计算
        - 数值稳定性好，适合后续特征值算法
    """
    n = A.shape[0]
    if n <= 2:
        # 2x2或更小的矩阵已经是上Hessenberg形式
        return A.copy(), np.eye(n)

    # 创建矩阵副本以避免修改原矩阵
    H = A.copy()
    # 初始化累积正交变换矩阵
    Q = np.eye(n)

    # 存储所有Householder变换，用于构建Q
    vs = []
    betas = []

    # 根据算法6.4.1，从第一列到倒数第三列应用Householder变换
    for k in range(n - 2):
        # 提取当前列的子向量，从k+1行开始（需要置零的元素）
        x = H[k + 1 :, k].copy()

        # 计算Householder变换
        v, beta = householder(x)
        vs.append((v, k + 1))
        betas.append(beta)

        # 应用变换到H的右侧: H = (I - beta*v*v^T) * H
        # 只需处理k+1行及以下、k列及以右的部分
        for j in range(k, n):
            # 提取H中要变换的列向量
            col = H[k + 1 :, j].copy()
            # 应用Householder变换
            H[k + 1 :, j] = apply_householder(col, v, beta)

        # 应用变换到H的左侧: H = H * (I - beta*v*v^T)
        # 注意左侧乘以(I - beta*v*v^T)相当于右侧乘以(I - beta*v*v^T)^T，即(I - beta*v*v^T)
        for i in range(n):
            # 提取H中要变换的行向量部分
            row = H[i, k + 1 :].copy()
            # 应用Householder变换
            H[i, k + 1 :] = apply_householder(row, v, beta)

    # 构建完整的正交矩阵Q
    for i in range(len(vs)):
        v, start_idx = vs[i]
        beta = betas[i]
        for j in range(n):
            # 提取Q的行向量要变换的部分
            row = Q[j, start_idx:].copy()
            # 应用Householder变换
            Q[j, start_idx:] = apply_householder(row, v, beta)

    return H, Q.T


def verify_hessenberg(H, tol=1e-10):
    """
    验证矩阵是否为上Hessenberg形式

    参数:
        H: 待验证的矩阵
        tol: 判断为零的容差

    返回:
        bool: 如果矩阵是上Hessenberg形式则返回True
    """
    n = H.shape[0]
    for i in range(2, n):
        for j in range(i - 1):
            if abs(H[i, j]) > tol:
                return False
    return True


def verify_decomposition(A, H, Q, tol=1e-10):
    """
    验证Hessenberg分解的正确性

    参数:
        A: 原始矩阵
        H: Hessenberg矩阵
        Q: 正交矩阵
        tol: 误差容差

    返回:
        bool: 如果A ≈ QHQ^T且Q^TQ ≈ I则返回True

    验证项:
        1. 矩阵H是否为上Hessenberg形式
        2. 矩阵Q是否为正交矩阵(Q^TQ = I)
        3. 是否满足A = QHQ^T
    """
    # 验证H是上Hessenberg形式
    if not verify_hessenberg(H, tol):
        print("H不是上Hessenberg矩阵")
        return False

    # 验证Q是正交矩阵
    I = np.eye(Q.shape[0])
    if not np.allclose(Q.T @ Q, I, rtol=tol, atol=tol):
        print("Q不是正交矩阵")
        return False

    # 验证A = QHQ^T
    A_reconstructed = Q @ H @ Q.T
    if not np.allclose(A, A_reconstructed, rtol=tol, atol=tol):
        print("A ≠ QHQ^T")
        return False

    return True
