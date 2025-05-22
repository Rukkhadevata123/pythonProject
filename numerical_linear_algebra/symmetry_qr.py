"""
对称矩阵特征值计算算法
==================

实现三个关键算法：
1. 三对角分解 (使用Householder变换)
2. Wilkinson位移QR迭代
3. 隐式对称QR算法

这些算法专门为计算对称矩阵的特征值而设计，比一般的QR算法更高效。
"""

import numpy as np
from householder import householder, apply_householder
from givens import givens, apply_givens_rotation


def tridiagonalize(A):
    """
    将对称矩阵A约化为三对角形式

    参数:
        A: 对称矩阵 (n x n)

    返回:
        T: 三对角矩阵
        U: 正交变换矩阵，满足 A = UTU^T

    算法步骤:
        使用Householder变换逐列消元，将对称矩阵约化为三对角形式
    """
    n = A.shape[0]
    T = A.copy()
    U = np.eye(n)

    for k in range(n - 2):
        # 提取下方元素形成向量
        x = T[k + 1 :, k].copy()

        # 如果向量全为零，跳过
        if np.linalg.norm(x) < 1e-14:
            continue

        # 计算Householder变换
        v, beta = householder(x)

        # 对每列应用Householder变换 (右侧变换)
        for j in range(k + 1, n):
            T[k + 1 :, j] = apply_householder(T[k + 1 :, j], v, beta)

        # 对每行应用Householder变换 (左侧变换)
        for i in range(k + 1, n):
            T[i, k + 1 :] = apply_householder(T[i, k + 1 :], v, beta)

        # 更新对称性
        for i in range(k + 1, n):
            for j in range(i + 1, n):
                T[i, j] = T[j, i]

        # 设置次对角线元素
        alpha = np.linalg.norm(x)
        T[k + 1, k] = alpha
        T[k, k + 1] = alpha

        # 将其余元素置零
        for i in range(k + 2, n):
            T[i, k] = 0.0
            T[k, i] = 0.0

        # 更新正交变换矩阵
        for i in range(n):
            U[i, k + 1 :] = apply_householder(U[i, k + 1 :], v, beta)

    # 清理微小的非零元素，确保严格三对角
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                T[i, j] = 0.0
            # 确保对称性
            if i > j:
                T[i, j] = T[j, i]

    return T, U


def wilkinson_shift(T, n):
    """
    计算Wilkinson位移

    参数:
        T: 对称三对角矩阵
        n: 矩阵维度

    返回:
        mu: Wilkinson位移值
    """
    # 获取右下角2x2子矩阵
    a = T[n - 2, n - 2]
    b = T[n - 2, n - 1]
    c = T[n - 1, n - 1]

    # 计算位移值
    d = (a - c) / 2
    # 避免除以零或引起不稳定的位移
    sgn_d = 1 if d >= 0 else -1
    mu = c - (sgn_d * b**2) / (abs(d) + np.sqrt(d**2 + b**2))

    return mu


def qr_iteration_with_wilkinson_shift(T):
    """
    带Wilkinson位移的QR迭代，处理对称三对角矩阵

    参数:
        T: 对称三对角矩阵

    返回:
        T_new: 一次迭代后的矩阵，接近对角形式
        Q: 正交变换矩阵
    """
    n = T.shape[0]
    T_new = T.copy()
    Q = np.eye(n)

    # 计算Wilkinson位移
    mu = wilkinson_shift(T_new, n)

    # 构造位移后的矩阵 T - μI
    T_shifted = T_new.copy()
    for i in range(n):
        T_shifted[i, i] -= mu

    # 使用Givens旋转进行QR分解
    for k in range(n - 1):
        # 计算Givens旋转系数
        c, s = givens(T_shifted[k, k], T_shifted[k + 1, k])

        # 应用旋转到矩阵中
        T_shifted = apply_givens_rotation(T_shifted, k, k + 1, c, s, left=True)
        T_shifted = apply_givens_rotation(T_shifted, k, k + 1, c, s, left=False)

        # 更新累积正交矩阵Q
        Q = apply_givens_rotation(Q, k, k + 1, c, s, left=False)

    # 计算最终矩阵 T_new = Q^T * T * Q
    T_new = Q.T @ T @ Q

    # 清理数值误差，保持三对角结构
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                T_new[i, j] = 0.0
            elif abs(T_new[i, j]) < 1e-14:
                T_new[i, j] = 0.0

    return T_new, Q


def implicit_symmetric_qr(A, tol=1e-12, max_iter=100):
    """
    隐式对称QR算法，计算对称矩阵的特征值

    参数:
        A: 对称矩阵
        tol: 收敛判据阈值
        max_iter: 最大迭代次数

    返回:
        eigenvalues: 特征值列表
        U: 特征向量矩阵(列向量是特征向量)
    """
    # 先进行三对角化
    T, U0 = tridiagonalize(A)
    n = T.shape[0]
    U = U0.copy()

    # 整体迭代计数
    iter_count = 0

    # 处理每个子矩阵，直到所有次对角线元素都足够小
    active_size = n

    while active_size > 1 and iter_count < max_iter:
        # 查找可以分离的小块
        for m in range(active_size - 1, 0, -1):
            if abs(T[m, m - 1]) <= tol * (abs(T[m - 1, m - 1]) + abs(T[m, m])):
                # 将次对角线元素置零
                T[m, m - 1] = 0.0
                T[m - 1, m] = 0.0
                break
        else:
            # 没有找到可分离的块，对整个活动子矩阵进行迭代
            m = 0

        # 处理活动子矩阵 T[m:active_size, m:active_size]
        if m < active_size - 1:
            # 确定子矩阵的大小
            submatrix_size = active_size - m

            # 对子矩阵应用一次QR迭代
            sub_T = T[m:active_size, m:active_size]
            sub_T_new, Q_sub = qr_iteration_with_wilkinson_shift(sub_T)

            # 更新整个矩阵
            T[m:active_size, m:active_size] = sub_T_new

            # 更新累积正交矩阵
            U[:, m:active_size] = U[:, m:active_size] @ Q_sub

            # 检查迭代后是否有次对角线元素足够接近零
            for i in range(m + 1, active_size):
                if abs(T[i, i - 1]) <= tol * (abs(T[i - 1, i - 1]) + abs(T[i, i])):
                    T[i, i - 1] = 0.0
                    T[i - 1, i] = 0.0
        else:
            # 已经处理到最上面的子矩阵，减少活动大小
            active_size = m

        iter_count += 1

    # 提取特征值
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1 or abs(T[i + 1, i]) < tol:
            # 1x1块 - 实特征值
            eigenvalues.append(T[i, i])
            i += 1
        else:
            # 2x2块 - 可能是复共轭对
            a, b = T[i, i], T[i, i + 1]
            c, d = T[i + 1, i], T[i + 1, i + 1]

            trace = a + d
            det = a * d - b * c
            disc = trace**2 - 4 * det

            if disc < 0:
                # 复共轭对
                real = trace / 2
                imag = np.sqrt(-disc) / 2
                eigenvalues.append(complex(real, imag))
                eigenvalues.append(complex(real, -imag))
            else:
                # 两个实特征值
                sqrt_disc = np.sqrt(disc)
                eigenvalues.append((trace + sqrt_disc) / 2)
                eigenvalues.append((trace - sqrt_disc) / 2)

            i += 2

    # 排序特征值（通常按从大到小）
    eigenvalues = sorted(eigenvalues, key=lambda x: float(np.real(x)), reverse=True)

    return eigenvalues, U


def compute_eigenvectors(A, eigenvalues, tol=1e-10):
    """
    通过求解线性方程组(A-λI)v=0直接计算特征向量

    参数:
        A: 对称矩阵
        eigenvalues: 已知的特征值列表
        tol: 数值容差

    返回:
        eigenvectors: 对应特征值的特征向量列表
    """
    n = A.shape[0]
    eigenvectors = []

    for lambd in eigenvalues:
        # 构造线性系统 (A - λI)
        A_lambda = A - lambd * np.eye(n)

        # 使用SVD计算零空间
        u, s, vh = np.linalg.svd(A_lambda)

        # 找到最小奇异值对应的右奇异向量
        min_idx = np.argmin(s)
        v = vh[min_idx]

        # 归一化
        v = v / np.linalg.norm(v)

        eigenvectors.append(v)

    return eigenvectors
