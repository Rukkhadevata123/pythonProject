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

    # 确保完全对称
    for i in range(n):
        for j in range(i + 1, n):
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

    # 对称矩阵保证b = c
    b = (b + T[n - 1, n - 2]) / 2  # 强制对称性

    delta = (a - c) / 2.0
    if delta == 0:
        # 处理delta为零的情况
        mu = c - abs(b)
    else:
        sign_delta = 1 if delta >= 0 else -1
        mu = c - (b**2) / (delta + sign_delta * np.sqrt(delta**2 + b**2))

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

    # 初始Givens旋转
    x = T_new[0, 0] - mu
    y = T_new[1, 0]
    c, s = givens(x, y)

    # 应用初始旋转
    T_new = apply_givens_rotation(T_new, 0, 1, c, s, left=True)
    T_new = apply_givens_rotation(T_new, 0, 1, c, s, left=False)
    Q = apply_givens_rotation(Q, 0, 1, c, s, left=False)

    # 执行隐式QR步骤
    for k in range(1, n - 1):
        # 计算下一个旋转
        x = T_new[k, k - 1]
        y = T_new[k + 1, k - 1]
        c, s = givens(x, y)

        # 应用旋转
        T_new = apply_givens_rotation(T_new, k, k + 1, c, s, left=True)
        T_new = apply_givens_rotation(T_new, k, k + 1, c, s, left=False)
        Q = apply_givens_rotation(Q, k, k + 1, c, s, left=False)

    # 维持三对角结构和对称性
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                T_new[i, j] = 0.0
            elif abs(T_new[i, j]) < 1e-14:
                T_new[i, j] = 0.0

    # 确保对称性
    for i in range(n - 1):
        T_new[i, i + 1] = T_new[i + 1, i]

    return T_new, Q


def implicit_symmetric_qr(A, tol=1e-12, max_iter=5000):
    """
    隐式对称QR算法，计算对称矩阵的特征值和特征向量

    参数:
        A: 对称矩阵
        tol: 收敛判据阈值
        max_iter: 最大迭代次数

    返回:
        eigenvalues: 特征值列表
        U: 特征向量矩阵(列向量是特征向量)
    """
    # 先进行三对角化
    T, U = tridiagonalize(A)
    n = T.shape[0]

    # 输出迭代信息的标志
    verbose = False  # 设为True可以输出迭代细节

    # 整体迭代计数
    iter_count = 0
    active_size = n

    # 跟踪未收敛的次对角线元素数量
    num_nonzero = n - 1

    while num_nonzero > 0 and iter_count < max_iter:
        # 查找可以分离的最大块
        m = 0
        for i in range(active_size - 1, 0, -1):
            if abs(T[i, i - 1]) <= tol * (abs(T[i - 1, i - 1]) + abs(T[i, i])):
                T[i, i - 1] = 0.0
                T[i - 1, i] = 0.0
                m = i
                break

        # 确定当前工作区
        if m > 0:
            # 可以分离矩阵块
            active_size = m

        # 对活动子矩阵应用一次QR迭代
        if active_size > 1:  # 只有当子矩阵大小>1时才需迭代
            sub_T = T[:active_size, :active_size]
            sub_T_new, Q_sub = qr_iteration_with_wilkinson_shift(sub_T)

            # 更新矩阵
            T[:active_size, :active_size] = sub_T_new

            # 更新特征向量矩阵
            U[:, :active_size] = U[:, :active_size] @ Q_sub

            if verbose and iter_count % 10 == 0:
                print(f"Iteration {iter_count}, active size: {active_size}")

        # 检查次对角线元素接近零的情况
        num_nonzero = 0
        for i in range(n - 1):
            if abs(T[i + 1, i]) > tol * (abs(T[i, i]) + abs(T[i + 1, i + 1])):
                num_nonzero += 1

        iter_count += 1

    if verbose:
        print(f"QR algorithm converged in {iter_count} iterations.")

    # 提取特征值(都是实数)
    eigenvalues = np.diag(T).real

    # 按降序排序特征值和对应特征向量
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]

    return eigenvalues, U


def compute_eigenvectors(A, eigenvalues, tol=1e-12):
    """针对对称矩阵计算特征向量，使用反迭代法"""
    n = A.shape[0]
    eigenvectors = np.zeros((n, n))

    for i, lambda_i in enumerate(eigenvalues):
        # 初始向量
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        # 反迭代
        for _ in range(20):  # 通常几次迭代就足够了
            # 求解线性方程组 (A-λI)w = v
            w = np.linalg.solve(A - lambda_i * np.eye(n) + 1e-10 * np.eye(n), v)
            w = w / np.linalg.norm(w)

            # 检查收敛
            if np.linalg.norm(w - v) < tol:
                break

            v = w

        eigenvectors[:, i] = w

    return eigenvectors


def eigensolver_symmetric(A, tol=1e-12):
    # 计算特征值
    eigenvalues, U_approx = implicit_symmetric_qr(A, tol)

    # 使用反迭代法计算准确特征向量
    eigenvectors = compute_eigenvectors(A, eigenvalues, tol)

    return eigenvalues, eigenvectors
