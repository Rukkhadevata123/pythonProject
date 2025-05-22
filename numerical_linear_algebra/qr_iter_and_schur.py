"""
QR迭代算法与Schur分解
==================

理论基础:
--------
QR迭代是一种计算矩阵特征值和特征向量的高效算法。Schur定理指出任意方阵A可以正交相似于
一个上三角矩阵，即存在正交矩阵Q使得Q^T*A*Q = T为上三角阵。实矩阵的实Schur分解得到的
是拟上三角矩阵，其对角线上包含1×1块(对应实特征值)或2×2块(对应复共轭特征值对)。

主要算法:
--------
1. 隐式QR算法: 避免显式QR分解，直接计算相似变换
2. 双重步位移: 使用二阶多项式位移保持计算过程中的实数性
3. 实Schur分解: 将矩阵约化为包含1×1和2×2块的拟上三角形式

数值特性:
--------
- 计算复杂度: 一般矩阵约O(10n³)，上Hessenberg矩阵约O(10n²)每次迭代
- 收敛性: 通常具有三次收敛特性，但受到特征值分布影响
- 位移策略: 采用Wilkinson位移或双重步位移提高收敛速度

应用场景:
--------
1. 特征值与特征向量计算
2. 线性系统的稳定性分析
3. 矩阵函数计算
4. 线性动态系统分析
"""

import numpy as np
from hessenberg import hessenberg_decomposition
from householder import householder, apply_householder


def double_shift_qr_iteration(H):
    """
    对上Hessenberg矩阵H进行一次双重步位移的QR迭代，使用apply_householder优化实现

    参数:
        H: 上Hessenberg矩阵

    返回:
        H_new: 一次迭代后的上Hessenberg矩阵
        P: 正交变换矩阵，满足H_new = P^T*H*P

    算法原理:
        使用隐式双重步位移QR迭代，通过特定的多项式位移提高收敛速度
        并保持计算过程中的实数性，即使原矩阵有复共轭特征值对
    """
    n = H.shape[0]
    if n <= 1:
        return H.copy(), np.eye(1)

    # 创建正交累积矩阵P
    P = np.eye(n)
    H_new = H.copy()

    # 使用最后的2×2子矩阵估计特征值位移
    m = n - 1
    # 计算s = trace 和 t = determinant
    s = H_new[m - 1, m - 1] + H_new[n - 1, n - 1]
    t = (
        H_new[m - 1, m - 1] * H_new[n - 1, n - 1]
        - H_new[m - 1, n - 1] * H_new[n - 1, m - 1]
    )

    # 计算初始bulge的元素
    # 这个矩阵乘法是隐式应用位移多项式p(A) = (A-μ₁I)(A-μ₂I)的第一列
    x = H_new[0, 0] * H_new[0, 0] + H_new[0, 1] * H_new[1, 0] - s * H_new[0, 0] + t
    y = H_new[1, 0] * (H_new[0, 0] + H_new[1, 1] - s)
    z = H_new[2, 1] * H_new[1, 0] if n >= 3 else 0

    # 逐步消去子对角线下方的元素
    for k in range(n - 2):
        # 构造Householder变换，用于消除y和z
        v_len = min(3, n - k)  # 确保不超出矩阵边界
        v_vec = np.zeros(v_len)
        v_vec[0] = x
        if v_len > 1:
            v_vec[1] = y
        if v_len > 2:
            v_vec[2] = z

        v, beta = householder(v_vec)

        # 计算变换应用范围
        col_start = max(0, k - 1)  # 应当从k-1开始以确保正确的bulge传播

        # 应用变换到H的右侧 - 使用apply_householder
        for j in range(col_start, n):
            H_new[k : k + v_len, j] = apply_householder(
                H_new[k : k + v_len, j], v, beta
            )

        # 应用变换到H的左侧 - 使用apply_householder
        for i in range(min(k + v_len + 1, n)):
            H_new[i, k : k + v_len] = apply_householder(
                H_new[i, k : k + v_len], v, beta
            )

        # 更新P矩阵 - 使用apply_householder
        for i in range(n):
            P[i, k : k + v_len] = apply_householder(P[i, k : k + v_len], v, beta)

        # 更新下一轮的x, y, z
        if k < n - 3:
            x = H_new[k + 1, k]
            y = H_new[k + 2, k]
            z = H_new[k + 3, k] if k + 3 < n else 0

    # 处理最后两行
    if n >= 2:
        k = n - 2
        x = H_new[k, k - 1]
        y = H_new[k + 1, k - 1]

        v_vec = np.array([x, y])
        v, beta = householder(v_vec)

        # 应用变换 - 使用apply_householder
        for j in range(k - 1, n):
            H_new[k : k + 2, j] = apply_householder(H_new[k : k + 2, j], v, beta)

        for i in range(k + 2):
            H_new[i, k : k + 2] = apply_householder(H_new[i, k : k + 2], v, beta)

        # 更新P - 使用apply_householder
        for i in range(n):
            P[i, k : k + 2] = apply_householder(P[i, k : k + 2], v, beta)

    # 清除数值噪声
    for i in range(2, n):
        for j in range(i - 1):
            if abs(H_new[i, j]) < 1e-14:
                H_new[i, j] = 0.0

    return H_new, P.T


def deflation_test(H, i, tol=1e-10):
    """
    检测上Hessenberg矩阵是否可以在位置i处进行分块(deflation)

    参数:
        H: 上Hessenberg矩阵
        i: 检查位置
        tol: 判断为零的阈值

    返回:
        bool: 如果H[i+1,i]足够小可以视为0，则可以分块
    """
    # 判断次对角线元素是否足够小可以置零
    # 使用Golub-Van Loan的判据: |h_{i+1,i}| ≤ u(|h_{i,i}| + |h_{i+1,i+1}|)
    # 其中u是机器精度或用户指定的阈值
    if abs(H[i, i - 1]) <= tol * (abs(H[i - 1, i - 1]) + abs(H[i, i])):
        return True
    return False


def real_schur_decomposition(A, max_iter=100, tol=1e-10):
    """
    计算实矩阵A的实Schur分解

    参数:
        A: 输入实矩阵 (n x n)
        max_iter: 最大迭代次数
        tol: 收敛判据阈值

    返回:
        T: 拟上三角Schur形式矩阵
        Q: 正交矩阵，满足A = QTQ^T
        blocks: 对角块的大小列表(1表示实特征值，2表示复特征值对)

    算法步骤:
        1. 将A转化为上Hessenberg形式
        2. 应用隐式QR迭代直到矩阵足够接近拟上三角形式
        3. 检测和处理2×2块(对应复共轭特征值对)
    """
    n = A.shape[0]
    if n <= 1:
        return A.copy(), np.eye(n), [1]

    # 1. 上Hessenberg化
    H, U0 = hessenberg_decomposition(A)
    Q = U0.copy()

    # 初始化对角块大小列表
    blocks = []

    # 改进的策略：按顺序处理子矩阵，而不是尝试一次处理整个矩阵
    # 这有助于更精确地识别小的块
    m = 0  # 已处理的行数

    while m < n:
        if m == n - 1:
            # 最后一个1×1块
            blocks.append(1)
            break

        # 尝试将当前2×2子矩阵当作一个块
        sub_size = min(n - m, 2)  # 确保不超出矩阵边界

        # 检查是否需要将当前位置作为1×1块处理
        if sub_size == 2 and abs(H[m + 1, m]) <= tol * (
            abs(H[m, m]) + abs(H[m + 1, m + 1])
        ):
            blocks.append(1)
            m += 1
            continue

        # 如果是最后两行，直接当作一个2×2块
        if m == n - 2:
            blocks.append(2)
            break

        # 对当前子矩阵执行多次QR迭代
        sub_converged = False
        sub_iter = 0

        while not sub_converged and sub_iter < max_iter:
            # 对当前子矩阵执行一次QR迭代
            H_sub = H[m:n, m:n]
            H_new, P = double_shift_qr_iteration(H_sub)
            H[m:n, m:n] = H_new
            Q[:, m:n] = Q[:, m:n] @ P

            # 检查是否可以分块
            for i in range(1, min(3, n - m)):  # 只检查前几个元素
                if abs(H[m + i, m + i - 1]) <= tol * (
                    abs(H[m + i - 1, m + i - 1]) + abs(H[m + i, m + i])
                ):
                    if i == 1:
                        blocks.append(1)
                        m += 1
                    else:
                        blocks.append(i)
                        m += i
                    sub_converged = True
                    break

            sub_iter += 1

        # 如果子矩阵没有收敛，强制进行分块
        if not sub_converged:
            print(f"警告: 子矩阵在{max_iter}次迭代后仍未收敛，强制分块")
            # 查找最小的次对角线元素
            min_idx = m
            min_val = float("inf")
            for i in range(m + 1, n):
                if abs(H[i, i - 1]) < min_val:
                    min_val = abs(H[i, i - 1])
                    min_idx = i - 1

            # 强制在最小元素处分块
            blocks.append(min_idx - m + 1)
            m = min_idx + 1

    return H, Q, blocks


def extract_eigenvalues(T, blocks):
    """
    从实Schur形式矩阵提取特征值

    参数:
        T: 拟上三角Schur形式矩阵
        blocks: 对角块的大小列表

    返回:
        eigenvalues: 特征值列表(复数形式)
    """
    eigenvalues = []
    idx = 0

    for block_size in blocks:
        if block_size == 1:
            # 1×1块直接提取对角线元素
            eigenvalues.append(T[idx, idx])
        elif block_size == 2:
            # 2×2块计算特征值
            a = T[idx, idx]
            b = T[idx, idx + 1]
            c = T[idx + 1, idx]
            d = T[idx + 1, idx + 1]

            # 计算特征方程 λ² - (a+d)λ + (ad-bc) = 0
            tr = a + d
            det = a * d - b * c
            disc = tr**2 - 4 * det

            if disc >= 0:
                # 两个实特征值
                sqrt_disc = np.sqrt(disc)
                eigenvalues.append((tr + sqrt_disc) / 2)
                eigenvalues.append((tr - sqrt_disc) / 2)
            else:
                # 一对复共轭特征值
                real_part = tr / 2
                imag_part = np.sqrt(-disc) / 2
                eigenvalues.append(complex(real_part, imag_part))
                eigenvalues.append(complex(real_part, -imag_part))
        else:
            # 对于大于2的块，需要直接计算子矩阵的特征值
            # 这种情况在实Schur分解中不应该出现，但如果出现，可以适当处理
            print("警告：进入大于2的块的分支")
            sub_matrix = T[idx : idx + block_size, idx : idx + block_size]
            sub_eigenvalues = np.linalg.eigvals(sub_matrix)
            eigenvalues.extend(sub_eigenvalues)

        idx += block_size

    return eigenvalues


def extract_eigenvectors(T, Q, blocks):
    """
    从实Schur分解直接计算特征向量

    参数:
        T: 拟上三角Schur形式矩阵
        Q: 正交变换矩阵
        blocks: 对角块大小

    返回:
        eigenvalues: 特征值列表
        eigenvectors: 特征向量列表
    """
    # 首先提取特征值
    eigenvalues = extract_eigenvalues(T, blocks)

    # 计算原始矩阵A = QTQ^T
    A = Q @ T @ Q.T

    # 直接使用解方程组的方法计算特征向量
    eigenvectors = []

    for lambd in eigenvalues:
        # 构造线性系统 (A - λI)
        A_lambda = A - lambd * np.eye(A.shape[0])

        # 使用SVD计算零空间
        u, s, vh = np.linalg.svd(A_lambda)

        # 找到最小奇异值对应的右奇异向量
        min_idx = np.argmin(s)
        v = vh[min_idx]

        # 归一化
        v = v / np.linalg.norm(v)

        eigenvectors.append(v)

    return eigenvalues, eigenvectors
