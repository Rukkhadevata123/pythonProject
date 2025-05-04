"""
共轭梯度法(Conjugate Gradient Method)求解线性方程组

算法步骤:
1. 初始化：设置初始猜测x0，计算初始残差r0 = b - Ax0，令k = 0
2. 迭代过程：
   a. k自增1
   b. 若k=1，令p0 = r0
      否则，计算beta_(k-2) = <r_(k-1), r_(k-1)> / <r_(k-2), r_(k-2)>
           更新搜索方向p_(k-1) = r_(k-1) + beta_(k-2) * p_(k-2)
   c. 计算步长alpha_(k-1) = <r_(k-1), r_(k-1)> / <p_(k-1), A*p_(k-1)>
   d. 更新解：x_k = x_(k-1) + alpha_(k-1) * p_(k-1)
   e. 更新残差：r_k = r_(k-1) - alpha_(k-1) * A*p_(k-1)
3. 当残差范数小于容差或达到最大迭代次数时停止

数学特性：
- 共轭梯度法是一种基于最速下降法改进的迭代方法
- 理论上，对于n维问题，最多n次迭代即可收敛到精确解
- 适用于大型稀疏对称正定矩阵，无需显式存储矩阵分解
- 每次迭代仅需一次矩阵-向量乘法，计算复杂度为O(n²)
"""

import numpy as np


def is_symmetric(A, tol=1e-8):
    """
    检查矩阵是否为对称矩阵

    参数:
        A: 待检查的矩阵
        tol: 容差阈值，用于浮点数比较

    返回:
        bool: 如果矩阵对称返回True，否则返回False

    说明:
        对称矩阵满足A = A^T，考虑浮点误差进行判断
    """
    return np.allclose(A, A.T, rtol=tol)


def is_positive_definite(A, tol=1e-8):
    """
    检查矩阵是否为正定矩阵

    参数:
        A: 待检查的矩阵
        tol: 正定性判断的阈值

    返回:
        bool: 如果矩阵对称且所有特征值大于tol则返回True

    说明:
        - 正定矩阵必须是对称矩阵
        - 正定矩阵的所有特征值严格大于0
        - 计算特征值是一个计算成本较高的操作，仅用于验证
    """
    if not is_symmetric(A, tol):
        return False
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > tol)


def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=None, return_history=False):
    """
    使用共轭梯度法求解线性方程组 Ax = b

    参数:
        A: 系数矩阵，必须是对称正定矩阵 (n x n)
        b: 右侧向量 (n)
        x0: 初始猜测解 (n)，默认为全零向量
        tol: 收敛容差，基于相对残差 ||r||/||b|| < tol
        max_iter: 最大迭代次数，默认为矩阵维度
        return_history: 是否返回收敛历史

    返回:
        x: 近似解
        iterations: 实际迭代次数
        residuals: 每次迭代的相对残差列表 (如果return_history=True)

    算法特点:
        - 每次迭代生成互相共轭的搜索方向
        - A-共轭意味着向量p_i和p_j满足p_i^T·A·p_j = 0 (i≠j)
        - 理论上n步内收敛到精确解，实践中通常更少
        - 不需要存储中间矩阵，空间复杂度为O(n)
    """
    if not is_symmetric(A):
        raise ValueError("矩阵A必须是对称矩阵")

    n = len(b)
    # 初始化猜测解，默认为零向量
    if x0 is None:
        x0 = np.zeros(n)
    # 设置最大迭代次数，理论上n次迭代内必定收敛
    if max_iter is None:
        max_iter = n

    # 初始化解向量和残差
    x = x0.copy()
    r = b - A @ x  # 初始残差
    p = r.copy()  # 初始搜索方向

    # 用于跟踪收敛过程
    residuals = []
    norm_b = np.linalg.norm(b)  # 计算右侧向量的范数用于归一化
    relative_residual = np.linalg.norm(r) / norm_b  # 计算相对残差
    residuals.append(relative_residual)

    # 主迭代循环
    for k in range(max_iter):
        Ap = A @ p  # 计算矩阵-向量乘积，最耗时的操作
        r_norm_squared = np.dot(r, r)  # 当前残差的内积 <r, r>

        # 计算最优步长，最小化搜索方向上的残差
        alpha = r_norm_squared / np.dot(p, Ap)  # α = <r, r> / <p, Ap>

        # 更新解和残差
        x = x + alpha * p  # 解向量沿搜索方向移动
        r_old = r.copy()  # 保存旧残差用于计算β
        r = r - alpha * Ap  # 更新残差: r_new = r_old - α·A·p

        # 计算新的相对残差并检查收敛性
        relative_residual = np.linalg.norm(r) / norm_b
        residuals.append(relative_residual)

        # 收敛判定：当相对残差小于容差时停止
        if relative_residual < tol:
            break

        # 计算下一个搜索方向的更新系数
        beta = np.dot(r, r) / r_norm_squared  # β = <r_new, r_new> / <r_old, r_old>
        p = r + beta * p  # 更新搜索方向: p_new = r_new + β·p_old

    # 返回结果
    if return_history:
        return x, k + 1, residuals  # 返回解向量、迭代次数和残差历史
    else:
        return x, k + 1  # 仅返回解向量和迭代次数
