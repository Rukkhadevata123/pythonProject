"""
最小二乘问题求解方法
=================

理论基础:
--------
对于超定线性系统Ax=b（方程数m大于未知量n），通常不存在精确解。
最小二乘法寻找向量x，使残差向量 r = b - Ax 的2-范数最小化。

数学表示:
--------
最小化 ||Ax - b||_2

解法策略:
--------
1. QR分解法
   * 将矩阵A分解为 A = QR，其中Q是正交矩阵，R是上三角矩阵
   * 最小二乘问题转化为: ||QRx - b||_2 = ||Rx - Q^T b||_2
   * 解上三角系统 Rx = Q^T b 的前n行

2. 正规方程法
   * 通过A^T乘以原方程，得到方程 A^T Ax = A^T b
   * 求解该n阶方程组得到解

比较:
----
- QR分解: 数值稳定性好，适用于一般情况
- 正规方程: 计算量小，但在A条件数大时数值稳定性差
"""

import numpy as np
from substitution import back_substitution
from qr import householder_qr, givens_qr
from lu_decomposition import solve_with_partial_pivoting


def ls_solve_qr(A, b, method="householder"):
    """
    使用QR分解求解最小二乘问题 min ||Ax - b||_2

    算法原理:
    --------
    1. 对矩阵A进行QR分解: A = QR
    2. 将问题转化为 ||QRx - b||_2 = ||Rx - Q^T b||_2
    3. 求解上三角系统 Rx = Q^T b 的前n行

    参数:
    ----
    A: 系数矩阵 (m x n), 通常 m > n
    b: 右侧向量 (m)
    method: QR分解方法，"householder"或"givens"

    返回:
    ----
    x: 最小二乘解 (n)
    residual: 残差范数 ||Ax - b||_2

    数值特性:
    --------
    - 数值稳定性好，条件数敏感度为κ(A)
    - Householder变换通常更快，Givens旋转适合稀疏矩阵
    - 计算复杂度: O(2mn² - 2n³/3)，包括QR分解和回代求解
    """
    # 使用指定方法进行QR分解
    if method == "householder":
        Q, R = householder_qr(A)  # Householder变换，适合密集矩阵
    elif method == "givens":
        Q, R = givens_qr(A)  # Givens旋转，适合稀疏矩阵
    else:
        raise ValueError("不支持的QR分解方法，请使用'householder'或'givens'")

    # 计算 Q^T * b - 利用正交性简化问题
    y = Q.T @ b

    # 提取前n行形成确定的上三角系统
    n = A.shape[1]
    R_square = R[:n, :n]  # 提取R的主要部分（n×n上三角矩阵）
    y_square = y[:n]  # 对应的右侧向量

    # 回代法求解上三角系统 Rx = y
    x = back_substitution(R_square, y_square)

    # 计算残差范数 ||Ax - b||_2
    # 这是最小二乘问题的目标函数值，衡量解的质量
    residual = np.linalg.norm(A @ x - b)

    return x, residual


def normal_equations_solve(A, b):
    """
    使用正规方程 (A^T A)x = A^T b 求解最小二乘问题

    算法原理:
    --------
    1. 将最小二乘问题转化为正规方程: A^T Ax = A^T b
    2. 直接求解该方程组得到解

    参数:
    ----
    A: 系数矩阵 (m x n)
    b: 右侧向量 (m)

    返回:
    ----
    x: 最小二乘解 (n)
    residual: 残差范数 ||Ax - b||_2

    数值特性:
    --------
    - 计算量小于QR分解法: O(mn² + n³/3)
    - 数值稳定性较差，条件数敏感度为κ(A)²
    - 适用于A为满秩且条件数不太大的情况
    - 在高精度要求下，不推荐用于病态问题
    """
    # 计算正规方程系统
    ATA = A.T @ A  # 构造系数矩阵 A^T A (n×n)
    ATb = A.T @ b  # 构造右侧向量 A^T b (n)

    # 求解正规方程 A^T A x = A^T b
    # 使用LU分解求解，也可使用Cholesky分解（对称正定情况）
    x = solve_with_partial_pivoting(ATA, ATb)

    # 计算残差范数 ||Ax - b||_2
    residual = np.linalg.norm(A @ x - b)

    return x, residual


def compare_ls_methods(A, b):
    """
    比较不同最小二乘求解方法的结果和性能

    功能:
    ----
    1. 分别使用QR分解和正规方程法求解最小二乘问题
    2. 比较解的精度、残差和计算时间

    参数:
    ----
    A: 系数矩阵 (m x n)
    b: 右侧向量 (m)

    返回:
    ----
    results: 包含各方法结果和性能指标的字典

    应用场景:
    --------
    - 用于评估不同方法在特定问题上的适用性
    - 帮助选择最适合的算法，平衡精度和效率
    """
    import time

    results = {}

    # 使用Householder QR分解
    start = time.time()
    x_hh, res_hh = ls_solve_qr(A, b, "householder")
    time_hh = time.time() - start
    results["householder"] = {"x": x_hh, "residual": res_hh, "time": time_hh}

    # 使用Givens QR分解
    start = time.time()
    x_gv, res_gv = ls_solve_qr(A, b, "givens")
    time_gv = time.time() - start
    results["givens"] = {"x": x_gv, "residual": res_gv, "time": time_gv}

    # 使用正规方程
    start = time.time()
    x_ne, res_ne = normal_equations_solve(A, b)
    time_ne = time.time() - start
    results["normal_equations"] = {"x": x_ne, "residual": res_ne, "time": time_ne}

    # 计算条件数，评估问题难度
    results["condition_number"] = np.linalg.cond(A)

    return results
