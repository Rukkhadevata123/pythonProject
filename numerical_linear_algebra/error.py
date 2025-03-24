"""
无穷范数误差估计
r=b-Ax_est
||x_real-x_est||_inf/||x_real||_inf<=cond(A)*||r||_inf/||b||_inf
Relative error estimation=||A^(-1)||_inf*||A||_inf*||r||_inf/||b||_inf
||A^(-1)||_inf can be estimated by the algorithm in cond.py

if estimation is not accurate, use iterative refinement
Newton iteration
r=b-Ax_est
Az=r to solve z using column pivoting
x_est+=z
if ||x-x_est||_inf<tolerance, break, otherwise continue
"""

import numpy as np
from cond import estimate_norm1, normInf_matrix, normInf


def estimate_relative_error(A, x_est, b):
    """
    估计解的相对误差上界

    参数:
    A -- 系数矩阵
    x_est -- 估计解
    b -- 右端向量

    返回:
    est_error -- 估计的相对误差上界
    """
    # 计算残差 r = b - A*x_est
    r = b - A @ x_est

    # 计算 ||r||_inf / ||b||_inf
    rel_residual = normInf(r) / normInf(b)

    # 计算或估计条件数
    norm_A = normInf_matrix(A)
    norm_A_inv = estimate_norm1(A)
    cond_A = norm_A * norm_A_inv

    # 估计相对误差上界
    est_error = cond_A * rel_residual

    return est_error


def iterative_refinement(A, b, x_est, max_iter=5, tol=1e-12):
    """
    通过迭代优化提高解的精度

    参数:
    A -- 系数矩阵
    b -- 右端向量
    x_est -- 初始估计解
    max_iter -- 最大迭代次数
    tol -- 收敛容差

    返回:
    x_refined -- 优化后的解
    iterations -- 实际迭代次数
    """
    from lu_decomposition import solve_with_partial_pivoting

    x = x_est.copy()

    for k in range(max_iter):
        # 计算残差
        r = b - A @ x

        # 如果残差足够小，停止迭代
        if normInf(r) < tol:
            return x, k

        # 求解修正方程 A*z = r
        z = solve_with_partial_pivoting(A, r)

        # 更新解
        x = x + z

    return x, max_iter
