"""
Least Squares
import qr
A=QR
b=[c1,c2]
c1=Q_1'*b
solve upper triangular matrix
Rx=c1
"""

import numpy as np
from substitution import back_substitution
from qr import householder_qr, givens_qr, compact_householder_qr


def ls_solve_qr(A, b, method="householder"):
    """
    使用 QR 分解求解最小二乘问题 min ||Ax - b||_2

    参数:
        A: 系数矩阵 (m x n), 通常 m > n
        b: 右侧向量 (m)
        method: QR分解方法，可选 "householder", "compact_householder", "givens"

    返回:
        x: 最小二乘解 (n)
        residual: 残差 ||Ax - b||_2
    """
    # 选择QR分解方法
    if method == "householder":
        Q, R = householder_qr(A)
    elif method == "compact_householder":
        Q, R = compact_householder_qr(A)
    elif method == "givens":
        Q, R = givens_qr(A)
    else:
        raise ValueError("不支持的QR分解方法")

    # 计算 Q^T * b
    y = Q.T @ b

    # 求解上三角系统 Rx = y (仅前 n 行)
    n = A.shape[1]
    R_square = R[:n, :n]
    y_square = y[:n]

    # 回代法求解上三角系统
    x = back_substitution(R_square, y_square)

    # 计算残差
    residual = np.linalg.norm(A @ x - b)

    return x, residual


def normal_equations_solve(A, b):
    """
    使用正规方程 (A^T A)x = A^T b 求解最小二乘问题

    参数:
        A: 系数矩阵 (m x n)
        b: 右侧向量 (m)

    返回:
        x: 最小二乘解 (n)
        residual: 残差 ||Ax - b||_2
    """
    ATA = A.T @ A
    ATb = A.T @ b

    # 求解正规方程
    x = np.linalg.solve(ATA, ATb)

    # 计算残差
    residual = np.linalg.norm(A @ x - b)

    return x, residual
