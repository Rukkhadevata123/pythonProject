"""
function: [c, s] = givens(a, b)
    if b = 0
        c = 1
        s = 0
    else
        if |b| > |a|
            tau = a/b
            s = 1/sqrt(1 + tau^2)
            c = s * tau
        else
            tau = b/a
            c = 1/sqrt(1 + tau^2)
            s = c * tau
        end
    end
"""

import numpy as np


def givens(a, b):
    """
    计算 Givens 旋转系数 c 和 s

    参数:
        a, b: 输入值

    返回:
        c, s: Givens 旋转系数，满足 [c s; -s c] 是一个正交矩阵
              并且 [c s; -s c] * [a; b] = [r; 0]，其中 r = sqrt(a^2 + b^2)
    """
    if b == 0:
        c = 1
        s = 0
    else:
        if abs(b) > abs(a):
            tau = a / b
            s = 1 / np.sqrt(1 + tau**2)
            c = s * tau
        else:
            tau = b / a
            c = 1 / np.sqrt(1 + tau**2)
            s = c * tau

    return c, s


def givens_matrix(n, i, j, c, s):
    """
    构造 Givens 旋转矩阵

    参数:
        n: 矩阵大小
        i, j: 需要旋转的行索引 (i < j)
        c, s: Givens 旋转系数

    返回:
        G: Givens 旋转矩阵，是一个正交矩阵
    """
    G = np.eye(n)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = s
    G[j, i] = -s
    return G


def givens_rotation(x, i, k):
    """
    计算 Givens 旋转系数并应用旋转到向量

    参数:
        x: 输入向量
        i, k: 需要旋转的索引 (i < k)

    返回:
        y: 旋转后的向量，y[k] = 0
        c, s: Givens 旋转系数
    """
    a = x[i]
    b = x[k]
    c, s = givens(a, b)

    y = x.copy()
    y[i] = c * a + s * b
    y[k] = -s * a + c * b
    return y, c, s
