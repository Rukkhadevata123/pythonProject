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
    # c, s是Givens 旋转系数，满足 [c s; -s c] 是一个正交矩阵
    # 并且 [c s; -s c] * [a; b] = [r; 0]，其中 r = sqrt(a^2 + b^2)
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
    # 构造旋转矩阵 G
    G = np.eye(n)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = s
    G[j, i] = -s
    return G


def givens_rotation(x, i, k):
    # 对向量 x 的第 i 和 k 个元素进行 Givens 旋转
    a = x[i]
    b = x[k]
    c, s = givens(a, b)

    y = x.copy()
    y[i] = c * a + s * b
    y[k] = -s * a + c * b
    return y, c, s
