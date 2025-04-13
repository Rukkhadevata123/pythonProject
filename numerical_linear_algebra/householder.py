"""
function [v, beta] = householder(x)
    n = length(x)
    ita = ||x||_inf; x = x / ita
    sigma = x(2:n)' * x(2:n) % 剩余部分的平方和
    v(2:n) = x(2:n) % 节省空间
    if sigma == 0
        beta = 0
    else
        alpha = sqrt(x(1)^2 + sigma) % 计算2-norm
        if x(1) <= 0
            v(1) = x(1) - alpha
        else
            v(1) = -sigma / (x(1) + alpha)
        end
        beta = 2 * v(1)^2 / (sigma + v(1)^2) % beta = 2 / (v' * v)
        v = v / v(1) % 归一化
    end
"""

import numpy as np


def householder(x):
    # 计算 Householder 变换的反射向量 v 和系数 beta
    # v: Householder 反射向量 beta: Householder 反射系数，满足 H = I - beta * v * v^T

    n = len(x)
    ita = np.max(np.abs(x))
    if ita > 0:
        x = x / ita

    v = np.zeros(n)
    sigma = np.sum(x[1:n] ** 2)
    v[1:n] = x[1:n]
    if sigma == 0:
        beta = 0
    else:
        alpha = np.sqrt(x[0] ** 2 + sigma)

        if x[0] <= 0:
            v[0] = x[0] - alpha
        else:
            v[0] = -sigma / (x[0] + alpha)

        # 提前归一化
        beta = 2 * v[0] ** 2 / (sigma + v[0] ** 2)
        v = v / v[0]

    return v, beta


def householder_matrix(v, beta):
    # 计算变换矩阵，直接带入公式
    n = len(v)
    H = np.eye(n) - beta * np.outer(v, v)
    return H
