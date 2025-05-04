"""
Householder 变换方法
=================

理论基础:
--------
Householder变换是一种形如 H = I - beta * v * v^T 的正交变换，
可以将向量x的第二个及之后的所有元素变为零，即计算 Hx = alpha * e_1。

数学原理:
--------
对于向量x，我们寻找一个单位向量v，使得 H = I - 2vv^T 将x变换为
与x长度相同但仅第一个元素非零的向量。

基本步骤伪代码:
--------------
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

应用场景:
--------
1. QR分解: 将矩阵A转化为上三角形式
2. 线性最小二乘问题: 将增广矩阵变换为上三角形式
3. 特征值计算: 将矩阵约化为Hessenberg形式
4. SVD分解: 双边约化为双对角形式
"""

import numpy as np


def householder(x):
    """
    计算向量x的Householder变换反射向量v和系数beta

    参数:
        x: 输入向量

    返回:
        v: Householder反射向量
        beta: 反射系数，满足 H = I - beta * v * v^T

    数学原理:
        选择v使得Hx只在第一个分量非零，即 Hx = alpha * e_1
        其中 H = I - beta * v * v^T 是正交矩阵，e_1是第一个单位向量

    数值稳定性考虑:
        - 采用缩放避免上溢/下溢
        - 选择v[0]符号时考虑避免取消误差
        - 通过特殊归一化方式提高精度
    """
    n = len(x)

    # 对x进行缩放，避免数值上溢或下溢
    ita = np.max(np.abs(x))
    if ita > 0:
        x = x / ita  # 归一化向量，不改变方向

    # 初始化反射向量v
    v = np.zeros(n)

    # 计算x的第2到n个元素的平方和
    sigma = np.sum(x[1:n] ** 2)

    # 将v的第2到n个元素设为x的对应元素
    v[1:n] = x[1:n]

    if sigma == 0:
        # 如果x的第2到n个元素已经都是0，则无需变换
        beta = 0
    else:
        # 计算x的2-范数，不直接使用norm函数以提高精度
        alpha = np.sqrt(x[0] ** 2 + sigma)

        # 选择v[0]的值以减小舍入误差
        # 当x[0]<=0时，直接使用 x[0]-alpha 避免两大数相减
        if x[0] <= 0:
            v[0] = x[0] - alpha
        else:
            # 当x[0]>0时，使用代数等价形式避免两大数相减
            v[0] = -sigma / (x[0] + alpha)

        # 计算beta值，beta = 2/v^T·v 确保H是正交矩阵
        beta = 2 * v[0] ** 2 / (sigma + v[0] ** 2)

        # 归一化反射向量，使v[0]=1以简化后续计算
        v = v / v[0]

    return v, beta


def householder_matrix(v, beta):
    """
    根据反射向量v和系数beta构造Householder变换矩阵

    参数:
        v: Householder反射向量
        beta: 反射系数

    返回:
        H: Householder变换矩阵 H = I - beta * v * v^T

    说明:
        - H是正交矩阵，满足H^T = H^(-1)
        - 应用Hx会使x主要投影到第一个坐标轴上

    应用建议:
        通常不显式计算H矩阵，而是通过v和beta直接应用变换
        以节省计算量和存储空间(尤其对于大型矩阵)
    """
    # 计算变换矩阵，直接带入公式
    n = len(v)
    H = np.eye(n) - beta * np.outer(v, v)
    return H


def apply_householder(x, v, beta):
    """
    应用Householder变换到向量x，不显式构造H矩阵

    参数:
        x: 输入向量
        v: Householder反射向量
        beta: 反射系数

    返回:
        y: 变换后的向量 y = Hx

    计算优化:
        使用公式 Hx = x - beta * v * (v^T * x) 直接计算，
        避免构造完整的H矩阵，将计算复杂度从O(n^2)降低到O(n)
    """
    # 计算 v^T * x
    v_dot_x = np.dot(v, x)

    # 应用Householder变换: y = x - beta * v * (v^T * x)
    y = x - beta * v * v_dot_x

    return y


def qr_householder(A):
    """
    使用Householder变换计算矩阵A的QR分解

    参数:
        A: 输入矩阵 (m x n)

    返回:
        Q: 正交矩阵 (m x m)
        R: 上三角矩阵 (m x n)

    算法步骤:
        1. 逐列应用Householder变换，使A变为上三角形式
        2. 累计变换矩阵得到Q

    特点:
        - 数值稳定性好
        - 适用于密集矩阵
        - 计算量为O(mn^2)，优于Givens旋转的O(mn^2)
    """
    m, n = A.shape
    k = min(m, n)  # 变换次数

    # 拷贝A作为R
    R = A.copy()

    # 初始化Q为单位矩阵
    Q = np.eye(m)

    # 存储Householder向量和系数
    vs = []
    betas = []

    # 逐列应用Householder变换
    for j in range(k):
        # 获取当前列从对角线元素开始的部分
        x = R[j:, j].copy()

        if len(x) > 1:  # 只有当有多个元素时才需要变换
            # 计算Householder变换
            v, beta = householder(x)
            vs.append(v)
            betas.append(beta)

            # 应用变换到R的剩余部分
            for i in range(j, n):
                x = R[j:, i]
                R[j:, i] = apply_householder(x, v, beta)

            # 累积变换到Q
            for i in range(m):
                x = Q[i, j:]
                Q[i, j:] = apply_householder(x, v, beta)

    # Q需要转置以满足A=QR
    return Q.T, R
