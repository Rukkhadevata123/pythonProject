"""
Givens 旋转方法
=============

理论基础:
--------
Givens 旋转是一种将矩阵中特定元素置零的正交变换技术。通过构造适当的
旋转矩阵 G，可以使得 G*x 中的特定元素变为零，同时保持向量范数不变。

数学原理:
--------
对于向量中的两个元素a和b，Givens旋转矩阵形式为:
G = [  c  s ]
    [ -s  c ]
其中 c^2 + s^2 = 1 (确保正交性)，且 G*[a; b] = [r; 0]，r = sqrt(a^2 + b^2)

计算过程:
--------
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

应用场景:
--------
1. QR分解: 逐步消除矩阵下三角的元素
2. 求解线性最小二乘问题
3. 对角化三对角矩阵(如SVD和特征值计算)
4. 稀疏矩阵计算，因为仅影响两行或两列
"""

import numpy as np


def givens(a, b):
    """
    计算Givens旋转系数c和s

    参数:
        a, b: 需要进行旋转的两个元素

    返回:
        c, s: 旋转系数，满足 [c s; -s c] * [a; b] = [r; 0]，r = sqrt(a^2 + b^2)

    算法原理:
        通过巧妙的数学变换，计算满足正交条件(c^2 + s^2 = 1)的旋转系数，
        使得经过旋转后第二个元素为0，同时避免数值溢出和精度损失

    数值稳定性考虑:
        - 当b为0时，无需旋转，直接返回c=1,s=0
        - 根据|a|和|b|的相对大小选择不同计算公式，确保数值稳定性
        - 避免直接计算平方根sqrt(a^2 + b^2)，减少计算误差
    """
    # c, s是Givens旋转系数，满足 [c s; -s c] 是一个正交矩阵
    # 并且 [c s; -s c] * [a; b] = [r; 0]，其中 r = sqrt(a^2 + b^2)
    if b == 0:
        # 如果b已经为0，无需旋转
        c = 1
        s = 0
    else:
        if abs(b) > abs(a):
            # 当|b| > |a|时，使用这种方式计算避免除以接近0的数
            tau = a / b
            s = 1 / np.sqrt(1 + tau**2)
            c = s * tau
        else:
            # 当|a| >= |b|时，使用这种方式更稳定
            tau = b / a
            c = 1 / np.sqrt(1 + tau**2)
            s = c * tau

    return c, s


def givens_matrix(n, i, j, c, s):
    """
    构造n维Givens旋转矩阵

    参数:
        n: 矩阵维度
        i, j: 要旋转的行/列索引(i < j)
        c, s: 旋转系数

    返回:
        G: n×n的Givens旋转矩阵，除了位置(i,i),(j,j),(i,j),(j,i)外都是单位矩阵

    矩阵结构:
        G 是单位矩阵在对应位置做如下修改:
        G[i,i] = c    G[i,j] = s
        G[j,i] = -s   G[j,j] = c

    应用:
        将矩阵G左乘到矩阵A上，可以使A的第j行第i列元素变为0
        G*A 会影响A的第i和j行
        A*G^T 会影响A的第i和j列
    """
    # 构造旋转矩阵 G
    G = np.eye(n)  # 以单位矩阵为基础
    G[i, i] = c  # 修改四个元素以形成旋转子矩阵
    G[j, j] = c
    G[i, j] = s
    G[j, i] = -s  # 注意这里是-s，确保G是正交矩阵
    return G


def givens_rotation(x, i, k):
    """
    对向量x的第i和k个元素应用Givens旋转

    参数:
        x: 输入向量
        i, k: 要旋转的元素索引(i < k)

    返回:
        y: 旋转后的向量，其中y[k] = 0
        c, s: 使用的旋转系数

    功能:
        对向量x的第i和k个元素应用旋转，使得第k个元素变为0

    优势:
        - 计算量小，每次只需O(1)的操作量计算旋转系数
        - 仅修改两个元素，适合处理稀疏结构
        - 数值稳定性好，适合各种条件下的应用

    应用示例:
        在QR分解中，逐个消除向量在特定位置的元素
    """
    # 对向量 x 的第 i 和 k 个元素进行 Givens 旋转
    a = x[i]  # 提取要旋转的两个元素
    b = x[k]
    c, s = givens(a, b)  # 计算旋转系数

    y = x.copy()  # 避免修改原向量
    # 应用旋转变换，只修改两个元素
    y[i] = c * a + s * b  # 新的第i个元素
    y[k] = -s * a + c * b  # 新的第k个元素，理论上应为0
    return y, c, s


def apply_givens_rotation(A, i, j, c, s, left=True):
    """
    对矩阵A应用Givens旋转

    参数:
        A: 输入矩阵
        i, j: 要旋转的行或列索引(i < j)
        c, s: 旋转系数
        left: 如果为True，左乘旋转矩阵(影响行)；否则右乘旋转矩阵(影响列)

    返回:
        B: 旋转后的矩阵

    操作说明:
        - left=True时: B = G * A，影响A的第i和j行
        - left=False时: B = A * G^T，影响A的第i和j列

    实现效率:
        不显式构造完整的旋转矩阵，只对受影响的行/列进行计算，
        时间复杂度从O(n^2)降低到O(n)
    """
    B = A.copy()  # 避免修改原矩阵

    if left:  # 左乘旋转矩阵，影响行
        for k in range(A.shape[1]):
            temp = c * A[i, k] + s * A[j, k]
            B[j, k] = -s * A[i, k] + c * A[j, k]
            B[i, k] = temp
    else:  # 右乘旋转矩阵的转置，影响列
        for k in range(A.shape[0]):
            temp = c * A[k, i] + s * A[k, j]
            B[k, j] = -s * A[k, i] + c * A[k, j]
            B[k, i] = temp

    return B
