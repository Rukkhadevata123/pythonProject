"""
盲人爬山法
估计矩阵的1范数
B=nxn矩阵
估计B的1范数
f(x)=||Bx||_1
D={x:||x||_1<=1}
f(x)在D上的最大值
f(x)是凸函数
D是凸集
伪代码
while True:
    w=Bx;v=sign(w);z=B^Tv
    if ||z||_inf<=z^Tx:
        find max
        break
    else:
        x = e_j;其中 ||z_j||=||z||_inf
        continue
||A^-1||_inf=||A^-T||_1
B=A^-T
w=Bx <=> A^Tw=x
z=B^Tv <=> Az=v
条件数 K(A)=||A||_inf*||A^-1||_inf, 后者可以用盲人爬山法估计
"""

import numpy as np
from lu_decomposition import solve_with_partial_pivoting
from substitution import forward_substitution, back_substitution


def sign(x):
    """计算向量元素的符号"""
    return np.sign(x)


def norm1(x):
    """计算向量的1-范数"""
    return np.sum(np.abs(x))


def normInf(x):
    """计算向量的无穷范数"""
    return np.max(np.abs(x))


def norm1_matrix(A):
    """计算矩阵的1-范数（列和范数）"""
    return np.max(np.sum(np.abs(A), axis=0))


def normInf_matrix(A):
    """计算矩阵的无穷范数（行和范数）"""
    return np.max(np.sum(np.abs(A), axis=1))


def estimate_norm1(A, t=2, max_iter=5):
    """
    使用盲人爬山法估计A^(-T)的1-范数，即A^(-1)的无穷范数

    原理：
    我们要估计||A^(-T)||_1，根据书本中的等式变换：
    - 设B = A^(-T)
    - 对于操作w = Bx，等价于求解A^T·w = x
    - 对于操作z = B^T·v，等价于求解A·z = v

    参数:
    A -- 输入矩阵 (n x n)
    t -- 使用的随机向量数量
    max_iter -- 最大迭代次数

    返回:
    est -- A^(-T)的1-范数的估计值，即A^(-1)的无穷范数
    """
    n = A.shape[0]

    # 初始化t个随机单位向量，这里第一个向量按书上的提示设置为全1/n,另一个向量随机生成
    X = np.zeros((n, t))
    for j in range(t):
        if j == 0:
            # 第一个向量全1，确保不低估
            X[:, j] = 1.0 / n
        else:
            # 随机生成-1,1向量
            X[:, j] = np.random.choice([-1.0, 1.0], size=n)
            X[:, j] /= norm1(X[:, j])  # 归一化

    k = 0
    est = 0.0

    while k < max_iter:
        k += 1
        # 对每个向量计算，如果只有一个向量，不需要下面的循环
        for j in range(t):
            # 计算 w = A^(-T)·x，通过求解 A^T·w = x
            try:
                # 我们使用列主元消去法解 A^T·w = x，这里导入前几周的代码
                w = solve_with_partial_pivoting(A.T, X[:, j])
            except Exception:
                # 如果矩阵奇异或求解失败，返回无穷大
                return float("inf")

            # 计算 ||A^(-T)·x||_1
            g = norm1(w)
            if g > est:
                est = g
            v = sign(w)

            # 计算 z = (A^(-T))^T·v = A^(-1)·v，通过求解 A·z = v
            try:
                # 使用列主元消去法解 A·z = v，同样使用前几周的代码
                z = solve_with_partial_pivoting(A, v)
            except Exception:
                return float("inf")

            # 检查终止条件
            h = np.dot(z, X[:, j])
            if normInf(z) <= h:
                # 已找到最大值
                break

            # 更新x为z的最大分量对应的单位向量
            ind = np.argmax(np.abs(z))
            e = np.zeros(n)
            e[ind] = 1.0 if z[ind] > 0 else -1.0
            X[:, j] = e

    return est


def condition_number(A):
    # 估计矩阵A的条件数
    # 计算公式：K(A) = ||A||_∞ · ||A^(-1)||_∞
    # 根据转置关系：||A^(-1)||_∞ = ||A^(-T)||_1

    # 计算A的无穷范数
    norm_A_inf = normInf_matrix(A)

    # 估计A^(-1)的无穷范数，通过计算A^(-T)的1-范数
    norm_A_inv_inf = estimate_norm1(A)

    # 计算条件数
    return norm_A_inf * norm_A_inv_inf
