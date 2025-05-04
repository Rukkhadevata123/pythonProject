import numpy as np


def forward_substitution(L, b):
    """
    前代法求解下三角系统 Ly = b

    参数:
        L: 下三角矩阵 (n x n)
        b: 右侧向量 (n x 1)

    返回:
        y: 解向量 (n x 1)

    算法:
        按顺序计算:
        y[0] = b[0]/L[0,0]
        y[1] = (b[1]-L[1,0]*y[0])/L[1,1]
        ...
    """
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        y[i] /= L[i, i]

    return y


def back_substitution(U, b):
    """
    回代法求解上三角系统 Ux = b

    参数:
        U: 上三角矩阵 (n x n)
        b: 右侧向量 (n x 1)

    返回:
        x: 解向量 (n x 1)

    算法:
        逆序计算:
        x[n-1] = b[n-1]/U[n-1,n-1]
        x[n-2] = (b[n-2]-U[n-2,n-1]*x[n-1])/U[n-2,n-2]
        ...
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(U[i, i + 1 :], x[i + 1 :])
        x[i] /= U[i, i]

    return x


def test_solvers():
    """测试前代和回代法"""
    # 测试前代法
    L = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
    b = np.array([1, 8, 32])
    y = forward_substitution(L, b)
    print(f"前代法测试: y = {y} (期望: [1. 2. 3.])")

    # 测试回代法
    U = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
    b = np.array([14, 23, 18])
    x = back_substitution(U, b)
    print(f"回代法测试: x = {x} (期望: [1. 2. 3.])")


if __name__ == "__main__":
    test_solvers()
