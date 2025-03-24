import numpy as np


def forward_substitution(L, b):
    """
    使用前代法求解下三角矩阵的线性方程组 Lx = b。

    参数:
    L -- 下三角矩阵 (n x n)
    b -- 右侧向量 (n x 1)

    返回:
    x -- 解向量 (n x 1)
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x


def test_forward_substitution():
    L = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
    b = np.array([1, 8, 32])
    x = forward_substitution(L, b)
    print(f'Expected: [1. 2. 3.], Got: {x}')


def back_substitution(U, b):
    """
    使用回代法求解上三角矩阵的线性方程组 Ux = b。

    参数:
    U -- 上三角矩阵 (n x n)
    b -- 右侧向量 (n x 1)

    返回:
    x -- 解向量 (n x 1)
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x


def test_back_substitution():
    U = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
    b = np.array([14, 23, 18])
    x = back_substitution(U, b)
    print(f'Expected: [1. 2. 3.], Got: {x}')


if __name__ == '__main__':
    test_forward_substitution()
    test_back_substitution()
