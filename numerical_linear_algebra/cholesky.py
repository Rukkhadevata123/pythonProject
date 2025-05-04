import numpy as np
from substitution import forward_substitution, back_substitution


def is_symmetric(A, tol=1e-8):
    """
    检查矩阵是否为对称矩阵

    参数:
        A: 待检查的矩阵
        tol: 容差阈值，用于浮点数比较

    返回:
        bool: 如果矩阵对称返回True，否则返回False

    说明:
        对称矩阵满足A = A^T，考虑浮点误差进行判断
    """
    return np.allclose(A, A.T, rtol=tol)


def is_positive_definite(A, tol=1e-8):
    """
    检查矩阵是否为正定矩阵

    参数:
        A: 待检查的矩阵
        tol: 正定性判断的阈值

    返回:
        bool: 如果所有特征值大于tol则返回True

    说明:
        - 正定矩阵的所有特征值必须严格大于0
        - 对于对称矩阵，正定性保证了Cholesky分解的可行性
    """
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > tol)


def cholesky_decomposition(A):
    """
    对正定对称矩阵A进行Cholesky分解，返回下三角矩阵L，使得A = L*L^T

    参数:
        A: 输入矩阵，必须是对称正定矩阵

    返回:
        L: 下三角矩阵，满足A = L*L^T

    算法步骤:
        对于矩阵元素L[i,j]，有：
        1. 当i=j时（对角线元素）：L[i,i] = sqrt(A[i,i] - sum(L[i,k]^2))，其中k从0到i-1
        2. 当i>j时（下三角元素）：L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k]))/L[j,j]，其中k从0到j-1

    复杂度:
        - 时间复杂度: O(n^3)
        - 空间复杂度: O(n^2)

    优势:
        - 比LU分解更高效，当处理对称正定矩阵时
        - 数值稳定性好，不需要进行行交换
    """
    # 检查矩阵是否为正定对称矩阵
    if not is_symmetric(A):
        raise ValueError("输入矩阵不是对称矩阵")
    if not is_positive_definite(A):
        raise ValueError("输入矩阵不是正定矩阵")

    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:  # 对角线元素
                s = sum(L[i, k] ** 2 for k in range(j))
                # 计算L[i,i]，使用平方根
                L[i, j] = np.sqrt(A[i, i] - s)
            else:  # 非对角线元素(下三角)
                s = sum(L[i, k] * L[j, k] for k in range(j))
                # 计算L[i,j]，需要除以对角线元素L[j,j]
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L


def solve_with_cholesky(A, b):
    """
    使用Cholesky分解求解线性方程组 Ax = b

    参数:
        A: 系数矩阵，必须是对称正定矩阵
        b: 右侧向量

    返回:
        x: 方程组的解

    求解步骤:
        1. 对A进行Cholesky分解，得到下三角矩阵L
        2. 将原方程Ax = b转化为LL^Tx = b
        3. 分两步求解:
           a. 前代法求解Ly = b，得到中间向量y
           b. 回代法求解L^Tx = y，得到最终解x

    优势:
        - 计算量约为LU分解的一半
        - 不需要选主元，数值稳定性较好
    """
    # 1.Cholesky分解 A = L*L^T
    L = cholesky_decomposition(A)

    # 2.前代法求解 Ly = b
    y = forward_substitution(L, b)

    # 3.回代法求解 L^T x = y
    x = back_substitution(L.T, y)

    return x


def test_cholesky():
    """
    测试Cholesky分解与求解线性方程组

    示例:
        - 使用3x3的正定对称矩阵演示分解过程
        - 用分解结果解线性方程组，验证精度

    验证项:
        1. 检查L*L^T是否等于原矩阵A
        2. 检查解向量x的残差||Ax-b||
    """
    # 创建一个正定对称矩阵
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)

    b = np.array([1, 2, 3], dtype=float)

    print("=== Cholesky分解测试 ===")
    L = cholesky_decomposition(A)
    print("原矩阵A:")
    print(A)
    print("\nCholesky分解得到的L:")
    print(L)
    print("\n验证 L*L^T == A:")
    print(np.allclose(L @ L.T, A))  # 验证分解结果是否正确

    print("\n=== 使用Cholesky分解求解线性方程组 ===")
    x = solve_with_cholesky(A, b)
    print(f"解向量 x = {x}")
    # 计算残差向量范数，验证解的精确度
    print(f"残差 ||Ax-b|| = {np.linalg.norm(A @ x - b)}")


if __name__ == "__main__":
    test_cholesky()
