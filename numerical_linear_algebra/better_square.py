import numpy as np
from substitution import forward_substitution, back_substitution


def is_symmetric(A, tol=1e-8):
    """检查矩阵是否为对称矩阵"""
    return np.allclose(A, A.T, rtol=tol)


def ldl_decomposition(A):
    """
    对对称矩阵A进行LDLT分解，返回单位下三角矩阵L和对角矩阵D
    使得 A = L*D*L^T

    参数:
    A: 输入矩阵 (n x n)，必须是对称矩阵

    返回:
    L: 单位下三角矩阵（对角线元素均为1）
    D: 对角矩阵，但这里只需要存储对角线元素，所以返回一个一维数组
    """
    # 检查矩阵是否为对称矩阵
    if not is_symmetric(A):
        raise ValueError("输入矩阵不是对称矩阵")

    n = len(A)
    L = np.eye(n)  # 初始化为单位矩阵
    D = np.zeros(n)  # 只存储对角线元素

    for j in range(n):
        # 计算D的对角线元素
        D[j] = A[j, j]
        for k in range(j):
            D[j] -= L[j, k] ** 2 * D[k]

        # 计算L的第j列元素
        for i in range(j + 1, n):
            L[i, j] = A[i, j]
            for k in range(j):
                L[i, j] -= L[i, k] * L[j, k] * D[k]
            L[i, j] /= D[j]

    return L, D


def solve_with_ldl(A, b):
    # 1. LDLT分解
    L, d = ldl_decomposition(A)

    # 2. 前代法求解 Ly = b
    y = forward_substitution(L, b)

    # 3. 对角系统求解 Dz = y
    z = np.zeros_like(y)
    for i in range(len(y)):
        z[i] = y[i] / d[i]

    # 4. 回代法求解 L^T x = z
    x = back_substitution(L.T, z)

    return x


def test_ldl():
    """测试LDLT分解与求解"""
    # 创建一个对称矩阵
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)

    b = np.array([1, 2, 3], dtype=float)

    print("=== LDLT分解测试 ===")
    L, d = ldl_decomposition(A)
    D = np.diag(d)  # 转换为对角矩阵便于显示

    print("原矩阵A:")
    print(A)
    print("\nLDLT分解得到的L:")
    print(L)
    print("\nLDLT分解得到的D (对角矩阵):")
    print(D)
    print("\n验证 L*D*L^T == A:")
    print(np.allclose(L @ D @ L.T, A))

    print("\n=== 使用LDLT分解求解线性方程组 ===")
    x = solve_with_ldl(A, b)
    print(f"解向量 x = {x}")
    print(f"残差 ||Ax-b|| = {np.linalg.norm(A @ x - b)}")

    # 与Cholesky分解结果比较
    from cholesky import solve_with_cholesky

    x_cholesky = solve_with_cholesky(A, b)
    print("\n=== 与Cholesky分解结果比较 ===")
    print(f"LDLT解:      {x}")
    print(f"Cholesky解:  {x_cholesky}")
    print(f"二者差异:    {np.linalg.norm(x - x_cholesky)}")


if __name__ == "__main__":
    test_ldl()
