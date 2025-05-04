import numpy as np


def complex_power_method(A, tol=1e-10, max_iter=1000, return_history=False):
    """
    使用幂法计算复数方阵A的模最大特征值及特征向量。

    参数:
        A: 输入方阵 (n x n)，支持复数。
        tol: 收敛容差（默认1e-10）。
        max_iter: 最大迭代次数（默认1000）。
        return_history: 是否返回特征值迭代历史（默认False）。

    返回:
        eigenvalue: 近似最大特征值（复数）。
        eigenvector: 对应的归一化特征向量（复数）。
        iterations: 实际迭代次数。
        history: 特征值迭代历史（仅当return_history=True时返回）。

    数学原理:
        1. 迭代公式: x_{k+1} = A y_k, 其中 y_k = x_k / ||x_k||。
        2. 特征值估计: λ ≈ (y_k^H A y_k) / (y_k^H y_k) (Rayleigh商)。
        3. 收敛条件: |λ_new - λ_old| < tol。
    """
    # 检查输入是否为方阵
    if A.shape[0] != A.shape[1]:
        raise ValueError("输入必须是方阵")

    n = A.shape[0]
    np.random.seed(42)  # 固定随机种子，确保结果可复现
    x = np.random.rand(n) + 1j * np.random.rand(n)  # 随机复数初始向量
    y = x / np.linalg.norm(x)  # 归一化

    lambda_history = []  # 记录特征值迭代历史
    lambda_old = 0  # 前一次迭代的特征值

    for k in range(max_iter):
        x_new = A @ y  # 计算 A y_k
        # Rayleigh商估计特征值 (y_k^H A y_k / y_k^H y_k)
        lambda_new = np.dot(y.conj(), A @ y) / np.dot(y.conj(), y)

        if return_history:
            lambda_history.append(lambda_new)

        # 检查收敛条件
        if abs(lambda_new - lambda_old) < tol:
            eigenvector = y / np.linalg.norm(y)  # 最终归一化特征向量
            if return_history:
                return lambda_new, eigenvector, k + 1, lambda_history
            else:
                return lambda_new, eigenvector, k + 1

        lambda_old = lambda_new
        y = x_new / np.linalg.norm(x_new)  # 更新归一化向量

    # 未收敛警告
    print(f"警告: 达到最大迭代次数 {max_iter}，但未达到收敛容差 {tol}")
    eigenvector = y / np.linalg.norm(y)

    if return_history:
        return lambda_old, eigenvector, max_iter, lambda_history
    else:
        return lambda_old, eigenvector, max_iter
