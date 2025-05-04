"""
线性系统求解误差分析与迭代求精
=============================

误差理论基础:
-----------
对于线性系统 Ax = b 的求解，如果 x_est 是计算得到的近似解，真实解为 x_real，则：

1. 前向误差界:
   ||x_real - x_est||_∞ / ||x_real||_∞ ≤ κ(A) * ||r||_∞ / ||b||_∞

   其中:
   - r = b - A*x_est 是残差向量
   - κ(A) = ||A||_∞ * ||A^(-1)||_∞ 是矩阵A的条件数
   - 这个界估计了解的相对误差上限

2. 后向误差分析:
   近似解 x_est 可视为一个稍微扰动的系统 (A+E)x_est = b 的精确解
   扰动矩阵 E 的大小反映了算法的数值稳定性

迭代求精方法:
-----------
当初始解精度不足时，可使用迭代求精(Iterative Refinement)提高精度:

1. 牛顿迭代思想:
   - 计算残差: r = b - A*x_est
   - 求解修正方程: A*z = r
   - 更新解: x_est = x_est + z
   - 重复直到收敛

2. 收敛特性:
   - 对于良态矩阵，通常几次迭代即可获得高精度解
   - 对于病态矩阵，需要更多迭代或使用混合精度计算
"""

import numpy as np
from cond import estimate_norm1, normInf_matrix, normInf


def estimate_relative_error(A, x_est, b):
    """
    估计线性系统Ax=b的近似解x_est的相对误差上界

    理论基础:
    --------
    对于近似解x_est与真实解x_real，相对误差上界为:
    ||x_real - x_est||_∞ / ||x_real||_∞ ≤ κ(A) * ||r||_∞ / ||b||_∞

    其中κ(A)是矩阵A的条件数，r = b - A*x_est是残差向量

    参数:
    ----
    A -- 系数矩阵 (n x n)
    x_est -- 估计解 (n)
    b -- 右端向量 (n)

    返回:
    ----
    est_error -- 估计的相对误差上界

    数值特性:
    --------
    - 估计值通常是实际误差的粗略上界，可能过于保守
    - 对于病态问题(条件数大)，误差估计尤其重要
    - 残差小不一定意味着误差小，这取决于矩阵的条件数
    """
    # 计算残差 r = b - A*x_est
    r = b - A @ x_est

    # 计算相对残差: ||r||_∞ / ||b||_∞
    # 这反映了解在满足方程Ax=b时的精确程度
    rel_residual = normInf(r) / normInf(b)

    # 估计条件数κ(A) = ||A||_∞ * ||A^(-1)||_∞
    # 矩阵范数直接计算，逆矩阵范数通过盲人爬山法估计
    norm_A = normInf_matrix(A)
    norm_A_inv = estimate_norm1(A)
    cond_A = norm_A * norm_A_inv

    # 估计相对误差上界
    # 这是基于扰动理论得出的经典误差界
    est_error = cond_A * rel_residual

    return est_error


def iterative_refinement(A, b, x_est, max_iter=5, tol=1e-12):
    """
    通过迭代求精(Iterative Refinement)提高线性系统解的精度

    算法原理:
    --------
    利用残差计算来不断修正近似解，类似于牛顿迭代:
    1. 计算当前残差: r = b - A*x
    2. 求解修正方程: A*z = r，得到误差向量z
    3. 更新解: x = x + z
    4. 重复以上步骤直到残差足够小

    参数:
    ----
    A -- 系数矩阵 (n x n)
    b -- 右端向量 (n)
    x_est -- 初始估计解 (n)
    max_iter -- 最大迭代次数，默认为5
    tol -- 收敛容差，默认为1e-12

    返回:
    ----
    x_refined -- 优化后的解 (n)
    iterations -- 实际迭代次数

    优势与应用:
    ----------
    - 可显著提高病态系统的解的精度
    - 能够纠正因舍入误差累积导致的精度损失
    - 在混合精度计算环境中尤其有效
    - 每次迭代的计算量与原线性系统求解相当

    注意事项:
    --------
    - 对于极度病态的问题，单纯迭代求精可能效果有限
    - 收敛速度取决于矩阵的条件数和初始解的质量
    """
    from lu_decomposition import solve_with_partial_pivoting

    # 复制初始估计解，避免修改原始数据
    x = x_est.copy()

    for k in range(max_iter):
        # 计算当前残差 r = b - A*x
        # 残差向量反映了当前解与真实解的偏差方向
        r = b - A @ x

        # 检查收敛条件
        # 如果残差范数小于设定阈值，认为已达到足够精度
        if normInf(r) < tol:
            return x, k  # 返回优化解和实际迭代次数

        # 求解修正方程 A*z = r
        # 这一步计算误差向量z，需要解一个与原问题相同的线性系统
        z = solve_with_partial_pivoting(A, r)

        # 更新解: x = x + z
        # 每次迭代都会减小解的误差，类似牛顿法的二次收敛特性
        x = x + z

        # 理论上，如果计算精确，迭代求精可以在一步内得到精确解
        # 但由于浮点运算误差，通常需要多次迭代

    # 如果达到最大迭代次数仍未收敛，返回当前解和最大迭代次数
    return x, max_iter


def test_error_estimation():
    """
    测试误差估计和迭代求精方法

    示例:
    ----
    1. 生成一个条件数已知的矩阵
    2. 构造一个精确解，并计算对应的右端向量
    3. 向精确解引入扰动，得到初始估计解
    4. 估计相对误差上界
    5. 应用迭代求精并比较改进效果

    输出:
    ----
    - 真实误差与估计误差的比较
    - 迭代求精前后的解精度比较
    - 收敛过程中残差的变化
    """
    # 生成测试矩阵和向量
    n = 5
    np.random.seed(42)

    # 构造一个有特定条件数的矩阵
    U, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0, 3, n)  # 奇异值在1到1000之间
    A = U @ np.diag(s) @ U.T

    # 生成真实解和对应的右端向量
    x_real = np.random.randn(n)
    b = A @ x_real

    # 向真实解引入扰动，模拟计算误差
    perturbation = 1e-4 * np.random.randn(n)
    x_est = x_real + perturbation

    # 计算真实相对误差
    true_error = normInf(x_real - x_est) / normInf(x_real)

    # 估计相对误差上界
    est_error = estimate_relative_error(A, x_est, b)

    print("=== 线性系统误差分析 ===")
    print(f"矩阵条件数: {np.linalg.cond(A, np.inf):.2e}")
    print(f"真实相对误差: {true_error:.6e}")
    print(f"估计误差上界: {est_error:.6e}")
    print(f"误差估计比率: {est_error/true_error:.2f}倍")

    # 应用迭代求精
    x_refined, iterations = iterative_refinement(A, b, x_est)
    refined_error = normInf(x_real - x_refined) / normInf(x_real)

    print("\n=== 迭代求精效果 ===")
    print(f"迭代次数: {iterations}")
    print(f"优化前误差: {true_error:.6e}")
    print(f"优化后误差: {refined_error:.6e}")
    print(f"误差改善比例: {true_error/refined_error:.2f}倍")


if __name__ == "__main__":
    test_error_estimation()
