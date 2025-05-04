"""
迭代法求解线性方程组
==================

本模块实现了三种经典迭代方法求解线性方程组 Ax = b：
1. Jacobi迭代法
2. Gauss-Seidel迭代法
3. 连续超松弛(SOR)迭代法

理论基础:
--------
对于线性方程组 Ax = b，我们将矩阵A分解为：
A = D - L - U
其中:
- D 是A的对角线元素组成的对角矩阵
- L 是A的严格下三角部分(乘以-1)
- U 是A的严格上三角部分(乘以-1)

各算法迭代公式:
-------------
1. Jacobi迭代:
   x(k+1) = D^(-1)(L + U)x(k) + D^(-1)b
   * 使用上一步迭代的全部结果计算新解

2. Gauss-Seidel迭代:
   x(k+1) = (D - L)^(-1)Ux(k) + (D - L)^(-1)b
   * 利用最新计算出的结果来更新后续分量
   * 理论上：(D - L)x(k+1) = Ux(k) + b

3. 连续超松弛(SOR)迭代:
   x(k+1) = (1-ω)x(k) + ω(D-ωL)^(-1)[(1-ω)D + ωU]x(k) + ω(D-ωL)^(-1)b
   * ω是松弛因子，控制收敛速度
   * ω=1时退化为Gauss-Seidel迭代
   * 1<ω<2时为超松弛，可加速收敛
   * 0<ω<1时为亚松弛，提高稳定性

收敛条件:
--------
1. 对角占优矩阵: 三种方法都收敛
2. 正定矩阵: Gauss-Seidel和SOR方法收敛
3. 收敛速度对比: SOR > Gauss-Seidel > Jacobi
"""

import numpy as np


def jacobi_iterator(A, b, x0=None, max_iter=100, tol=1e-8):
    """
    Jacobi迭代法求解线性方程组 Ax = b

    算法原理:
    --------
    对于方程 Ax = b 的第i个方程: ∑(a_ij * x_j) = b_i，解出 x_i:
    x_i = (b_i - ∑(a_ij * x_j, j≠i)) / a_ii

    迭代公式:
    x_i^(k+1) = (b_i - ∑(a_ij * x_j^(k), j≠i)) / a_ii

    特点:
    ----
    - 每一轮迭代使用上一轮的所有结果
    - 可并行计算，适合大规模并行处理
    - 收敛速度较慢，但实现简单

    参数:
    ----
    A: 系数矩阵 (n x n)
    b: 右侧向量 (n)
    x0: 初始猜测解 (n)，默认为全零向量
    max_iter: 最大迭代次数
    tol: 收敛容差，基于相对残差 ||Ax-b||/||b||

    返回:
    ----
    x: 近似解向量
    iterations: 实际迭代次数
    residuals: 每次迭代的相对残差列表

    收敛条件:
    --------
    矩阵A严格对角占优时保证收敛: |a_ii| > ∑(|a_ij|, j≠i)
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()
    x_new = np.zeros_like(x)
    residuals = []

    for iteration in range(max_iter):
        # 计算相对残差: ||Ax-b||/||b||
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        residuals.append(residual)

        # 收敛检查
        if residual < tol:
            return x, iteration, residuals

        # Jacobi迭代核心: 对每个方程求解x_i
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]  # 计算非对角元素的贡献
            x_new[i] = (b[i] - sigma) / A[i, i]  # 更新xi，使用上一轮的所有x值

        # 整体更新解向量，这是Jacobi方法的关键特点
        x = x_new.copy()

    # 达到最大迭代次数仍未收敛
    return x, max_iter, residuals


def gauss_seidel_iterator(A, b, x0=None, max_iter=100, tol=1e-8):
    """
    Gauss-Seidel迭代法求解线性方程组 Ax = b

    算法原理:
    --------
    类似于Jacobi方法，但在计算x_i^(k+1)时使用最新计算出的x_1^(k+1)...x_(i-1)^(k+1)

    迭代公式:
    x_i^(k+1) = (b_i - ∑(a_ij * x_j^(k+1), j<i) - ∑(a_ij * x_j^(k), j>i)) / a_ii

    特点:
    ----
    - 利用最新计算出的结果立即更新后续分量
    - 无需存储上一轮的完整结果，节省内存
    - 通常比Jacobi方法收敛更快

    参数:
    ----
    A: 系数矩阵 (n x n)
    b: 右侧向量 (n)
    x0: 初始猜测解 (n)，默认为全零向量
    max_iter: 最大迭代次数
    tol: 收敛容差，基于相对残差 ||Ax-b||/||b||

    返回:
    ----
    x: 近似解向量
    iterations: 实际迭代次数
    residuals: 每次迭代的相对残差列表

    收敛条件:
    --------
    - 矩阵A严格对角占优时保证收敛
    - 矩阵A对称正定时也保证收敛
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()
    residuals = []

    for iteration in range(max_iter):
        # 计算相对残差
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        residuals.append(residual)

        # 收敛检查
        if residual < tol:
            return x, iteration, residuals

        # Gauss-Seidel迭代核心: 使用最新的x值
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]  # 使用最新的x值计算非对角元素的贡献
            x[i] = (b[i] - sigma) / A[i, i]  # 更新x[i]并立即在后续计算中使用

    # 达到最大迭代次数仍未收敛
    return x, max_iter, residuals


def sor_iterator(A, b, omega=1.5, x0=None, max_iter=100, tol=1e-8):
    """
    连续超松弛(SOR)迭代法求解线性方程组 Ax = b

    算法原理:
    --------
    在Gauss-Seidel方法的基础上引入松弛因子ω，对解向量施加加权平均

    迭代公式:
    x_i^(k+1) = (1-ω)x_i^(k) + ω(b_i - ∑(a_ij * x_j^(k+1), j<i) - ∑(a_ij * x_j^(k), j>i)) / a_ii

    特点:
    ----
    - ω=1时退化为Gauss-Seidel方法
    - 1<ω<2时为超松弛，可加速收敛
    - 0<ω<1时为亚松弛，增加稳定性
    - 选择最优的ω可大幅提高收敛速度

    参数:
    ----
    A: 系数矩阵 (n x n)
    b: 右侧向量 (n)
    omega: 松弛因子，通常在(0,2)范围内，默认1.5
    x0: 初始猜测解 (n)，默认为全零向量
    max_iter: 最大迭代次数
    tol: 收敛容差，基于相对残差 ||Ax-b||/||b||

    返回:
    ----
    x: 近似解向量
    iterations: 实际迭代次数
    residuals: 每次迭代的相对残差列表

    收敛条件:
    --------
    - 矩阵A对称正定时，当0<ω<2时SOR方法收敛
    - 最优松弛因子ω_opt通常与问题特性有关，可通过理论计算或试验确定
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()
    residuals = []

    for iteration in range(max_iter):
        # 计算相对残差
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        residuals.append(residual)

        # 收敛检查
        if residual < tol:
            return x, iteration, residuals

        # SOR迭代核心
        for i in range(n):
            sigma = 0
            # 使用已更新的元素(下三角部分)
            for j in range(i):
                sigma += A[i, j] * x[j]
            # 使用上一轮的元素(上三角部分)
            for j in range(i + 1, n):
                sigma += A[i, j] * x[j]

            # SOR更新公式: 新值 = (1-ω)*旧值 + ω*Gauss-Seidel新值
            x_new_gs = (b[i] - sigma) / A[i, i]  # Gauss-Seidel新值
            x[i] = (1 - omega) * x[i] + omega * x_new_gs  # 应用松弛因子

    # 达到最大迭代次数仍未收敛
    return x, max_iter, residuals


def is_diagonally_dominant(A):
    """
    检查矩阵A是否严格对角占优

    定义:
    ----
    矩阵A严格对角占优，当且仅当对于每一行i:
    |a_ii| > ∑(|a_ij|, j≠i)

    重要性:
    ------
    - 对角占优是许多迭代方法收敛的充分条件
    - Jacobi和Gauss-Seidel方法对严格对角占优矩阵一定收敛

    参数:
    ----
    A: 输入矩阵

    返回:
    ----
    bool: 如果矩阵严格对角占优则返回True，否则返回False
    """
    n = A.shape[0]

    for i in range(n):
        diagonal = abs(A[i, i])  # 对角元素绝对值
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)  # 非对角元素绝对值之和

        if diagonal <= row_sum:  # 不满足严格对角占优条件
            return False

    return True


def auto_select_omega(A):
    """
    根据矩阵特性自动选择SOR方法的最优松弛因子

    算法原理:
    --------
    1. 对于三对角矩阵，使用理论最优值:
       ω_opt = 2 / (1 + √(1 - ρ²))
       其中ρ是Jacobi迭代矩阵的谱半径

    2. 对于一般矩阵，使用经验值1.5

    特殊矩阵优化:
    -----------
    - 三对角矩阵: 使用理论计算的最优ω
    - Jacobi矩阵: D^(-1)(L+U)，用于计算谱半径

    参数:
    ----
    A: 输入矩阵

    返回:
    ----
    omega: 推荐的松弛因子

    参考文献:
    --------
    数值分析教材，通常在三对角矩阵的SOR方法章节约第127页
    """
    n = A.shape[0]

    # 判断是否为三对角矩阵(主对角线及其相邻两条对角线以外的元素都接近于0)
    is_tridiagonal = True
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(A[i, j]) > 1e-10:
                is_tridiagonal = False
                break

    if is_tridiagonal:
        # 对于三对角矩阵，构造Jacobi迭代矩阵 D^(-1)(L+U)
        jacobi_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    jacobi_matrix[i, j] = -A[i, j] / A[i, i]

        # 计算Jacobi矩阵的谱半径(最大特征值的模)
        try:
            eigenvalues = np.linalg.eigvals(jacobi_matrix)
            rho = max(abs(eigenvalues))
            # 利用理论公式计算最优松弛因子
            omega_opt = 2 / (1 + np.sqrt(1 - rho**2))
            return omega_opt
        except:
            # 如果特征值计算失败(如矩阵奇异或数值不稳定)，使用默认值
            return 1.5

    # 对于一般矩阵，使用经验值
    return 1.5
