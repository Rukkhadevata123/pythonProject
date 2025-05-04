"""
盲人爬山法估计矩阵条件数
========================

理论基础:
---------
盲人爬山法(Blind Mountaineering Algorithm)用于估计矩阵的1范数，而无需显式计算矩阵的逆。

关键思想:
---------
1. 要估计B=A^(-T)的1范数，即最大化问题:
   max ||Bx||_1 subject to ||x||_1 <= 1

2. 令f(x) = ||Bx||_1，D = {x: ||x||_1 <= 1}
   * f(x)在D上的最大值对应矩阵B的1范数
   * f(x)是凸函数，D是凸集，最大值在D的顶点处取得

算法框架:
---------
1. 初始化: 选择t个单位1范数的向量x_j
2. 迭代过程:
   a. 计算w = Bx
   b. 计算v = sign(w)
   c. 计算z = B^T v
   d. 若||z||_∞ <= z^T x，则已找到最大值，退出
   e. 否则，更新x为z最大分量对应的单位向量e_j，继续迭代

实现技巧:
---------
由于我们要估计A^(-T)的1范数，直接计算矩阵逆效率低下且不稳定，所以:
* 计算w = Bx 等价于求解线性方程组 A^T w = x
* 计算z = B^T v 等价于求解线性方程组 A z = v

条件数计算:
---------
矩阵A的条件数 K(A) = ||A||_∞ * ||A^(-1)||_∞
其中||A^(-1)||_∞ = ||A^(-T)||_1，后者可以用盲人爬山法估计
"""

import numpy as np
from lu_decomposition import solve_with_partial_pivoting
from substitution import forward_substitution, back_substitution


def sign(x):
    """
    计算向量元素的符号

    参数:
        x: 输入向量

    返回:
        与x同形的向量，元素为对应元素的符号(1, 0, 或-1)

    数学表示:
        sign(x_i) = { 1  if x_i > 0
                    { 0  if x_i = 0
                    { -1 if x_i < 0
    """
    return np.sign(x)


def norm1(x):
    """
    计算向量的1-范数(各元素绝对值之和)

    参数:
        x: 输入向量

    返回:
        ||x||_1 = 累加|x_i|

    应用:
        在盲人爬山法中用于归一化向量和评估进展
    """
    return np.sum(np.abs(x))


def normInf(x):
    """
    计算向量的无穷范数(元素绝对值的最大值)

    参数:
        x: 输入向量

    返回:
        ||x||_∞ = max|x_i|

    应用:
        在盲人爬山法中用于终止条件判断和选择新的迭代向量
    """
    return np.max(np.abs(x))


def norm1_matrix(A):
    """
    计算矩阵的1-范数(列和范数)

    参数:
        A: 输入矩阵

    返回:
        ||A||_1 = max_j(累加|a_ij|)，即各列绝对值之和的最大值

    数学等价:
        ||A||_1 = max_{||x||_1=1} ||Ax||_1

    应用:
        在条件数计算中作为矩阵大小的度量
    """
    return np.max(np.sum(np.abs(A), axis=0))


def normInf_matrix(A):
    """
    计算矩阵的无穷范数(行和范数)

    参数:
        A: 输入矩阵

    返回:
        ||A||_∞ = max_i(累加|a_ij|)，即各行绝对值之和的最大值

    数学等价:
        ||A||_∞ = max_{||x||_∞=1} ||Ax||_∞

    应用:
        在条件数计算中作为矩阵A的度量
    """
    return np.max(np.sum(np.abs(A), axis=1))


def estimate_norm1(A, t=2, max_iter=5):
    """
    使用盲人爬山法估计A^(-T)的1-范数，即A^(-1)的无穷范数

    算法原理:
    ---------
    我们要估计||A^(-T)||_1，利用等式变换简化计算:
    - 设B = A^(-T)
    - 对于操作w = Bx，等价于求解A^T·w = x
    - 对于操作z = B^T·v，等价于求解A·z = v

    参数:
    ------
    A -- 输入矩阵 (n x n)
    t -- 使用的随机向量数量，默认为2
         增加t可能提高估计精度，但会增加计算成本
    max_iter -- 最大迭代次数，默认为5
         限制算法的计算时间，通常5次迭代足够得到良好估计

    返回:
    ------
    est -- A^(-T)的1-范数的估计值，即A^(-1)的无穷范数
          如果矩阵奇异或计算过程中出错，返回无穷大

    算法效率:
    ---------
    - 时间复杂度: O(n^3 * max_iter)，主要成本在求解线性方程组
    - 空间复杂度: O(n^2)，存储矩阵A和中间结果

    注意事项:
    ---------
    - 该方法是一种估计，结果可能略小于实际1-范数
    - 如果矩阵接近奇异，结果可能不可靠
    """
    n = A.shape[0]

    # 初始化t个随机单位向量
    X = np.zeros((n, t))
    for j in range(t):
        if j == 0:
            # 第一个向量设为全1/n，确保能给出一个合理的下界估计
            # 这是因为对于某些特殊矩阵，随机向量可能会低估范数
            X[:, j] = 1.0 / n
        else:
            # 随机生成-1,1向量，覆盖不同的搜索方向
            X[:, j] = np.random.choice([-1.0, 1.0], size=n)
            X[:, j] /= norm1(X[:, j])  # 归一化确保||x||_1 = 1

    k = 0  # 迭代计数器
    est = 0.0  # 当前范数估计值

    while k < max_iter:
        k += 1
        # 对每个向量独立进行盲人爬山迭代
        for j in range(t):
            # 步骤1: 计算 w = A^(-T)·x，通过求解 A^T·w = x
            try:
                # 使用列主元消去法解线性方程组以提高数值稳定性
                w = solve_with_partial_pivoting(A.T, X[:, j])
            except Exception:
                # 如果矩阵接近奇异或求解失败，返回无穷大表示无法估计
                return float("inf")

            # 步骤2: 计算当前向量对应的范数值并更新估计
            g = norm1(w)  # g = ||A^(-T)·x||_1
            if g > est:
                est = g  # 保留最大值作为范数估计

            # 步骤3: 计算最佳上升方向
            v = sign(w)  # v是使得v^T·w = ||w||_1的向量

            # 步骤4: 计算 z = (A^(-T))^T·v = A^(-1)·v，通过求解 A·z = v
            try:
                z = solve_with_partial_pivoting(A, v)
            except Exception:
                return float("inf")

            # 步骤5: 检查是否达到局部最大值
            h = np.dot(z, X[:, j])  # h = z^T·x
            if normInf(z) <= h:
                # 若||z||_∞ <= z^T·x，则当前x已是最优向量
                break

            # 步骤6: 更新搜索向量，选择z最大分量对应的单位向量
            ind = np.argmax(np.abs(z))
            e = np.zeros(n)
            e[ind] = 1.0 if z[ind] > 0 else -1.0  # 单位向量，方向与z[ind]一致
            X[:, j] = e  # 更新搜索向量

    return est


def condition_number(A):
    """
    估计矩阵A的条件数

    定义:
    -----
    条件数K(A)度量了矩阵A对输入扰动的敏感程度:
    - K(A)接近1表示矩阵良态(well-conditioned)
    - K(A)很大表示矩阵病态(ill-conditioned)

    参数:
    -----
    A -- 输入矩阵 (n x n)

    返回:
    -----
    cond -- 矩阵A的条件数估计值K(A)

    计算方法:
    ---------
    K(A) = ||A||_∞ · ||A^(-1)||_∞

    利用转置关系简化计算:
    ||A^(-1)||_∞ = ||A^(-T)||_1，后者通过盲人爬山法估计

    应用:
    -----
    - 预测线性方程组Ax=b解的精度
    - 评估数值算法的稳定性
    - 当K(A)大时，解可能对b或A的扰动非常敏感
    """
    # 计算A的无穷范数
    norm_A_inf = normInf_matrix(A)

    # 估计A^(-1)的无穷范数，通过计算A^(-T)的1-范数
    norm_A_inv_inf = estimate_norm1(A)

    # 计算条件数
    return norm_A_inf * norm_A_inv_inf


def test_condition_number():
    """
    测试条件数估计功能

    示例:
    -----
    1. 希尔伯特矩阵(Hilbert matrix)：经典的病态矩阵
    2. 随机矩阵：通常条件数适中
    3. 对角矩阵：条件数等于最大对角元素与最小对角元素之比

    输出:
    -----
    各测试矩阵的条件数估计值和numpy计算的参考值
    """
    # 测试1: 希尔伯特矩阵(著名的病态矩阵)
    n = 5
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)

    print("=== 希尔伯特矩阵条件数估计 ===")
    print(f"矩阵大小: {n}x{n}")
    est_cond = condition_number(H)
    true_cond = np.linalg.cond(H, np.inf)
    print(f"估计条件数: {est_cond}")
    print(f"NumPy计算条件数: {true_cond}")
    print(f"相对误差: {abs(est_cond-true_cond)/true_cond:.4e}")

    # 测试2: 随机矩阵
    A = np.random.randn(n, n)
    print("\n=== 随机矩阵条件数估计 ===")
    print(f"矩阵大小: {n}x{n}")
    est_cond = condition_number(A)
    true_cond = np.linalg.cond(A, np.inf)
    print(f"估计条件数: {est_cond}")
    print(f"NumPy计算条件数: {true_cond}")
    print(f"相对误差: {abs(est_cond-true_cond)/true_cond:.4e}")


if __name__ == "__main__":
    test_condition_number()
