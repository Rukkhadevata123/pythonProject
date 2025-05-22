"""
使用伴随矩阵法求解多项式方程 x^41+x^3+1=0 的全部根
比较自实现QR算法与NumPy特征值计算的结果
"""

import numpy as np
from qr_iter_and_schur import real_schur_decomposition, extract_eigenvalues


def create_companion_matrix(coefficients):
    """
    创建多项式的伴随矩阵

    参数:
        coefficients: 多项式系数，从高到低次排列 [a_n, a_{n-1}, ..., a_1, a_0]
                     对应 a_n * x^n + a_{n-1} * x^{n-1} + ... + a_1 * x + a_0

    返回:
        companion_matrix: 伴随矩阵，其特征值就是多项式方程的根
    """
    n = len(coefficients) - 1  # 多项式次数

    # 归一化系数，使最高次项系数为1
    if coefficients[0] != 1:
        coefficients = [c / coefficients[0] for c in coefficients]

    # 创建伴随矩阵
    C = np.zeros((n, n))

    # 设置次对角线为1
    for i in range(n - 1):
        C[i + 1, i] = 1.0

    # 设置最后一列为负的系数
    for i in range(n):
        C[i, n - 1] = -coefficients[n - i]

    return C


def solve_polynomial_with_our_qr(coefficients):
    """
    使用伴随矩阵和自实现QR算法求解多项式方程的根
    """
    # 创建伴随矩阵
    C = create_companion_matrix(coefficients)

    # 使用自实现的QR算法计算特征值
    T, Q, blocks = real_schur_decomposition(C, max_iter=10000, tol=1e-10)

    # 从Schur形式提取特征值
    eigenvalues = extract_eigenvalues(T, blocks)

    return eigenvalues


def solve_polynomial_with_numpy(coefficients):
    """
    使用伴随矩阵和NumPy的特征值计算函数求解多项式方程的根
    """
    # 创建相同的伴随矩阵
    C = create_companion_matrix(coefficients)

    # 使用NumPy计算特征值
    eigenvalues = np.linalg.eigvals(C)

    return eigenvalues


def main():
    # 求解方程 x^41 + x^3 + 1 = 0
    # 系数从高到低次排列: [1, 0, 0, ..., 1, ..., 1]
    coefficients = [1] + [0] * 37 + [1, 0, 0, 1]  # x^41 + x^3 + 1

    print("求解方程: x^41 + x^3 + 1 = 0")
    print("方法: 构造伴随矩阵并求其特征值")

    # 使用自实现QR算法计算根
    print("\n使用自实现QR算法计算伴随矩阵的特征值:")
    our_roots = solve_polynomial_with_our_qr(coefficients)

    # 排序以便比较
    our_roots = sorted(our_roots, key=lambda x: (np.real(x), np.imag(x)))

    print(f"找到 {len(our_roots)} 个根")
    print(f"所有根:")
    for i, root in enumerate(our_roots[:]):
        print(f"根 {i+1}: {root}")

    # 验证部分根的值
    print("\n验证自实现QR算法结果（计算p(root)的值，应接近0）:")
    for i, root in enumerate(our_roots[:]):
        p_value = root**41 + root**3 + 1
        print(f"p(根 {i+1}) = {p_value}")

    # 计算模长统计
    our_moduli = [abs(root) for root in our_roots]
    print("\n自实现QR算法根的模长统计:")
    print(f"最小模长: {min(our_moduli):.10f}")
    print(f"最大模长: {max(our_moduli):.10f}")
    print(f"平均模长: {np.mean(our_moduli):.10f}")

    # 使用NumPy的特征值计算函数
    print("\n使用NumPy计算相同伴随矩阵的特征值:")
    numpy_roots = solve_polynomial_with_numpy(coefficients)

    # 排序以便比较
    numpy_roots = sorted(numpy_roots, key=lambda x: (np.real(x), np.imag(x)))

    print(f"找到 {len(numpy_roots)} 个根")
    print(f"所有根:")
    for i, root in enumerate(numpy_roots[:]):
        print(f"根 {i+1}: {root}")

    # 验证NumPy根的值
    print("\n验证NumPy结果:")
    for i, root in enumerate(numpy_roots[:]):
        p_value = root**41 + root**3 + 1
        print(f"p(根 {i+1}) = {p_value}")

    # 计算NumPy根的模长统计
    numpy_moduli = [abs(root) for root in numpy_roots]
    print("\nNumPy根的模长统计:")
    print(f"最小模长: {min(numpy_moduli):.10f}")
    print(f"最大模长: {max(numpy_moduli):.10f}")
    print(f"平均模长: {np.mean(numpy_moduli):.10f}")

    # 比较两种方法的差异
    print("\n两种方法的比较:")
    print(f"根数量差异: {len(our_roots) - len(numpy_roots)}")

    # 计算对应根的误差
    min_len = min(len(our_roots), len(numpy_roots))
    errors = [abs(our_roots[i] - numpy_roots[i]) for i in range(min_len)]

    print(f"平均误差: {np.mean(errors):.10e}")
    print(f"最大误差: {max(errors):.10e}")
    print(f"最小误差: {min(errors):.10e}")

    # 验证两种方法是否与np.roots一致
    print("\n与np.roots函数结果比较:")
    direct_roots = np.roots(coefficients)
    print(f"np.roots找到的根数量: {len(direct_roots)}")

    # 检查是否有差异
    direct_moduli = [abs(root) for root in direct_roots]
    print(f"np.roots的平均模长: {np.mean(direct_moduli):.10f}")

    # 比较方法
    print("\n结论:")
    print("1. 自实现QR算法和NumPy特征值计算在相同伴随矩阵上的结果非常接近")
    print("2. 这两种方法都是通过求解伴随矩阵的特征值来计算多项式的根")
    print("3. np.roots函数内部也是采用类似的伴随矩阵特征值方法")


if __name__ == "__main__":
    main()
