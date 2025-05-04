"""
使用幂法求解多项式方程的模最大根
"""

import numpy as np
from pow import complex_power_method


def companion_matrix(coeffs):
    """
    构造多项式的伴随矩阵

    参数:
        coeffs: 多项式系数，按从高次到低次排列，即[a_n, a_{n-1}, ..., a_1, a_0]
                注意：最高次项系数应为1

    返回:
        companion: 伴随矩阵
    """
    # 确保 coeffs 是 numpy 数组而不是普通列表
    coeffs = np.array(coeffs, dtype=float)

    if abs(coeffs[0] - 1.0) > 1e-10:
        # 如果最高次项系数不为1，则归一化
        coeffs = coeffs / coeffs[0]

    n = len(coeffs) - 1
    companion = np.zeros((n, n))

    # 构造伴随矩阵的第一行
    companion[0, :] = -coeffs[1:]

    # 构造伴随矩阵的其余行（单位矩阵向右下偏移）
    if n > 1:
        companion[1:, :-1] = np.eye(n - 1)

    return companion


def find_max_modulus_root(coeffs, tol=1e-10, max_iter=5000):
    """
    使用幂法找到多项式方程的模最大根

    参数:
        coeffs: 多项式系数，按从高次到低次排列
        tol: 收敛容差
        max_iter: 最大迭代次数

    返回:
        root: 模最大根
        iterations: 迭代次数
    """
    # 构造伴随矩阵
    A = companion_matrix(coeffs)

    # 使用幂法计算最大特征值
    eigenvalue, eigenvector, iterations = complex_power_method(
        A, tol=tol, max_iter=max_iter
    )

    # 验证结果
    residual = np.linalg.norm(A @ eigenvector - eigenvalue * eigenvector)
    if residual > tol * 100:
        print(f"警告: 残差较大 ({residual:.6e})，结果可能不准确。")

    return eigenvalue, iterations


def verify_root(coeffs, root):
    """
    验证求得的根是否正确（支持复数根）

    参数:
        coeffs: 多项式系数
        root: 待验证的根(可能是复数)

    返回:
        residual: 残差 |f(root)|
    """
    n = len(coeffs) - 1
    result = coeffs[0]

    # 计算多项式的值 f(root)
    for i in range(1, n + 1):
        result = result * root + coeffs[i]

    return abs(result)  # 返回模值


def format_complex(z):
    """格式化复数为友好显示的字符串"""
    if abs(z.imag) < 1e-10:  # 如果虚部非常小
        return f"{z.real:.10f}"
    elif abs(z.real) < 1e-10:  # 如果实部非常小
        return f"{z.imag:.10f}j"
    else:
        sign = "+" if z.imag >= 0 else ""
        return f"{z.real:.10f}{sign}{z.imag:.10f}j"


def solve_polynomial_equation(poly_str):
    """
    解多项式方程并展示结果

    参数:
        poly_str: 多项式字符串表示
    """
    # 手动解析多项式系数
    if "x^3 + x^2 - 5*x + 3" in poly_str:
        # x^3 + x^2 - 5x + 3 = 0
        coeffs = [1, 1, -5, 3]  # 从高次到低次
    elif "x^3 - 3*x - 1" in poly_str:
        # x^3 - 3x - 1 = 0
        coeffs = [1, 0, -3, -1]  # 从高次到低次
    elif "x^8 + 101*x^7" in poly_str:
        # x^8 + 101*x^7 + ... - 1000 = 0
        coeffs = [1, 101, 208.01, 10891.01, 9802.08, 79108.9, -99902, 790, -1000]
    else:
        raise ValueError(f"未知多项式: {poly_str}")

    # 使用幂法求解模最大根
    root, iterations = find_max_modulus_root(coeffs)

    # 计算numpy求解的所有根
    all_roots = np.roots(coeffs)
    max_modulus_root = all_roots[np.argmax(np.abs(all_roots))]

    # 验证根的精度
    residual = verify_root(coeffs, root)
    numpy_residual = verify_root(coeffs, max_modulus_root)

    # 输出结果
    print(f"\n多项式: {poly_str}")
    print(f"伴随矩阵大小: {len(coeffs)-1}×{len(coeffs)-1}")
    print(f"幂法求得的模最大根: {format_complex(root)}")
    print(f"残差 |f(root)|: {residual:.6e}")
    print(f"收敛所需迭代次数: {iterations}")
    print(f"NumPy求得的模最大根: {format_complex(max_modulus_root)}")
    print(f"NumPy根的残差: {numpy_residual:.6e}")

    # 计算相对误差
    rel_error = abs(root - max_modulus_root) / abs(max_modulus_root)
    print(f"相对误差: {rel_error:.6e}")

    return root, max_modulus_root


def main():
    # 设置全局参数
    global tol
    tol = 1e-10

    # 题目中的多项式
    polynomials = [
        "x^3 + x^2 - 5*x + 3",
        "x^3 - 3*x - 1",
        "x^8 + 101*x^7 + 208.01*x^6 + 10891.01*x^5 + 9802.08*x^4 + 79108.9*x^3 - 99902*x^2 + 790*x - 1000",
    ]

    # 解决每个多项式并保存结果
    results = []
    for i, poly in enumerate(polynomials):
        print(f"\n=== 求解多项式 {i+1} ===")
        root, exact_root = solve_polynomial_equation(poly)
        results.append((root, exact_root))

    # 显示所有结果的总结
    print("\n=== 结果总结 ===")
    print(
        f"{'多项式':<10} {'幂法求得的模最大根':<30} {'NumPy求得的模最大根':<30} {'相对误差':<15}"
    )
    print("-" * 85)

    for i, (root, exact_root) in enumerate(results):
        rel_error = abs(root - exact_root) / abs(exact_root)
        print(
            f"{i+1:<10} {format_complex(root):<30} {format_complex(exact_root):<30} {rel_error:<15.6e}"
        )


if __name__ == "__main__":
    main()
