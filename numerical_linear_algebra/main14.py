import numpy as np
from qr_iter_and_schur import real_schur_decomposition, extract_eigenvalues


def compute_eigenvalues(x):
    """
    计算矩阵A在参数x下的特征值

    参数:
        x: 矩阵中的参数值

    返回:
        our_eigenvalues: 自实现算法计算的特征值
        numpy_eigenvalues: NumPy计算的特征值
    """
    # 构造矩阵
    A = np.array(
        [
            [9.1, 3.0, 2.6, 4.0],
            [4.2, 5.3, 1.7, 1.6],
            [3.2, 1.7, 9.4, x],
            [6.1, 4.9, 3.5, 6.2],
        ]
    )

    # 使用自实现的QR迭代算法计算特征值
    T, Q, blocks = real_schur_decomposition(A, max_iter=10000, tol=1e-12)
    our_eigenvalues = extract_eigenvalues(T, blocks)

    # 使用NumPy计算特征值
    numpy_eigenvalues = np.linalg.eigvals(A)

    return our_eigenvalues, numpy_eigenvalues


def format_complex(z):
    """格式化复数输出"""
    if abs(z.imag) < 1e-10:
        return f"{z.real:.10f}"
    return f"{z.real:.10f} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.10f}j"


def main():
    print("矩阵A的参数敏感性分析\n")
    print("A = [[9.1, 3.0, 2.6, 4.0],")
    print("     [4.2, 5.3, 1.7, 1.6],")
    print("     [3.2, 1.7, 9.4, x],")
    print("     [6.1, 4.9, 3.5, 6.2]]\n")

    # 测试三个不同的x值
    x_values = [0.9, 1.0, 1.1]

    results = {}
    for x in x_values:
        our_eigenvalues, numpy_eigenvalues = compute_eigenvalues(x)

        # 对特征值排序以便比较
        our_eigenvalues = sorted(
            our_eigenvalues,
            key=lambda val: (
                -abs(val),
                -val.real if abs(val.imag) < 1e-10 else -val.imag,
            ),
        )
        numpy_eigenvalues = sorted(
            numpy_eigenvalues,
            key=lambda val: (
                -abs(val),
                -val.real if abs(val.imag) < 1e-10 else -val.imag,
            ),
        )

        results[x] = (our_eigenvalues, numpy_eigenvalues)

    # 打印结果
    for x in x_values:
        our_eigenvalues, numpy_eigenvalues = results[x]

        print(f"\n当 x = {x} 时:")
        print("-" * 50)
        print("| {:^20} | {:^20} |".format("自实现QR算法", "NumPy"))
        print("-" * 50)

        for i in range(len(our_eigenvalues)):
            our_val = format_complex(our_eigenvalues[i])
            numpy_val = format_complex(numpy_eigenvalues[i])
            print("| {:^20} | {:^20} |".format(our_val, numpy_val))

        print("-" * 50)

        # 计算误差
        errors = [
            abs(our_eigenvalues[i] - numpy_eigenvalues[i])
            for i in range(len(our_eigenvalues))
        ]
        avg_error = np.mean(errors)
        max_error = np.max(errors)

        print(f"平均误差: {avg_error:.4e}")
        print(f"最大误差: {max_error:.4e}")

    # 分析特征值随x变化的趋势
    print("\n\n特征值随参数x变化的趋势分析:")
    print("-" * 60)
    print(
        "| {:^10} | {:^10} | {:^10} | {:^10} |".format(
            "特征值", "x=0.9", "x=1.0", "x=1.1"
        )
    )
    print("-" * 60)

    for i in range(4):  # 矩阵是4×4的，所以有4个特征值
        values = [results[x][0][i] for x in x_values]
        print(
            "| {:^10} | {:^10} | {:^10} | {:^10} |".format(
                f"λ{i+1}",
                f"{values[0].real:.4f}{'+' + str(values[0].imag)[:5] + 'j' if abs(values[0].imag) > 1e-10 else ''}",
                f"{values[1].real:.4f}{'+' + str(values[1].imag)[:5] + 'j' if abs(values[1].imag) > 1e-10 else ''}",
                f"{values[2].real:.4f}{'+' + str(values[2].imag)[:5] + 'j' if abs(values[2].imag) > 1e-10 else ''}",
            )
        )

    print("-" * 60)
    print("\n结论: 通过观察可以看出，随着参数x的微小变化，特征值也相应发生变化，")
    print("      但变化幅度较小，表明该矩阵的特征值对参数x的扰动具有一定的稳定性。")


if __name__ == "__main__":
    main()
