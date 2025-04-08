"""
y=at^2+bt+c
残向量2-范数最小
t_i = -1, -0.75, -0.5, 0, 0.25, 0.5, 0.75
y_i = 1.00, 0.8125, 0.75, 1.00, 1.3125, 1.75, 2.3125
"""

import numpy as np
import matplotlib.pyplot as plt
from LS import ls_solve_qr, normal_equations_solve


def main():
    # 数据点
    t = np.array([-1, -0.75, -0.5, 0, 0.25, 0.5, 0.75])
    y = np.array([1.00, 0.8125, 0.75, 1.00, 1.3125, 1.75, 2.3125])

    # 构建系数矩阵 A，每行是 [t_i^2, t_i, 1]
    A = np.column_stack((t**2, t, np.ones_like(t)))

    print("=== 二次多项式拟合问题 y = at² + bt + c ===")
    print("数据点:")
    for i in range(len(t)):
        print(f"t = {t[i]:.2f}, y = {y[i]:.4f}")

    print("\n系数矩阵 A:")
    print(A)

    # 使用QR分解求解最小二乘问题
    x_qr, residual_qr = ls_solve_qr(A, y, method="householder")
    a_qr, b_qr, c_qr = x_qr

    print("\n=== 使用QR分解求解 ===")
    print(f"参数估计: a = {a_qr:.6f}, b = {b_qr:.6f}, c = {c_qr:.6f}")
    print(f"拟合多项式: y = {a_qr:.6f}t² + {b_qr:.6f}t + {c_qr:.6f}")
    print(f"残差的2-范数: {residual_qr:.10f}")

    # 使用正规方程求解最小二乘问题
    x_normal, residual_normal = normal_equations_solve(A, y)
    a_normal, b_normal, c_normal = x_normal

    print("\n=== 使用正规方程求解 ===")
    print(f"参数估计: a = {a_normal:.6f}, b = {b_normal:.6f}, c = {c_normal:.6f}")
    print(f"拟合多项式: y = {a_normal:.6f}t² + {b_normal:.6f}t + {c_normal:.6f}")
    print(f"残差的2-范数: {residual_normal:.10f}")

    # 计算拟合值和误差
    y_fit_qr = A @ x_qr
    errors_qr = y - y_fit_qr

    print("\n=== 拟合精度分析 ===")
    print("   t      y(实际)    y(拟合)    误差")
    for i in range(len(t)):
        print(f"{t[i]:6.2f}  {y[i]:8.4f}  {y_fit_qr[i]:8.4f}  {errors_qr[i]:8.4f}")

    # 绘制结果
    plt.figure(figsize=(10, 6))

    # 绘制原始数据点
    plt.scatter(t, y, color="blue", label="data points")

    # 绘制拟合曲线
    t_fine = np.linspace(min(t) - 0.2, max(t) + 0.2, 100)
    y_fine_qr = a_qr * t_fine**2 + b_qr * t_fine + c_qr
    plt.plot(t_fine, y_fine_qr, "r-", label="QR Fit")

    # 绘制每个数据点的误差线
    for i in range(len(t)):
        plt.plot([t[i], t[i]], [y[i], y_fit_qr[i]], "g--", alpha=0.5)

    plt.title("Polynomial Fit using QR Decomposition")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("polynomial_fit.png")
    plt.show()

    # 解析解验证
    print("\n=== 解析解验证 ===")
    try:
        exact_sol = np.array([1.0, 1.0, 1.0])  # 假设真实解为 y = t² + t + 1
        predicted_y = A @ exact_sol
        print("假设真实多项式为: y = t² + t + 1")
        print("代入数据点计算的理论值:")
        for i in range(len(t)):
            print(f"t = {t[i]:.2f}: 理论值 = {predicted_y[i]:.4f}, 实际值 = {y[i]:.4f}")

        # 验证误差
        error = np.linalg.norm(predicted_y - y)
        print(f"理论值与实际值的残差2-范数: {error:.10f}")

        # 比较解与理论解的误差
        qr_error = np.linalg.norm(x_qr - exact_sol)
        print(f"QR解与理论解的差异: {qr_error:.10f}")

        if error < 1e-10:
            print("\n结论: 数据点完全符合 y = t² + t + 1")
        else:
            print("\n结论: 数据点可能包含误差或不完全符合二次多项式")
    except Exception as e:
        print(f"解析验证失败: {e}")


if __name__ == "__main__":
    main()
