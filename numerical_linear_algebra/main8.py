"""
房产估价线性模型
y=x_0+a1x1+a2x2+...+anxn
n=11
共28组数据
"""

import numpy as np
import matplotlib.pyplot as plt
from LS import ls_solve_qr, normal_equations_solve


def main():
    # 读取房价数据（y值）
    y = np.array(
        [
            25.9,
            29.5,
            27.9,
            25.9,
            29.9,
            29.9,
            30.9,
            28.9,
            84.9,
            82.9,
            35.9,
            31.5,
            31.0,
            30.9,
            30.0,
            28.9,
            36.9,
            41.9,
            40.5,
            43.9,
            37.5,
            37.9,
            44.5,
            37.9,
            38.9,
            36.9,
            45.8,
            41.0,
        ]
    )

    # 读取特征数据
    X = np.array(
        [
            [4.9176, 1.0, 3.4720, 0.9980, 1.0, 7, 4, 42, 3, 1, 0],
            [5.0208, 1.0, 3.5310, 1.5000, 2.0, 7, 4, 62, 1, 1, 0],
            [4.5429, 1.0, 2.2750, 1.1750, 1.0, 6, 3, 40, 2, 1, 0],
            [4.5573, 1.0, 4.0500, 1.2320, 1.0, 6, 3, 54, 4, 1, 0],
            [5.0597, 1.0, 4.4550, 1.1210, 1.0, 6, 3, 42, 3, 1, 0],
            [3.8910, 1.0, 4.4550, 0.9880, 1.0, 6, 3, 56, 2, 1, 0],
            [5.8980, 1.0, 5.8500, 1.2400, 1.0, 7, 3, 51, 2, 1, 1],
            [5.6039, 1.0, 9.5200, 1.5010, 0.0, 6, 3, 32, 1, 1, 0],
            [15.4202, 2.5, 9.8000, 3.4200, 2.0, 10, 5, 42, 2, 1, 1],
            [14.4598, 2.5, 12.800, 3.0000, 2.0, 9, 5, 11, 4, 1, 1],
            [5.8282, 1.0, 6.4350, 1.2250, 2.0, 6, 3, 32, 1, 1, 0],
            [5.3003, 1.0, 4.9883, 1.5520, 1.0, 6, 3, 30, 1, 2, 0],
            [6.2712, 1.0, 5.5200, 0.9750, 1.0, 5, 2, 30, 1, 2, 0],
            [5.9592, 1.0, 6.6660, 1.1210, 2.0, 6, 3, 32, 2, 1, 0],
            [5.0500, 1.0, 5.0000, 1.0200, 0.0, 5, 2, 46, 4, 1, 1],
            [5.6039, 1.0, 9.5200, 1.5010, 0.0, 6, 3, 32, 1, 1, 0],
            [8.2464, 1.5, 5.1500, 1.6640, 2.0, 8, 4, 50, 4, 1, 0],
            [6.6969, 1.5, 6.0920, 1.4880, 1.5, 7, 3, 22, 1, 1, 1],
            [7.7841, 1.5, 7.1020, 1.3760, 1.0, 6, 3, 17, 2, 1, 0],
            [9.0384, 1.0, 7.8000, 1.5000, 1.5, 7, 3, 23, 3, 3, 0],
            [5.9894, 1.0, 5.5200, 1.2560, 2.0, 6, 3, 40, 4, 1, 1],
            [7.5422, 1.5, 4.0000, 1.6900, 1.0, 6, 3, 22, 1, 1, 0],
            [8.7951, 1.5, 9.8900, 1.8200, 2.0, 8, 4, 50, 1, 1, 1],
            [6.0931, 1.5, 6.7265, 1.6520, 1.0, 6, 3, 44, 4, 1, 0],
            [8.3607, 1.5, 9.1500, 1.7770, 2.0, 8, 4, 48, 1, 1, 1],
            [8.1400, 1.0, 8.0000, 1.5040, 2.0, 7, 3, 3, 1, 3, 0],
            [9.1416, 1.5, 7.3262, 1.8310, 1.5, 8, 4, 31, 4, 1, 0],
            [12.000, 1.5, 5.0000, 1.2000, 2.0, 6, 3, 30, 3, 1, 1],
        ]
    )

    # 添加常数项（截距）
    A = np.column_stack([np.ones(len(X)), X])

    # 特征名称
    feature_names = [
        "税",
        "浴室数目",
        "占地面积",
        "居住面积",
        "车库数目",
        "房屋数目",
        "居室数目",
        "房龄",
        "建筑类型",
        "户型",
        "壁炉数目",
    ]

    print("=== 房产估价线性回归模型 ===")
    print(f"数据集大小: {len(y)} 样本, {X.shape[1]} 特征")

    # 使用QR分解求解最小二乘问题
    coefs_qr, residual_qr = ls_solve_qr(A, y, method="householder")

    print("\n=== 使用QR分解求解 ===")
    print("回归系数:")
    for i, (name, coef) in enumerate(zip(feature_names, coefs_qr)):
        print(f"{name}: {coef:.6f}")
    print(f"残差平方和: {residual_qr**2:.6f}")
    print(f"均方残差: {(residual_qr**2)/len(y):.6f}")

    # 使用正规方程求解最小二乘问题
    coefs_normal, residual_normal = normal_equations_solve(A, y)

    print("\n=== 使用正规方程求解 ===")
    print("回归系数:")
    for i, (name, coef) in enumerate(zip(feature_names, coefs_normal)):
        print(f"{name}: {coef:.6f}")
    print(f"残差平方和: {residual_normal**2:.6f}")
    print(f"均方残差: {(residual_normal**2)/len(y):.6f}")

    # 计算R²（决定系数）
    y_mean = np.mean(y)
    total_sum_squares = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (residual_qr**2) / total_sum_squares

    print(f"\nR²决定系数: {r_squared:.6f}")

    # 计算调整R²
    n = len(y)
    p = X.shape[1] + 1  # 特征数加上截距
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - p)

    print(f"调整R²: {adj_r_squared:.6f}")

    # 预测vs实际值分析
    y_pred = A @ coefs_qr
    errors = y - y_pred

    print("\n=== 预测分析 ===")
    print("     实际价格    预测价格    误差    相对误差(%)")
    for i in range(len(y)):
        print(
            f"{y[i]:10.2f}  {y_pred[i]:10.2f}  {errors[i]:7.2f}  {100*abs(errors[i]/y[i]):10.2f}"
        )

    # 计算平均绝对误差(MAE)和平均相对误差(MAPE)
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / y)) * 100

    print(f"\n平均绝对误差(MAE): {mae:.4f}")
    print(f"平均相对误差(MAPE): {mape:.2f}%")

    # 绘制残差分析图
    plt.figure(figsize=(15, 10))

    # 1. 预测值vs实际值
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, y)
    plt.plot([min(y), max(y)], [min(y), max(y)], "r--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Actual Price")
    plt.title("Predicted vs Actual Price")

    # 2. 残差图
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, errors)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Distribution")

    # 3. 残差直方图
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=10, alpha=0.7)
    plt.axvline(x=0, color="r", linestyle="--")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residual Histogram")

    # 4. 特征重要性(绝对系数值)
    plt.subplot(2, 2, 4)
    # 跳过常数项
    importance = np.abs(coefs_qr[1:])
    feature_indices = np.arange(len(importance))
    plt.bar(feature_indices, importance)
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Importance (|Coefficient|)")
    plt.title("Feature Importance Analysis")
    plt.xticks(feature_indices, [f"F{i+1}" for i in feature_indices], rotation=45)

    plt.tight_layout()
    plt.savefig("house_pricing_analysis.png")
    plt.show()

    # 特征重要性详细分析
    print("\n=== 特征重要性分析 ===")
    # 特征及其对应系数
    feature_names_with_constant = ["常数项"] + feature_names
    coefficients_with_names = list(zip(feature_names_with_constant, coefs_qr))

    # 确认列表长度匹配
    print(
        f"系数数量: {len(coefs_qr)}, 特征数量(含常数项): {len(feature_names_with_constant)}"
    )
    print("QR求解的所有系数:")
    for name, coef in coefficients_with_names:
        print(f"{name}: {coef:.6f}")

    # 按照系数绝对值大小排序（不包括常数项）
    coefficients = coefs_qr[1:]  # 从索引1开始，跳过常数项
    features = feature_names  # 特征名称列表（不含常数项）
    importance_order = np.argsort(-np.abs(coefficients))

    print("\n按重要性排序的特征:")
    for i, idx in enumerate(importance_order):
        print(f"{i+1}. {features[idx]}: {coefficients[idx]:.6f}")

    # 多元线性回归模型方程
    print("\n=== 多元线性回归模型方程 ===")
    equation = f"房价 = {coefs_qr[0]:.4f}"
    for name, coef in zip(feature_names, coefs_qr[1:]):
        if coef >= 0:
            equation += f" + {coef:.4f}×{name}"
        else:
            equation += f" - {abs(coef):.4f}×{name}"

    print(equation)

    # 模型交叉验证
    print("\n=== 简单交叉验证 ===")
    # 将数据集随机分为训练集(80%)和测试集(20%)
    np.random.seed(42)
    indices = np.random.permutation(len(y))
    train_size = int(0.8 * len(y))

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    A_train, y_train = A[train_idx], y[train_idx]
    A_test, y_test = A[test_idx], y[test_idx]

    # 在训练集上拟合模型
    coefs_train, _ = ls_solve_qr(A_train, y_train)

    # 在测试集上评估模型
    y_test_pred = A_test @ coefs_train
    test_errors = y_test - y_test_pred
    test_mse = np.mean(test_errors**2)
    test_rmse = np.sqrt(test_mse)
    test_r2 = 1 - np.sum(test_errors**2) / np.sum((y_test - np.mean(y_test)) ** 2)

    print(f"测试集大小: {len(y_test)} 样本")
    print(f"测试集均方误差(MSE): {test_mse:.6f}")
    print(f"测试集均方根误差(RMSE): {test_rmse:.6f}")
    print(f"测试集R²: {test_r2:.6f}")

    return coefs_qr, r_squared


if __name__ == "__main__":
    main()
