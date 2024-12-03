import numpy as np

# 假设银行需要根据顾客的年龄、薪资、当前债务来决定是否给予信用卡
# 数据输入：年龄, 薪资（月）, 债务（万）
X = np.array([[16, 0, 0],
              [24, 5000, 0],
              [23, 6000, 10],
              [28, 5000, 50],
              [40, 15000, 200]])
# 目标输出（0: 否, 1: 是）
y = np.array([0, 1, 1, 0, 1])

# 数据标准化
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 初始化权重W=[w1, w2, w3], 偏置b
weights = np.random.randn(3)
bias = np.random.randn()


# 激活函数选择Sigmoid函数
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# 实现梯度下降求解，更新权重和偏置，并记录最终的权重和偏置
def gradient_descent(_X, _y, _weights, _bias, lr=0.01, epochs=1000):
    m = _X.shape[0]
    for epoch in range(epochs):
        # 前向传播
        z = np.dot(_X, _weights) + _bias
        predictions = sigmoid(z)

        # 计算损失
        loss = -1/m * np.sum(_y * np.log(predictions + 1e-8) +
                             (1 - _y) * np.log(1 - predictions + 1e-8))

        # 反向传播
        dw = 1/m * np.dot(_X.T, (predictions - _y))
        db = 1/m * np.sum(predictions - _y)

        # 更新权重和偏置
        _weights -= lr * dw
        _bias -= lr * db

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return _weights, _bias


# 训练模型
weights, bias = gradient_descent(X, y, weights, bias)


# 判断自己（50, 12000, 30）是否获批。当输出y<0.5，未获批，否则获批
def predict(_X, _weights, _bias):
    return sigmoid(np.dot(_X, _weights) + _bias)


# 输入数据
new_customer = np.array([50, 12000, 30])
new_customer = (new_customer - np.mean(X, axis=0)) / np.std(X, axis=0)
result = predict(new_customer, weights, bias)

if result < 0.5:
    print("未获批")
else:
    print("获批")