# 尝试导入 PyTorch
try:
    import torch.nn as nn
    pytorch_available = True
except ImportError as e:
    print(f"Failed to import PyTorch: {e}")
    pytorch_available = False

if pytorch_available:
    # 定义 CNN_1 模型（类似于 AlexNet）
    class CNN_1(nn.Module):
        def __init__(self, num_classes=7):
            super(CNN_1, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Flatten(),
                nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes)
            )

        def forward(self, x):
            x = self.net(x)
            return x

    # 定义 CNN_2 模型
    class CNN_2(nn.Module):
        def __init__(self, num_classes=7):
            super(CNN_2, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(256 * 28 * 28, 1024), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(1024, num_classes)
            )

        def forward(self, x):
            x = self.net(x)
            return x

    # 定义 CNN_3 模型
    class CNN_3(nn.Module):
        def __init__(self, num_classes=7):
            super(CNN_3, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, 512), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.net(x)
            return x

    # 定义 CNN_4 模型
    class CNN_4(nn.Module):
        def __init__(self, num_classes=7):
            super(CNN_4, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(256 * 14 * 14, 1024), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.net(x)
            return x

# 尝试导入 TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tensorflow_available = True
except ImportError as e:
    print(f"Failed to import TensorFlow: {e}")
    tensorflow_available = False

if tensorflow_available:
    # 定义 CNN_1_tf 模型（类似于 AlexNet）
    def CNN_1_tf(input_shape=(224, 224, 1), num_classes=7):
        model = models.Sequential()
        model.add(layers.Conv2D(96, kernel_size=11, strides=4, padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=3, strides=2))
        model.add(layers.Conv2D(256, kernel_size=5, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=3, strides=2))
        model.add(layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=3, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    # 定义 CNN_2_tf 模型
    def CNN_2_tf(input_shape=(224, 224, 1), num_classes=7):
        model = models.Sequential()
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    # 定义 CNN_3_tf 模型
    def CNN_3_tf(input_shape=(224, 224, 1), num_classes=7):
        model = models.Sequential()
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    # 定义 CNN_4_tf 模型
    def CNN_4_tf(input_shape=(224, 224, 1), num_classes=7):
        model = models.Sequential()
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

# 尝试导入 PaddlePaddle
try:
    import paddle.nn as pnn
    paddle_available = True
except ImportError as e:
    print(f"Failed to import PaddlePaddle: {e}")
    paddle_available = False

if paddle_available:
    # 定义 CNN_1_pp 模型（类似于 AlexNet）
    class CNN_1_pp(pnn.Layer):
        def __init__(self, num_classes=7):
            super(CNN_1_pp, self).__init__()
            self.net = pnn.Sequential(
                pnn.Conv2D(1, 96, kernel_size=11, stride=4, padding=2), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=3, stride=2),
                pnn.Conv2D(96, 256, kernel_size=5, padding=2), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=3, stride=2),
                pnn.Conv2D(256, 384, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(384, 384, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(384, 256, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=3, stride=2),
                pnn.Flatten(),
                pnn.Linear(256 * 6 * 6, 4096), pnn.ReLU(),
                pnn.Dropout(p=0.5),
                pnn.Linear(4096, 4096), pnn.ReLU(),
                pnn.Dropout(p=0.5),
                pnn.Linear(4096, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    # 定义 CNN_2_pp 模型
    class CNN_2_pp(pnn.Layer):
        def __init__(self, num_classes=7):
            super(CNN_2_pp, self).__init__()
            self.net = pnn.Sequential(
                pnn.Conv2D(1, 64, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(64, 64, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Conv2D(64, 128, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(128, 128, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Conv2D(128, 256, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(256, 256, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(256, 256, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Flatten(),
                pnn.Linear(256 * 28 * 28, 1024), pnn.ReLU(),
                pnn.Dropout(p=0.5),
                pnn.Linear(1024, 1024), pnn.ReLU(),
                pnn.Dropout(p=0.5),
                pnn.Linear(1024, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    # 定义 CNN_3_pp 模型
    class CNN_3_pp(pnn.Layer):
        def __init__(self, num_classes=7):
            super(CNN_3_pp, self).__init__()
            self.net = pnn.Sequential(
                pnn.Conv2D(1, 32, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(32, 32, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Conv2D(32, 64, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(64, 64, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Conv2D(64, 128, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.Conv2D(128, 128, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Flatten(),
                pnn.Linear(128 * 28 * 28, 512), pnn.ReLU(),
                pnn.Dropout(p=0.5),
                pnn.Linear(512, 512), pnn.ReLU(),
                pnn.Dropout(p=0.5),
                pnn.Linear(512, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    # 定义 CNN_4_pp 模型
    class CNN_4_pp(pnn.Layer):
        def __init__(self, num_classes=7):
            super(CNN_4_pp, self).__init__()
            self.net = pnn.Sequential(
                pnn.Conv2D(1, 32, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Conv2D(32, 64, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Conv2D(64, 128, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Conv2D(128, 256, kernel_size=3, padding=1), pnn.ReLU(),
                pnn.MaxPool2D(kernel_size=2, stride=2),
                pnn.Flatten(),
                pnn.Linear(256 * 14 * 14, 1024), pnn.ReLU(),
                pnn.Dropout(p=0.5),
                pnn.Linear(1024, 512), pnn.ReLU(),
                pnn.Dropout(p=0.5),
                pnn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.net(x)