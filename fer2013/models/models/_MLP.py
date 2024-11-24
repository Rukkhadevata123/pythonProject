# 尝试导入 PyTorch
try:
    import torch.nn as nn
    pytorch_available = True
except ImportError as e:
    print(f"Failed to import PyTorch: {e}")
    pytorch_available = False

if pytorch_available:
    # 定义 MLP 模型
    class MLP_1(nn.Module):
        def __init__(self, num_classes=7):
            super(MLP_1, self).__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(48 * 48 * 1, 512),  # 修改输入尺寸
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)
    
    class MLP_2(nn.Module):
        def __init__(self, num_classes=7):
            super(MLP_2, self).__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(48 * 48 * 1, 1024),  # 修改输入尺寸
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)
    
    class MLP_3(nn.Module):
        def __init__(self, num_classes=7):
            super(MLP_3, self).__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(48 * 48 * 1, 2048),  # 修改输入尺寸
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)
    
    class MLP_4(nn.Module):
        def __init__(self, num_classes=7):
            super(MLP_4, self).__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(48 * 48 * 1, 256),  # 修改输入尺寸
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)

# 尝试导入 TensorFlow
try:
    from tensorflow.keras import layers, models
    tensorflow_available = True
except ImportError as e:
    print(f"Failed to import TensorFlow: {e}")
    tensorflow_available = False

if tensorflow_available:
    # 定义 TensorFlow MLP 模型
    def MLP_tf_1(input_shape=(48, 48, 1), num_classes=7):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    def MLP_tf_2(input_shape=(48, 48, 1), num_classes=7):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(512, activation='relu'))  
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))  
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))  
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    def MLP_tf_3(input_shape=(48, 48, 1), num_classes=7):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(1024, activation='relu'))  
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))  
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))  
        model.add(layers.Dense(64, activation='relu'))  
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    def MLP_tf_4(input_shape=(48, 48, 1), num_classes=7):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(128, activation='relu'))  
        model.add(layers.Dense(64, activation='relu'))  
        model.add(layers.Dense(32, activation='relu'))  
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

# 尝试导入 PaddlePaddle
try:
    import paddle.nn as pnn
    import paddle.nn.functional as F
    paddle_available = True
except ImportError as e:
    print(f"Failed to import PaddlePaddle: {e}")
    paddle_available = False

if paddle_available:
    # 定义 MLP 模型
    class MLP_1_pp(pnn.Layer):
        def __init__(self, num_classes=7):
            super(MLP_1_pp, self).__init__()
            self.net = pnn.Sequential(
                pnn.Flatten(),
                pnn.Linear(48 * 48 * 1, 512),
                pnn.ReLU(),
                pnn.Linear(512, 128),
                pnn.ReLU(),
                pnn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    class MLP_2_pp(pnn.Layer):
        def __init__(self, num_classes=7):
            super(MLP_2_pp, self).__init__()
            self.net = pnn.Sequential(
                pnn.Flatten(),
                pnn.Linear(48 * 48 * 1, 1024),
                pnn.ReLU(),
                pnn.Dropout(0.5),
                pnn.Linear(1024, 512),
                pnn.ReLU(),
                pnn.Dropout(0.5),
                pnn.Linear(512, 128),
                pnn.ReLU(),
                pnn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    class MLP_3_pp(pnn.Layer):
        def __init__(self, num_classes=7):
            super(MLP_3_pp, self).__init__()
            self.net = pnn.Sequential(
                pnn.Flatten(),
                pnn.Linear(48 * 48 * 1, 2048),
                pnn.ReLU(),
                pnn.Dropout(0.5),
                pnn.Linear(2048, 1024),
                pnn.ReLU(),
                pnn.Dropout(0.5),
                pnn.Linear(1024, 512),
                pnn.ReLU(),
                pnn.Linear(512, 128),
                pnn.ReLU(),
                pnn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    class MLP_4_pp(pnn.Layer):
        def __init__(self, num_classes=7):
            super(MLP_4_pp, self).__init__()
            self.net = pnn.Sequential(
                pnn.Flatten(),
                pnn.Linear(48 * 48 * 1, 256),
                pnn.ReLU(),
                pnn.Linear(256, 128),
                pnn.ReLU(),
                pnn.Linear(128, 64),
                pnn.ReLU(),
                pnn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)