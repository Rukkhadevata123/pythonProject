import torch.nn as nn

# 定义 MLP 模型
class MLP_1(nn.Module):
    def __init__(self, num_classes=7):
        super(MLP_1, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224 * 1, 512),  # 修改输入尺寸
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
            nn.Linear(224 * 224 * 1, 1024),  # 修改输入尺寸
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
            nn.Linear(224 * 224 * 1, 2048),  # 修改输入尺寸
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
            nn.Linear(224 * 224 * 1, 256),  # 修改输入尺寸
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    

# 定义 TensorFlow MLP 模型
import tensorflow as tf
from tensorflow.keras import layers, models

def MLP_tf_1(input_shape=(224, 224, 1), num_classes=7):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))  # 减少神经元数量
    model.add(layers.Dense(64, activation='relu'))  # 减少神经元数量
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def MLP_tf_2(input_shape=(224, 224, 1), num_classes=7):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(512, activation='relu'))  # 减少神经元数量
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))  # 减少神经元数量
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))  # 减少神经元数量
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def MLP_tf_3(input_shape=(224, 224, 1), num_classes=7):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(1024, activation='relu'))  # 减少神经元数量
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))  # 减少神经元数量
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))  # 减少神经元数量
    model.add(layers.Dense(64, activation='relu'))  # 减少神经元数量
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def MLP_tf_4(input_shape=(224, 224, 1), num_classes=7):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(128, activation='relu'))  # 减少神经元数量
    model.add(layers.Dense(64, activation='relu'))  # 减少神经元数量
    model.add(layers.Dense(32, activation='relu'))  # 减少神经元数量
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model