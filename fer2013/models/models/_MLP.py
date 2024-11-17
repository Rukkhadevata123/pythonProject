import torch.nn as nn

# 定义 MLP 模型
class MLP_1(nn.Module):
    def __init__(self, num_classes=7):
        super(MLP_1, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48 * 3, 512),
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
            nn.Linear(48 * 48 * 3, 1024),
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
    pass

class MLP_4(nn.Module):
    pass