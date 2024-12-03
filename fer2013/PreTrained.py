import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# 自定义数据集类
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, self_transform=None, mode='train', test_size=0.2):
        self.data = pd.read_csv(csv_file)
        self.transform = self_transform
        self.mode = mode  # 根据传入的 mode 参数设置模式

        # 将数据划分为训练集和测试集
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=42, shuffle=True)
        self.train_data = train_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)

    def __len__(self):
        # 返回当前数据集（训练集或测试集）的长度
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'test':
            return len(self.test_data)
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'test'.")

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'test'.")

        img = data.iloc[idx, 1]
        img = np.fromstring(img, sep=' ').reshape(48, 48).astype(np.uint8)
        img = Image.fromarray(img).convert('L')
        img = img.resize((224, 224), Image.BILINEAR)  # 调整图像尺寸为 224x224，使用双线性插值
        label = int(data.iloc[idx, 0])  # 标签通常位于 CSV 文件的第一列

        if self.transform:
            img = self.transform(img)

        return {'image': img, 'label': label}

# 数据增强和预处理
transform_pytorch = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BILINEAR),  # 调整图像尺寸为 224x224，使用双线性插值
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建相对路径
csv_file = os.path.join(current_dir, '.', 'csvs', 'fer2013.csv')

# 加载和划分数据集
train_dataset = FER2013Dataset(csv_file=csv_file, self_transform=transform_pytorch, mode='train', test_size=0.2)
test_dataset = FER2013Dataset(csv_file=csv_file, self_transform=transform_pytorch, mode='test', test_size=0.2)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 使用预训练模型
class PretrainedModel(nn.Module):
    def __init__(self, num_classes=7):
        super(PretrainedModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 训练和评估函数
def train_model(model, train_loader_, test_loader_, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data in tqdm(train_loader_, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            x, y = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader_):.4f}, Accuracy: {100 * correct / total:.4f}%")

        # Test accuracy and confusion matrix
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            confusion_matrix = np.zeros((7, 7), dtype=int)
            all_labels = []
            all_preds = []
            for data in test_loader_:
                x, y = data['image'].to(device), data['label'].to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                for t, p in zip(y.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())
            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy:.4f}%")

            # Calculate TP, FP, TN, FN for each class
            for i in range(7):
                TP = confusion_matrix[i, i]
                FP = confusion_matrix[:, i].sum() - TP
                FN = confusion_matrix[i, :].sum() - TP
                TN = confusion_matrix.sum() - (TP + FP + FN)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                print(f"Class {i} ({emotion_labels[i]}): TP={TP}, FP={FP}, TN={TN}, FN={FN}")
            print(f"Overall: Accuracy={accuracy:.4f}%, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}")

            # 创建保存 ROC 曲线的目录
            roc_dir = os.path.join(current_dir, '..', 'ROC')
            if not os.path.exists(roc_dir):
                os.makedirs(roc_dir)

            # 绘制并保存 ROC 曲线
            all_labels = label_binarize(all_labels, classes=[0, 1, 2, 3, 4, 5, 6])
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(7):
                fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], np.array(all_preds)[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure()
            for i in range(7):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} ({emotion_labels[i]}) (AUC = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(roc_dir, f'roc_curve_{epoch + 1}.png'))
            plt.close()

# 初始化模型
model = PretrainedModel(num_classes=7)

# 训练和评估模型
train_model(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001)