import os
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from PyQt6.QtCore import QThread, pyqtSignal


# 自定义数据集类
class FER2013Dataset_paddle(Dataset):
    def __init__(self, csv_file, transform=None, mode='train', test_size=0.2):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
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
        img = img.resize((48, 48))
        label = int(data.iloc[idx, 0])  # 标签通常位于 CSV 文件的第一列

        if self.transform:
            img = self.transform(img)

        return img, label


# 数据增强和预处理
transform_paddle = paddle.vision.transforms.Compose([
    paddle.vision.transforms.Resize((48, 48)),  # 调整图像尺寸为 224x224，使用线性插值
    paddle.vision.transforms.RandomHorizontalFlip(),
    paddle.vision.transforms.RandomRotation(10, fill=(0,)),  # 添加 fill 参数
    paddle.vision.transforms.ColorJitter(0.1, 0.1, 0.1),  # 添加 ColorJitter
    paddle.vision.transforms.ToTensor(),
    paddle.vision.transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_paddle_test = paddle.vision.transforms.Compose([
    paddle.vision.transforms.Resize((48, 48)),
    paddle.vision.transforms.ToTensor(),
    paddle.vision.transforms.Normalize(mean=[0.5], std=[0.5])
])


class TrainThread_paddle(QThread):
    update_text = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        num_epochs=25,
        learning_rate=0.001,
    ):
        super(TrainThread_paddle, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.stop_training = False
        self.paused = False
        self.save_checkpoint_flag = False
        self.save_best_model_flag = False

    def run(self):
        device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
        self.model.to(device)
        criterion = nn.CrossEntropyLoss()

        if self.optimizer == 'Adam':
            optimizer = optim.Adam(parameters=self.model.parameters(), learning_rate=self.learning_rate)
        elif self.optimizer == 'AdamW':
            optimizer = optim.AdamW(parameters=self.model.parameters(), learning_rate=self.learning_rate)
        elif self.optimizer == 'Adamax':
            optimizer = optim.Adamax(parameters=self.model.parameters(), learning_rate=self.learning_rate)
        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(parameters=self.model.parameters(), learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer == 'Adagrad':
            optimizer = optim.Adagrad(parameters=self.model.parameters(), learning_rate=self.learning_rate)
        
        scheduler = optim.lr.StepDecay(learning_rate=self.learning_rate, step_size=7, gamma=0.1)
        best_accuracy = 0.0

        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        for epoch in range(self.num_epochs):
            if self.stop_training:
                break
            while self.paused:
                self.msleep(100)
            self.update_text.emit(f"正在训练第 {epoch + 1} 轮...\n")
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')):
                if self.stop_training:
                    break
                while self.paused:
                    self.msleep(100)
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer.clear_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                predicted = paddle.argmax(outputs, axis=1)
                total += y.shape[0]
                correct += (predicted == y).numpy().sum()
                self.update_progress.emit(int((i + 1) / len(self.train_loader) * 100))
            scheduler.step()
            epoch_info = (
                f"Epoch {epoch + 1}/{self.num_epochs}, "
                f"Loss: {running_loss / len(self.train_loader):.4f}, "
                f"Accuracy: {100 * correct / total:.4f}%\n"
            )
            self.update_text.emit(epoch_info)

            # 保存断点
            if self.save_checkpoint_flag:
                checkpoint_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '..',
                    'result_paddle',
                    'checkpoint.pdparams'
                )
                paddle.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': running_loss / len(self.train_loader),
                    'accuracy': 100 * correct / total
                }, checkpoint_path)
                self.update_text.emit(f"Checkpoint saved at epoch {epoch + 1}.\n")
                self.save_checkpoint_flag = False  # 重置标志

            # 测试准确率和混淆矩阵
            self.model.eval()
            with paddle.no_grad():
                correct = 0
                total = 0
                confusion_matrix = np.zeros((7, 7), dtype=int)
                all_labels = []
                all_preds = []
                for data in self.test_loader:
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    outputs = self.model(x)
                    predicted = paddle.argmax(outputs, axis=1)
                    total += y.shape[0]
                    correct += (predicted == y).numpy().sum()
                    for t, p in zip(y.numpy(), predicted.numpy()):
                        confusion_matrix[t, p] += 1
                    all_labels.extend(y.numpy())
                    all_preds.extend(outputs.numpy())
                accuracy = 100 * correct / total
                test_info = f"Test Accuracy: {accuracy:.4f}%\n"
                self.update_text.emit(test_info)

                # 保存最优模型
                if self.save_best_model_flag and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '..',
                        'result_paddle',
                        'best_model.pdparams'
                    )
                    paddle.save(self.model.state_dict(), best_model_path)
                    self.update_text.emit(f"Best model saved with accuracy: {best_accuracy:.4f}%\n")

                # 计算每个类别的 TP、FP、TN、FN
                for i in range(7):
                    TP = confusion_matrix[i, i]
                    FP = confusion_matrix[:, i].sum() - TP
                    FN = confusion_matrix[i, :].sum() - TP
                    TN = confusion_matrix.sum() - (TP + FP + FN)
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    class_info = (
                        f"Class {i} ({emotion_labels[i]}): TP={TP}, FP={FP}, TN={TN}, FN={FN}, "
                        f"Accuracy={(TP + TN) / (TP + FP + TN + FN):.4f}, Precision={precision:.4f}, "
                        f"Recall={recall:.4f}, F1 Score={f1_score:.4f}\n"
                    )
                    self.update_text.emit(class_info)

                # 创建保存 ROC 曲线的目录
                roc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ROC')
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

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.stop_training = True
        self.paused = False  # 防止线程被暂停

    def enable_save_checkpoint(self):
        self.save_checkpoint_flag = True

    def enable_save_best_model(self):
        self.save_best_model_flag = True