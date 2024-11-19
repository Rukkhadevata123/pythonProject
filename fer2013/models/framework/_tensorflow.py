import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from PyQt6.QtCore import QThread, pyqtSignal
import matplotlib.pyplot as plt

# 配置 GPU 内存使用比例和设备日志记录
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置每个 GPU 的内存使用上限
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # 例如，设置为 4096MB
    except RuntimeError as e:
        print(e)

# 启用设备日志记录
# tf.debugging.set_log_device_placement(True)

# 自定义数据集类
class FER2013Dataset_tf:
    def __init__(self, csv_file, mode='train', test_size=0.1):
        self.data = pd.read_csv(csv_file)
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
            data = self.train_data.iloc[idx]
        elif self.mode == 'test':
            data = self.test_data.iloc[idx]
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'test'.")

        img = np.fromstring(data['pixels'], sep=' ').reshape(48, 48).astype(np.uint8)
        img = np.stack((img,) * 1, axis=-1)  # 转换为灰度图像
        img = tf.image.resize(img, [224, 224])  # 调整图像尺寸为 224x224
        label = data['emotion']
        return img, label

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # 归一化
    return image, label

def create_tf_dataset_from_csv(csv_file, batch_size, mode='train', test_size=0.1, shuffle=True):
    data = pd.read_csv(csv_file)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42, shuffle=True)
    if mode == 'train':
        data = train_data
    elif mode == 'test':
        data = test_data
    else:
        raise ValueError("Invalid mode. Must be 'train' or 'test'.")

    def generator():
        for _, row in data.iterrows():
            img = np.fromstring(row['pixels'], sep=' ').reshape(48, 48).astype(np.uint8)
            img = np.stack((img,) * 1, axis=-1)  # 转换为灰度图像
            img = tf.image.resize(img, [224, 224])  # 调整图像尺寸为 224x224
            label = row['emotion']
            yield img, label

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(224, 224, 1), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32)))

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, len(data)

# 训练线程类
class TrainThread_tensorflow(QThread):
    update_text = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        train_size,
        test_size,
        optimizer,
        num_epochs=25,
        learning_rate=0.001,
        batch_size=16,  # 添加批量大小参数
    ):
        super(TrainThread_tensorflow, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_size = train_size
        self.test_size = test_size
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # 保存批量大小
        self.stop_training = False
        self.paused = False
        self.save_checkpoint_flag = False
        self.save_best_model_flag = False

    def compile_model(self):
        if self.optimizer == 'Adam':
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'AdamW':
            optimizer = optimizers.AdamW(learning_rate=self.learning_rate)
        elif self.optimizer == 'Adamax':
            optimizer = optimizers.Adamax(learning_rate=self.learning_rate)
        elif self.optimizer == 'SGD':
            optimizer = optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)

        self.model.compile(optimizer=optimizer,
                           loss=losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train_step(self, images, labels):
        return self.model.train_on_batch(images, labels)

    def run(self):
        self.compile_model()

        best_accuracy = 0.0
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        for epoch in range(self.num_epochs):
            if self.stop_training:
                break
            while self.paused:
                self.msleep(100)
            self.update_text.emit(f"正在训练第 {epoch + 1} 轮...\n")

            running_loss = 0.0
            correct = 0
            total = 0
            num_batches = self.train_size // self.batch_size

            for i, (images, labels) in enumerate(self.train_dataset):
                if self.stop_training:
                    break
                while self.paused:
                    self.msleep(100)

                history = self.train_step(images, labels)
                loss = history[0].numpy()  # 确保在 tf.function 外部进行 NumPy 操作
                accuracy = history[1].numpy()  # 确保在 tf.function 外部进行 NumPy 操作

                running_loss += loss
                correct += accuracy * len(labels)
                total += len(labels)

                # 更新进度条
                progress = int((i + 1) / num_batches * 100)
                self.update_progress.emit(progress)

            epoch_loss = running_loss / num_batches
            epoch_accuracy = correct / total * 100
            epoch_info = f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%\n"
            self.update_text.emit(epoch_info)

            # 保存断点
            if self.save_checkpoint_flag:
                checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result_h5', 'checkpoint.weights.h5')
                self.model.save_weights(checkpoint_path)
                self.update_text.emit(f"Checkpoint saved at epoch {epoch + 1}.\n")
                self.save_checkpoint_flag = False  # 重置标志

            # 测试准确率和混淆矩阵
            test_loss, test_accuracy = self.model.evaluate(self.test_dataset, verbose=0)
            self.update_text.emit(f"Test Accuracy: {test_accuracy * 100:.4f}%\n")

            # 保存最优模型
            if self.save_best_model_flag and test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result_h5', 'best_model.weights.h5')
                self.model.save(best_model_path)
                self.update_text.emit(f"Best model saved with accuracy: {best_accuracy * 100:.4f}%\n")

            # 计算每个类别的 TP、FP、TN、FN
            y_true = []
            y_pred = []
            for images, labels in self.test_dataset:
                predictions = self.model.predict(images)
                y_true.extend(labels.numpy())  # 确保在 tf.function 外部进行 NumPy 操作
                y_pred.extend(np.argmax(predictions, axis=1))

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=7).numpy()

            for i in range(7):
                TP = confusion_matrix[i, i]
                FP = confusion_matrix[:, i].sum() - TP
                FN = confusion_matrix[i, :].sum() - TP
                TN = confusion_matrix.sum() - (TP + FP + FN)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                class_info = f"Class {i} ({emotion_labels[i]}): TP={TP}, FP={FP}, TN={TN}, FN={FN}\n"
                self.update_text.emit(class_info)
            overall_info = f"Overall: Accuracy={test_accuracy * 100:.4f}%, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}\n"
            self.update_text.emit(overall_info)

            # 创建保存 ROC 曲线的目录
            roc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ROC')
            if not os.path.exists(roc_dir):
                os.makedirs(roc_dir)

            # 绘制并保存 ROC 曲线
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6])
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(7):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred == i)
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