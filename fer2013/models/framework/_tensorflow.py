import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
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
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=4096)])  # 例如，设置为 4096MB
    except RuntimeError as e:
        print(e)

# 不启用设备日志记录
# tf.debugging.set_log_device_placement(True)

# 启动实验性NumPy功能
tf.experimental.numpy.experimental_enable_numpy_behavior()


# 自定义数据集类
class FER2013Dataset_tf(tf.keras.utils.Sequence):
    def __init__(self,
                 csv_file,
                 mode='train',
                 batch_size=32,
                 test_size=0.2,
                 shuffle=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 将数据划分为训练集和测试集
        train_data, test_data = train_test_split(self.data,
                                                 test_size=test_size,
                                                 shuffle=True)
        self.train_data = train_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)

        # 数据增强和预处理
        if mode == 'train':
            self.datagen = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True,
                rotation_range=10
            )
        else:
            self.datagen = ImageDataGenerator(rescale=1./255)

        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        if self.mode == 'train':
            return int(np.ceil(len(self.train_data) / self.batch_size))
        elif self.mode == 'test':
            return int(np.ceil(len(self.test_data) / self.batch_size))
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'test'.")

    def __getitem__(self, idx):
        if self.mode == 'train':
            batch_data = self.train_data.iloc[
                idx * self.batch_size:(idx + 1) * self.batch_size]
        elif self.mode == 'test':
            batch_data = self.test_data.iloc[
                idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'test'.")

        images = []
        labels = []
        for _, row in batch_data.iterrows():
            pixels = row['pixels']
            img = np.fromstring(pixels, sep=' ')
            img = img.reshape(48, 48).astype(np.uint8)
            img = np.expand_dims(img, axis=-1)  # 保持单通道
            img = tf.image.resize(img, [224, 224])  # 调整图像尺寸为 224x224
            images.append(img)
            labels.append(row['emotion'])

        images = np.array(images)
        labels = np.array(labels)
        if images.size == 0 or labels.size == 0:
            return self.__getitem__(
                (idx + 1) % self.__len__()
                )  # 跳过空批次，继续处理下一个批次
        labels = tf.keras.utils.to_categorical(labels,
                                               num_classes=7
                                               )  # 转换为 one-hot 编码
        return next(self.datagen.flow(images,
                                      labels,
                                      batch_size=self.batch_size
                                      )
                    )

    def on_epoch_end(self):
        if self.shuffle:
            if self.mode == 'train':
                self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)
            elif self.mode == 'test':
                self.test_data = self.test_data.sample(frac=1).reset_index(drop=True)


# 自定义回调函数
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, update_text_signal):
        super().__init__()
        self.test_dataset = test_dataset
        self.update_text_signal = update_text_signal

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        for i in range(len(self.test_dataset)):
            images, labels = self.test_dataset[i]
            predictions = self.model.predict(images)
            y_true.extend(np.argmax(labels, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_true, y_pred)

        # 提取TP, FP, TN, FN
        TP = np.diag(conf_matrix)
        FP = conf_matrix.sum(axis=0) - TP
        FN = conf_matrix.sum(axis=1) - TP
        TN = conf_matrix.sum() - (FP + FN + TP)

        # 计算准确率、精确率、召回率和F1值
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = np.divide(TP,
                              (TP + FP),
                              out=np.zeros_like(TP, dtype=float),
                              where=(TP + FP) != 0
                              )
        recall = np.divide(TP,
                           (TP + FN),
                           out=np.zeros_like(TP, dtype=float),
                           where=(TP + FN) != 0
                           )
        f1_score = np.divide(2 * (precision * recall),
                             (precision + recall),
                             out=np.zeros_like(precision, dtype=float),
                             where=(precision + recall) != 0
                             )

        # 输出每种情绪的TP, FP, TN, FN以及准确率、精确率、召回率和F1值
        for i in range(7):
            class_info = (
                f"Class {i} - TP: {TP[i]}, FP: {FP[i]}, TN: {TN[i]}, FN: {FN[i]}, "
                f"Accuracy: {accuracy[i]:.4f}, Precision: {precision[i]:.4f}, "
                f"Recall: {recall[i]:.4f}, F1 Score: {f1_score[i]:.4f}"
            )
            print(class_info)
            self.update_text_signal.emit(class_info + "\n")

        # 绘制并保存ROC曲线
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5, 6])

        plt.figure()
        for i in range(7):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC Curve")
        plt.legend(loc='lower right')
        roc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..',
                               'ROC'
                               )
        if not os.path.exists(roc_dir):
            os.makedirs(roc_dir)
        plt.savefig(os.path.join(roc_dir, "roc_curve.png"))
        plt.close()


# 训练线程类
class TrainThread_tensorflow(QThread):
    update_text = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        optimizer,
        num_epochs=25,
        learning_rate=0.001,
    ):
        super(TrainThread_tensorflow, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
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
            optimizer = optimizers.SGD(learning_rate=self.learning_rate,
                                       momentum=0.9
                                       )
        else:
            optimizer = None

        self.model.compile(optimizer=optimizer,
                           loss=losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def run(self):
        self.compile_model()
        best_accuracy = 0.0
        # 定义模型检查点回调函数
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'result_h5',
            'checkpoint.weights.h5'
        )
        best_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'result_h5',
            'best_model.keras'
        )

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1,
            save_freq='epoch')  # 每个 epoch 保存一次

        best_model_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1)

        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self,
                         total_batches,
                         update_progress_signal,
                         train_thread
                         ):
                super().__init__()
                self.total_batches = total_batches
                self.update_progress_signal = update_progress_signal
                self.train_thread = train_thread

            def on_batch_end(self, batch, logs=None):
                while self.train_thread.paused:
                    self.train_thread.msleep(100)
                if self.train_thread.stop_training:
                    self.model.stop_training = True
                progress = int((batch + 1) / self.total_batches * 100)
                self.update_progress_signal.emit(progress)

        total_batches = len(self.train_dataset)
        progress_callback = ProgressCallback(
            total_batches,
            self.update_progress,
            self
            )

        # 添加 ConfusionMatrixCallback 回调
        confusion_matrix_callback = ConfusionMatrixCallback(
            self.test_dataset,
            self.update_text
            )

        for epoch in range(self.num_epochs):
            if self.stop_training:
                break
            self.update_text.emit(f"正在训练第 {epoch + 1} 轮...\n")

            # 训练一个 epoch
            history = self.model.fit(self.train_dataset,
                                     epochs=1,
                                     validation_data=self.test_dataset,
                                     callbacks=[cp_callback,
                                                best_model_callback,
                                                progress_callback,
                                                confusion_matrix_callback
                                                ]
                                     )

            # 更新最佳准确率
            val_accuracy = history.history['val_accuracy'][-1]
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # 保存断点
            if self.save_checkpoint_flag:
                self.model.save_weights(checkpoint_path)
                self.update_text.emit(
                    f"Checkpoint saved at epoch {epoch + 1}.\n"
                    )
                self.save_checkpoint_flag = False  # 重置标志

            # 测试准确率
            test_loss, test_accuracy = self.model.evaluate(
                self.test_dataset,
                verbose=0
                )
            self.update_text.emit(
                f"Test Accuracy: {test_accuracy * 100:.4f}%\n"
                )

            # 保存最优模型
            if self.save_best_model_flag and test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                self.model.save(best_model_path)
                self.update_text.emit(
                    f"Best model saved with accuracy: {best_accuracy * 100:.4f}%\n"
                    )

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