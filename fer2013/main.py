import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from PyQt6.QtCore import pyqtSlot
from models.framework._pytorch import TrainThread_pytorch, FER2013Dataset_pytorch, transform_pytorch
from models.models._MLP import MLP_1, MLP_2
# from models.models._CNN import CNN_1, CNN_2, CNN_3, CNN_4

from PyQt6 import QtWidgets, QtCore
from ui.fer2013_ui import Ui_MainWindow

current_dir = os.path.dirname(os.path.abspath(__file__))

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setupButtonGroups()
        self.setupResetButtons()
        self.setupComboBox()

        self.train_thread = None

        self.load.clicked.connect(self.load_data)
        self.start.clicked.connect(self.start_training)
        self.pause.clicked.connect(self.pause_training)
        self.resume.clicked.connect(self.resume_training)
        self.stop.clicked.connect(self.stop_training)
        self.save_checkpoint.clicked.connect(self.enable_save_checkpoint)
        self.save_best_model.clicked.connect(self.enable_save_best_model)
        self.final_test.clicked.connect(self.test_model)

    def setupButtonGroups(self):
        # 创建框架按钮组
        self.framework_group = QtWidgets.QButtonGroup(self)
        self.framework_group.addButton(self.pytorch)
        self.framework_group.addButton(self.tensorflow)
        self.framework_group.addButton(self.sklearn)
        self.framework_group.addButton(self.paddlepaddle)
        self.framework_group.setExclusive(True)
        self.pytorch.setChecked(True)  # 默认选中 PyTorch

        # 创建优化器按钮组
        self.optimizer_group = QtWidgets.QButtonGroup(self)
        self.optimizer_group.addButton(self.adam)
        self.optimizer_group.addButton(self.adam8)
        self.optimizer_group.addButton(self.sgd)
        self.optimizer_group.addButton(self.adagrad)
        self.optimizer_group.addButton(self.momentum)
        self.optimizer_group.setExclusive(True)
        self.adam.setChecked(True)  # 默认选中 Adam

    def setupResetButtons(self):
        # 设置默认值
        self.default_lr = "0.001"
        self.default_batch = "32"
        self.default_epoch = "25"

        # 连接按钮点击事件到重置函数
        self.resume_lr.clicked.connect(self.reset_lr)
        self.resume_batch.clicked.connect(self.reset_batch)
        self.resume_epoch.clicked.connect(self.reset_epoch)

    def reset_lr(self):
        self.lr_text.setText(self.default_lr)

    def reset_batch(self):
        self.batch_text.setText(self.default_batch)

    def reset_epoch(self):
        self.epoch_text.setText(self.default_epoch)

    def setupComboBox(self):
        self.comboBox.currentIndexChanged.connect(self.updateModelDescription)

    def updateModelDescription(self):
        model_name = self.comboBox.currentText()
        _translate = QtCore.QCoreApplication.translate
        if model_name == "MLP_1":
            self.cur_model.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "hr { height: 1px; border-width: 0; }\n"
                                              "li.unchecked::marker { content: \"\\2610\"; }\n"
                                              "li.checked::marker { content: \"\\2612\"; }\n"
                                              "</style></head><body style=\" font-family:\'Ubuntu Sans\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">隐含层 输出=64 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">隐含层 输出=64 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">隐含层 输出=64 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">输出层 输出=7 激活=Softmax</p></body></html>"))
        elif model_name == "MLP_2":
            self.cur_model.setHtml("# TODO")
        elif model_name == "MLP_3":
            self.cur_model.setHtml("# TODO3")
        elif model_name == "MLP_4":
            self.cur_model.setHtml("# TODO4")
        elif model_name == "CNN_1":
            self.cur_model.setHtml("# TODO5")
        elif model_name == "CNN_2":
            self.cur_model.setHtml("# TODO6")
        elif model_name == "CNN_3":
            self.cur_model.setHtml("# TODO7")
        elif model_name == "CNN_4":
            self.cur_model.setHtml("# TODO8")

    def setupModel(self):
        model_name = self.comboBox.currentText()
        if model_name == "MLP_1":
            self.model = MLP_1(num_classes=7)
        elif model_name == "MLP_2":
            self.model = MLP_2(num_classes=7)
        # elif model_name == "MLP_3":
        #     self.model = MLP_3(num_classes=7)
        # elif model_name == "MLP_4":
        #     self.model = MLP_4(num_classes=7)
        # elif model_name == "CNN_1":
        #     self.model = CNN_1(num_classes=7)
        # elif model_name == "CNN_2":
        #     self.model = CNN_2(num_classes=7)
        # elif model_name == "CNN_3":
        #     self.model = CNN_3(num_classes=7)
        # elif model_name == "CNN_4":
        #     self.model = CNN_4(num_classes=7)

    @pyqtSlot()
    def load_data(self):
        self.setupModel()
        try:
            csv_file = 'csvs/fer2013.csv'

            try:
                test_size = float(self.test_text.text())
                if not (0 < test_size < 1):
                    raise ValueError
            except ValueError:
                self.process_text.append("Invalid test size. Please enter a float between 0 and 1.\n")
                return
            self.train_text.setText(str(1 - test_size))

            try:
                batch_size = int(self.batch_text.text())
                if batch_size <= 0:
                    raise ValueError
            except ValueError:
                self.process_text.append("Invalid batch size. Please enter a valid positive integer.\n")
                return

            try:
                num_epochs = int(self.epoch_text.text())
                if num_epochs <= 1:
                    raise ValueError
            except ValueError:
                self.process_text.append("Invalid number of epochs. Please enter an integer greater than 1.\n")
                return

            try:
                learning_rate = float(self.lr_text.text())
                if learning_rate <= 0:
                    raise ValueError
            except ValueError:
                self.process_text.append("Invalid learning rate. Please enter a float greater than 0.\n")
                return

            if self.framework_group.checkedButton().text() == "PyTorch":
                # 加载和划分数据集
                dataset = FER2013Dataset_pytorch(csv_file=csv_file, self_transform=transform_pytorch, test_size=test_size)

                # 设置为训练模式并创建训练数据加载器
                dataset.set_mode('train')
                self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # 设置为测试模式并创建测试数据加载器
                dataset.set_mode('test')
                self.test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            elif self.framework_group.checkedButton().text() == "TensorFlow":
                pass
            #   TODO
            elif self.framework_group.checkedButton().text() == "Scikit-Learn":
                pass
            #   TODO
            elif self.framework_group.checkedButton().text() == "PaddlePaddle":
                pass
            #   TODO

            self.process_text.append("Data loaded successfully.\n")
            self.load.setEnabled(False)
        except Exception as e:
            self.process_text.append(f"Failed to load data: {e}\n")

    @pyqtSlot()
    def start_training(self):
        if not hasattr(self, 'train_loader') or not hasattr(self, 'test_loader'):
            self.process_text.append("Please load data first.\n")
            return

        if self.train_thread is None or not self.train_thread.isRunning():
            self.process_text.append("Training started...\n")
            if self.framework_group.checkedButton().text() == "PyTorch":
                self.train_thread = TrainThread_pytorch(
                    model=self.model,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader,
                    optimizer=self.optimizer_group.checkedButton().text(),
                    num_epochs=int(self.epoch_text.text()),
                    learning_rate=float(self.lr_text.text())
                )
                self.gpu_true = torch.cuda.is_available()
            elif self.framework_group.checkedButton().text() == "TensorFlow":
                pass
            #   TODO
            elif self.framework_group.checkedButton().text() == "Scikit-Learn":
                pass
            #   TODO
            elif self.framework_group.checkedButton().text() == "PaddlePaddle":
                pass
            #   TODO
            self.train_thread.update_text.connect(self.append_text)
            self.train_thread.update_progress.connect(self.update_progress)
            self.train_thread.start()

    @pyqtSlot()
    def pause_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.process_text.append("Training paused...\n")
            self.train_thread.pause()

    @pyqtSlot()
    def resume_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.process_text.append("Training resumed...\n")
            self.train_thread.resume()

    @pyqtSlot()
    def stop_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.process_text.append("Training stopped...\n")
            self.train_thread.stop()
            self.load.setEnabled(True)

    @pyqtSlot()
    def enable_save_checkpoint(self):
        if self.train_thread and self.train_thread.isRunning():
            self.process_text.append("Checkpoint saving enabled for the next epoch...\n")
            self.train_thread.enable_save_checkpoint()

    @pyqtSlot()
    def enable_save_best_model(self):
        if self.train_thread and self.train_thread.isRunning():
            self.process_text.append("Best model saving enabled...\n")
            self.train_thread.enable_save_best_model()

    @pyqtSlot()
    def test_model(self):
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.pause()
            self.process_text.append("Training paused for model testing...\n")

        best_model_path = 'models/result_pth/best_model.pth'
        checkpoint_path = 'models/result_pth/checkpoint.pth'
        model_path = None

        if os.path.exists(best_model_path):
            model_path = best_model_path
            self.process_text.append("Using best_model.pth for testing...\n")
        elif os.path.exists(checkpoint_path):
            model_path = checkpoint_path
            self.process_text.append("Using checkpoint.pth for testing...\n")
        else:
            self.process_text.append("No model found for testing.\n")
            return

        to_be_predicted_dir = 'to_be_predicted'
        if not os.path.exists(to_be_predicted_dir):
            os.makedirs(to_be_predicted_dir)
            self.process_text.append("Please put images to be predicted in the to_be_predicted folder.\n")
            return
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        
        if self.framework_group.checkedButton().text() == "PyTorch":
            if model_path == checkpoint_path:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(torch.load(model_path))

            transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()

            images_found = False
            for img_name in os.listdir(to_be_predicted_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images_found = True
                    img_path = os.path.join(to_be_predicted_dir, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = self.model(img)
                        _, predicted = torch.max(outputs, 1)
                        self.process_text.append(f"Prediction for {img_name}: {emotion_labels[predicted.item()]}\n")
            if not images_found:
                self.process_text.append("No images found for prediction.\n")

            if self.train_thread and self.train_thread.isRunning():
                self.train_thread.resume()
                self.process_text.append("Training resumed...\n")

    def append_text(self, text):
        self.process_text.append(text)

    def update_progress(self, value):
        self.progressBar.setValue(value)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    window = MainWindow()
    window.show()
    #    print(QStyleFactory.keys())
    sys.exit(app.exec())
