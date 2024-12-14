import numpy as np
import os
from PIL import Image
from PyQt6.QtCore import pyqtSlot
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
        self.setFixedSize(self.width(), self.height())

        self.train_thread = None

        self.load.clicked.connect(self.load_data)
        self.start.clicked.connect(self.start_training)
        self.pause.clicked.connect(self.pause_training)
        self.resume.clicked.connect(self.resume_training)
        self.stop.clicked.connect(self.stop_training)
        self.save_checkpoint.clicked.connect(self.enable_save_checkpoint)
        self.save_best_model.clicked.connect(self.enable_save_best_model)
        self.final_test.clicked.connect(self.test_model)
        self.final_test.setEnabled(False)

    def setupButtonGroups(self):
        # 创建框架按钮组
        self.framework_group = QtWidgets.QButtonGroup(self)
        self.framework_group.addButton(self.pytorch)
        self.framework_group.addButton(self.tensorflow)
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
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "模型名称: MLP_1</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输入层: 输入尺寸=48x48x1</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层1: 输出=512, 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层2: 输出=128, 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输出层: 输出=7, 激活=Softmax</p>\n"
                                              "</body></html>"))
        elif model_name == "MLP_2":
            self.cur_model.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "hr { height: 1px; border-width: 0; }\n"
                                              "li.unchecked::marker { content: \"\\2610\"; }\n"
                                              "li.checked::marker { content: \"\\2612\"; }\n"
                                              "</style></head><body style=\" font-family:\'Ubuntu Sans\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "模型名称: MLP_2</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输入层: 输入尺寸=48x48x1</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层1: 输出=1024, 激活=ReLU, Dropout=0.5</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层2: 输出=512, 激活=ReLU, Dropout=0.5</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层3: 输出=128, 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输出层: 输出=7, 激活=Softmax</p>\n"
                                              "</body></html>"))
        elif model_name == "MLP_3":
            self.cur_model.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "hr { height: 1px; border-width: 0; }\n"
                                              "li.unchecked::marker { content: \"\\2610\"; }\n"
                                              "li.checked::marker { content: \"\\2612\"; }\n"
                                              "</style></head><body style=\" font-family:\'Ubuntu Sans\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "模型名称: MLP_3</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输入层: 输入尺寸=48x48x1</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层1: 输出=2048, 激活=ReLU, Dropout=0.5</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层2: 输出=1024, 激活=ReLU, Dropout=0.5</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层3: 输出=512, 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层4: 输出=128, 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输出层: 输出=7, 激活=Softmax</p>\n"
                                              "</body></html>"))
        elif model_name == "MLP_4":
            self.cur_model.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "hr { height: 1px; border-width: 0; }\n"
                                              "li.unchecked::marker { content: \"\\2610\"; }\n"
                                              "li.checked::marker { content: \"\\2612\"; }\n"
                                              "</style></head><body style=\" font-family:\'Ubuntu Sans\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "模型名称: MLP_4</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输入层: 输入尺寸=48x48x1</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层1: 输出=256, 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层2: 输出=128, 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "隐含层3: 输出=64, 激活=ReLU</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输出层: 输出=7, 激活=Softmax</p>\n"
                                              "</body></html>"))
        elif model_name == "CNN_1":
            self.cur_model.setHtml(_translate("MainWindow",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "hr { height: 1px; border-width: 0; }\n"
                                            "li.unchecked::marker { content: \"\\2610\"; }\n"
                                            "li.checked::marker { content: \"\\2612\"; }\n"
                                            "</style></head><body style=\" font-family:\'Ubuntu Sans\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "模型名称: CNN_1</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "输入层: 输入尺寸=48x48x1</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层1: 输出通道=32, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层1: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层2: 输出通道=64, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层2: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层3: 输出通道=128, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层3: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "全连接层1: 输出=512, 激活=ReLU, Dropout</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "全连接层2: 输出=128, 激活=ReLU, Dropout</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "输出层: 输出=7, 激活=Softmax</p>\n"
                                            "</body></html>"))
        elif model_name == "CNN_2":
            self.cur_model.setHtml(_translate("MainWindow",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "hr { height: 1px; border-width: 0; }\n"
                                            "li.unchecked::marker { content: \"\\2610\"; }\n"
                                            "li.checked::marker { content: \"\\2612\"; }\n"
                                            "</style></head><body style=\" font-family:\'Ubuntu Sans\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "模型名称: CNN_2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "输入层: 输入尺寸=48x48x1</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层1: 输出通道=32, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层2: 输出通道=32, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层1: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层3: 输出通道=64, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层4: 输出通道=64, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层2: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层5: 输出通道=128, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层6: 输出通道=128, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层3: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "全连接层1: 输出=512, 激活=ReLU, Dropout</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "全连接层2: 输出=128, 激活=ReLU, Dropout</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "输出层: 输出=7, 激活=Softmax</p>\n"
                                            "</body></html>"))
        elif model_name == "CNN_3":
            self.cur_model.setHtml(_translate("MainWindow",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "hr { height: 1px; border-width: 0; }\n"
                                            "li.unchecked::marker { content: \"\\2610\"; }\n"
                                            "li.checked::marker { content: \"\\2612\"; }\n"
                                            "</style></head><body style=\" font-family:\'Ubuntu Sans\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "模型名称: CNN_3</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "输入层: 输入尺寸=48x48x1</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层1: 输出通道=64, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层2: 输出通道=64, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层1: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层3: 输出通道=128, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层4: 输出通道=128, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层2: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层5: 输出通道=256, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "卷积层6: 输出通道=256, 卷积核=3x3, 激活=ReLU</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "池化层3: 池化核=2x2, 步幅=2</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "全连接层1: 输出=512, 激活=ReLU, Dropout</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "全连接层2: 输出=128, 激活=ReLU, Dropout</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                            "输出层: 输出=7, 激活=Softmax</p>\n"
                                            "</body></html>"))
        elif model_name == "CNN_4":
            self.cur_model.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "hr { height: 1px; border-width: 0; }\n"
                                              "li.unchecked::marker { content: \"\\2610\"; }\n"
                                              "li.checked::marker { content: \"\\2612\"; }\n"
                                              "</style></head><body style=\" font-family:\'Ubuntu Sans\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "模型名称: CNN_4</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输入层: 输入尺寸=48x48x1</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "卷积层1: 输出通道=64, 核大小=3x3, 填充=1, 激活=LeakyReLU(0.1), 批量规范化</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "卷积层2: 输出通道=64, 核大小=3x3, 填充=1, 激活=LeakyReLU(0.1), 批量规范化</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "池化层1: 核大小=2x2, 步幅=2</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "卷积层3: 输出通道=128, 核大小=3x3, 填充=1, 激活=LeakyReLU(0.1), 批量规范化</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "卷积层4: 输出通道=128, 核大小=3x3, 填充=1, 激活=LeakyReLU(0.1), 批量规范化</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "池化层2: 核大小=2x2, 步幅=2</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "卷积层5: 输出通道=256, 核大小=3x3, 填充=1, 激活=LeakyReLU(0.1), 批量规范化</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "卷积层6: 输出通道=256, 核大小=3x3, 填充=1, 激活=LeakyReLU(0.1), 批量规范化</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "池化层3: 核大小=2x2, 步幅=2</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "全连接层1: 输出=512, 激活=LeakyReLU(0.1), 批量规范化, Dropout</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "全连接层2: 输出=128, 激活=LeakyReLU(0.1), 批量规范化, Dropout</p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                                              "输出层: 输出=7, 激活=Softmax</p>\n"
                                              "</body></html>"))


    def setupModel(self):
        model_name = self.comboBox.currentText()
        framework = self.framework_group.checkedButton().text()

        if framework == "PyTorch":
            try:
                from models.models._MLP import MLP_1, MLP_2, MLP_3, MLP_4
                from models.models._CNN import CNN_1, CNN_2, CNN_3, CNN_4
            except ImportError as e:
                self.process_text.append(f"Failed to import PyTorch models: {e}\n")
                return

            if model_name == "MLP_1":
                self.model = MLP_1(num_classes=7)
            elif model_name == "MLP_2":
                self.model = MLP_2(num_classes=7)
            elif model_name == "MLP_3":
                self.model = MLP_3(num_classes=7)
            elif model_name == "MLP_4":
                self.model = MLP_4(num_classes=7)
            elif model_name == "CNN_1":
                self.model = CNN_1(num_classes=7)
            elif model_name == "CNN_2":
                self.model = CNN_2(num_classes=7)
            elif model_name == "CNN_3":
                self.model = CNN_3(num_classes=7)
            elif model_name == "CNN_4":
                self.model = CNN_4(num_classes=7)

        elif framework == "Tensorflow":
            try:
                from models.models._MLP import MLP_tf_1, MLP_tf_2, MLP_tf_3, MLP_tf_4
                from models.models._CNN import CNN_1_tf, CNN_2_tf, CNN_3_tf, CNN_4_tf
            except ImportError as e:
                self.process_text.append(f"Failed to import Tensorflow models: {e}\n")
                return

            if model_name == "MLP_1":
                self.model = MLP_tf_1(num_classes=7)
            elif model_name == "MLP_2":
                self.model = MLP_tf_2(num_classes=7)
            elif model_name == "MLP_3":
                self.model = MLP_tf_3(num_classes=7)
            elif model_name == "MLP_4":
                self.model = MLP_tf_4(num_classes=7)
            elif model_name == "CNN_1":
                self.model = CNN_1_tf(num_classes=7)
            elif model_name == "CNN_2":
                self.model = CNN_2_tf(num_classes=7)
            elif model_name == "CNN_3":
                self.model = CNN_3_tf(num_classes=7)
            elif model_name == "CNN_4":
                self.model = CNN_4_tf(num_classes=7)

        elif framework == "PaddlePaddle":
            try:
                from models.models._MLP import MLP_1_pp, MLP_2_pp, MLP_3_pp, MLP_4_pp
                from models.models._CNN import CNN_1_pp, CNN_2_pp, CNN_3_pp, CNN_4_pp
            except ImportError as e:
                self.process_text.append(f"Failed to import PaddlePaddle models: {e}\n")
                return

            if model_name == "MLP_1":
                self.model = MLP_1_pp(num_classes=7)
            elif model_name == "MLP_2":
                self.model = MLP_2_pp(num_classes=7)
            elif model_name == "MLP_3":
                self.model = MLP_3_pp(num_classes=7)
            elif model_name == "MLP_4":
                self.model = MLP_4_pp(num_classes=7)
            elif model_name == "CNN_1":
                self.model = CNN_1_pp(num_classes=7)
            elif model_name == "CNN_2":
                self.model = CNN_2_pp(num_classes=7)
            elif model_name == "CNN_3":
                self.model = CNN_3_pp(num_classes=7)
            elif model_name == "CNN_4":
                self.model = CNN_4_pp(num_classes=7)

    @pyqtSlot()
    def load_data(self):
        self.setupModel()
        self.final_test.setEnabled(True)

        try:
            csv_file = 'csvs/fer2013.csv'

            try:
                test_size = float(self.test_text.text())
                self.test_text.setReadOnly(True)
                if not (0 < test_size < 1):
                    raise ValueError
            except ValueError:
                self.process_text.append(
                    "Invalid test size. Please enter a float between 0 and 1.\n")
                self.test_text.setText("0.2")
                self.test_text.setReadOnly(False)
                return
            self.train_text.setText(f"{1 - test_size:.2f}")

            try:
                batch_size = int(self.batch_text.text())
                self.batch_text.setReadOnly(True)
                if batch_size <= 0:
                    raise ValueError
            except ValueError:
                self.process_text.append(
                    "Invalid batch size. Please enter a valid positive integer.\n")
                self.batch_text.setText("32")
                self.batch_text.setReadOnly(False)
                return

            try:
                num_epochs = int(self.epoch_text.text())
                self.epoch_text.setReadOnly(True)
                if num_epochs <= 1:
                    raise ValueError
            except ValueError:
                self.process_text.append(
                    "Invalid number of epochs. Please enter an integer greater than 1.\n")
                self.epoch_text.setText("25")
                self.epoch_text.setReadOnly(False)
                return

            try:
                learning_rate = float(self.lr_text.text())
                self.lr_text.setReadOnly(True)
                if learning_rate <= 0:
                    raise ValueError
            except ValueError:
                self.process_text.append(
                    "Invalid learning rate. Please enter a float greater than 0.\n")
                self.lr_text.setText("0.001")
                self.lr_text.setReadOnly(False)
                return

            framework = self.framework_group.checkedButton().text()
            if framework == "PyTorch":
                try:
                    from models.framework._pytorch import FER2013Dataset_pytorch, transform_pytorch, transform_pytorch_test
                    from torch.utils.data import DataLoader
                except ImportError as e:
                    self.process_text.append(f"Failed to import PyTorch dataset: {e}\n")
                    return

                # 加载和划分数据集
                train_dataset = FER2013Dataset_pytorch(csv_file=csv_file, self_transform=transform_pytorch, mode='train', test_size=test_size)
                test_dataset = FER2013Dataset_pytorch(csv_file=csv_file, self_transform=transform_pytorch_test, mode='test', test_size=test_size)

                # 创建训练数据加载器
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # 创建测试数据加载器
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            elif framework == "Tensorflow":
                try:
                    from models.framework._tensorflow import FER2013Dataset_tf
                except ImportError as e:
                    self.process_text.append(f"Failed to import Tensorflow dataset: {e}\n")
                    return
            
                # 加载和划分数据集
                self.train_loader = FER2013Dataset_tf(csv_file=csv_file, mode='train', batch_size=batch_size, test_size=test_size, shuffle=True)
                self.test_loader = FER2013Dataset_tf(csv_file=csv_file, mode='test', batch_size=batch_size, test_size=test_size, shuffle=False)

            elif framework == "PaddlePaddle":
                try:
                    from models.framework._paddlepaddle import FER2013Dataset_paddle, transform_paddle, transform_paddle_test
                    from paddle.io import DataLoader as pDataLoader
                except ImportError as e:
                    self.process_text.append(f"Failed to import PaddlePaddle dataset: {e}\n")
                    return

                train_dataset = FER2013Dataset_paddle(csv_file=csv_file, transform=transform_paddle, mode='train', test_size=test_size)
                test_dataset = FER2013Dataset_paddle(csv_file=csv_file, transform=transform_paddle_test, mode='test', test_size=test_size)

                self.train_loader = pDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                self.test_loader = pDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # 禁用框架按钮组中的所有按钮
            for button in self.framework_group.buttons():
                button.setEnabled(False)

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
            framework = self.framework_group.checkedButton().text()
            if framework == "PyTorch":
                try:
                    from models.framework._pytorch import TrainThread_pytorch
                    import torch
                except ImportError as e:
                    self.process_text.append(f"Failed to import PyTorch training thread: {e}\n")
                    return

                self.train_thread = TrainThread_pytorch(
                    model=self.model,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader,
                    optimizer=self.optimizer_group.checkedButton().text(),
                    num_epochs=int(self.epoch_text.text()),
                    learning_rate=float(self.lr_text.text())
                )
                self.gpu_true.setText(str(torch.cuda.is_available()))
            elif framework == "Tensorflow":
                try:
                    from models.framework._tensorflow import TrainThread_tensorflow
                    import tensorflow as tf
                except ImportError as e:
                    self.process_text.append(f"Failed to import Tensorflow training thread: {e}\n")
                    return

                self.train_thread = TrainThread_tensorflow(
                    model=self.model,
                    train_dataset=self.train_loader,
                    test_dataset=self.test_loader,
                    optimizer=self.optimizer_group.checkedButton().text(),
                    num_epochs=int(self.epoch_text.text()),
                    learning_rate=float(self.lr_text.text())
                )
                self.gpu_true.setText(str(tf.test.is_gpu_available()))

            elif framework == "PaddlePaddle":
                try:
                    import paddle
                    from models.framework._paddlepaddle import TrainThread_paddle
                except ImportError as e:
                    self.process_text.append(f"Failed to import PaddlePaddle training thread: {e}\n")
                    return

                self.train_thread = TrainThread_paddle(
                    model=self.model,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader,
                    optimizer=self.optimizer_group.checkedButton().text(),
                    num_epochs=int(self.epoch_text.text()),
                    learning_rate=float(self.lr_text.text())
                )
                self.gpu_true.setText(str(paddle.is_compiled_with_cuda()))
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
            # 恢复框架按钮组中的所有按钮
            for button in self.framework_group.buttons():
                button.setEnabled(True)
            self.test_text.setReadOnly(False)
            self.batch_text.setReadOnly(False)
            self.epoch_text.setReadOnly(False)
            self.lr_text.setReadOnly(False)

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
    
        framework = self.framework_group.checkedButton().text()
        if framework == "PyTorch":
            best_model_path = 'models/result_pth/best_model.pth'
            checkpoint_path = 'models/result_pth/checkpoint.pth'
        elif framework == "Tensorflow":
            best_model_path = 'models/result_h5/best_model.keras'
            checkpoint_path = 'models/result_h5/checkpoint.weights.h5'
        elif framework == "PaddlePaddle":
            best_model_path = 'models/result_paddle/best_model.pdparams'
            checkpoint_path = 'models/result_paddle/checkpoint.pdparams'
        else:
            self.process_text.append("Unsupported framework for testing.\n")
            return
    
        model_path = None
    
        if os.path.exists(best_model_path):
            model_path = best_model_path
            self.process_text.append(f"Using {best_model_path} for testing...\n")
        elif os.path.exists(checkpoint_path):
            model_path = checkpoint_path
            self.process_text.append(f"Using {checkpoint_path} for testing...\n")
        else:
            self.process_text.append("No model found for testing.\n")
            return
    
        to_be_predicted_dir = 'to_be_predicted'
        if not os.path.exists(to_be_predicted_dir):
            os.makedirs(to_be_predicted_dir)
            self.process_text.append("Please put images to be predicted in the to_be_predicted folder.\n")
            return
    
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
        if framework == "PyTorch":
            try:
                import torch
                from torchvision import transforms
            except ImportError as e:
                self.process_text.append(f"Failed to import PyTorch: {e}\n")
                return
    
            if model_path == checkpoint_path:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(torch.load(model_path))
    
            transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
    
            images_found = False
            for img_name in os.listdir(to_be_predicted_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images_found = True
                    img_path = os.path.join(to_be_predicted_dir, img_name)
                    img = Image.open(img_path).convert('L')
                    img = transform(img).unsqueeze(0).to(device)
                    outputs = self.model(img)
                    _, predicted = torch.max(outputs.data, 1)
                    emotion = emotion_labels[predicted.item()]
                    self.process_text.append(f"Image: {img_name}, Predicted Emotion: {emotion}\n")
    
            if not images_found:
                self.process_text.append("No images found in the to_be_predicted folder.\n")
    
        elif framework == "Tensorflow":
            try:
                import tensorflow as tf
            except ImportError as e:
                self.process_text.append(f"Failed to import Tensorflow: {e}\n")
                return
    
            model = self.model
            model.load_weights(model_path)
    
            images_found = False
            for img_name in os.listdir(to_be_predicted_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images_found = True
                    img_path = os.path.join(to_be_predicted_dir, img_name)
                    img = Image.open(img_path).convert('L')
                    img = img.resize((48, 48))
                    img = np.array(img).reshape((1, 48, 48, 1)) / 255.0
                    predictions = model.predict(img)
                    predicted = np.argmax(predictions, axis=1)
                    emotion = emotion_labels[predicted[0]]
                    self.process_text.append(f"Image: {img_name}, Predicted Emotion: {emotion}\n")
    
            if not images_found:
                self.process_text.append("No images found in the to_be_predicted folder.\n")
    
        elif framework == "PaddlePaddle":
            try:
                import paddle
                import paddle.vision.transforms as T
            except ImportError as e:
                self.process_text.append(f"Failed to import PaddlePaddle: {e}\n")
                return
    
            model = self.model
            model.set_state_dict(paddle.load(model_path))
    
            transform = T.Compose([
                T.Resize((48, 48)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
            device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
            model.to(device)
            model.eval()
    
            images_found = False
            for img_name in os.listdir(to_be_predicted_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images_found = True
                    img_path = os.path.join(to_be_predicted_dir, img_name)
                    img = Image.open(img_path).convert('L')
                    img = transform(img).unsqueeze(0).to(device)
                    outputs = model(img)
                    predicted = paddle.argmax(outputs, axis=1)
                    emotion = emotion_labels[predicted.item()]
                    self.process_text.append(f"Image: {img_name}, Predicted Emotion: {emotion}\n")
    
            if not images_found:
                self.process_text.append("No images found in the to_be_predicted folder.\n")
    
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
