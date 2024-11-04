import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt

# 定义主窗口类，继承自QMainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle("简单的 PyQt6 界面")
        # 设置窗口大小和位置
        self.setGeometry(100, 100, 280, 80)

        # 创建一个标签，并设置其对齐方式为居中
        self.label = QLabel("Hello, PyQt6!", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 创建一个按钮，并连接其点击事件到自定义的槽函数
        self.button = QPushButton("点击我", self)
        self.button.clicked.connect(self.on_button_click)

        # 创建一个垂直布局，并将标签和按钮添加到布局中
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        # 创建一个容器部件，并设置其布局
        container = QWidget()
        container.setLayout(layout)
        # 将容器部件设置为主窗口的中央部件
        self.setCentralWidget(container)

    # 定义按钮点击事件的槽函数
    def on_button_click(self):
        # 修改标签的文本
        self.label.setText("按钮被点击了!")

# 主程序入口
if __name__ == "__main__":
    # 创建QApplication对象
    app = QApplication(sys.argv)
    # 创建主窗口对象
    window = MainWindow()
    # 显示主窗口
    window.show()
    # 进入应用程序的主循环
    sys.exit(app.exec())