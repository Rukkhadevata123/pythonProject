import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from scipy.ndimage import label

app = QApplication(sys.argv)
win = QMainWindow()
win.setGeometry(400, 200, 300, 200)
win.setWindowTitle("Hello PyQt6")

label = QLabel("Hello PyQt6", win)
label.resize(200, 50)
label.setText("Hello this is PyQt6")
label.move(50, 50)

win.show()
sys.exit(app.exec())