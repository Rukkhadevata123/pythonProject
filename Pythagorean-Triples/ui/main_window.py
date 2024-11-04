from PyQt6.QtWidgets import QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QTextEdit
import sys

sys.path.append('..')
from pythagorean import generate_pythagorean_triples


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pythagorean Triples Generator")

        self.label = QLabel("Enter a positive integer:")
        self.input_field = QLineEdit()
        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_triples)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.result_text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def generate_triples(self) -> None:
        try:
            x = int(self.input_field.text())
            if x <= 0:
                raise ValueError
            triples = generate_pythagorean_triples(x)
            result_text = "\n".join([str(triple) for triple in triples])
            self.result_text_edit.setText(result_text)
        except ValueError:
            self.result_text_edit.setText("Please enter a positive integer.")