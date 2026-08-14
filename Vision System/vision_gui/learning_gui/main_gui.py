# from PySide6.QtWidgets import QWidget, QApplication, QPushButton

# app = QApplication()

# window = QWidget()
# window.show()

# button = QPushButton("Click Me")


# import sys

from PySide6.QtWidgets import QApplication, QWidget


app = QApplication()

window = QWidget()
window.setWindowTitle("My first GUI")
window.resize(400, 300)

window.show()
app.exec()
# sys.exit(app.exec())