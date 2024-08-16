import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Visual Function")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel("please choose HSI file", self)
        self.label.setGeometry(50, 50, 700, 400)
        self.label.setScaledContents(True)

        self.open_button = QPushButton("Open", self)
        self.open_button.setGeometry(50, 280, 100, 30)
        self.open_button.clicked.connect(self.open_file)

        self.rgb_button = QPushButton("RGB", self)
        self.rgb_button.setGeometry(200, 500, 100, 30)
        self.rgb_button.clicked.connect(self.show_rgb)

        # 其他按钮
        self.ndvi_button = QPushButton("NDVI", self)
        self.ndvi_button.setGeometry(350, 500, 100, 30)
        self.ndvi_button.clicked.connect(self.show_ndvi)

        self.evi_button = QPushButton("EVI", self)
        self.evi_button.setGeometry(500, 500, 100, 30)
        self.evi_button.clicked.connect(self.show_evi)

        self.mcari_button = QPushButton("MCARI", self)
        self.mcari_button.setGeometry(650, 500, 100, 30)
        self.mcari_button.clicked.connect(self.show_mcari)

        # 其他可视化方案按钮可以按类似方式添加

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open", "", "(*.hdr *.bli)")
        if file_name:
            self.label.setPixmap(QPixmap(file_name))
            self.image_path = file_name

    def show_rgb(self):
        # 这里添加显示RGB图像的代码
        pass

    def show_ndvi(self):
        # 这里添加显示NDVI图像的代码
        pass

    def show_evi(self):
        # 这里添加显示EVI图像的代码
        pass

    def show_mcari(self):
        # 这里添加显示MCARI图像的代码
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())