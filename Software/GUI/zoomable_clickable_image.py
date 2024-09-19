from PyQt6.QtWidgets import QLabel, QMenu
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QAction
from PyQt6.QtCore import Qt, QRect, QPoint, QSize

import platform
import os
import sys
if platform.system() == 'Windows':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Software')))
else:
    # On macOS or other Unix-like systems, keep the original path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.Hypercube_Spectrum.Spectrum_plot import plot_spectrum
from PyQt6.QtWidgets import QLabel, QMenu, QWidget
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent, QAction, QMouseEvent
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal

class ZoomableClickableImage(QWidget):
    right_clicked = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.scale = 1.0
        self.pos = QPoint(0, 0)
        self.mouse_pressed = False
        self.last_mouse_pos = QPoint(0, 0)
        self.setMouseTracking(True)

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            
            scaled_pixmap = self.pixmap.scaled(
                self.pixmap.size() * self.scale,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            paint_x = self.pos.x() + (self.width() - scaled_pixmap.width()) / 2
            paint_y = self.pos.y() + (self.height() - scaled_pixmap.height()) / 2
            
            painter.drawPixmap(QPoint(int(paint_x), int(paint_y)), scaled_pixmap)

    def wheelEvent(self, event: QWheelEvent):
        old_pos = event.position()
        old_scale = self.scale
        
        if event.angleDelta().y() > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        
        self.scale = max(0.1, min(10.0, self.scale))
        
        new_pos = event.position()
        moved = new_pos - old_pos
        self.pos += moved.toPoint() * self.scale - moved.toPoint() * old_scale
        
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_pos = event.pos()
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.mouse_pressed:
            delta = event.pos() - self.last_mouse_pos
            self.pos += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = False

    def reset_view(self):
        self.scale = 1.0
        self.pos = QPoint(0, 0)
        self.update()