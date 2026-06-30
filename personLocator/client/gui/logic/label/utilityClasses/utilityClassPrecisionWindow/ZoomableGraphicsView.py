from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class ZoomableGraphicsView(QGraphicsView):
    """Grafik-Ansicht mit Unterstützung für Mausrad-Zoom und Klick-Koordinaten."""
    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal()
    mouse_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet("background-color: #050505; border: 1px solid #333;")
        self.setMouseTracking(True)
        self.first_frame = True

    def set_pixmap(self, pixmap: QPixmap):
        """Setzt das angezeigte Bild und passt die Ansicht an, wenn es das erste Bild ist."""
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        if self.first_frame:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.first_frame = False

    def wheelEvent(self, event):
        """Zoomt die Ansicht basierend auf der Mausradbewegung."""
        zoom_factor = 1.15
        if event.angleDelta().y() < 0:
            zoom_factor = 1 / zoom_factor
        self.scale(zoom_factor, zoom_factor)

    def mouseMoveEvent(self, event):
        """Sendet die aktuellen Mauskoordinaten im Bild an das Hauptfenster, um die Lupen-Position zu aktualisieren."""
        scene_pos = self.mapToScene(event.pos())
        self.mouse_moved.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        """Sendet die Klick-Koordinaten oder das Rechtsklick-Signal an das Hauptfenster."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.left_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)