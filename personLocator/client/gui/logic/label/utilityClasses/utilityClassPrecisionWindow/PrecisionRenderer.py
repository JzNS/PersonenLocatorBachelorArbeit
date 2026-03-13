from PyQt6.QtCore import QRect, QPoint, Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush


class PrecisionRenderer:
    @staticmethod
    def create_loupe_pixmap(image: QImage, x: int, y: int) -> QPixmap:
        """Erzeugt das vergrößerte Lupen-Bild mit Fadenkreuz."""
        crop_rect = QRect(x - 20, y - 20, 40, 40)
        scaled_crop = QPixmap.fromImage(image.copy(crop_rect)).scaled(160, 160, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                                      Qt.TransformationMode.FastTransformation)

        painter = QPainter(scaled_crop)
        painter.setPen(QPen(QColor(255, 0, 0, 150), 2))
        painter.drawLine(80, 0, 80, 160)
        painter.drawLine(0, 80, 160, 80)
        painter.end()
        return scaled_crop

    @staticmethod
    def render_main_image(raw_qimage: QImage, overlay_pixmap: QPixmap, opacity: float,
                          rectangles_data: list, draw_mode: str = None, temp_points: list = None) -> QPixmap:
        """Zeichnet alle Boxen, Punkte, Texte und Live-Zonen auf das Vorschaubild."""
        base_pixmap = QPixmap.fromImage(raw_qimage)
        painter = QPainter(base_pixmap)

        # Anti-Aliasing aktivieren
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if overlay_pixmap and not overlay_pixmap.isNull() and overlay_pixmap.size() == base_pixmap.size():
            painter.setOpacity(opacity)
            painter.drawPixmap(0, 0, overlay_pixmap)
            painter.setOpacity(1.0)

        # 2. Rechtecke & Gitter
        for rect in rectangles_data:
            if not rect.get("is_active", True) and not rect.get("is_main_calib") and not rect.get("is_zone"):
                continue

            is_main = rect.get("is_main_calib", False)
            color = PrecisionRenderer._get_color_for_rect(rect)
            painter.setPen(QPen(color, 2 if is_main else 1))

            if is_main:
                PrecisionRenderer._draw_room_mesh(painter, rect["corners"])
            else:
                PrecisionRenderer._draw_custom_rect(painter, rect["corners"])

            # 3. Punkte & Beschriftungen
            for i, c in enumerate(rect["corners"]):
                if c["px"] is not None:
                    p = QPoint(int(c["px"]), int(c["py"]))
                    painter.setBrush(QBrush(color))

                    painter.drawEllipse(p, 2, 2)

                    lbl = c["label"].split()[0] if is_main else f"{rect['display_id']}.{i + 1}"
                    painter.drawText(p.x() + 6, p.y() - 6, lbl)
            # 4. Live-Zonen (aktuell im Zeichenmodus)
        if draw_mode and temp_points:
            zone_color = QColor(255, 50, 50) if draw_mode == "Dead Zone" else QColor(50, 100, 255)

            painter.setPen(QPen(zone_color, 2, Qt.PenStyle.SolidLine))
            for i in range(len(temp_points) - 1):
                p1 = QPoint(int(temp_points[i][0]), int(temp_points[i][1]))
                p2 = QPoint(int(temp_points[i + 1][0]), int(temp_points[i + 1][1]))
                painter.drawLine(p1, p2)

            if len(temp_points) == 3:
                dash_pen = QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine)
                painter.setPen(dash_pen)
                p_last = QPoint(int(temp_points[2][0]), int(temp_points[2][1]))
                p_first = QPoint(int(temp_points[0][0]), int(temp_points[0][1]))
                painter.drawLine(p_last, p_first)

            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(zone_color))
            for i, pt in enumerate(temp_points):
                p = QPoint(int(pt[0]), int(pt[1]))
                painter.drawEllipse(p, 5, 5)
                painter.drawText(p.x() + 8, p.y() - 8, f"{i + 1}/4")

        painter.end()
        return base_pixmap

    @staticmethod
    def _get_color_for_rect(rect: dict) -> QColor:
        """Bestimmt die Farbe für ein Rechteck basierend auf seinen Eigenschaften."""
        if rect.get("is_main_calib"): return QColor(0, 255, 0)
        if rect.get("type") == "Dead Zone": return QColor(255, 50, 50)  # Rot
        if rect.get("type") == "Mirror Zone": return QColor(50, 150, 255)  # Blau
        if rect.get("type") == "Typ 1": return QColor(0, 255, 255)
        if rect.get("type") == "Typ 2": return QColor(255, 0, 255)
        return QColor(255, 255, 0)

    @staticmethod
    def _draw_room_mesh(painter: QPainter, corners: list):
        """Zeichnet das Gitter für die Hauptkalibrierung (8 Punkte, 12 Verbindungen)."""
        connections = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 6), (6, 7), (7, 5), (5, 4), (0, 4), (1, 6), (2, 5), (3, 7)]
        for s, e in connections:
            c1, c2 = corners[s], corners[e]
            if c1["px"] is not None and c2["px"] is not None:
                painter.drawLine(int(c1["px"]), int(c1["py"]), int(c2["px"]), int(c2["py"]))

    @staticmethod
    def _draw_custom_rect(painter: QPainter, corners: list):
        """Zeichnet ein Rechteck oder eine Zone basierend auf den gültigen Eckpunkten."""
        valid_points = [QPoint(c["px"], c["py"]) for c in corners if c["px"] is not None]
        if len(valid_points) > 1:
            poly = list(valid_points)
            if len(poly) == 4: poly.append(poly[0])
            painter.drawPolyline(poly)