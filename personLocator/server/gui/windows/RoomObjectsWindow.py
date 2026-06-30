from typing import Any, Optional, List, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QColorDialog, QMessageBox, QCheckBox, QWidget, QLabel
)


COLUMN_HEADERS: list[str] = [
    "ID", "Name", "Typ", "X [cm]", "Y [cm]", "Z [cm]",
    "Breite", "Höhe", "Tiefe", "Yaw [°]", "Farbe", "Sichtbar"
]


class RoomObjectsWindow(QDialog):
    """
    Dialog zum Hinzufügen, Bearbeiten und Löschen platzierter Raum-Objekte.

    Die Objekte werden in der Tabelle ``room_objects`` der SystemDatabase persistiert
    und nach dem Speichern unmittelbar in den ServerController-Cache neu geladen, sodass
    sie sofort im Master-View des Dashboards sichtbar sind.
    """

    def __init__(self, controller: Any, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("📦 Raum-Objekte")
        self.resize(1000, 520)
        self.controller: Any = controller

        self.objects: list[dict[str, Any]] = []
        self.deleted_ids: list[int] = []

        layout: QVBoxLayout = QVBoxLayout(self)

        hint: QLabel = QLabel(
            "Koordinaten in Zentimetern. (X, Y, Z) ist der Mittelpunkt der Bodenfläche; "
            "(Breite, Höhe, Tiefe) ist die Ausdehnung; Yaw ist die Drehung um die Y-Achse in Grad."
        )
        hint.setStyleSheet("color: #aaa; padding: 4px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table: QTableWidget = QTableWidget(0, len(COLUMN_HEADERS), self)
        self.table.setHorizontalHeaderLabels(COLUMN_HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        btn_row: QHBoxLayout = QHBoxLayout()
        self.btn_add: QPushButton = QPushButton("➕ Hinzufügen")
        self.btn_remove: QPushButton = QPushButton("🗑️ Löschen")
        self.btn_pick_color: QPushButton = QPushButton("🎨 Farbe wählen…")
        self.btn_save: QPushButton = QPushButton("💾 Speichern")
        self.btn_cancel: QPushButton = QPushButton("Abbrechen")

        self.btn_add.setStyleSheet("background-color: #1a6f3a; color: white; padding: 6px; font-weight: bold;")
        self.btn_remove.setStyleSheet("background-color: #8b2727; color: white; padding: 6px; font-weight: bold;")
        self.btn_pick_color.setStyleSheet("background-color: #444; color: white; padding: 6px;")
        self.btn_save.setStyleSheet("background-color: #1a4d8f; color: white; padding: 6px; font-weight: bold;")
        self.btn_cancel.setStyleSheet("padding: 6px;")

        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_pick_color)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_save)
        layout.addLayout(btn_row)

        self.btn_add.clicked.connect(self._on_add_row)
        self.btn_remove.clicked.connect(self._on_remove_row)
        self.btn_pick_color.clicked.connect(self._on_pick_color)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_cancel.clicked.connect(self.reject)

        self._load_from_controller()

    def _load_from_controller(self) -> None:
        try:
            self.controller.reload_room_objects()
            self.objects = [dict(o) for o in (self.controller.room_objects or [])]
        except Exception:
            self.objects = []

        self.table.setRowCount(0)
        for obj in self.objects:
            self._append_row(obj)

    def _append_row(self, obj: dict[str, Any]) -> None:
        row: int = self.table.rowCount()
        self.table.insertRow(row)

        def _set_text(col: int, value: Any, editable: bool = True, align_right: bool = False) -> None:
            item: QTableWidgetItem = QTableWidgetItem(str(value))
            flags = item.flags()
            if not editable:
                flags &= ~Qt.ItemFlag.ItemIsEditable
            item.setFlags(flags)
            if align_right:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, col, item)

        _set_text(0, obj.get("id", 0), editable=False, align_right=True)
        _set_text(1, obj.get("name", "Objekt"))
        _set_text(2, obj.get("obj_type", "box"))
        _set_text(3, float(obj.get("pos_x", 0.0)), align_right=True)
        _set_text(4, float(obj.get("pos_y", 0.0)), align_right=True)
        _set_text(5, float(obj.get("pos_z", 0.0)), align_right=True)
        _set_text(6, float(obj.get("size_w", 50.0)), align_right=True)
        _set_text(7, float(obj.get("size_h", 50.0)), align_right=True)
        _set_text(8, float(obj.get("size_d", 50.0)), align_right=True)
        _set_text(9, float(obj.get("rotation_yaw", 0.0)), align_right=True)

        color_hex: str = str(obj.get("color_hex", "#FFAA00"))
        color_item: QTableWidgetItem = QTableWidgetItem(color_hex)
        color_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            color_item.setBackground(QColor(color_hex))
            color_item.setForeground(QColor(0, 0, 0))
        except Exception:
            pass
        self.table.setItem(row, 10, color_item)

        visible_widget: QWidget = QWidget(self.table)
        v_layout: QHBoxLayout = QHBoxLayout(visible_widget)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chk: QCheckBox = QCheckBox()
        chk.setChecked(bool(obj.get("is_visible", True)))
        v_layout.addWidget(chk)
        self.table.setCellWidget(row, 11, visible_widget)

    def _on_add_row(self) -> None:
        defaults: dict[str, Any] = {
            "id": 0, "name": f"Objekt_{self.table.rowCount() + 1}", "obj_type": "box",
            "pos_x": 100.0, "pos_y": 0.0, "pos_z": 100.0,
            "size_w": 50.0, "size_h": 80.0, "size_d": 50.0,
            "color_hex": "#FFAA00", "rotation_yaw": 0.0, "is_visible": True,
        }
        self._append_row(defaults)

    def _on_remove_row(self) -> None:
        row: int = self.table.currentRow()
        if row < 0:
            return
        id_item: Optional[QTableWidgetItem] = self.table.item(row, 0)
        try:
            db_id: int = int(id_item.text()) if id_item else 0
        except Exception:
            db_id = 0
        if db_id > 0:
            self.deleted_ids.append(db_id)
        self.table.removeRow(row)

    def _on_pick_color(self) -> None:
        row: int = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Farbe wählen", "Bitte zuerst eine Zeile in der Tabelle auswählen.")
            return
        item: Optional[QTableWidgetItem] = self.table.item(row, 10)
        current_hex: str = item.text() if item else "#FFAA00"
        try:
            initial: QColor = QColor(current_hex)
        except Exception:
            initial = QColor("#FFAA00")
        col: QColor = QColorDialog.getColor(initial, self, "Objekt-Farbe wählen")
        if col.isValid():
            hex_str: str = col.name().upper()
            new_item: QTableWidgetItem = QTableWidgetItem(hex_str)
            new_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            new_item.setBackground(col)
            new_item.setForeground(QColor(0, 0, 0))
            self.table.setItem(row, 10, new_item)

    def _row_to_object(self, row: int) -> Optional[dict[str, Any]]:
        try:
            id_text: str = self.table.item(row, 0).text() if self.table.item(row, 0) else "0"
            db_id: int = int(id_text) if id_text.strip() else 0

            def _flt(col: int, default: float) -> float:
                item: Optional[QTableWidgetItem] = self.table.item(row, col)
                if not item:
                    return default
                try:
                    return float(str(item.text()).replace(",", "."))
                except Exception:
                    return default

            def _txt(col: int, default: str) -> str:
                item: Optional[QTableWidgetItem] = self.table.item(row, col)
                return item.text() if item and item.text() else default

            visible_widget: Optional[QWidget] = self.table.cellWidget(row, 11)
            is_visible: bool = True
            if visible_widget:
                chk: Optional[QCheckBox] = visible_widget.findChild(QCheckBox)
                if chk is not None:
                    is_visible = chk.isChecked()

            return {
                "id": db_id,
                "name": _txt(1, f"Objekt_{row + 1}"),
                "obj_type": _txt(2, "box"),
                "pos_x": _flt(3, 0.0),
                "pos_y": _flt(4, 0.0),
                "pos_z": _flt(5, 0.0),
                "size_w": max(1.0, _flt(6, 50.0)),
                "size_h": max(1.0, _flt(7, 50.0)),
                "size_d": max(1.0, _flt(8, 50.0)),
                "rotation_yaw": _flt(9, 0.0),
                "color_hex": _txt(10, "#FFAA00"),
                "is_visible": is_visible,
            }
        except Exception:
            return None

    def _on_save(self) -> None:
        if not self.controller or not getattr(self.controller, "system_db", None):
            QMessageBox.critical(self, "Speichern", "Kein gültiger ServerController verfügbar.")
            return

        try:
            for db_id in self.deleted_ids:
                self.controller.system_db.delete_room_object(int(db_id))
            self.deleted_ids.clear()

            for row in range(self.table.rowCount()):
                obj: Optional[dict[str, Any]] = self._row_to_object(row)
                if not obj:
                    continue
                new_id: int = self.controller.system_db.save_room_object(obj)
                if new_id > 0:
                    obj["id"] = new_id

            self.controller.reload_room_objects()
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(e))
