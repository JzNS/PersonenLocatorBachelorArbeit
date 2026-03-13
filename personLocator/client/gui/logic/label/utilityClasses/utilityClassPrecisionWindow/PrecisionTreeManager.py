import uuid
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QComboBox, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal


class PrecisionTreeManager(QWidget):
    """
    Verwaltet die Liste der Vierecke und Zonen.
    Kapselt die gesamte Filter-, Zeichen- und Editier-Logik für den QTreeWidget.
    """
    needs_redraw = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rectangles_data = []
        self.deleted_rect_ids = []
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        filter_layout = QHBoxLayout()
        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "Alle anzeigen", "Nur Boden", "Nur Breite", "Nur Tiefe",
            "Ausgefüllt", "Unvollständig / Neu", "Nur Dead Zones", "Nur Mirror Zones"
        ])
        self.combo_filter.currentTextChanged.connect(self.apply_filters)

        self.search_filter = QLineEdit()
        self.search_filter.setPlaceholderText("Größe (z.B. 10)...")
        self.search_filter.setFixedWidth(130)
        self.search_filter.textChanged.connect(self.apply_filters)

        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(self.combo_filter)
        filter_layout.addWidget(self.search_filter)
        self.layout.addLayout(filter_layout)

        # --- Tree Widget ---
        self.tree = QTreeWidget()
        self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(["ID", "Name/Ecke", "Raum X", "Raum Y", "Raum Z", "Pixel X", "Pixel Y"])
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemChanged.connect(self.on_tree_item_changed)
        self.layout.addWidget(self.tree, stretch=1)

        # --- Delete Button ---
        self.btn_delete = QPushButton("Ausgewähltes löschen")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.layout.addWidget(self.btn_delete)


    def load_data(self, data_list: list):
        """Lädt eine frische Liste an Vierecken ins Widget."""
        self.rectangles_data = data_list
        self.tree.clear()
        for rect in self.rectangles_data:
            self._insert_rect_into_tree(rect)
        self.apply_filters()

    def add_rectangle(self, rect_data: dict):
        """Fügt ein neues Element zur Liste hinzu (z.B. Zonen oder Custom Rects)."""
        self.rectangles_data.append(rect_data)
        self._insert_rect_into_tree(rect_data)
        self.needs_redraw.emit()

    def process_pixel_click(self, x, y, is_delete=False):
        """Weist einem in der Liste markierten Eckpunkt einen Pixelklick zu."""
        self.apply_filters()
        selected = self.tree.selectedItems()
        if not selected: return

        item = selected[0]
        is_parent = item.parent() is None
        parent_item = item if is_parent else item.parent()

        rect_data = next(
            (r for r in self.rectangles_data if r["internal_id"] == parent_item.data(0, Qt.ItemDataRole.UserRole)),
            None)
        if not rect_data: return

        self.tree.itemChanged.disconnect(self.on_tree_item_changed)
        val_str = str(x) if not is_delete else "-"

        if not is_parent:
            idx = parent_item.indexOfChild(item)
            rect_data["corners"][idx]["px"] = x
            rect_data["corners"][idx]["py"] = y
            item.setText(5, val_str)
            item.setText(6, str(y) if not is_delete else "-")
            if not is_delete: self.select_next_item()
        else:
            for i, c in enumerate(rect_data["corners"]):
                if is_delete or c["px"] is None:
                    c["px"] = x
                    c["py"] = y
                    parent_item.child(i).setText(5, val_str)
                    parent_item.child(i).setText(6, str(y) if not is_delete else "-")
                    if not is_delete:
                        self.tree.setCurrentItem(parent_item.child(i))
                        break

        self.tree.itemChanged.connect(self.on_tree_item_changed)
        self.needs_redraw.emit()

    def select_next_item(self):
        """Wählt das nächste Element in der Liste aus (für schnellere Pixelzuweisung)."""
        current = self.tree.currentItem()
        if not current:
            if self.tree.topLevelItemCount() > 0: self.tree.setCurrentItem(self.tree.topLevelItem(0))
            return
        next_item = self.tree.itemBelow(current)
        self.tree.setCurrentItem(next_item if next_item else self.tree.topLevelItem(0))

    def delete_selected(self):
        """Entfernt das ausgewählte Rechteck
        oder die Zone aus der Liste (nur wenn es kein Hauptkalibrierungsrechteck ist)."""
        selected = self.tree.selectedItems()
        if not selected: return
        item = selected[0]
        parent = item.parent() if item.parent() else item

        rect_id = parent.data(0, Qt.ItemDataRole.UserRole)
        if rect_id == "MAIN_ROOM_CALIB":
            QMessageBox.warning(self, "Verboten", "Die Haupt-Raumkalibrierung kann nicht gelöscht werden!")
            return

        if rect_id not in self.deleted_rect_ids:
            self.deleted_rect_ids.append(rect_id)

        self.rectangles_data = [r for r in self.rectangles_data if r["internal_id"] != rect_id]
        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(parent))
        self.needs_redraw.emit()

    def apply_filters(self):
        """Wendet die ausgewählten Filter (Boden, Breite, Tiefe, Ausgefüllt, Dead/Mirror Zones)
        und die Suchfunktion auf die Liste an."""
        filter_type = self.combo_filter.currentText()
        search_text = self.search_filter.text().lower()

        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            rect_id = item.data(0, Qt.ItemDataRole.UserRole)
            rect_data = next((r for r in self.rectangles_data if r["internal_id"] == rect_id), None)
            if not rect_data: continue

            show_item = True
            item_text = item.text(1)

            if filter_type == "Nur Boden" and "[Boden]" not in item_text:
                show_item = False
            elif filter_type == "Nur Breite" and "[Breite]" not in item_text:
                show_item = False
            elif filter_type == "Nur Tiefe" and "[Tiefe]" not in item_text:
                show_item = False
            elif filter_type == "Nur Dead Zones" and rect_data.get("type") != "Dead Zone":
                show_item = False
            elif filter_type == "Nur Mirror Zones" and rect_data.get("type") != "Mirror Zone":
                show_item = False
            elif filter_type in ["Ausgefüllt", "Unvollständig / Neu"]:
                is_filled = all(c.get("px") is not None and c.get("py") is not None for c in rect_data["corners"])
                if filter_type == "Ausgefüllt" and not is_filled:
                    show_item = False
                elif filter_type == "Unvollständig / Neu" and is_filled:
                    show_item = False

            if search_text and search_text not in item_text.lower(): show_item = False
            if rect_data.get("is_main_calib") and filter_type != "Alle anzeigen": show_item = False

            item.setHidden(not show_item)

    def on_tree_item_changed(self, item, col):
        """Reagiert auf Änderungen im Tree Widget
        (z.B. Checkboxen, Textänderungen) und
        aktualisiert die Daten entsprechend."""
        rect_id = (item if not item.parent() else item.parent()).data(0, Qt.ItemDataRole.UserRole)
        rect_data = next((r for r in self.rectangles_data if r["internal_id"] == rect_id), None)
        if not rect_data: return

        if not item.parent():
            if col == 0:
                if not rect_data.get("is_main_calib") and not rect_data.get("is_zone"):
                    is_active = (item.checkState(0) == Qt.CheckState.Checked)
                    if rect_data.get("is_active", True) != is_active:
                        rect_data["is_active"] = is_active
                        self.needs_redraw.emit()

                new_id = item.text(0)
                if rect_data.get("display_id") != new_id:
                    rect_data["display_id"] = new_id
                    self.tree.itemChanged.disconnect(self.on_tree_item_changed)
                    for i in range(item.childCount()):
                        item.child(i).setText(0, f"{new_id}.{i + 1}")
                    self.tree.itemChanged.connect(self.on_tree_item_changed)

            elif col == 1:
                raw_text = item.text(1)
                for suffix in [" [Neu]", " [Boden]", " [Breite]", " [Tiefe]", " [Schräg]"]:
                    raw_text = raw_text.replace(suffix, "")
                rect_data["type"] = raw_text.strip()
                self._update_rect_label(item, rect_data)
            return

        parent_item = item.parent()
        if rect_data:
            corner = rect_data["corners"][parent_item.indexOfChild(item)]
            try:
                val = float(item.text(col).replace(',', '.'))
                if col == 2:
                    corner["x"] = val
                elif col == 3:
                    corner["y"] = val
                elif col == 4:
                    corner["z"] = val

                if col in (2, 3, 4): self._update_rect_label(parent_item, rect_data)
            except ValueError:
                pass

    def _insert_rect_into_tree(self, rect_data: dict):
        """Fügt ein Rechteck oder eine Zone als neuen Eintrag in den Tree ein."""
        self.tree.itemChanged.disconnect(self.on_tree_item_changed)
        parent = QTreeWidgetItem(self.tree)
        parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsEditable)
        parent.setText(0, rect_data["display_id"])

        is_main = rect_data.get("is_main_calib", False)
        is_zone = rect_data.get("is_zone", False)
        parent.setData(0, Qt.ItemDataRole.UserRole, rect_data["internal_id"])

        if not is_main and not is_zone:
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            is_active = rect_data.get("is_active", True)
            parent.setCheckState(0, Qt.CheckState.Checked if is_active else Qt.CheckState.Unchecked)

        if is_main or is_zone: parent.setText(1, rect_data['type'])

        bg, fg = (QColor("#1a331a"), QColor("#00ff00")) if is_main else \
            (QColor("#331111") if rect_data.get("type") == "Dead Zone" else QColor("#111133"),
             QColor("#ffffff")) if is_zone else \
                (QColor("#222"), QColor("#fff"))

        for col in range(7):
            parent.setBackground(col, bg)
            parent.setForeground(col, fg)
            font = parent.font(col)
            font.setBold(True)
            parent.setFont(col, font)

        for i, corner in enumerate(rect_data["corners"]):
            child = QTreeWidgetItem(parent)
            child.setText(0, f"{rect_data['display_id']}.{i + 1}")
            child.setText(1, corner["label"])

            if is_zone:
                child.setText(2, "-");
                child.setText(3, "-");
                child.setText(4, "-")
            else:
                child.setText(2, str(corner["x"]));
                child.setText(3, str(corner["y"]));
                child.setText(4, str(corner["z"]))

            child.setText(5, str(corner["px"]) if corner["px"] is not None else "-")
            child.setText(6, str(corner["py"]) if corner["py"] is not None else "-")
            child.setFlags(child.flags() | Qt.ItemFlag.ItemIsEditable)

        self.tree.itemChanged.connect(self.on_tree_item_changed)
        self._update_rect_label(parent, rect_data)

    def _update_rect_label(self, parent_item, rect_data):
        """Aktualisiert die Beschriftung eines Rechtecks basierend auf seinen Eckdaten"""
        if rect_data.get("is_main_calib") or rect_data.get("is_zone"): return

        xs = [c.get("x", 0.0) for c in rect_data["corners"]]
        ys = [c.get("y", 0.0) for c in rect_data["corners"]]
        zs = [c.get("z", 0.0) for c in rect_data["corners"]]

        if max(xs) == 0 and max(ys) == 0 and max(zs) == 0:
            orientation = " [Neu]"
        elif max(ys) - min(ys) < 0.001:
            orientation = " [Boden]"
        elif max(zs) - min(zs) < 0.001:
            orientation = " [Breite]"
        elif max(xs) - min(xs) < 0.001:
            orientation = " [Tiefe]"
        else:
            orientation = " [Schräg]"

        base_name = rect_data.get("type", "Custom Viereck")
        for suffix in [" [Neu]", " [Boden]", " [Breite]", " [Tiefe]", " [Schräg]"]:
            base_name = base_name.replace(suffix, "")

        rect_data["type"] = base_name.strip()
        self.tree.itemChanged.disconnect(self.on_tree_item_changed)
        parent_item.setText(1, f"{rect_data['type']}{orientation}")
        self.tree.itemChanged.connect(self.on_tree_item_changed)
        self.apply_filters()