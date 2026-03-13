import logging
from typing import List

import numpy as np
from PyQt6.QtWidgets import QTableWidget, QHeaderView, QTableWidgetItem
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


class CalibrationCoordinatesTable(QTableWidget):
    """
    Tabelle für Live-Personen-Daten:
    Zeigt Keypoints, Metriken (Größe, Breite, Rotation) und Live-Farben an.
    Passt sich automatisch an den 3D- oder Performance-Modus an!
    """

    value_changed = pyqtSignal(int, str, float)

    COLUMNS_SKEL = ["Körperteil / Metrik", "Wert / Position", "Status"]

    STYLE_SHEET = """
        QTableWidget { 
            background-color: #1a1a1a; color: #ececec; 
            gridline-color: #333; font-size: 11px; border: none;
        }
        QHeaderView::section { 
            background-color: #2c2c2c; color: #aaa; padding: 4px; border: 1px solid #111;
        }
        QTableWidget::item:selected { background-color: #3d3d3d; }
    """

    def __init__(self):
        super().__init__(0, 3)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setHorizontalHeaderLabels(self.COLUMNS_SKEL)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet(self.STYLE_SHEET)

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self.itemChanged.connect(self._on_cell_edited)

    def update_skeleton_data(self, keypoints: list, metrics: dict = None):
        """Zeigt Keypoints, Metriken UND die Farbprofile (nebeneinander)."""
        self.blockSignals(True)

        KP_NAMES = {
            0: "Nase", 5: "Schulter Li", 6: "Schulter Re",
            11: "Hüfte Li", 12: "Hüfte Re", 13: "Knie Li", 14: "Knie Re",
            15: "Fuß Li", 16: "Fuß Re",
            -1: "Torso / Oberteil"
        }

        visible_kps = [k for k in keypoints if k['id'] in KP_NAMES]

        has_metrics = metrics and "total_height" in metrics
        metric_rows = 6 if has_metrics else 0

        color_profiles_raw = metrics.get("color_profiles", {}) if metrics else {}

        stable_raw = metrics.get("stable_colors", {}) if metrics else {}

        if stable_raw:
            fallback_colors_raw = dict(stable_raw)
        else:
            fallback_colors_raw = metrics.get("joint_colors", {}) if metrics else {}


        color_profiles = {}
        if isinstance(color_profiles_raw, dict):
            for view, joints in color_profiles_raw.items():
                parsed_joints = {}
                if isinstance(joints, dict):
                    for k, v in joints.items():
                        try:
                            parsed_joints[int(k)] = v
                        except ValueError:
                            pass
                color_profiles[view] = parsed_joints

        fallback_colors = {}
        if isinstance(fallback_colors_raw, dict):
            for k, v in fallback_colors_raw.items():
                try:
                    fallback_colors[int(k)] = v
                except ValueError:
                    pass

        if not has_metrics:
            color_profiles = {}
            if "Front" in fallback_colors_raw or "Side" in fallback_colors_raw:
                fallback_colors_raw = metrics.get("joint_colors", {})

        joint_ids_to_show = set()
        if color_profiles:
            for view_map in color_profiles.values():
                joint_ids_to_show.update(view_map.keys())
        elif fallback_colors:
            joint_ids_to_show.update(fallback_colors.keys())

        if not has_metrics and not joint_ids_to_show:
            joint_ids_to_show = {5, 6, 11, 12, 13, 14, 15, 16}

        sorted_j_ids = [jid for jid in KP_NAMES.keys() if jid in joint_ids_to_show]

        color_rows_count = 0
        if sorted_j_ids:
            if color_profiles:
                color_rows_count = 1 + (len(sorted_j_ids) * 2)
            else:
                color_rows_count = 1 + len(sorted_j_ids)

        total_rows = len(visible_kps) + metric_rows + color_rows_count

        if self.rowCount() != total_rows:
            self.setRowCount(total_rows)
            self.clearSpans()

        row = 0

        # ==========================================
        # 1. KEYPOINTS
        # ==========================================
        for kp in visible_kps:
            kp_id = kp['id']
            name = KP_NAMES.get(kp_id, "?")
            conf = kp.get('c', 0.0)

            self._set_smart(row, 0, name, editable=False)
            self._set_smart(row, 1, f"x:{int(kp['x'])} y:{int(kp['y'])}", editable=False)

            c_item = self._set_smart(row, 2, f"{int(conf * 100)}%", editable=False)
            if conf > 0.6:
                c_item.setForeground(QColor("#859900"))
            elif conf < 0.3:
                c_item.setForeground(QColor("#dc322f"))
            else:
                c_item.setForeground(QColor("#b58900"))
            row += 1

        # ==========================================
        # 2. BODY METRICS (NUR IM 3D MODUS)
        # ==========================================
        if has_metrics:
            self._add_metric(row, "---", "METRIKEN", "#586e75")
            row += 1

            h_val = metrics.get("total_height") or 0
            self._add_metric(row, "Gesamtgröße", f"{int(h_val)}", "#b58900", editable=True, key="fixed_height")
            self._set_smart(row, 2, "(Editierbar)", False).setForeground(QColor("#555"))
            row += 1

            rot_deg = metrics.get("orientation_angle") or 0.0
            rot_text = metrics.get("orientation", "-")
            self._add_metric(row, "Rotation", f"{int(rot_deg)}°", "#268bd2")
            self._set_smart(row, 2, rot_text, False).setForeground(QColor("#888"))
            row += 1

            conf_val = metrics.get("orientation_confidence") or 0.0
            conf_color = "#859900" if conf_val > 0.7 else ("#b58900" if conf_val > 0.4 else "#dc322f")
            self._add_metric(row, "Sicherheit", f"{int(conf_val * 100)}%", conf_color)
            self._set_smart(row, 2, "Ansichts-Check", False).setForeground(QColor("#555"))
            row += 1

            live_w = metrics.get('shoulder_width') or 0
            learned_w = metrics.get('learned_width') or 0

            self._add_metric(row, "Schulterbreite (Live)", f"{live_w} cm")
            row += 1
            self._add_metric(row, "Breite (Gelernt)", f"{learned_w} cm", "#586e75")
            row += 1

        # ==========================================
        # 3. FARBEN (SMART)
        # ==========================================
        if sorted_j_ids:
            if color_profiles:
                self._add_metric(row, "FARB-PROFILE", "Front  |  Seite  |  Rücken", "#586e75")
                self.item(row, 1).setForeground(QColor("#aaa"))
                row += 1

                for j_id in sorted_j_ids:
                    name = KP_NAMES.get(j_id, f"ID {j_id}")
                    c_front = color_profiles.get("Front", {}).get(j_id)
                    c_side = color_profiles.get("Side", {}).get(j_id)
                    c_back = color_profiles.get("Back", {}).get(j_id)

                    lbl_item = self._set_smart(row, 0, f"{name}", False, bg="#222")
                    lbl_item.setForeground(QColor("#eee"))
                    self.setSpan(row, 0, 1, 3)
                    row += 1

                    f_item = self._set_smart(row, 0, "Front" if c_front else "-", False)
                    if c_front:
                        f_item.setBackground(QColor(c_front))
                        f_item.setForeground(QColor(c_front))

                    s_item = self._set_smart(row, 1, "Seite" if c_side else "-", False)
                    if c_side:
                        s_item.setBackground(QColor(c_side))
                        s_item.setForeground(QColor(c_side))

                    b_item = self._set_smart(row, 2, "Rücken" if c_back else "-", False)
                    if c_back:
                        b_item.setBackground(QColor(c_back))
                        b_item.setForeground(QColor(c_back))
                    row += 1

            else:
                self._add_metric(row, "FARB-PROFILE", "Global (Performance)", "#586e75")
                self.item(row, 1).setForeground(QColor("#aaa"))  # Macht die Schrift etwas heller
                row += 1
                for j_id in sorted_j_ids:
                    hex_col = fallback_colors.get(j_id) if fallback_colors else None
                    label_name = KP_NAMES.get(j_id, "?")
                    self._set_smart(row, 0, label_name, False)

                    if hex_col:
                        v_item = self._set_smart(row, 1, hex_col, False)
                        v_item.setForeground(QColor("#aaaaaa"))
                        c_item = self._set_smart(row, 2, "", False)
                        c_item.setBackground(QColor(hex_col))
                    else:
                        v_item = self._set_smart(row, 1, "-", False)
                        v_item.setForeground(QColor("#555555"))
                        self._set_smart(row, 2, "", False).setBackground(QColor("#073642"))
                    row += 1

        self.blockSignals(False)
    #Hilfsmehtoden
    def _add_metric(self, row, label, value, color_hex=None, editable=False, key=None):
        self._set_smart(row, 0, label, False).setBackground(QColor("#073642"))
        v_item = self._set_smart(row, 1, value, editable)
        v_item.setBackground(QColor("#073642"))

        if key: v_item.setData(Qt.ItemDataRole.UserRole, key)

        if color_hex:
            v_item.setForeground(QColor(color_hex))
            f = v_item.font()
            f.setBold(True)
            v_item.setFont(f)

        if not editable:
            self._set_smart(row, 2, "", False).setBackground(QColor("#073642"))

    def _set_smart(self, row, col, text, editable=True, bg=None) -> QTableWidgetItem:
        item = self.item(row, col)
        if not item:
            item = QTableWidgetItem()
            self.setItem(row, col, item)

        if item.text() != str(text):
            item.setText(str(text))

        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable: flags |= Qt.ItemFlag.ItemIsEditable
        if item.flags() != flags: item.setFlags(flags)

        if bg: item.setBackground(QColor(bg))
        return item

    def _on_cell_edited(self, item: QTableWidgetItem):
        row, col = item.row(), item.column()
        special_key = item.data(Qt.ItemDataRole.UserRole)

        if special_key == "fixed_height":
            try:
                val = float(item.text().replace(" cm", "").replace(",", "."))
                self.value_changed.emit(row, "fixed_height", val)
            except ValueError:
                pass

