from PyQt6.QtWidgets import QMainWindow, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QHeaderView
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush
import time


class PersonDetailsWindow(QMainWindow):
    KP_NAMES = {
        0: "Nase", 1: "Auge Li", 2: "Auge Re", 3: "Ohr Li", 4: "Ohr Re",
        5: "Schulter Li", 6: "Schulter Re", 7: "Ellenbogen Li", 8: "Ellenbogen Re",
        9: "Hand Li", 10: "Hand Re", 11: "Hüfte Li", 12: "Hüfte Re",
        13: "Knie Li", 14: "Knie Re", 15: "Fuß Li", 16: "Fuß Re"
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Sensor-Fusion & Matrix Inspector")
        self.resize(1100, 900)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)

        self.tree = QTreeWidget()
        self.tree.setStyleSheet("""
            QTreeWidget { background-color: #1a1a1a; color: #ececec; font-size: 13px; }
            QTreeWidget::item:selected { background-color: #3d3d3d; }
            QTreeWidget::item { border-bottom: 1px solid #2a2a2a; padding: 2px; }
            QHeaderView::section { background-color: #2c2c2c; padding: 4px; border: 1px solid #111; font-weight: bold; }
        """)

        layout.addWidget(self.tree)

        self.tree_items = {}
        self.current_headers = []

    def update_data(self, global_persons):
        """Aktualisiert die Baumstruktur mit den neuesten Daten aus der Fusion. Es werden nur Personen angezeigt, die in den letzten 2 Sekunden ein Update hatten."""
        now = time.time()
        active_gps = [gp for gp in global_persons if now - gp.last_update <= 2.0]

        active_cams = set()
        for gp in active_gps:
            for cam, obs in gp.client_observations.items():
                if now - obs["last_seen"] <= 2.0:
                    active_cams.add(cam)
        active_cams = sorted(list(active_cams))

        headers = ["Metrik / Eigenschaft", "Server (Fusion)"] + active_cams
        if headers != self.current_headers:
            self.tree.setHeaderLabels(headers)
            self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            for i in range(1, len(headers)):
                self.tree.header().setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            self.current_headers = headers

        visited_keys = set()

        def get_item(item_key, parent_node, default_text):
            """Holt das QTreeWidgetItem für den gegebenen Schlüssel. Wenn es noch nicht existiert, wird es erstellt und mit default_text initialisiert."""
            is_first_access_this_frame = item_key not in visited_keys
            visited_keys.add(item_key)

            if item_key not in self.tree_items:
                it = QTreeWidgetItem([default_text])
                if parent_node is None:
                    self.tree.addTopLevelItem(it)
                else:
                    parent_node.addChild(it)
                self.tree_items[item_key] = it
                if "skel" not in item_key and "color" not in item_key:
                    it.setExpanded(True)
                is_first_access_this_frame = True
            else:
                it = self.tree_items[item_key]
                it.setText(0, default_text)

            if is_first_access_this_frame:
                for col in range(1, len(headers)):
                    it.setText(col, "-")
                    it.setForeground(col, QBrush(QColor("#ececec")))
                    it.setBackground(col, QBrush(Qt.GlobalColor.transparent))

            return it

        for gp in active_gps:
            p_key = f"p_{gp.id}"
            name_str = gp.name if gp.name != "Unknown" else f"Unbekannt (ID {gp.id})"

            root = get_item(p_key, None, f"👤 {name_str}")
            for col in range(len(headers)):
                root.setBackground(col, QBrush(QColor("#003300")))
            root.setText(1, f"Live ID: {gp.id}")

            pos_it = get_item(f"{p_key}_pos", root, "📍 3D Position")
            pos_it.setText(1, f"X:{int(gp.pos[0])} Y:{int(gp.pos[1])} Z:{int(gp.pos[2])}")

            h_it = get_item(f"{p_key}_height", root, "📏 Körpergröße")
            h_it.setText(1, f"{int(gp.height)} cm")

            w_it = get_item(f"{p_key}_width", root, "↔️ Schulterbreite")
            rot_it = get_item(f"{p_key}_rot", root, "🔄 Rotation & Neigung")

            color_root = get_item(f"{p_key}_colors", root, "🎨 Farb-Profile (Perspektiven)")
            skel_root = get_item(f"{p_key}_skel", root, "🦴 3D-Skelett & Memory")

            for cam in active_cams:
                if cam not in gp.client_observations: continue
                obs = gp.client_observations[cam]
                raw = obs.get("raw_data", {})
                col_idx = headers.index(cam)

                conf = raw.get('pos_confidence', 0)
                pos_it.setText(col_idx,
                               f"X:{int(obs['pos'][0])} Y:{int(obs['pos'][1])} Z:{int(obs['pos'][2])} ({int(conf * 100)}%)")
                h_it.setText(col_idx, f"{int(raw.get('height', 0))} cm")

                sh_live, sh_learn = raw.get('shoulder_width_live', 0), raw.get('shoulder_width_learned', 0)
                w_it.setText(col_idx, f"L: {sh_live}cm | Max: {sh_learn}cm")

                ang, tilt = raw.get('orientation_angle', 0), raw.get('tilt_angle', 0)
                rot_it.setText(col_idx, f"{ang}° (Tilt: {tilt}°)")

                colors_all = raw.get('colors_all', {})
                for view_name, view_colors in colors_all.items():
                    if not view_colors: continue

                    view_node = get_item(f"{p_key}_c_{view_name}", color_root, f"  Ansicht: {view_name}")

                    for j_id_str, hex_c in view_colors.items():
                        j_name = self.KP_NAMES.get(int(j_id_str), f"Gelenk {j_id_str}")
                        c_it = get_item(f"{p_key}_color_{view_name}_{j_id_str}", view_node, f"    {j_name}")
                        c_it.setText(col_idx, hex_c)
                        c_it.setForeground(col_idx, QBrush(QColor(hex_c)))

                # --- Skelett eintragen ---
                last_kps = obs.get("last_kps", {})
                last_skel = obs.get("last_skel_3d", {})

                for j_id in range(17):
                    if j_id not in last_kps: continue
                    kp = last_kps[j_id]
                    pos3d = last_skel.get(j_id, {})
                    j_name = self.KP_NAMES.get(j_id, f"Gelenk {j_id}")

                    sk_it = get_item(f"{p_key}_skel_{j_id}", skel_root, f"  {j_name}")

                    conf = kp.get("c", 0.0)
                    if conf > 0:
                        status = f"Live ({int(conf * 100)}%)"
                        text_color = QColor("#00ff00") if conf > 0.6 else QColor("#ffaa00")
                    else:
                        status = "Verdeckt (Memory)"
                        text_color = QColor("#ff4444")

                    pos_text = f"[{status}]  2D: {int(kp.get('x', 0))},{int(kp.get('y', 0))}"
                    if pos3d:
                        pos_text += f" | 3D: X:{int(pos3d.get('x', 0))} Y:{int(pos3d.get('y', 0))} Z:{int(pos3d.get('z', 0))}cm"

                    sk_it.setText(col_idx, pos_text)
                    sk_it.setForeground(col_idx, QBrush(text_color))

        for old_key in list(self.tree_items.keys()):
            if old_key not in visited_keys:
                item = self.tree_items[old_key]
                try:
                    parent = item.parent()
                    if parent:
                        parent.removeChild(item)
                    else:
                        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
                except RuntimeError:
                    pass
                del self.tree_items[old_key]