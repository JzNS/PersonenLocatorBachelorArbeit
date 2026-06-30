import time
import math
import numpy as np
import cv2
from PyQt6.QtWidgets import (QMainWindow, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QHBoxLayout, QWidget, QHeaderView, QLabel, QScrollArea, QGridLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush, QImage, QPixmap


class PersonInfoWindow(QMainWindow):
    KP_NAMES = {
        0: "Nase", 1: "Auge Li", 2: "Auge Re", 3: "Ohr Li", 4: "Ohr Re",
        5: "Schulter Li", 6: "Schulter Re", 7: "Ellenbogen Li", 8: "Ellenbogen Re",
        9: "Hand Li", 10: "Hand Re", 11: "Hüfte Li", 12: "Hüfte Re",
        13: "Knie Li", 14: "Knie Re", 15: "Fuß Li", 16: "Fuß Re"
    }

    TEMPLATE_SKEL = {
        0: (150, 30), 1: (138, 22), 2: (161, 22),
        3: (127, 30), 4: (172, 30),
        5: (105, 82), 6: (195, 82),
        7: (67, 142), 8: (232, 142),
        9: (37, 202), 10: (262, 202),
        11: (120, 210), 12: (180, 210),
        13: (120, 315), 14: (180, 315),
        15: (120, 420), 16: (180, 420)
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("👤 Live Personen Monitor (Dauerhaftes Tracking & Echt-Kompass)")
        self.resize(1450, 900)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        self.tree = QTreeWidget()
        self.tree.setStyleSheet("""
            QTreeWidget { background-color: #11111b; color: #cdd6f4; font-size: 14px; border: 1px solid #313244; }
            QTreeWidget::item:selected { background-color: #313244; }
            QTreeWidget::item { border-bottom: 1px solid #1e1e2e; padding: 6px; }
            QHeaderView::section { background-color: #181825; padding: 10px; border: none; font-weight: bold; color: #00FF96; }
        """)
        main_layout.addWidget(self.tree, stretch=2)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #11111b; }")

        self.skeletons_container = QWidget()
        self.skeletons_container.setStyleSheet("background-color: #11111b;")

        self.skeletons_layout = QGridLayout(self.skeletons_container)
        self.skeletons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.skeletons_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

        self.tree_items = {}
        self.current_headers = []

        self.smoothed_colors = {}
        self.smoothed_dirs = {}
        self.last_update_time = {}

        self.first_seen_time = {}
        self.known_triangulated_ids = set()

        self.skeleton_widgets = {}

    def _reorganize_grid(self):
        for i in reversed(range(self.skeletons_layout.count())):
            item = self.skeletons_layout.itemAt(i)
            if item.widget():
                self.skeletons_layout.removeWidget(item.widget())

        active_widgets = [w["container"] for w in self.skeleton_widgets.values()]
        if not active_widgets: return

        cols = math.ceil(math.sqrt(len(active_widgets)))
        if cols < 2 and len(active_widgets) > 1:
            cols = 2

        for i, widget in enumerate(active_widgets):
            row = i // cols
            col = i % cols
            self.skeletons_layout.addWidget(widget, row, col)

    def update_data(self, global_persons):
        now = time.time()
        safe_gps = list(global_persons)

        raw_active_gps = [gp for gp in safe_gps if now - gp.last_update <= 3.0]

        valid_gps = []
        active_cams_global = set()

        for gp in raw_active_gps:
            cams_for_this_person = [cam for cam, obs in gp.client_observations.items() if now - obs.last_seen <= 3.0]

            f_skel = getattr(gp, 'fusion_skel', {})
            s_skel = getattr(gp, 'skeleton_3d', {})
            print(f"     [DEBUG SOURCE] Person {gp.id}: fusion_skel keys={list(f_skel.keys())}, skeleton_3d keys={list(s_skel.keys())}")
            
            raw_skel = f_skel if f_skel else s_skel
            
            if not raw_skel:
                print(f"     [DEBUG SOURCE] Suche in client_observations...")
                for cam in cams_for_this_person:
                    obs = gp.client_observations[cam]
                    c_skel = getattr(obs, 'last_skel_3d', {})
                    print(f"       -> Kamera {cam}: last_skel_3d keys={list(c_skel.keys() if c_skel else [])}")
                    if c_skel:
                        raw_skel = c_skel
                        break

            clean_skel = {}
            for j_id, pos in raw_skel.items():
                try:
                    j_int = int(j_id)
                    if j_int in [5, 6]:
                        print(f"     [DEBUG SKEL] Gelenk {j_int} (Roh-ID-Typ: {type(j_id)}): {pos} (Typ: {type(pos)})")

                    if isinstance(pos, dict):
                        clean_skel[j_int] = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)],
                                                     dtype=np.float32)
                    elif isinstance(pos, (list, tuple)):
                        clean_skel[j_int] = np.array(pos, dtype=np.float32)
                    elif isinstance(pos, np.ndarray):
                        clean_skel[j_int] = pos
                except Exception as e:
                    print(f"     [DEBUG SKEL] Fehler bei Gelenk {j_id}: {e}")
                    pass

            if 5 in clean_skel and 6 in clean_skel:
                print(f"     [DEBUG SKEL] Schultern 5 & 6 erfolgreich in clean_skel!")
            else:
                missing = [j for j in [5, 6] if j not in clean_skel]
                print(f"     [DEBUG SKEL] FEHLENDE Gelenke in clean_skel: {missing}. Keys vorhanden: {list(clean_skel.keys())}")

            is_triangulated_now = (len(cams_for_this_person) >= 2 and len(clean_skel) >= 4)

            if is_triangulated_now:
                self.known_triangulated_ids.add(gp.id)


            if gp.id not in self.first_seen_time:
                self.first_seen_time[gp.id] = now
            if now - self.first_seen_time[gp.id] < 0.5:
                continue

            valid_gps.append((gp, cams_for_this_person, clean_skel))
            active_cams_global.update(cams_for_this_person)

        if not valid_gps and safe_gps:
             print(f"[DEBUG] Keine validen Personen gefunden, obwohl {len(safe_gps)} im Tracker sind.")

        valid_gps.sort(key=lambda x: x[0].id)
        active_cams = sorted(list(active_cams_global))

        headers = ["Eigenschaft / Gelenk", "Fusion (Server)"] + active_cams
        if headers != self.current_headers:
            self.tree.setHeaderLabels(headers)
            self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            for i in range(1, len(headers)):
                self.tree.header().setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            self.current_headers = headers

        visited_keys = set()
        needs_relayout = False

        def get_item(item_key, parent_node, default_text):
            is_first_access = item_key not in visited_keys
            visited_keys.add(item_key)

            if item_key not in self.tree_items:
                it = QTreeWidgetItem([default_text])
                if parent_node is None:
                    self.tree.addTopLevelItem(it)
                else:
                    parent_node.addChild(it)
                self.tree_items[item_key] = it
                it.setExpanded(True)
                is_first_access = True
            else:
                it = self.tree_items[item_key]
                it.setText(0, default_text)

            if is_first_access:
                for col in range(1, len(headers)):
                    it.setText(col, "-")
                    it.setForeground(col, QBrush(QColor("#777777")))
                    it.setBackground(col, QBrush(Qt.GlobalColor.transparent))
            return it

        for gp, cams_for_this_person, clean_skel in valid_gps:
            p_key = f"p_{gp.id}"

            dt = now - self.last_update_time.get(gp.id, now)
            self.last_update_time[gp.id] = now
            blend_factor = min(1.0, max(0.0, 5.0 * dt))

            if gp.id not in self.smoothed_colors:
                self.smoothed_colors[gp.id] = {}

            calculated_rot_str = "Suche Ausrichtung..."
            rotation_angle = 0.0
            ROOM_COMPASS_OFFSET = 0.0
            has_direction = False

            forward_vec = getattr(gp, 'forward_vec', None)
            if forward_vec is not None:
                fx, fz = forward_vec[0], forward_vec[2]
                raw_angle = np.degrees(np.arctan2(fx, fz)) % 360.0
                rotation_angle = (raw_angle + ROOM_COMPASS_OFFSET) % 360.0
                has_direction = True

            elif 5 in clean_skel and 6 in clean_skel:
                s_li = clean_skel[5]
                s_re = clean_skel[6]
                dx = s_li[0] - s_re[0]
                dz = s_li[2] - s_re[2]
                norm = np.linalg.norm([dz, -dx])
                if norm > 0.001:
                    target_fx = dz / norm
                    target_fz = -dx / norm

                    if gp.id not in self.smoothed_dirs:
                        self.smoothed_dirs[gp.id] = np.array([target_fx, target_fz])
                    else:
                        curr_dir = self.smoothed_dirs[gp.id]
                        new_dir = curr_dir * (1.0 - blend_factor) + np.array([target_fx, target_fz]) * blend_factor
                        norm_new = np.linalg.norm(new_dir)
                        if norm_new > 0:
                            self.smoothed_dirs[gp.id] = new_dir / norm_new

                    smooth_fx, smooth_fz = self.smoothed_dirs[gp.id]
                    raw_angle = np.degrees(np.arctan2(smooth_fx, smooth_fz)) % 360.0
                    rotation_angle = (raw_angle + ROOM_COMPASS_OFFSET) % 360.0
                    has_direction = True

            if has_direction:
                dirs = ["Nord", "Nord-Ost", "Ost", "Süd-Ost", "Süd", "Süd-West", "West", "Nord-West", "Nord"]
                idx = int(round((rotation_angle % 360) / 45.0))
                calculated_rot_str = f"{dirs[idx]} ({rotation_angle:.0f}°)"
            else:
                calculated_rot_str = "Schultern verdeckt"

            final_height = getattr(gp, 'height', 175.0)
            final_width = getattr(gp, 'fused_width', 45.0)

            # Falls der Tracker 'fused_height' setzt, hat diese Vorrang
            if hasattr(gp, 'fused_height') and gp.fused_height > 0:
                final_height = gp.fused_height

            px, py, pz = gp.pos[0], gp.pos[1], gp.pos[2] if np.any(gp.pos) else (0, 0, 0)

            root = get_item(p_key, None, f"👤 Bestätigte Person (ID: {gp.id})")
            for col in range(len(headers)):
                root.setBackground(col, QBrush(QColor("#15241b")))

            if len(cams_for_this_person) >= 2:
                root.setText(1, "FUSIONIERT (3D)")
                root.setForeground(1, QBrush(QColor("#00FF96")))
            else:
                root.setText(1, "SINGLE (Zonen-Tracking)")
                root.setForeground(1, QBrush(QColor("#00FFFF")))

            pos_node = get_item(f"{p_key}_pos", root, "📍 3D Raum-Position")
            pos_node.setText(1, f"X: {px:.1f} cm | Y: {py:.1f} cm | Z: {pz:.1f} cm")
            pos_node.setForeground(1, QBrush(QColor("#ffffff")))

            metrics_node = get_item(f"{p_key}_metrics", root, "📏 Berechnete 3D-Maße")
            metrics_node.setText(1, f"Größe: {final_height:.1f} cm | Schultern: {final_width:.1f} cm")
            metrics_node.setForeground(1, QBrush(QColor("#89b4fa")))

            rot_node = get_item(f"{p_key}_rot", root, "🧭 Blickrichtung")
            rot_node.setText(1, calculated_rot_str)
            rot_node.setForeground(1, QBrush(QColor("#f9e2af")) if has_direction else QBrush(QColor("#FFA500")))

            id_node = get_item(f"{p_key}_ids", root, "🔗 Kamera IDs (Mapping)")
            id_node.setText(1, "Verbunden")

            joint_colors_collected = {j: [] for j in range(17)}

            for cam in cams_for_this_person:
                obs = gp.client_observations[cam]
                col_idx = headers.index(cam)
                cam_id = obs.id
                id_node.setText(col_idx, f"ID {cam_id}")
                id_node.setForeground(col_idx, QBrush(QColor("#ffffff")))

                raw_data = obs.raw_data
                stable_cols = raw_data.get("metrics", {}).get("stable_colors", {})
                kps = raw_data.get("keypoints", [])

                for j_id in range(17):
                    c_hex = stable_cols.get(str(j_id), stable_cols.get(j_id))
                    if c_hex and isinstance(c_hex, str) and len(c_hex) >= 7:
                        conf = next((k.get("c", 0.0) for k in kps if k.get("id") == j_id), 0.0)
                        joint_colors_collected[j_id].append((col_idx, c_hex, conf))

            color_root = get_item(f"{p_key}_colors", root, "🎨 Sensorfusion: Gelenk-Farben")

            for j_id in range(17):
                colors_for_this_joint = joint_colors_collected[j_id]
                if not colors_for_this_joint: continue

                j_name = self.KP_NAMES.get(j_id, f"Gelenk {j_id}")
                j_node = get_item(f"{p_key}_color_{j_id}", color_root, f"  {j_name}")

                rgb_list = []
                for col_idx, c_hex, conf in colors_for_this_joint:
                    display_text = f"{c_hex} ({int(conf * 100)}%)"
                    j_node.setText(col_idx, display_text)
                    j_node.setForeground(col_idx, QBrush(QColor(c_hex)))
                    try:
                        rgb_list.append([int(c_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)])
                    except ValueError:
                        pass

                if rgb_list:
                    target_rgb = np.mean(rgb_list, axis=0)
                    if j_id not in self.smoothed_colors[gp.id]:
                        self.smoothed_colors[gp.id][j_id] = target_rgb
                    else:
                        current_rgb = self.smoothed_colors[gp.id][j_id]
                        self.smoothed_colors[gp.id][j_id] = current_rgb * (
                                1.0 - blend_factor) + target_rgb * blend_factor

                    final_rgb = self.smoothed_colors[gp.id][j_id].astype(int)
                    fused_hex = f"#{final_rgb[0]:02x}{final_rgb[1]:02x}{final_rgb[2]:02x}"

                    j_node.setText(1, fused_hex)
                    j_node.setForeground(1, QBrush(QColor(fused_hex)))
                else:
                    j_node.setText(1, "-")

            if gp.id not in self.skeleton_widgets:
                container = QWidget()
                container.setStyleSheet(
                    "background-color: #181825; border: 1px solid #313244; border-radius: 8px; margin-bottom: 5px;")
                layout = QVBoxLayout(container)

                lbl_metrics = QLabel()
                lbl_metrics.setStyleSheet("color: #00FF96; font-size: 14px; font-weight: bold; padding: 5px;")
                lbl_metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)

                lbl_img = QLabel()
                lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl_img.setStyleSheet("background-color: #11111b; border-radius: 5px;")

                layout.addWidget(lbl_metrics)
                layout.addWidget(lbl_img)

                self.skeleton_widgets[gp.id] = {"container": container, "metrics": lbl_metrics, "img": lbl_img}
                needs_relayout = True

            widget_dict = self.skeleton_widgets[gp.id]
            mode_str = "Fusioniert" if len(cams_for_this_person) >= 2 else "Single"
            widget_dict["metrics"].setText(
                f"👤 Person {gp.id} ({mode_str}) | {calculated_rot_str}\nHöhe: {final_height:.0f} cm | Pos: X {px:.0f} Z {pz:.0f}")

            pixmap = self.render_demo_skeleton(self.smoothed_colors[gp.id], angle=rotation_angle)
            widget_dict["img"].setPixmap(pixmap)

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

                pid = int(old_key.split('_')[1]) if '_' in old_key else -1

                if pid in self.smoothed_colors: del self.smoothed_colors[pid]
                if pid in self.smoothed_dirs: del self.smoothed_dirs[pid]

                if pid in self.last_update_time and now - self.last_update_time[pid] > 3.0:
                    print(f"[DEBUG MONITOR] Person ID {pid} wird aus allen Listen gelöscht (3s Timeout)!")
                    del self.last_update_time[pid]
                    if pid in self.first_seen_time: del self.first_seen_time[pid]
                    if pid in self.known_triangulated_ids: self.known_triangulated_ids.remove(pid)

                if pid in self.skeleton_widgets:
                    widget = self.skeleton_widgets[pid]["container"]
                    for i in range(self.skeletons_layout.count()):
                        item = self.skeletons_layout.itemAt(i)
                        if item and item.widget() == widget:
                            self.skeletons_layout.removeWidget(widget)
                            break
                    widget.deleteLater()
                    del self.skeleton_widgets[pid]
                    needs_relayout = True

        if needs_relayout:
            self._reorganize_grid()

    def render_demo_skeleton(self, colors, angle=None):
        W, H = 280, 480
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        links = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11),
                 (6, 12)]
        x_offset = -10
        y_offset = -30

        if angle is not None:
            cx, cy = int(W / 2), H - 55
            radius = 35

            cv2.circle(canvas, (cx, cy), radius, (30, 30, 35), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), radius, (80, 80, 90), 1, cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, "N", (cx - 5, cy - radius - 5), font, 0.4, (0, 255, 150), 1, cv2.LINE_AA)
            cv2.putText(canvas, "S", (cx - 4, cy + radius + 12), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(canvas, "W", (cx - radius - 15, cy + 4), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(canvas, "O", (cx + radius + 5, cy + 4), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

            rad = np.radians(angle - 90)
            ex = int(cx + np.cos(rad) * radius * 0.9)
            ey = int(cy + np.sin(rad) * radius * 0.9)

            cv2.arrowedLine(canvas, (cx, cy), (ex, ey), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)

        for s, e in links:
            if s in self.TEMPLATE_SKEL and e in self.TEMPLATE_SKEL:
                c1 = colors.get(s, np.array([80, 80, 80]))
                c2 = colors.get(e, np.array([80, 80, 80]))
                c_avg = (c1 + c2) / 2
                color_bgr = (int(c_avg[2]), int(c_avg[1]), int(c_avg[0]))

                pt1 = (self.TEMPLATE_SKEL[s][0] + x_offset, self.TEMPLATE_SKEL[s][1] + y_offset)
                pt2 = (self.TEMPLATE_SKEL[e][0] + x_offset, self.TEMPLATE_SKEL[e][1] + y_offset)

                cv2.line(canvas, pt1, pt2, color_bgr, 5, cv2.LINE_AA)

        if 0 in self.TEMPLATE_SKEL and 5 in self.TEMPLATE_SKEL and 6 in self.TEMPLATE_SKEL:
            mid_shoulder = (
                int((self.TEMPLATE_SKEL[5][0] + self.TEMPLATE_SKEL[6][0]) / 2) + x_offset,
                int((self.TEMPLATE_SKEL[5][1] + self.TEMPLATE_SKEL[6][1]) / 2) + y_offset
            )
            head_pt = (self.TEMPLATE_SKEL[0][0] + x_offset, self.TEMPLATE_SKEL[0][1] + y_offset)
            cv2.line(canvas, mid_shoulder, head_pt, (120, 120, 120), 5, cv2.LINE_AA)

        for j, pt in self.TEMPLATE_SKEL.items():
            c = colors.get(j, np.array([200, 200, 200]))
            color_bgr = (int(c[2]), int(c[1]), int(c[0]))
            adj_pt = (pt[0] + x_offset, pt[1] + y_offset)
            cv2.circle(canvas, adj_pt, 6, color_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, adj_pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

        return self.numpy_to_pixmap(canvas)

    def numpy_to_pixmap(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(q_img)