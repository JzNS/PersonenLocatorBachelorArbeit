from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QCheckBox,
                             QLabel, QPushButton, QScrollArea, QWidget, QFrame, QComboBox)
from PyQt6.QtCore import Qt


class MasterSettingsWindow(QDialog):
    JOINT_NAMES = {
        0: "Nase", 1: "Auge L", 2: "Auge R", 3: "Ohr L", 4: "Ohr R",
        5: "Schulter L", 6: "Schulter R", 7: "Ellenbogen L", 8: "Ellenbogen R",
        9: "Hand L", 10: "Hand R", 11: "Hüfte L", 12: "Hüfte R",
        13: "Knie L", 14: "Knie R", 15: "Fuß L", 16: "Fuß R"
    }

    def __init__(self, current_settings, active_cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Master View Einstellungen & Fusion Override")
        self.resize(1100, 700)
        self.setStyleSheet("background-color: #222; color: #EEE;")

        self.settings = current_settings
        if "force_single_person" not in self.settings:
            self.settings["force_single_person"] = False

        self.cameras = [cam for cam in active_cameras if cam.startswith("CAMERA_") or cam == "MASTER_FUSION"]
        self.cameras.sort()

        self.ui_checkboxes = {}
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        override_frame = QFrame()
        override_frame.setStyleSheet(
            "background-color: #442222; border: 2px solid #AA5555; border-radius: 8px; margin-bottom: 10px;")
        override_layout = QVBoxLayout(override_frame)

        lbl_fusion = QLabel("🔥 MANUELLER FUSION OVERRIDE")
        lbl_fusion.setStyleSheet("font-weight: bold; font-size: 16px; color: #FFAAAA; border: none;")
        override_layout.addWidget(lbl_fusion)

        self.chk_force_single = QCheckBox("Alle erkannten Personen als EINE Person behandeln (Erzwingt ID 1)")
        self.chk_force_single.setStyleSheet("font-size: 14px; font-weight: bold; color: white; border: none;")
        self.chk_force_single.setChecked(self.settings.get("force_single_person", False))
        self.chk_force_single.stateChanged.connect(self.on_force_single_toggle)
        override_layout.addWidget(self.chk_force_single)

        main_layout.addWidget(override_frame)

        info = QLabel("Detaillierte Kamera- und Gelenk-Filter:")
        info.setStyleSheet("font-size: 12px; color: #AAA;")
        main_layout.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        scroll_content = QWidget()
        self.columns_layout = QHBoxLayout(scroll_content)

        self.build_column("ALL CAMERAS", is_global=True)

        for cam_name in self.cameras:
            if cam_name not in self.settings or isinstance(self.settings[cam_name], bool):
                self.settings[cam_name] = {
                    "master": True,
                    "show_rays": True,
                    "show_single_rays": True,
                    "show_points": True,
                    "show_bones": cam_name == "MASTER_FUSION",
                    "show_arm_extension": False,
                    "joints": {i: True for i in range(17)}
                }

            if not self.settings[cam_name].get("joints"):
                self.settings[cam_name]["joints"] = {i: True for i in range(17)}
            self.build_column(cam_name, is_global=False)

        self.columns_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        btn_close = QPushButton("Einstellungen Übernehmen")
        btn_close.setStyleSheet("background-color: #008080; padding: 12px; font-weight: bold; font-size: 14px;")
        btn_close.clicked.connect(self.accept)
        main_layout.addWidget(btn_close)

    def build_column(self, title, is_global):
        col_frame = QFrame()
        col_frame.setStyleSheet("background-color: #333; border: 1px solid #555; border-radius: 5px;")
        col_layout = QVBoxLayout(col_frame)
        col_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        ui_title = "FUSION" if title == "MASTER_FUSION" else title
        lbl_title = QLabel(ui_title)
        if title == "MASTER_FUSION":
            lbl_title.setStyleSheet("font-weight: bold; color: #FF69B4;")
        elif is_global:
            lbl_title.setStyleSheet("font-weight: bold; color: #00FFFF;")
        else:
            lbl_title.setStyleSheet("font-weight: bold; color: #FFA500;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col_layout.addWidget(lbl_title)

        if title not in self.ui_checkboxes: self.ui_checkboxes[title] = {}

        keys = [("master", "Aktiv", "white"), ("show_rays", "Strahlen", "#00FFFF"),
                ("show_points", "Punkte", "#FFA500"), ("show_bones", "Knochen", "#FF69B4"),
                ("show_arm_extension", "Arm-Verlängerung", "#ADFF2F")]

        for key, text, color in keys:
            chk = QCheckBox(text)
            chk.setStyleSheet(f"font-weight: bold; color: {color};")
            if not is_global:
                chk.setChecked(self.settings[title].get(key, True if key not in ["show_bones", "show_arm_extension"] else False))
                chk.stateChanged.connect(
                    lambda state, k=key, c=title: self.on_setting_change(c, k, state == Qt.CheckState.Checked.value))
            else:
                chk.stateChanged.connect(
                    lambda state, k=key: self.apply_to_all_bool(k, state == Qt.CheckState.Checked.value))
            self.ui_checkboxes[title][key] = chk
            col_layout.addWidget(chk)

        if not is_global and title != "MASTER_FUSION":
            col_layout.addWidget(QLabel("Kamera-Typ:"))
            type_combo = QComboBox()
            type_combo.addItems(["Standard", "Top-Down"])
            current_type = self.settings[title].get("camera_type", "Standard")
            type_combo.setCurrentText(current_type)
            type_combo.currentTextChanged.connect(
                lambda text, c=title: self.on_setting_change(c, "camera_type", text))
            col_layout.addWidget(type_combo)

        self.ui_checkboxes[title]["joints"] = {}
        for j_id, j_name in self.JOINT_NAMES.items():
            chk_joint = QCheckBox(f"{j_id}: {j_name}")
            if not is_global:
                chk_joint.setChecked(self.settings[title].get("joints", {}).get(j_id, True))
                chk_joint.stateChanged.connect(
                    lambda state, c=title, j=j_id: self.on_joint_toggle(c, j, state == Qt.CheckState.Checked.value))
            else:
                chk_joint.stateChanged.connect(
                    lambda state, j=j_id: self.apply_to_all_joint(j, state == Qt.CheckState.Checked.value))
            self.ui_checkboxes[title]["joints"][j_id] = chk_joint
            col_layout.addWidget(chk_joint)

        self.columns_layout.addWidget(col_frame)

    def on_force_single_toggle(self, state):
        self.settings["force_single_person"] = (state == Qt.CheckState.Checked.value)

    def on_setting_change(self, cam_name, key, is_active):
        if cam_name not in self.settings: self.settings[cam_name] = {}
        self.settings[cam_name][key] = is_active

    def on_joint_toggle(self, cam_name, j_id, is_active):
        if "joints" not in self.settings[cam_name]: self.settings[cam_name]["joints"] = {}
        self.settings[cam_name]["joints"][j_id] = is_active

    def apply_to_all_bool(self, key, is_active):
        for cam_name in self.cameras:
            self.on_setting_change(cam_name, key, is_active)
            if key in self.ui_checkboxes[cam_name]: self.ui_checkboxes[cam_name][key].setChecked(is_active)

    def apply_to_all_joint(self, j_id, is_active):
        for cam_name in self.cameras:
            self.on_joint_toggle(cam_name, j_id, is_active)
            self.ui_checkboxes[cam_name]["joints"][j_id].setChecked(is_active)

    def get_settings(self):
        return self.settings