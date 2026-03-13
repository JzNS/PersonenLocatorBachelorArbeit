from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QComboBox, QVBoxLayout, QSlider, QLabel, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from client.utils.ConfigManager import ConfigManager  # <--- NEU IMPORTIERT


class SettingsDialog(QDialog):
    capacity_changed = pyqtSignal(int)

    def __init__(self, parent, camera_name, settings):
        super().__init__(parent)
        self.setWindowTitle(f"Einstellungen: {camera_name}")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.inputs = {}

        self.slider_capacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_capacity.setRange(10, 100)
        initial_cap = settings.get("render_capacity", 100)
        self.slider_capacity.setValue(initial_cap)
        self.lbl_cap_val = QLabel(f"{initial_cap}%")
        self.slider_capacity.valueChanged.connect(self._on_slider_moved)

        cap_layout = QHBoxLayout()
        cap_layout.addWidget(self.slider_capacity)
        cap_layout.addWidget(self.lbl_cap_val)
        form.addRow("YOLO Render Capacity", cap_layout)

        # ==========================================
        # GLOBALE RAUM-MAßE
        # ==========================================
        all_configs = ConfigManager.load_camera_config()
        global_data = all_configs.get("Camera_ALL", {})

        room_dims = global_data.get("room_dimensions", {"width": 320.0, "height": 250.0, "depth": 470.0})

        self.inp_room_w = QLineEdit(str(room_dims.get("width", 320.0)))
        self.inp_room_h = QLineEdit(str(room_dims.get("height", 250.0)))
        self.inp_room_d = QLineEdit(str(room_dims.get("depth", 470.0)))

        val_room = QDoubleValidator(10.0, 10000.0, 1)
        val_room.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.inp_room_w.setValidator(val_room)
        self.inp_room_h.setValidator(val_room)
        self.inp_room_d.setValidator(val_room)

        form.addRow("🌍 Raum Breite (X cm)", self.inp_room_w)
        form.addRow("🌍 Raum Höhe (Y cm)", self.inp_room_h)
        form.addRow("🌍 Raum Tiefe (Z cm)", self.inp_room_d)

        self.combo_profile = QComboBox()
        profiles = [k for k, v in global_data.items() if isinstance(v, dict) and "camera_matrix" in v]
        if not profiles: profiles = ["Default"]
        self.combo_profile.addItems(profiles)

        current_profile = settings.get("active_lens_profile", "Default")
        if current_profile not in profiles: self.combo_profile.addItem(current_profile)
        self.combo_profile.setCurrentText(current_profile)
        form.addRow("Objektiv-Profil", self.combo_profile)

        # --- Dynamische Felder mit Validatoren ---
        fields_config = [
            ("Kamera Index", "camera_index", "int"),
            ("Ziel FPS", "target_fps", "int"),
            ("Zoom Faktor", "zoom", "float"),
            ("Rotation", "rotation", "int"),
        ]

        for label, key, dtype in fields_config:
            val = settings.get(key, 0)
            widget = QLineEdit(str(val))

            if dtype == "int":
                widget.setValidator(QIntValidator(0, 999))
            else:
                validator = QDoubleValidator(0.1, 10.0, 2)
                validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                widget.setValidator(validator)

            form.addRow(label, widget)
            self.inputs[key] = (widget, dtype)

        # --- Auflösung ---
        res_options = ["640x480", "1280x720", "1920x1080"]
        current_res_list = settings.get("resolution", [1280, 720])
        current_res_str = f"{current_res_list[0]}x{current_res_list[1]}"
        if current_res_str not in res_options: res_options.insert(0, current_res_str)
        self.combo_res = QComboBox()
        self.combo_res.addItems(res_options)
        self.combo_res.setCurrentText(current_res_str)
        form.addRow("Auflösung", self.combo_res)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_slider_moved(self, val):
        self.lbl_cap_val.setText(f"{val}%")
        self.capacity_changed.emit(val)

    def get_data(self):
        """Gibt die Daten bereits im richtigen Datentyp zurück."""
        results = {}
        for key, (widget, dtype) in self.inputs.items():
            # Komma durch Punkt ersetzen (Europa-Fix)
            raw_text = widget.text().replace(',', '.')
            if not raw_text: raw_text = "0"

            try:
                if dtype == "int":
                    results[key] = int(float(raw_text))
                else:
                    results[key] = float(raw_text)
            except ValueError:
                results[key] = 0 if dtype == "int" else 1.0

        res_str = self.combo_res.currentText()
        results["resolution"] = [int(x) for x in res_str.split('x')]
        results["render_capacity"] = self.slider_capacity.value()
        results["active_lens_profile"] = self.combo_profile.currentText()
        results["room_dimensions"] = {
            "width": float(self.inp_room_w.text().replace(',', '.') or 320.0),
            "height": float(self.inp_room_h.text().replace(',', '.') or 250.0),
            "depth": float(self.inp_room_d.text().replace(',', '.') or 470.0)
        }
        return results