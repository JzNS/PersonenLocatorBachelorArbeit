from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QGroupBox)


class TrackingSettingsWindow(QDialog):
    def __init__(self, current_mode: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Tracking & Sensor-Fusion Einstellungen")
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)

        self.selected_mode = current_mode

        layout = QVBoxLayout(self)

        group_algo = QGroupBox("1. Aktiver Fusions-Algorithmus")
        group_algo.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        layout_algo = QVBoxLayout(group_algo)

        self.combo_algo = QComboBox()
        self.combo_algo.addItem("🧠 Deep SORT Light (Hungarian + Color + 3D)", "hungarian")
        self.combo_algo.addItem("⚡ Greedy Matching (Schnelle Distanz-Baseline)", "greedy")

        self.combo_algo.setStyleSheet("""
            QComboBox { padding: 8px; font-weight: bold; background-color: #2A2A2A; color: white; border-radius: 4px; border: 1px solid #555; }
            QComboBox::drop-down { border: 0px; }
        """)

        index = self.combo_algo.findData(current_mode)
        if index >= 0:
            self.combo_algo.setCurrentIndex(index)

        self.lbl_description = QLabel()
        self.lbl_description.setWordWrap(True)
        self.lbl_description.setStyleSheet("color: #AAAAAA; font-style: italic; margin-top: 10px;")

        layout_algo.addWidget(self.combo_algo)
        layout_algo.addWidget(self.lbl_description)
        layout.addWidget(group_algo)

        self.combo_algo.currentIndexChanged.connect(self._on_algo_changed)
        self._on_algo_changed()

        layout.addStretch()
        group_filter = QGroupBox("2. Skelett-Glättung (Smoothing)")
        group_filter.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; margin-top: 10px; }")
        layout_filter = QVBoxLayout(group_filter)

        layout.addWidget(group_filter)
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("🏃‍♂️ 1-Euro Filter (Null Lag, Perfekt für Fitness)", "one_euro")
        self.combo_filter.addItem("🛸 Kalman Filter (Weich, Gut für langsames Gehen)", "kalman")

        self.combo_filter.setStyleSheet("""
                    QComboBox { padding: 8px; font-weight: bold; background-color: #2A2A2A; color: white; border-radius: 4px; border: 1px solid #555; }
                    QComboBox::drop-down { border: 0px; }
                """)

        current_filter = "one_euro"
        if hasattr(parent, 'current_smoothing_filter'):
            current_filter = parent.current_smoothing_filter

        index_filter = self.combo_filter.findData(current_filter)
        if index_filter >= 0:
            self.combo_filter.setCurrentIndex(index_filter)

        layout_filter.addWidget(self.combo_filter)

        group_ik = QGroupBox("3. Skelett-Anatomie (Inverse Kinematik)")
        group_ik.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; margin-top: 10px; }")
        layout_ik = QVBoxLayout(group_ik)

        self.combo_ik = QComboBox()
        self.combo_ik.addItem("🦴 FABRIK IK (Organisch, Perfekte Knochen, Industrie-Standard)", "fabrik")
        self.combo_ik.addItem("📏 Classic Constraints (Hartes Clipping, Alte Version)", "classic")

        self.combo_ik.setStyleSheet("""
                    QComboBox { padding: 8px; font-weight: bold; background-color: #2A2A2A; color: white; border-radius: 4px; border: 1px solid #555; }
                    QComboBox::drop-down { border: 0px; }
                """)

        current_ik = "fabrik"
        if hasattr(parent, 'current_ik_mode'):
            current_ik = parent.current_ik_mode

        index_ik = self.combo_ik.findData(current_ik)
        if index_ik >= 0:
            self.combo_ik.setCurrentIndex(index_ik)

        layout_ik.addWidget(self.combo_ik)
        layout.addWidget(group_ik)

        group_tri = QGroupBox("4. Triangulations-Methode (3D-Schnitt)")
        group_tri.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; margin-top: 10px; }")
        layout_tri = QVBoxLayout(group_tri)

        self.combo_tri = QComboBox()
        self.combo_tri.addItem("🎯 Levenberg-Marquardt (Pixel-Reprojektion, Goldstandard)", "lm")
        self.combo_tri.addItem("⚡ Weighted Least Squares (Sichtstrahlen, Sehr Schnell)", "wls")

        self.combo_tri.setStyleSheet("""
                    QComboBox { padding: 8px; font-weight: bold; background-color: #2A2A2A; color: white; border-radius: 4px; border: 1px solid #555; }
                    QComboBox::drop-down { border: 0px; }
                """)

        current_tri = "lm"
        if hasattr(parent, 'current_triangulation_mode'):
            current_tri = parent.current_triangulation_mode

        index_tri = self.combo_tri.findData(current_tri)
        if index_tri >= 0:
            self.combo_tri.setCurrentIndex(index_tri)

        layout_tri.addWidget(self.combo_tri)
        layout.addWidget(group_tri)
        layout_buttons = QHBoxLayout()
        btn_cancel = QPushButton("Abbrechen")
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("💾 Speichern & Anwenden")
        btn_save.setStyleSheet(
            "background-color: #0055AA; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        btn_save.clicked.connect(self.accept)

        layout_buttons.addStretch()
        layout_buttons.addWidget(btn_cancel)
        layout_buttons.addWidget(btn_save)
        layout.addLayout(layout_buttons)

    def _on_algo_changed(self):
        mode_data = self.combo_algo.currentData()

        if mode_data == "hungarian":
            self.lbl_description.setText(
                "Nutzt den Hungarian Algorithmus. Löst Verdeckungen extrem sicher auf. (Fest optimiert auf den Industrie-Standard: 70% Spatial-Distanz / 30% RGB-Farbe)")
        elif mode_data == "greedy":
            self.lbl_description.setText(
                "Nimmt einfach den nächstbesten 3D-Punkt. Sehr rechenleicht, tauscht aber IDs aus, wenn Personen direkt aneinander vorbeilaufen.")
        else:
            self.lbl_description.setText("Dieser Algorithmus ist noch in Entwicklung und aktuell gesperrt.")

    def get_selected_triangulation(self) -> str:
        return self.combo_tri.currentData()
    def get_selected_ik(self) -> str:
        return self.combo_ik.currentData()
    def get_selected_filter(self) -> str:
        return self.combo_filter.currentData()
    def get_selected_mode(self) -> str:
        if self.combo_algo.currentData() == "voxel_future":
            return "hungarian"
        return self.combo_algo.currentData()