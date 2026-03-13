from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QProgressBar, \
    QTabWidget
from PyQt6.QtCore import Qt


class PersonDetailTab(QWidget):
    """
    Eine Unter-Komponente, die genau EINE Person detailliert anzeigt.
    Wird vom KeypointWidget als Tab verwaltet.
    """

    # Vollständige COCO-Keypoint Liste (0-16)
    ALL_KEYPOINTS = {
        0: "Nase", 1: "Auge L", 2: "Auge R", 3: "Ohr L", 4: "Ohr R",
        5: "Schulter L", 6: "Schulter R",
        7: "Ellbogen L", 8: "Ellbogen R",
        9: "Handgelenk L", 10: "Handgelenk R",
        11: "Hüfte L", 12: "Hüfte R",
        13: "Knie L", 14: "Knie R",
        15: "Fuß L", 16: "Fuß R"
    }

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        # Status Label
        self.lbl_status = QLabel("Initialisiere...")
        self.lbl_status.setStyleSheet("color: #00ff7f; font-weight: bold; font-size: 12px;")
        self.layout.addWidget(self.lbl_status)

        # Tabelle für alle Gelenke
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Gelenk", "Sicherheit"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)

        # Dark Mode Design
        self.table.setStyleSheet("""
            QTableWidget { background-color: #222; color: white; gridline-color: #444; border: none; }
            QHeaderView::section { background-color: #333; color: white; padding: 4px; border: 1px solid #222; }
            QScrollBar:vertical { background: #111; width: 10px; }
            QScrollBar::handle:vertical { background: #555; }
        """)
        self.layout.addWidget(self.table)

        self.table.setRowCount(len(self.ALL_KEYPOINTS))
        for row, name in self.ALL_KEYPOINTS.items():
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, item)

            self.table.setItem(row, 1, QTableWidgetItem("-"))

    def update_view(self, person_data: dict):
        """Aktualisiert die Daten für diese spezifische Person."""
        status = person_data.get('status', 'Unbekannt')
        height = person_data.get('height', 0)
        self.lbl_status.setText(f"Status: {status} | Höhe: {height:.1f}cm")

        kp_map = {k['id']: k['c'] for k in person_data.get('keypoints', [])}

        for row_id, name in self.ALL_KEYPOINTS.items():
            conf = kp_map.get(row_id, 0.0)
            pbar = QProgressBar()
            pbar.setValue(int(conf * 100))
            pbar.setFormat(f"{int(conf * 100)}%")
            pbar.setAlignment(Qt.AlignmentFlag.AlignCenter)

            color = "#00ff00" if conf > 0.8 else ("#ffff00" if conf > 0.5 else "#ff3333")

            pbar.setStyleSheet(f"""
                QProgressBar {{ border: 0px; background-color: #444; color: black; border-radius: 2px; text-align: center; }}
                QProgressBar::chunk {{ background-color: {color}; }}
            """)

            self.table.setCellWidget(row_id, 1, pbar)