from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
                             QPushButton, QHBoxLayout, QLabel, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from typing import Any, List, Optional, Dict
import numpy as np

class PointersWindow(QDialog):
    """
    Monitor-Fenster für Interaktionen (Pointing Rays).
    Zeigt für jede Person an, wohin sie zeigt und was sie trifft.
    """

    def __init__(self, controller: Any, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("🎯 Interaction & Pointer Monitor")
        self.resize(700, 500)

        layout = QVBoxLayout(self)
        
        info_label = QLabel("Übersicht aller aktiven Pointing-Vektoren (FUSION-Ebene)")
        info_label.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(info_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Person", "Arm", "Ziel-Typ", "Objekt / Name", "Entfernung", "Koordinaten (X, Y, Z)"])
        self.tree.setColumnWidth(0, 100)
        self.tree.setColumnWidth(1, 60)
        self.tree.setColumnWidth(2, 80)
        self.tree.setColumnWidth(3, 120)
        self.tree.setColumnWidth(4, 80)
        
        self.tree.setStyleSheet("""
            QTreeWidget { background-color: #f9f9f9; }
            QTreeWidget::item { padding: 5px; border-bottom: 1px solid #ddd; }
        """)
        layout.addWidget(self.tree)

        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Manuell Aktualisieren")
        self.btn_refresh.clicked.connect(self.update_ui)
        btn_layout.addWidget(self.btn_refresh)
        
        self.btn_close = QPushButton("Schließen")
        self.btn_close.clicked.connect(self.hide)
        btn_layout.addWidget(self.btn_close)
        
        layout.addLayout(btn_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)

    def update_ui(self) -> None:
        if not self.controller or not hasattr(self.controller, 'tracker'):
            return


        persons = self.controller.tracker.global_persons

        self.tree.clear()
        
        for gp in persons:
            if not hasattr(gp, 'current_pointers') or not gp.current_pointers:
                continue
                
            p_name = f"ID {gp.id} ({gp.name})"
            
            for side, data in gp.current_pointers.items():
                arm_text = "Links" if side == "left" else "Rechts"
                hit_type = data.get("type", "wall")
                hit_label = data.get("label", "Wand")
                hit_dist = data.get("dist", 0.0)
                hit_pos = data.get("pos")
                
                type_icon = "🧱"
                if hit_type == "person": type_icon = "👤"
                elif hit_type == "object": type_icon = "📦"
                
                coords_str = "N/A"
                if hit_pos is not None:
                    coords_str = f"{hit_pos[0]:.0f}, {hit_pos[1]:.0f}, {hit_pos[2]:.0f}"

                item = QTreeWidgetItem([
                    p_name,
                    arm_text,
                    f"{type_icon} {hit_type.upper()}",
                    hit_label,
                    f"{hit_dist:.1f} cm",
                    coords_str
                ])

                if hit_type == "person":
                    item.setBackground(3, Qt.GlobalColor.yellow)
                elif hit_type == "object":
                    item.setBackground(3, Qt.GlobalColor.cyan)
                    
                self.tree.addTopLevelItem(item)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.update_ui()
