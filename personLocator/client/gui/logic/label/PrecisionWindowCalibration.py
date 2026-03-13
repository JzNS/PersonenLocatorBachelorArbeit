import uuid
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSlider, QLineEdit, QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent
from PyQt6.QtCore import Qt

from client.gui.logic.label.utilityClasses.utilityClassPrecisionWindow.PrecisionDataManager import PrecisionDataManager
from client.gui.logic.label.utilityClasses.utilityClassPrecisionWindow.PrecisionInteractionHandler import \
    PrecisionInteractionHandler
from client.gui.logic.label.utilityClasses.utilityClassPrecisionWindow.PrecisionRenderer import PrecisionRenderer
from client.gui.logic.label.utilityClasses.utilityClassPrecisionWindow.PrecisionTreeManager import PrecisionTreeManager
from client.gui.logic.label.utilityClasses.utilityClassPrecisionWindow.ZoomableGraphicsView import ZoomableGraphicsView
from client.utils.ConfigManager import ConfigManager



class PrecisionWindowCalibration(QDialog):
    def __init__(self, camera_name: str, parent=None):
        super().__init__(parent)
        self.camera_name = camera_name if isinstance(camera_name, str) else "CAMERA_1"
        self.setWindowTitle(f"Präzisions-Kalibrierung ({self.camera_name})")
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
        self.showMaximized()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.current_live_image = None
        self.overlay_pixmap = None
        self.opacity_level = 0.5

        self.tree_manager = PrecisionTreeManager(self)
        self.interaction_handler = PrecisionInteractionHandler()

        self._setup_ui()
        self._connect_signals()
        self._load_data()

    def _setup_ui(self):
        self.main_layout = QHBoxLayout(self)

        # Links: Bild-Ansicht
        self.image_view = ZoomableGraphicsView()
        self.main_layout.addWidget(self.image_view, stretch=3)

        # Rechts: Sidebar
        self.sidebar_layout = QVBoxLayout()
        self._build_loupe_widget()
        self._build_room_dims_widget()
        self._build_overlay_widget()
        self._build_add_buttons()

        # TreeManager als Widget einfügen
        self.sidebar_layout.addWidget(self.tree_manager, stretch=1)

        self.close_btn = QPushButton("Speichern & Schließen")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold; padding: 10px;")
        self.sidebar_layout.addWidget(self.close_btn)

        self.main_layout.addLayout(self.sidebar_layout, stretch=1)

    def _connect_signals(self):
        """Verdrahtet alle Manager und Widgets miteinander."""
        self.image_view.left_clicked.connect(self.interaction_handler.handle_left_click)
        self.image_view.right_clicked.connect(self.interaction_handler.handle_right_click)
        self.image_view.mouse_moved.connect(self.update_loupe)

        self.interaction_handler.zone_info_changed.connect(self.lbl_zone_info.setText)
        self.interaction_handler.cursor_changed.connect(self.setCursor)
        self.interaction_handler.redraw_requested.connect(self._force_redraw)
        self.interaction_handler.zone_completed.connect(self.tree_manager.add_rectangle)

        self.interaction_handler.pixel_click_requested.connect(self.tree_manager.process_pixel_click)

        self.tree_manager.needs_redraw.connect(self._force_redraw)

    def _build_loupe_widget(self):
        """Erstellt das Widget für die Live-Pixel-Lupe."""
        grp = QGroupBox("Live Pixel-Lupe")
        layout = QVBoxLayout()
        self.lbl_loupe = QLabel()
        self.lbl_loupe.setFixedSize(100, 100)
        self.lbl_loupe.setStyleSheet("background-color: black; border: 2px solid #555;")
        layout.addWidget(self.lbl_loupe, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_loupe_coords = QLabel("X: 0 | Y: 0")
        self.lbl_loupe_coords.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_loupe_coords.setStyleSheet("color: #00ff00; font-family: monospace; font-weight: bold;")
        font = self.lbl_loupe_coords.font()
        font.setPointSize(9)
        self.lbl_loupe_coords.setFont(font)

        layout.addWidget(self.lbl_loupe_coords)
        grp.setLayout(layout)
        self.sidebar_layout.addWidget(grp)

    def _build_room_dims_widget(self):
        """Erstellt das Widget für die Eingabe der Raum-Dimensionen."""
        grp = QGroupBox("Raum-Dimensionen (cm)")
        layout = QGridLayout()
        self.edit_width = QLineEdit()
        self.edit_height = QLineEdit()
        self.edit_depth = QLineEdit()
        layout.addWidget(QLabel("Breite:"), 0, 0);
        layout.addWidget(self.edit_width, 0, 1)
        layout.addWidget(QLabel("Höhe:"), 1, 0);
        layout.addWidget(self.edit_height, 1, 1)
        layout.addWidget(QLabel("Tiefe:"), 2, 0);
        layout.addWidget(self.edit_depth, 2, 1)
        grp.setLayout(layout)
        self.sidebar_layout.addWidget(grp)

    def _build_overlay_widget(self):
        """Erstellt das Widget für die Overlay-Steuerung."""
        grp = QGroupBox("Overlay Steuerung")
        layout = QVBoxLayout()
        btn_snap = QPushButton("📸 Snapshot erstellen & laden")
        btn_snap.clicked.connect(self.take_snapshot)
        btn_snap.setStyleSheet("background-color: #444488; color: white; padding: 5px;")

        box_slider = QHBoxLayout()
        box_slider.addWidget(QLabel("Deckkraft:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.valueChanged.connect(self._on_opacity_slider_changed)
        box_slider.addWidget(slider)

        layout.addWidget(btn_snap)
        layout.addLayout(box_slider)
        grp.setLayout(layout)
        self.sidebar_layout.addWidget(grp)

    def _build_add_buttons(self):
        """Erstellt die Buttons zum Hinzufügen von Zonen und Vierecken."""
        grp = QGroupBox("Zonen & Vierecke hinzufügen")
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)

        self.btn_add_rect = QPushButton("➕ Neues Custom-Viereck")
        self.btn_add_rect.setStyleSheet("background-color: #225522;")
        self.btn_add_rect.clicked.connect(self.create_new_rectangle)
        layout.addWidget(self.btn_add_rect)

        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #555;")
        layout.addWidget(line)

        hbox_zones = QHBoxLayout()
        self.btn_dead = QPushButton("⛔ Dead Zone")
        self.btn_dead.setStyleSheet("background-color: #552222;")
        self.btn_dead.clicked.connect(lambda: self.interaction_handler.start_zone_drawing("Dead Zone"))

        self.btn_mirror = QPushButton("🪞 Mirror Zone")
        self.btn_mirror.setStyleSheet("background-color: #222255;")
        self.btn_mirror.clicked.connect(lambda: self.interaction_handler.start_zone_drawing("Mirror Zone"))

        hbox_zones.addWidget(self.btn_dead)
        hbox_zones.addWidget(self.btn_mirror)
        layout.addLayout(hbox_zones)

        self.lbl_zone_info = QLabel("")
        self.lbl_zone_info.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.lbl_zone_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_zone_info)

        grp.setLayout(layout)
        self.sidebar_layout.addWidget(grp)

    # --- DATENLADEN ---
    def _load_data(self):
        """Lädt alle Rechtecke, Zonen und Raummaße aus der Datenbank und Konfiguration."""
        data = PrecisionDataManager.load_all_rectangles(self.camera_name)

        cam_settings = ConfigManager.get_camera_settings(self.camera_name)
        for z_pts in cam_settings.get("dead_zones", []):
            data.append(self.interaction_handler._create_zone_dict("Dead Zone", z_pts))
        for z_pts in cam_settings.get("mirror_zones", []):
            data.append(self.interaction_handler._create_zone_dict("Mirror Zone", z_pts))

        self.tree_manager.load_data(data)

        # Raummaße laden
        all_configs = ConfigManager.load_camera_config()
        room_dims = all_configs.get("Camera_ALL", {}).get("room_dimensions",
                                                          {"width": 600.0, "height": 250.0, "depth": 800.0})
        self.edit_width.setText(str(room_dims.get("width", 600.0)))
        self.edit_height.setText(str(room_dims.get("height", 250.0)))
        self.edit_depth.setText(str(room_dims.get("depth", 800.0)))

        snap_path = ConfigManager.SNAPSHOT_FILE
        if snap_path.exists():
            self.overlay_pixmap = QPixmap(str(snap_path))

    def create_new_rectangle(self):
        """Fügt ein neues, leeres Rechteck mit Standardwerten hinzu und öffnet es zur Bearbeitung."""
        rect_data = {
            "internal_id": str(uuid.uuid4()), "display_id": str(len(self.tree_manager.rectangles_data)),
            "type": "Neues Viereck", "size_cm": 0,
            "corners": [
                {"label": "Oben-Links", "x": 0.0, "y": 0.0, "z": 0.0, "px": None, "py": None},
                {"label": "Oben-Rechts", "x": 0.0, "y": 0.0, "z": 0.0, "px": None, "py": None},
                {"label": "Unten-Rechts", "x": 0.0, "y": 0.0, "z": 0.0, "px": None, "py": None},
                {"label": "Unten-Links", "x": 0.0, "y": 0.0, "z": 0.0, "px": None, "py": None}
            ]
        }
        self.tree_manager.add_rectangle(rect_data)

    # --- RENDERING ---
    def _force_redraw(self):
        if self.current_live_image and not self.current_live_image.isNull():
            self.update_frame(self.current_live_image)

    def _on_opacity_slider_changed(self, value):
        self.opacity_level = value / 100.0
        self._force_redraw()

    def update_frame(self, raw_qimage: QImage):
        if raw_qimage.isNull():
            self._unlock_worker()
            return

        self.current_live_image = raw_qimage.copy()
        rendered_pixmap = PrecisionRenderer.render_main_image(
            raw_qimage, self.overlay_pixmap, self.opacity_level,
            self.tree_manager.rectangles_data,
            self.interaction_handler.draw_mode,
            self.interaction_handler.temp_points
        )
        self.image_view.set_pixmap(rendered_pixmap)
        self._unlock_worker()

    def _unlock_worker(self):
        if self.parent() and hasattr(self.parent(), 'worker'):
            self.parent().worker.lens_window_ready = True

    def update_loupe(self, x, y):
        if not self.current_live_image or self.current_live_image.isNull(): return
        self.lbl_loupe_coords.setText(f"X: {x} | Y: {y}")
        if 0 <= x < self.current_live_image.width() and 0 <= y < self.current_live_image.height():
            self.lbl_loupe.setPixmap(PrecisionRenderer.create_loupe_pixmap(self.current_live_image, x, y))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.tree_manager.select_next_item()
        elif event.key() == Qt.Key.Key_Delete:
            self.tree_manager.delete_selected()
        else:
            super().keyPressEvent(event)

    def take_snapshot(self):
        if self.current_live_image and not self.current_live_image.isNull():
            save_path = ConfigManager.SNAPSHOT_FILE
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if self.current_live_image.save(str(save_path)):
                self.overlay_pixmap = QPixmap.fromImage(self.current_live_image)

    def closeEvent(self, event):
        """Speichert alle Rechtecke, Zonen und Raummaße in der Datenbank und Konfiguration, bevor das Fenster geschlossen wird."""
        dead_zones, mirror_zones = PrecisionDataManager.extract_zones(self.tree_manager.rectangles_data)

        try:
            new_dims = {
                "width": float(self.edit_width.text().replace(',', '.')),
                "height": float(self.edit_height.text().replace(',', '.')),
                "depth": float(self.edit_depth.text().replace(',', '.'))
            }
        except ValueError:
            new_dims = None

        if self.parent() and hasattr(self.parent(), 'sender') and self.parent().sender:
            sender = self.parent().sender

            # Payloads für Server bauen
            rects_payload, pixel_payload = PrecisionDataManager.build_save_payloads(
                self.tree_manager.rectangles_data, self.tree_manager.deleted_rect_ids
            )

            if rects_payload or self.tree_manager.deleted_rect_ids:
                sender.send_db_global_rectangles(rects_payload)
                sender.send_db_camera_pixels(self.camera_name, pixel_payload)

            ConfigManager.update_camera_settings(self.camera_name, {
                "dead_zones": dead_zones,
                "mirror_zones": mirror_zones
            })

            full_settings = ConfigManager.get_camera_settings(self.camera_name)
            sender.send_db_camera_settings(self.camera_name, full_settings)

            if new_dims:
                sender.send_db_global_room(new_dims["width"], new_dims["height"], new_dims["depth"])

        super().closeEvent(event)