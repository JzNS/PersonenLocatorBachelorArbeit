import time
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QLabel, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QHeaderView, QGridLayout,
                             QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox, QSizePolicy,
                             QLineEdit, QFileDialog, QScrollArea, QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QColor, QBrush, QPainter, QPen, QFont

import pyqtgraph as pg
from server.gui.windows.QuickEvalDialog import QuickEvalDialog


class PieChartWidget(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.data = {}
        self.setMinimumSize(300, 300)

    def set_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#121212"))

        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(QRectF(0, 10, self.width(), 30), Qt.AlignmentFlag.AlignCenter, self.title)

        if not self.data or sum(v[0] for v in self.data.values()) == 0:
            painter.setPen(QColor("#777777"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sammle Historische Daten...")
            return

        total = sum(v[0] for v in self.data.values())
        margin = 60
        pie_size = min(self.width() - 250, self.height() - margin * 2)
        pie_rect = QRectF(20, margin, pie_size, pie_size)

        start_angle = 90 * 16
        legend_x = pie_rect.right() + 30
        legend_y = margin + 10

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))

        for label, (value, color_hex) in self.data.items():
            if value == 0: continue
            span_angle = int(-(value / total) * 360 * 16)
            painter.setBrush(QColor(color_hex))
            painter.setPen(QPen(QColor("#121212"), 3))
            painter.drawPie(pie_rect, start_angle, span_angle)
            start_angle += span_angle

            painter.setBrush(QColor(color_hex))
            painter.drawRect(int(legend_x), int(legend_y), 15, 15)

            painter.setPen(QColor("#e0e0e0"))
            percentage = (value / total) * 100
            text = f"{label}\n{int(value):,} ({percentage:.1f}%)"
            painter.drawText(QRectF(legend_x + 25, legend_y - 5, 200, 40), Qt.AlignmentFlag.AlignLeft, text)
            legend_y += 45


class EvaluationWindow(QMainWindow):
    KP_NAMES = {
        0: "Nase", 1: "Auge Li", 2: "Auge Re", 3: "Ohr Li", 4: "Ohr Re",
        5: "Schulter Li", 6: "Schulter Re", 7: "Ellenbogen Li", 8: "Ellenbogen Re",
        9: "Hand Li", 10: "Hand Re", 11: "Hüfte Li", 12: "Hüfte Re",
        13: "Knie Li", 14: "Knie Re", 15: "Fuß Li", 16: "Fuß Re"
    }

    CAM_COLORS = ["#00FFFF", "#FF00FF", "#FFFF00", "#00FF00", "#FFA500"]

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("📈 System-Evaluierung & 3D-Filter-Diagnostik")
        self.resize(1450, 950)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', '#d3d3d3')

        self.start_time = time.time()
        self.history_size = 300
        self.time_data = deque(maxlen=self.history_size)

        self.error_data = deque(maxlen=self.history_size)
        self.kalman_smooth_history = deque(maxlen=self.history_size)
        self.ik_resize_history = deque(maxlen=self.history_size)

        self.camera_packet_count = {}
        self.cam_arrival_times = {}
        self.fps_history = {}
        self.jitter_history = {}
        self.fps_curves = {}
        self.jitter_curves = {}

        self.joint_conf_sum = {}
        self.joint_conf_count = {}

        self.historical_ghosts = 0
        self.historical_valid = 0
        self.tree_items = {}

        self.stat_ticks = 0
        self.all_time_kalman_smooth_cm = 0.0
        self.all_time_ik_correction_cm = 0.0
        self.last_frame_sigs = {}

        self.client_fps_history = {}
        self.server_tri_fps_history = deque(maxlen=self.history_size)
        self.client_fps_curves = {}
        self.server_tri_fps_curve = None

        self.setup_ui(layout)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_packets)
        self.poll_timer.start(10)

        self.gui_timer = QTimer(self)
        self.gui_timer.timeout.connect(self.update_stats)
        self.gui_timer.start(500)

    def setup_ui(self, main_layout):
        self._build_controls_bar(main_layout)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { background: #1a1a2e; color: #a6accd; padding: 12px 20px; font-weight: bold; font-size: 14px; border-right: 1px solid #111;}
            QTabBar::tab:selected { background: #00FF96; color: #11111b; border-radius: 4px;}
            QTabWidget::pane { border: none; background: #121212; }
        """)
        main_layout.addWidget(self.tabs)

        self._build_tab_graphs()
        self._build_tab_filters()
        self._build_tab_algo()
        self._build_tab_latency()
        self._build_tab_fps()
        self._build_tab_pies()
        self._build_tab_details()
        self._build_tab_guide()
        self._build_tab_sqpnp()

    def _build_controls_bar(self, main_layout):
        bar = QGroupBox("Test-Runner Controls")
        bar.setStyleSheet("""
            QGroupBox { background: #1a1a2e; color: #a6accd; font-size: 13px; font-weight: bold;
                         border: 1px solid #313244; border-radius: 6px; margin-top: 6px; padding: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #00FFFF; }
            QComboBox, QSpinBox, QDoubleSpinBox { background: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a;
                 border-radius: 4px; padding: 4px 8px; font-size: 13px; min-width: 120px; }
            QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                 border-radius: 4px; padding: 6px 14px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background: #45475a; }
            QLabel { color: #a6accd; font-size: 12px; }
        """)
        row = QHBoxLayout()
        row.setSpacing(12)

        row.addWidget(QLabel("Test-Block:"))
        self.combo_test_block = QComboBox()
        self.combo_test_block.addItems([
            "Block_1_Inferenzlatenz",
            "Block_2_Netzwerklatenz",
            "Block_3_MultiKamera",
            "Block_4_Kalibrierung",
            "Block_5_Lokalisierung",
            "Block_6_Heatmap",
            "Block_7_Skelett_3D",
            "Block_8_ID_Switches",
            "Block_9_Stabilitaet",
        ])
        self.combo_test_block.currentTextChanged.connect(self._on_test_block_changed)
        row.addWidget(self.combo_test_block)

        row.addWidget(QLabel("Filter:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["one_euro", "kalman", "none"])
        self.combo_filter.currentTextChanged.connect(self._on_filter_changed)
        row.addWidget(self.combo_filter)

        row.addWidget(QLabel("Triangulation:"))
        self.combo_tri = QComboBox()
        self.combo_tri.addItems(["wls", "lm"])
        self.combo_tri.currentTextChanged.connect(self._on_tri_changed)
        row.addWidget(self.combo_tri)

        row.addWidget(QLabel("Tracking:"))
        self.combo_tracking = QComboBox()
        self.combo_tracking.addItems(["hungarian", "greedy"])
        self.combo_tracking.currentTextChanged.connect(self._on_tracking_changed)
        row.addWidget(self.combo_tracking)

        row.addWidget(QLabel("Frames:"))
        self.spin_frame_limit = QSpinBox()
        self.spin_frame_limit.setRange(0, 100000)
        self.spin_frame_limit.setValue(0)
        self.spin_frame_limit.setSpecialValueText("∞")
        self.spin_frame_limit.setToolTip("0 = unbegrenzt")
        self.spin_frame_limit.setFixedWidth(80)
        row.addWidget(self.spin_frame_limit)

        row.addWidget(QLabel("Zeit (s):"))
        self.spin_time_limit = QDoubleSpinBox()
        self.spin_time_limit.setRange(0.0, 3600.0)
        self.spin_time_limit.setValue(0.0)
        self.spin_time_limit.setSpecialValueText("∞")
        self.spin_time_limit.setDecimals(1)
        self.spin_time_limit.setFixedWidth(80)
        row.addWidget(self.spin_time_limit)

        self.btn_record = QPushButton("🔴 REC")
        self.btn_record.setFixedWidth(90)
        self.btn_record.setStyleSheet("background: #313244; color: #FF5555; font-size: 14px; font-weight: bold;")
        self.btn_record.clicked.connect(self._on_record_clicked)
        row.addWidget(self.btn_record)

        self.lbl_rec_status = QLabel("⬛ Bereit")
        self.lbl_rec_status.setStyleSheet("color: #a6accd; font-size: 12px; min-width: 140px;")
        row.addWidget(self.lbl_rec_status)

        self.btn_eval_block = QPushButton("📊 Block auswerten")
        self.btn_eval_block.setFixedWidth(150)
        self.btn_eval_block.setStyleSheet("background: #1a2e1a; color: #a6e3a1; font-size: 13px; font-weight: bold; border-radius: 4px; border: 1px solid #45475a;")
        self.btn_eval_block.setToolTip("Wertet die gesamte Block-CSV aus (alle Sessions kombiniert)")
        self.btn_eval_block.clicked.connect(self._on_eval_block_clicked)
        row.addWidget(self.btn_eval_block)

        row.addStretch()

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("Log-Ordner:"))
        self.edit_log_dir = QLineEdit("logs")
        self.edit_log_dir.setStyleSheet("background: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px 8px; font-size: 12px;")
        self.edit_log_dir.setMinimumWidth(280)
        self.edit_log_dir.setReadOnly(True)
        row2.addWidget(self.edit_log_dir)
        btn_browse = QPushButton("📁 Ordner wählen")
        btn_browse.clicked.connect(self._on_browse_log_dir)
        row2.addWidget(btn_browse)
        row2.addStretch()

        bar_layout = QVBoxLayout()
        bar_layout.setSpacing(6)
        bar_layout.addLayout(row)
        bar_layout.addLayout(row2)
        bar.setLayout(bar_layout)
        main_layout.addWidget(bar)

        self._is_recording = False

    def _on_test_block_changed(self, text: str):
        if self.controller and hasattr(self.controller, "set_test_block"):
            self.controller.set_test_block(text)

    def _on_filter_changed(self, text: str):
        if self.controller and hasattr(self.controller, "set_filter_mode"):
            self.controller.set_filter_mode(text)

    def _on_tri_changed(self, text: str):
        if self.controller and hasattr(self.controller, "set_triangulation_mode"):
            self.controller.set_triangulation_mode(text)

    def _on_tracking_changed(self, text: str):
        if self.controller and hasattr(self.controller, "set_tracking_mode"):
            self.controller.set_tracking_mode(text)

    def _on_eval_block_clicked(self):
        """Wertet die gesamte Block-CSV aus (alle Sessions fuer diesen Block)."""
        import os
        if not self.controller:
            return
        block = getattr(self.controller, "current_test_block", "Block_1_Inferenzlatenz")
        log_dir = getattr(self.controller, "_log_dir", "logs")
        block_slug = block.replace(" ", "_")
        block_csv = os.path.join(log_dir, f"eval_log_{block_slug}.csv")
        if not os.path.exists(block_csv):
            self.lbl_rec_status.setText("Keine Block-CSV gefunden")
            self.lbl_rec_status.setStyleSheet("color: #f38ba8; font-size: 11px;")
            return
        try:
            dlg = QuickEvalDialog(block_csv, block, parent=self)
            dlg.setWindowTitle(f"Block-Gesamtauswertung \u2014 {block}")
            dlg.show()
        except Exception as e:
            self.lbl_rec_status.setText(f"Fehler: {e}")
    def _on_browse_log_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Log-Verzeichnis wählen", self.edit_log_dir.text())
        if folder:
            self.edit_log_dir.setText(folder)
            if self.controller and hasattr(self.controller, "set_log_dir"):
                self.controller.set_log_dir(folder)

    def _on_record_clicked(self):
        if not self.controller:
            return
        if not self._is_recording:
            frame_limit = self.spin_frame_limit.value()
            time_limit = self.spin_time_limit.value()
            session_file = ""
            if hasattr(self.controller, "start_recording"):
                session_file = self.controller.start_recording(frame_limit=frame_limit, time_limit=time_limit) or ""
            self._last_session_file = session_file
            self._is_recording = True
            self.btn_record.setText("⬛ STOP")
            self.btn_record.setStyleSheet("background: #3d1515; color: #FF5555; font-size: 14px; font-weight: bold;")
            limit_str = f"{frame_limit}F" if frame_limit > 0 else (f"{time_limit:.0f}s" if time_limit > 0 else "∞")
            fname = session_file.split("\\")[-1].split("/")[-1] if session_file else ""
            self.lbl_rec_status.setText(f"🔴 {limit_str} → {fname}")
            self.lbl_rec_status.setStyleSheet("color: #FF5555; font-size: 11px;")
        else:
            if hasattr(self.controller, "stop_recording"):
                self.controller.stop_recording()
            self._is_recording = False
            self.btn_record.setText("🔴 REC")
            self.btn_record.setStyleSheet("background: #313244; color: #FF5555; font-size: 14px; font-weight: bold;")
            frames = getattr(self.controller, '_recording_session_frames', 0)
            self.lbl_rec_status.setText(f"✅ Fertig ({frames} F)")
            self.lbl_rec_status.setStyleSheet("color: #00FF96; font-size: 12px;")
            import os
            session_file = getattr(self, '_last_session_file', '')
            if session_file and os.path.exists(session_file):
                dlg = QuickEvalDialog(session_file, self.controller.current_test_block, parent=self)
                dlg.show()

    def _build_tab_graphs(self):
        tab_graphs = QWidget()
        layout_graphs = QVBoxLayout(tab_graphs)
        header_layout = QGridLayout()

        self.lbl_through = QLabel("Paket-Durchsatz: Berechne...")
        self.lbl_through.setStyleSheet(
            "background: #1e1e2e; color: #00FFFF; padding: 15px; font-size: 15px; border-radius: 6px; font-weight: bold;")

        self.lbl_ghosts = QLabel("Epipolar-Geister Blockiert: 0")
        self.lbl_ghosts.setStyleSheet(
            "background: #311b1b; color: #FF5555; padding: 15px; font-size: 15px; border-radius: 6px; font-weight: bold;")

        self.lbl_valid = QLabel("Bestätigte Fusionen: 0")
        self.lbl_valid.setStyleSheet(
            "background: #153120; color: #00FF96; padding: 15px; font-size: 15px; border-radius: 6px; font-weight: bold;")

        self.lbl_ray_dev = QLabel("Ø Epipolar-Abweichung: 0.0 cm")
        self.lbl_ray_dev.setStyleSheet(
            "background: #2d2615; color: #FFAA00; padding: 15px; font-size: 15px; border-radius: 6px; font-weight: bold;")

        header_layout.addWidget(self.lbl_through, 0, 0)
        header_layout.addWidget(self.lbl_ghosts, 0, 1)
        header_layout.addWidget(self.lbl_valid, 0, 2)
        header_layout.addWidget(self.lbl_ray_dev, 0, 3)
        layout_graphs.addLayout(header_layout)

        self.plot_error = pg.PlotWidget(title="📡 Epipolar-Geometrie & Triangulations-Fehler (cm)")
        self.plot_error.setLabel('left', 'Abweichung (cm)')
        self.curve_error = self.plot_error.plot(pen=pg.mkPen(color='#FFAA00', width=3))
        self.plot_error.showGrid(x=True, y=True, alpha=0.3)
        layout_graphs.addWidget(self.plot_error)
        self.tabs.addTab(tab_graphs, "📐 3D Fehlerquote (Sensorfusion)")

    def _build_tab_filters(self):
        tab_filters = QWidget()
        layout_ik = QVBoxLayout(tab_filters)

        lbl_ik_desc = QLabel("Diagnostik aller aktiven Filter (FABRIK, Kalman, 1-Euro & Acausal Curve)")
        lbl_ik_desc.setStyleSheet("color: #a6accd; font-size: 14px; margin-bottom: 5px; font-weight: bold;")
        layout_ik.addWidget(lbl_ik_desc)

        ik_header_layout = QGridLayout()

        self.lbl_ik_count = QLabel("🦴 FABRIK IK Korrekturen:\n0")
        self.lbl_ik_count.setStyleSheet(
            "background: #1f1b31; color: #DDAAFF; padding: 15px; font-size: 15px; border-radius: 6px; font-weight: bold; text-align: center;")

        self.lbl_kalman_block = QLabel("🛡️ Signal-Glitches Blockiert:\n0")
        self.lbl_kalman_block.setStyleSheet(
            "background: #311b1b; color: #FF5555; padding: 15px; font-size: 15px; border-radius: 6px; font-weight: bold; text-align: center;")

        self.lbl_filter_modes = QLabel("Aktive Pipeline:\nLade...")
        self.lbl_filter_modes.setStyleSheet(
            "background: #1b2631; color: #89b4fa; padding: 15px; font-size: 15px; border-radius: 6px; font-weight: bold; text-align: center;")

        ik_header_layout.addWidget(self.lbl_ik_count, 0, 0)
        ik_header_layout.addWidget(self.lbl_kalman_block, 0, 1)
        ik_header_layout.addWidget(self.lbl_filter_modes, 0, 2)
        layout_ik.addLayout(ik_header_layout)

        self.plot_correction = pg.PlotWidget(title="Live Glättungs-Intensität der Filter (cm pro Frame)")
        self.plot_correction.setLabel('left', 'Korrektur (cm)')
        self.plot_correction.setLabel('bottom', 'Zeit (s)')
        self.plot_correction.showGrid(x=True, y=True, alpha=0.3)
        self.plot_correction.addLegend()
        self.curve_smooth = self.plot_correction.plot(pen=pg.mkPen(color='#00FF96', width=3),
                                                      name="Ø Rausch-Filter Glättung (cm)")
        self.curve_ik_dist = self.plot_correction.plot(
            pen=pg.mkPen(color='#DDAAFF', width=2, style=Qt.PenStyle.DashLine), name="FABRIK IK Längen-Anpassung (cm)")
        layout_ik.addWidget(self.plot_correction, stretch=2)

        self.lbl_approx = QLabel("Warte auf Daten zur Approximation...")
        self.lbl_approx.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; padding: 15px; font-size: 14px; border: 1px solid #313244; border-radius: 6px; font-family: 'Cascadia Code', monospace;")
        layout_ik.addWidget(self.lbl_approx, stretch=1)
        self.tabs.addTab(tab_filters, "🦴 Kinematik & Rauschfilter")

    def _build_tab_algo(self):
        tab_algo = QWidget()
        layout_algo = QVBoxLayout(tab_algo)
        lbl_algo_info = QLabel("Bewertung der ID-Assoziation (DeepSORT / Hungarian vs. Greedy)")
        lbl_algo_info.setStyleSheet("color: #a6accd; font-size: 14px; margin-bottom: 10px; font-weight: bold;")
        layout_algo.addWidget(lbl_algo_info)

        algo_header_layout = QGridLayout()

        self.lbl_algo_mode = QLabel("Aktiver Modus:\nUNBEKANNT")
        self.lbl_algo_mode.setStyleSheet(
            "background: #1e1e2e; color: #00FFFF; padding: 15px; font-size: 16px; border-radius: 6px; font-weight: bold; text-align: center;")

        self.lbl_err_hungary = QLabel("Hungarian (SORT) ID-Switches:\n0")
        self.lbl_err_hungary.setStyleSheet(
            "background: #153120; color: #00FF96; padding: 15px; font-size: 16px; border-radius: 6px; font-weight: bold; text-align: center;")

        self.lbl_err_greedy = QLabel("Greedy Baseline ID-Switches:\n0")
        self.lbl_err_greedy.setStyleSheet(
            "background: #311b1b; color: #FF5555; padding: 15px; font-size: 16px; border-radius: 6px; font-weight: bold; text-align: center;")

        self.lbl_id_switches = QLabel("Echte ID-Switches (Color-Trigger):\n0")
        self.lbl_id_switches.setStyleSheet(
            "background: #1e1e2e; color: #FFAA00; padding: 15px; font-size: 16px; border-radius: 6px; font-weight: bold; text-align: center;")

        algo_header_layout.addWidget(self.lbl_algo_mode, 0, 0)
        algo_header_layout.addWidget(self.lbl_err_hungary, 0, 1)
        algo_header_layout.addWidget(self.lbl_err_greedy, 0, 2)
        algo_header_layout.addWidget(self.lbl_id_switches, 0, 3)
        layout_algo.addLayout(algo_header_layout)

        self.lbl_algo_desc = QLabel(
            "Nutze diesen Bereich für Experimente:\n"
            "1. Schalte im Tracker auf 'Greedy' und lasse 2 Personen sich kreuzen.\n"
            "2. Beobachte die Ausfallquote (ID-Switches).\n"
            "3. Schalte auf 'Hungarian' (SORT) um das Problem durch Kuhn-Munkres Optimierung zu lösen."
        )
        self.lbl_algo_desc.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; padding: 15px; font-size: 15px; border: 1px solid #313244; border-radius: 6px;")
        layout_algo.addWidget(self.lbl_algo_desc, stretch=1)
        self.tabs.addTab(tab_algo, "🧠 Tracking-KI Evaluierung")

    def _build_tab_latency(self):
        tab_latency = QWidget()
        layout_latency = QVBoxLayout(tab_latency)

        graph_layout = QHBoxLayout()
        self.plot_fps = pg.PlotWidget(title="Live Paket-Frequenz (Server FPS)")
        self.plot_fps.setLabel('left', 'FPS')
        self.plot_fps.showGrid(x=True, y=True, alpha=0.3)
        self.plot_fps.setYRange(0, 40)
        self.plot_fps.addLegend()
        graph_layout.addWidget(self.plot_fps)

        self.plot_jitter = pg.PlotWidget(title="Netzwerk-Jitter (Paketverzögerungs-Schwankung)")
        self.plot_jitter.setLabel('left', 'Jitter (ms)')
        self.plot_jitter.showGrid(x=True, y=True, alpha=0.3)
        self.plot_jitter.setYRange(0, 100)
        self.plot_jitter.addLegend()
        graph_layout.addWidget(self.plot_jitter)
        layout_latency.addLayout(graph_layout, stretch=1)

        self.tree_latency = QTreeWidget()
        self.tree_latency.setHeaderLabels(
            ["Kamera / Pipeline", "FPS", "Client (t2-t1)", "Netz (t3-t2)", "Server (t4-t3)", "Total (t4-t1)"])
        self.tree_latency.setStyleSheet("""
            QTreeWidget { background-color: #1e1e2e; color: #cdd6f4; font-size: 14px; border: 1px solid #313244; }
            QHeaderView::section { background-color: #11111b; padding: 10px; color: #89b4fa; font-weight: bold; border: none; }
        """)
        self.tree_latency.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        layout_latency.addWidget(self.tree_latency, stretch=1)
        self.tabs.addTab(tab_latency, "⏱️ Netzwerk Latenz & Jitter")


    def _build_tab_fps(self):
        tab_fps = QWidget()
        layout_fps = QVBoxLayout(tab_fps)

        lbl_desc = QLabel("Live FPS Vergleich: Client-Kamera FPS vs. Server Triangulations-FPS")
        lbl_desc.setStyleSheet("color: #a6accd; font-size: 14px; margin-bottom: 5px; font-weight: bold;")
        layout_fps.addWidget(lbl_desc)

        fps_header = QGridLayout()
        self.lbl_server_tri_fps = QLabel("Server Tri-FPS:\n0.0")
        self.lbl_server_tri_fps.setStyleSheet(
            "background: #1e1e2e; color: #00FF96; padding: 15px; font-size: 15px; border-radius: 6px; font-weight: bold;")
        fps_header.addWidget(self.lbl_server_tri_fps, 0, 0)
        layout_fps.addLayout(fps_header)

        self.plot_fps_compare = pg.PlotWidget(title="Client FPS vs. Server Triangulations-FPS")
        self.plot_fps_compare.setLabel("left", "FPS")
        self.plot_fps_compare.setLabel("bottom", "Zeit (s)")
        self.plot_fps_compare.showGrid(x=True, y=True, alpha=0.3)
        self.plot_fps_compare.setYRange(0, 40)
        self.plot_fps_compare.addLegend()
        self.server_tri_fps_curve = self.plot_fps_compare.plot(
            pen=pg.mkPen(color="#00FF96", width=3), name="Server Tri-FPS")
        layout_fps.addWidget(self.plot_fps_compare, stretch=2)

        self.tree_fps = QTreeWidget()
        self.tree_fps.setHeaderLabels(["Kamera / System", "Client FPS", "Ziel FPS", "Server Tri-FPS", "Status"])
        self.tree_fps.setStyleSheet("""
            QTreeWidget { background-color: #1e1e2e; color: #cdd6f4; font-size: 14px; border: 1px solid #313244; }
            QHeaderView::section { background-color: #11111b; padding: 10px; color: #89b4fa; font-weight: bold; border: none; }
        """)
        self.tree_fps.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        layout_fps.addWidget(self.tree_fps, stretch=1)
        self.tabs.addTab(tab_fps, "⚡ FPS Monitor")

    def _build_tab_pies(self):
        tab_pies = QWidget()
        layout_pies = QHBoxLayout(tab_pies)
        self.pie_filter = PieChartWidget("Sensorfusion: Ghost-Filter Historie")
        self.pie_cams = PieChartWidget("Traffic: Kamera-Datenpakete")
        layout_pies.addWidget(self.pie_filter)
        layout_pies.addWidget(self.pie_cams)
        self.tabs.addTab(tab_pies, "📊 System-Auslastung")

    def _build_tab_details(self):
        tab_details = QWidget()
        layout_details = QVBoxLayout(tab_details)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            ["Hierarchie (Person > Kamera > Gelenk)", "3D Position (X,Y,Z)", "Ø Historische Sicherheit",
             "Pakete / Letztes Update"])
        self.tree.setStyleSheet("""
            QTreeWidget { background-color: #1e1e2e; color: #cdd6f4; font-size: 13px; border: none; }
            QHeaderView::section { background-color: #11111b; padding: 10px; color: #00FF96; font-weight: bold; border: none; }
        """)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout_details.addWidget(self.tree)
        self.tabs.addTab(tab_details, "📋 3D Gelenk-Live-Hierarchie")


    def _build_tab_guide(self):
        tab = QWidget()
        tab.setStyleSheet("background: #121212;")
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(0)

        title = QLabel("📋  Test-Block Übersicht  —  Wann welchen Block verwenden?")
        title.setStyleSheet("color: #00FFFF; font-size: 16px; font-weight: bold; margin-bottom: 12px;")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #121212; }")
        inner = QWidget()
        inner.setStyleSheet("background: #121212;")
        grid = QGridLayout(inner)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        BLOCKS = [
            ("Block_1_Inferenzlatenz",
             "Forschungsfrage 1: Wie groß ist die YOLO-Inferenzlatenz auf dem Edge-Knoten?",
             "1 Person steht ruhig im Kamerabild (statische Referenz). "
             "REC starten, 90 Sek. aufnehmen, STOP. "
             "Wiederholung 3x (Reihenfolge randomisieren gegen thermisches Throttling). "
             "VARIANTE A — Inferenz-Skalierung: In den Client-Einstellungen die Frame-Skalierung "
             "auf 1.0 → 0.75 → 0.5 setzen und je 90 Sek. aufnehmen (ROI-Modus: Full-Scan). "
             "VARIANTE B — ROI-Modus: Sticky vs. Full-Scan bei Skalierung 1.0 vergleichen. "
             "Hardware: RTX 3050 (ONNX CUDA).",
             "inference_ms, client_fps, inference_scale",
             "#89b4fa"),
            ("Block_2_Netzwerklatenz",
             "Forschungsfrage 1: Wie groß sind die 4 Pipeline-Zeitstufen t1→t2→t3→t4?",
             "1 Person steht still. Verbindung über Gigabit-Ethernet (kein WLAN). "
             "REC starten, 90 Sek., STOP. Mindestens 3 Wiederholungen. "
             "Die 4 Zeitstempel t1 (Frame-Erfassung), t2 (nach YOLO), "
             "t3 (Empfang Server), t4 (nach Triangulation) werden automatisch geloggt. "
             "E2E = t4 - t1. Netz = t3 - t2. Inferenz = t2 - t1. Server = t4 - t3.",
             "inference_ms, network_ms, server_ms, e2e_ms, t1–t4",
             "#cba6f7"),
            ("Block_3_MultiKamera",
             "Forschungsfrage 1: Wie skaliert die Serverlatenz mit N ∈ {1..5} Kameras?",
             "Gleiche Person, gleiche Position. Zunächst 1 Kamera verbinden → REC 60 Sek. → STOP. "
             "Dann 2. Kamera anschließen → REC 60 Sek. → STOP. Schrittweise bis 5 Kameras. "
             "camera_count wird automatisch aus der Anzahl verbundener Clients geloggt — "
             "kein manueller Eintrag nötig. Jede Stufe ist ein eigener REC-Durchlauf.",
             "server_ms, server_triangulation_fps, e2e_ms, camera_count",
             "#94e2d5"),
            ("Block_4_Kalibrierung",
             "Forschungsfrage 2: Wie groß ist der Reprojektionsfehler nach Schachbrett-Kalibrierung?",
             "Schachbrettmuster (liegt auf dem Client) aus mind. 20 verschiedenen Winkeln und Abständen "
             "fotografieren — Kamera kippen, drehen, nah (30 cm) und weit (1,5 m) halten. "
             "Kalibrierung in der Client-GUI starten. Reprojektionsfehler und calibration.json "
             "werden automatisch gespeichert. "
             "ZUSATZ SQPnP: Nach der Kalibrierung im Tab 'SQPnP-Genauigkeit' die echten "
             "Kamerapositionen (P1–P6, mit Laser eingemessen) eintragen und mit den "
             "berechneten Positionen vergleichen → Positionsfehler zeigt Kalibrierungsqualität.",
             "repro_error_px, epipolar_error_avg, sqpnp_pos_error_cm",
             "#f9e2af"),
            ("Block_5_Lokalisierung",
             "Forschungsfrage 2: Wie genau ist die 3D-Ortung gegenüber Bodenmarkierungen P1–P6?",
             "Person stellt sich nacheinander auf die 6 eingemessenen Bodenmarkierungen P1–P6 "
             "(mit Laser-Distanzmessgerät auf ±2 mm genau eingemessen). "
             "Pro Markierung ~15 Sek. still stehen. "
             "Ground-Truth-Koordinaten in evaluate.py als GT_POINTS eintragen. "
             "VARIANTE: Triangulationsverfahren wechseln (WLS ↔ LM-iterativ) "
             "und je einen Block aufnehmen. Mindestens 3 Wiederholungen pro Verfahren.",
             "loc_error_cm, loc_rmse_cm, pos_x/y/z vs. Ground-Truth",
             "#a6e3a1"),
            ("Block_6_Heatmap",
             "Forschungsfrage 2: Wie ist der Lokalisierungsfehler räumlich verteilt?",
             "Person läuft den Raum (3,20 m × 4,70 m) systematisch ab — Längsreihen "
             "von vorne nach hinten, dann Querreihen (Rasenmäher-Muster). "
             "Ca. 5–8 Minuten. Zeigt welche Raumbereiche durch Kameraüberlappung "
             "präzise geortet werden und wo Abdeckungslücken entstehen.",
             "pos_x/z → räumliche Fehlerverteilung (Heatmap)",
             "#fab387"),
            ("Block_7_Skelett_3D",
             "Forschungsfrage 3: Wie wirken sich Filter auf Glättung und Latenz aus?",
             "Person macht 3 definierte Posen je ~20 Sek. (langsam, damit Filter-Effekt sichtbar): "
             "(1) beide Arme gerade nach oben strecken, "
             "(2) in die Hocke gehen (Knie ~90°), "
             "(3) T-Pose (Arme waagerecht). "
             "Filtermodus in den Einstellungen wechseln: 1-Euro-Filter → Kalman → deaktiviert. "
             "Je einen eigenen Block aufnehmen. Misst Glättungstiefe d_smooth [cm] und FABRIK-Rate.",
             "kalman_smoothing_cm_delta, ik_correction_cm_delta, ik_resizes_delta",
             "#f38ba8"),
            ("Block_8_ID_Switches",
             "Forschungsfrage 3: Wie viele ID-Switches entstehen bei kreuzenden Personen?",
             "2 Personen starten an gegenüberliegenden Enden des Raums. "
             "Laufen 8–10 Mal in gerader Linie aneinander vorbei — bei jeder Kreuzung "
             "kann das System die IDs verwechseln. "
             "Tracking-Modus wechseln (Hungarian ↔ Greedy) und je einen eigenen Block aufnehmen. "
             "Hungarian (Kuhn-Munkres) sollte deutlich weniger Switches zeigen als Greedy. "
             "Mindestens 3 Durchläufe pro Modus.",
             "id_switches_delta, error_hungarian_delta, error_greedy_delta",
             "#eba0ac"),
            ("Block_9_Stabilitaet",
             "Forschungsfrage 3: Bleibt der Health-Index über alle Versuchsblöcke stabil?",
             "System 10–15 Minuten ohne Eingriff laufen lassen. "
             "1–2 Personen bewegen sich dabei normal (gehen, stehen, kurz hinsetzen). "
             "Testet ob Speicher- oder FPS-Einbrüche auftreten. "
             "Health-Index sollte konstant ≥ 90 % bleiben. "
             "Dieser Block wird idealerweise am Ende jeder Versuchsreihe als Kontrollmessung "
             "wiederholt um thermisches Throttling zu erkennen.",
             "health_index, server_triangulation_fps, kalman_smoothing_cm_delta",
             "#b4befe"),
        ]

        HEADERS = ["Block", "Fragestellung", "Szenario", "Schlüssel-Metriken"]
        HDR_STYLE = "background: #1e1e2e; color: #00FFFF; font-weight: bold; font-size: 13px; padding: 8px 12px; border: 1px solid #313244; border-radius: 4px;"
        for col, h in enumerate(HEADERS):
            lbl = QLabel(h)
            lbl.setStyleSheet(HDR_STYLE)
            grid.addWidget(lbl, 0, col)

        CELL_BASE = "padding: 8px 12px; font-size: 12px; border: 1px solid #1e1e2e; border-radius: 3px;"
        for row_i, (block, question, scenario, metrics, accent) in enumerate(BLOCKS, start=1):
            bg = "#1a1a2e" if row_i % 2 == 0 else "#16161e"
            name_lbl = QLabel(block.replace("_", " "))
            name_lbl.setStyleSheet(f"background: {bg}; color: {accent}; font-weight: bold; {CELL_BASE}")
            name_lbl.setWordWrap(True)
            grid.addWidget(name_lbl, row_i, 0)

            q_lbl = QLabel(question)
            q_lbl.setStyleSheet(f"background: {bg}; color: #cdd6f4; {CELL_BASE}")
            q_lbl.setWordWrap(True)
            grid.addWidget(q_lbl, row_i, 1)

            s_lbl = QLabel(scenario)
            s_lbl.setStyleSheet(f"background: {bg}; color: #a6accd; {CELL_BASE}")
            s_lbl.setWordWrap(True)
            grid.addWidget(s_lbl, row_i, 2)

            m_lbl = QLabel(metrics)
            m_lbl.setStyleSheet(f"background: {bg}; color: #89dceb; font-family: monospace; {CELL_BASE}")
            m_lbl.setWordWrap(True)
            grid.addWidget(m_lbl, row_i, 3)

        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 4)
        grid.setColumnStretch(3, 3)

        note = QLabel(
            "💡  Tipp: Jeder Block schreibt in eine eigene Datei  "
            "(eval_log_Block_X_….csv  +  skeleton_log_Block_X_….jsonl).  "
            "python evaluate.py  wertet automatisch alle vorhandenen Dateien aus."
        )
        note.setStyleSheet("color: #a6accd; font-size: 12px; margin-top: 12px; padding: 10px; "
                           "background: #1e1e2e; border-radius: 6px; border: 1px solid #313244;")
        note.setWordWrap(True)

        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)
        outer.addWidget(note)

        self.tabs.addTab(tab, "📋 Block-Guide")

    _KNOWN_CAMS = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4", "CAMERA_5"]

    def _build_tab_sqpnp(self):
        tab = QWidget()
        tab.setStyleSheet("background: #121212;")
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(8)

        title = QLabel("\u00d7 SQPnP-Positionsgenauigkeit  —  Berechnete vs. eingemessene Kamerapositionen (K1–K5)")
        title.setStyleSheet("color: #00FFFF; font-size: 15px; font-weight: bold; margin-bottom: 4px;")
        outer.addWidget(title)

        desc = QLabel(
            "Tr\u00e4gt die mit dem Laser eingemessenen Referenzpositionen K1–K5 ein. "
            "Nach Klick auf 'Berechnen' wird die aus R,t abgeleitete Kameraposition (SQPnP) "
            "mit den Referenzwerten verglichen. "
            "Metriken: Euklidischer Abstand \u0394 3D [cm]  |  \u0394 XZ-Ebene [cm] (Grundriss)  |  \u0394 Y (H\u00f6he) [cm]"
        )
        desc.setStyleSheet("color: #a6accd; font-size: 12px; padding: 6px; "
                           "background: #1a1a2e; border-radius: 4px; margin-bottom: 6px;")
        desc.setWordWrap(True)
        outer.addWidget(desc)

        self._sqpnp_table = QTableWidget()
        self._sqpnp_table.setStyleSheet("""
            QTableWidget { background: #1e1e2e; color: #cdd6f4; font-size: 12px;
                           gridline-color: #313244; border: 1px solid #313244; }
            QHeaderView::section { background: #11111b; color: #89b4fa; font-weight: bold;
                                   padding: 6px; border: none; }
            QTableWidget::item { padding: 4px 8px; }
            QTableWidget::item:selected { background: #313244; }
        """)
        headers = ["Kamera", "GT X [cm]", "GT Y [cm]", "GT Z [cm]",
                   "Ber. X [cm]", "Ber. Y [cm]", "Ber. Z [cm]",
                   "\u0394 XZ [cm]", "\u0394 Y [cm]", "\u0394 3D [cm]"]
        self._sqpnp_table.setColumnCount(len(headers))
        self._sqpnp_table.setHorizontalHeaderLabels(headers)
        self._sqpnp_table.setRowCount(len(self._KNOWN_CAMS))
        self._sqpnp_table.verticalHeader().setVisible(False)
        hh = self._sqpnp_table.horizontalHeader()
        for i in range(len(headers)):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        self._sqpnp_gt_spins = {}
        for row, cam in enumerate(self._KNOWN_CAMS):
            name_item = QTableWidgetItem(cam)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name_item.setForeground(QColor("#00FFFF"))
            self._sqpnp_table.setItem(row, 0, name_item)
            spins = []
            for col in range(1, 4):
                spin = QDoubleSpinBox()
                spin.setRange(-500.0, 2000.0)
                spin.setDecimals(1)
                spin.setSingleStep(1.0)
                spin.setValue(0.0)
                spin.setStyleSheet("background: #1a1a2e; color: #cdd6f4; border: 1px solid #45475a; "
                                   "padding: 2px 4px; font-size: 12px;")
                self._sqpnp_table.setCellWidget(row, col, spin)
                spins.append(spin)
            self._sqpnp_gt_spins[cam] = tuple(spins)
            for col in range(4, 10):
                item = QTableWidgetItem("—")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                self._sqpnp_table.setItem(row, col, item)

        self._sqpnp_table.setMinimumHeight(220)
        outer.addWidget(self._sqpnp_table)

        btn_row = QHBoxLayout()
        btn_calc = QPushButton("  Berechnen")
        btn_calc.setStyleSheet("background:#1a2e1a;color:#a6e3a1;padding:8px 20px;"
                               "font-size:13px;font-weight:bold;border-radius:4px;border:1px solid #45475a;")
        btn_calc.clicked.connect(self._on_sqpnp_calculate)

        btn_refresh = QPushButton("  Pos. aus Kalibrierung laden")
        btn_refresh.setStyleSheet("background:#1e1e2e;color:#89b4fa;padding:8px 20px;"
                                  "font-size:13px;border-radius:4px;border:1px solid #45475a;")
        btn_refresh.clicked.connect(self._on_sqpnp_refresh)

        btn_export = QPushButton("  Als CSV exportieren")
        btn_export.setStyleSheet("background:#1e1e2e;color:#cdd6f4;padding:8px 20px;"
                                 "font-size:13px;border-radius:4px;border:1px solid #45475a;")
        btn_export.clicked.connect(self._on_sqpnp_export)

        btn_row.addWidget(btn_calc)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_export)

        btn_export_calib = QPushButton("💾 calibration.json exportieren")
        btn_export_calib.setStyleSheet("background:#1e1e2e;color:#f9e2af;padding:8px 20px;"
                                       "font-size:13px;border-radius:4px;border:1px solid #45475a;")
        btn_export_calib.setToolTip("Exportiert alle Linsendaten aus der DB als calibration.json")
        btn_export_calib.clicked.connect(self._on_export_calibration_json)
        btn_row.addWidget(btn_export_calib)

        btn_row.addStretch()
        outer.addLayout(btn_row)

        self.lbl_sqpnp_summary = QLabel("Noch keine Berechnung durchgeführt.")
        self.lbl_sqpnp_summary.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; font-size: 13px; "
            "padding: 12px 16px; border-radius: 6px; border: 1px solid #313244;")
        self.lbl_sqpnp_summary.setWordWrap(True)
        outer.addWidget(self.lbl_sqpnp_summary)
        outer.addStretch()

        self.tabs.addTab(tab, "\U0001f4d0 SQPnP-Genauigkeit")

    def _on_sqpnp_refresh(self):
        """Lädt die berechneten Kamerapositionen aus dem config_cache."""
        if not self.controller:
            return
        cache = getattr(self.controller, "config_cache", {})
        for row, cam in enumerate(self._KNOWN_CAMS):
            cam_data = cache.get(cam, {})
            pos3d = cam_data.get("cam_3d_data", {}).get("pos", None)
            if pos3d is not None:
                try:
                    import numpy as np
                    p = np.asarray(pos3d).flatten()
                    for col, val in zip(range(4, 7), p[:3]):
                        item = self._sqpnp_table.item(row, col)
                        if item:
                            item.setText(f"{float(val):.1f}")
                            item.setForeground(QColor("#cdd6f4"))
                except Exception:
                    pass
            else:
                for col in range(4, 7):
                    item = self._sqpnp_table.item(row, col)
                    if item:
                        item.setText("n/a")
                        item.setForeground(QColor("#585b70"))

    def _on_sqpnp_calculate(self):
        """Berechnet Fehlermetriken: Delta XZ, Delta Y, Delta 3D pro Kamera."""
        import math
        self._on_sqpnp_refresh()

        d3d_vals, dxz_vals, dy_vals = [], [], []
        any_data = False

        for row, cam in enumerate(self._KNOWN_CAMS):
            spins = self._sqpnp_gt_spins.get(cam)
            if not spins:
                continue
            gt_x = spins[0].value()
            gt_y = spins[1].value()
            gt_z = spins[2].value()

            bx_item = self._sqpnp_table.item(row, 4)
            by_item = self._sqpnp_table.item(row, 5)
            bz_item = self._sqpnp_table.item(row, 6)

            if not bx_item or bx_item.text() in ("—", "n/a", ""):
                for col in range(7, 10):
                    self._sqpnp_table.item(row, col).setText("—")
                continue

            try:
                bx = float(bx_item.text())
                by = float(by_item.text())
                bz = float(bz_item.text())
            except ValueError:
                continue

            dxz = math.sqrt((bx - gt_x) ** 2 + (bz - gt_z) ** 2)
            dy  = abs(by - gt_y)
            d3d = math.sqrt((bx - gt_x) ** 2 + (by - gt_y) ** 2 + (bz - gt_z) ** 2)

            dxz_vals.append(dxz); dy_vals.append(dy); d3d_vals.append(d3d)
            any_data = True

            def _color(v):
                if v < 5:   return "#a6e3a1"
                if v < 15:  return "#f9e2af"
                return "#f38ba8"

            for col, val, clr in [
                (7, dxz, _color(dxz)),
                (8, dy,  _color(dy)),
                (9, d3d, _color(d3d)),
            ]:
                item = self._sqpnp_table.item(row, col)
                if item:
                    item.setText(f"{val:.2f}")
                    item.setForeground(QColor(clr))

        if not any_data:
            self.lbl_sqpnp_summary.setText(
                "Keine berechneten Positionen gefunden. "
                "Zuerst Kalibrierung durchführen, dann 'Pos. aus Kalibrierung laden' klicken.")
            return

        def _fmt(vals, label):
            return (f"{label}  Mittel: {sum(vals)/len(vals):.2f} cm  "
                    f"Max: {max(vals):.2f} cm  Min: {min(vals):.2f} cm")

        n = len(d3d_vals)
        mean3d = sum(d3d_vals) / n
        status = "GUT" if mean3d < 5 else ("OK" if mean3d < 15 else "VERBESSERUNGSBEDARF")
        color  = "#a6e3a1" if mean3d < 5 else ("#f9e2af" if mean3d < 15 else "#f38ba8")

        summary = (
            f"Auswertung über {n} Kameras   |   Status: {status}"
            f"{_fmt(d3d_vals, 'Δ 3D    ')}"
            f"{_fmt(dxz_vals, 'Δ XZ    ')}"
            f"{_fmt(dy_vals,  'Δ Y     ')}"
        )
        self.lbl_sqpnp_summary.setText(summary.replace("\n", "<br>"))
        self.lbl_sqpnp_summary.setStyleSheet(
            f"background:#1e1e2e;color:{color};font-size:13px;"
            "padding:12px 16px;border-radius:6px;border:1px solid #313244;")

    def _on_sqpnp_export(self):
        import csv as _csv
        path, _ = QFileDialog.getSaveFileName(
            self, "SQPnP-Auswertung exportieren", "sqpnp_accuracy.csv",
            "CSV (*.csv)")
        if not path:
            return
        rows = []
        for row, cam in enumerate(self._KNOWN_CAMS):
            spins = self._sqpnp_gt_spins.get(cam)
            if not spins:
                continue
            row_data = {
                "camera": cam,
                "gt_x": spins[0].value(), "gt_y": spins[1].value(), "gt_z": spins[2].value(),
            }
            for col, key in [(4,"calc_x"),(5,"calc_y"),(6,"calc_z"),
                             (7,"delta_xz_cm"),(8,"delta_y_cm"),(9,"delta_3d_cm")]:
                item = self._sqpnp_table.item(row, col)
                row_data[key] = item.text() if item else ""
            rows.append(row_data)
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        except Exception as e:
            print(f"SQPnP export error: {e}")


    def _on_export_calibration_json(self):
        """Exportiert alle Linsendaten aus der DB in eine calibration.json."""
        import json as _json
        if not self.controller:
            return
        try:
            profiles = self.controller.system_db.get_all_lens_profiles()
        except Exception as e:
            self.lbl_sqpnp_summary.setText(f"DB-Fehler: {e}")
            return
        if not profiles:
            self.lbl_sqpnp_summary.setText("Keine Linsenprofile in der DB gefunden.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "calibration.json exportieren", "calibration.json", "JSON (*.json)")
        if not path:
            return
        import time as _time
        export = {
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lens_profiles": {}
        }
        for pid, data in profiles.items():
            export["lens_profiles"][pid] = {
                "camera_matrix": data.get("camera_matrix", []),
                "dist_coeffs": data.get("dist_coeffs", []),
                "reprojection_error": data.get("reprojection_error", None)
            }
        try:
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(export, f, indent=2, default=str)
            n = len(profiles)
            self.lbl_sqpnp_summary.setText(
                f"✅ {n} Linsenprofile exportiert nach: {path}")
        except Exception as e:
            self.lbl_sqpnp_summary.setText(f"Export-Fehler: {e}")
    def poll_packets(self):
        if not self.controller: return
        parent_dash = self.parent()
        if not parent_dash or not hasattr(parent_dash, 'cache_timestamps'): return

        now = time.time()
        for cam, ts in parent_dash.cache_timestamps.items():
            if not cam.startswith("CAMERA_"): continue

            if cam not in self.cam_arrival_times:
                self.cam_arrival_times[cam] = deque(maxlen=30)
                self.camera_packet_count[cam] = 0
                self.last_frame_sigs[cam] = 0.0

            if ts != self.last_frame_sigs[cam]:
                self.last_frame_sigs[cam] = ts
                self.cam_arrival_times[cam].append(now)
                self.camera_packet_count[cam] += 1

    def update_stats(self):
        if not self.controller or not hasattr(self.controller, 'tracker'):
            return

        # Auto-Stop Sync: falls der Controller die Session beendet hat, GUI aktualisieren
        if self._is_recording and hasattr(self.controller, '_session_log_file') and self.controller._session_log_file is None:
            self._is_recording = False
            self.btn_record.setText("🔴 REC")
            self.btn_record.setStyleSheet("background: #313244; color: #FF5555; font-size: 14px; font-weight: bold;")
            frames = getattr(self.controller, '_recording_session_frames', 0)
            self.lbl_rec_status.setText(f"✅ Fertig ({frames} F)")
            self.lbl_rec_status.setStyleSheet("color: #00FF96; font-size: 12px;")
            import os
            session_file = getattr(self, '_last_session_file', '')
            if session_file and os.path.exists(session_file):
                dlg = QuickEvalDialog(session_file, self.controller.current_test_block, parent=self)
                dlg.show()

        now = time.time()
        tracker = self.controller.tracker
        current_t = now - self.start_time
        self.time_data.append(current_t)
        self.stat_ticks += 1

        ik_count = kalman_blocked = err_hungary = err_greedy = epipolar_ghosts = id_switches = 0
        avg_smooth_cm = avg_ik_corr_cm = real_avg_err = reprojection_error = health_index = 0.0

        current_mode = "UNBEKANNT"
        smooth_mode = "UNBEKANNT"
        ik_mode = "UNBEKANNT"
        triang_mode = "UNBEKANNT"

        cam_latencies = {}
        latencies = {}

        try:
            parent_dash = self.parent()
            if parent_dash and hasattr(parent_dash, 'view_master') and parent_dash.view_master:
                master_renderer = parent_dash.view_master.renderer
                stats = master_renderer.get_filter_stats()

                cam_latencies = stats.get("cam_latencies", {})

                ik_count = stats.get('ik_resizes', 0)
                kalman_blocked = stats.get('kalman_glitches_blocked', 0)

                # Werte vom Backend sind kumulativ -> Delta für den Graphen berechnen
                total_smooth_cm = stats.get('kalman_smoothing_cm', 0.0)
                total_ik_corr_cm = stats.get('ik_correction_cm', 0.0)

                avg_smooth_cm = total_smooth_cm - getattr(self, 'last_total_smooth', 0.0)
                avg_ik_corr_cm = total_ik_corr_cm - getattr(self, 'last_total_ik', 0.0)

                self.last_total_smooth = total_smooth_cm
                self.last_total_ik = total_ik_corr_cm

                err_hungary = stats.get('error_hungarian', 0)
                err_greedy = stats.get('error_greedy', 0)
                id_switches = stats.get('id_switches', 0)
                reprojection_error = stats.get('reprojection_error', 0.0)
                health_index = stats.get('health_index', 100.0)

                current_mode = master_renderer.tracking_mode.upper()
                smooth_mode = master_renderer.smoothing_mode.upper()
                ik_mode = master_renderer.ik_mode.upper()
                triang_mode = master_renderer.triangulation_mode.upper()

                real_avg_err = stats.get('epipolar_error_avg', 0.0)
                epipolar_ghosts = stats.get('epipolar_ghosts', 0)

                ghosts_delta = epipolar_ghosts - getattr(self, 'last_epipolar_ghosts', 0)
                self.last_epipolar_ghosts = epipolar_ghosts

                for k in ["t1", "t2", "t3", "t4", "inference_time_ms", "network_latency_ms", "server_latency_ms", "end_to_end_latency_ms"]:
                    if k in stats:
                        latencies[k] = stats[k]

        except Exception as e:
            pass

        self.lbl_algo_mode.setText(f"Aktiver Modus:\n{current_mode}")
        self.lbl_err_hungary.setText(f"Hungarian (SORT) ID-Switches:\n{err_hungary}")
        self.lbl_err_greedy.setText(f"Greedy Baseline ID-Switches:\n{err_greedy}")

        if hasattr(self, 'lbl_id_switches'):
            self.lbl_id_switches.setText(f"Echte ID-Switches (Color-Trigger):\n{id_switches}")

        self.lbl_ik_count.setText(f"🦴 {ik_mode} IK Korrekturen:\n{ik_count:,} Eingriffe")
        self.lbl_kalman_block.setText(f"🛡️ {smooth_mode} Signal-Filterung:\n{kalman_blocked:,} Glitches blockiert")

        self.lbl_filter_modes.setText(f"Triangulation: {triang_mode}\nRepro-Fehler: {reprojection_error:.2f} px")

        self.kalman_smooth_history.append(avg_smooth_cm)
        self.ik_resize_history.append(avg_ik_corr_cm)

        t_list_full = list(self.time_data)
        self.curve_smooth.setData(t_list_full, list(self.kalman_smooth_history))
        self.curve_ik_dist.setData(t_list_full, list(self.ik_resize_history))

        run_time_min = max(current_t / 60.0, 0.01)
        approx_mean_smooth = getattr(self, 'last_total_smooth', 0.0) / max(1, self.stat_ticks)
        approx_mean_ik = getattr(self, 'last_total_ik', 0.0) / max(1, self.stat_ticks)
        ik_per_min = ik_count / run_time_min
        glitches_per_min = kalman_blocked / run_time_min

        if health_index > 0:
            health_score = health_index
        else:
            health_score = 100.0 - (approx_mean_smooth * 0.8) - (glitches_per_min * 0.05) - (ik_per_min * 0.01)
        health_score = max(0.0, min(100.0, health_score))

        health_color = "#00FF96" if health_score >= 90 else ("#FFAA00" if health_score >= 70 else "#FF3333")

        approx_text = (
            f"Vollständige Pipeline Approximation (Laufzeit: {current_t:.1f}s)\n"
            f"----------------------------------------------------------\n"
            f"➤ Aktiver Rauschfilter  : {smooth_mode}\n"
            f"➤ Ø Filter-Glättung     : {approx_mean_smooth:.3f} cm pro Gelenk (All-Time)\n"
            f"➤ Ø IK-Knochendehnung   : {approx_mean_ik:.3f} cm pro Gelenk (All-Time)\n"
            f"➤ Acausal Curve Delay   : Exakt 2 Frames (ca. 66ms Latenz)\n"
            f"➤ Rate: Kamera-Glitches : {glitches_per_min:.1f} Fehler / Minute\n"
            f"➤ Rate: Anatomie-Brüche : {ik_per_min:.1f} Korrekturen / Minute\n\n"
            f"Hardware-Stabilitäts-Index: <span style='color:{health_color}; font-size:18px; font-weight:bold;'>{health_score:.2f} %</span>"
        )
        self.lbl_approx.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_approx.setText(approx_text.replace('\n', '<br>'))

        self.tree_latency.clear()

        if latencies:
            sys_item = QTreeWidgetItem(self.tree_latency, ["PIPELINE (Global Ø)", "-", "-", "-", "-", "AKTIV"])
            sys_item.setBackground(0, QBrush(QColor("#11111b")))
            
            QTreeWidgetItem(sys_item, ["  Server-Verarbeitung (t4-t3)", "-", "-", "-", f"{latencies.get('server_latency_ms', 0):.1f} ms", "FUSION"])

        for cam_idx, cam in enumerate(self.cam_arrival_times.keys()):
            if cam not in self.fps_history:
                self.fps_history[cam] = deque(maxlen=self.history_size)
                self.jitter_history[cam] = deque(maxlen=self.history_size)
                color = self.CAM_COLORS[cam_idx % len(self.CAM_COLORS)]
                self.fps_curves[cam] = self.plot_fps.plot(pen=pg.mkPen(color=color, width=2), name=cam)
                self.jitter_curves[cam] = self.plot_jitter.plot(pen=pg.mkPen(color=color, width=2), name=cam)

            current_fps, avg_iat_ms, jitter_ms = 0.0, 0.0, 0.0
            status_text, status_col = "Verbinde...", "#777777"

            cam_key_clean = cam.replace("CAMERA_", "")
            lat = cam_latencies.get(cam, cam_latencies.get(cam_key_clean, {}))
            
            cli_ms = lat.get("client_latency_ms", 0.0)
            net_ms = lat.get("network_latency_ms", 0.0)
            srv_ms = lat.get("server_latency_ms", 0.0)
            e2e_ms = lat.get("end_to_end_latency_ms", 0.0)
            inf_ms = lat.get("inference_time_ms", 0.0)

            if len(self.cam_arrival_times[cam]) >= 2:
                arrivals = list(self.cam_arrival_times[cam])
                deltas = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
                avg_delta = np.mean(deltas)
                avg_iat_ms = avg_delta * 1000.0

                if avg_delta > 0: current_fps = 1.0 / avg_delta
                if len(deltas) >= 2: jitter_ms = np.std(deltas) * 1000.0

                if current_fps >= 15.0 and jitter_ms < 30.0:
                    status_text, status_col = "Stabil", "#00FF96"
                elif current_fps >= 10.0 and jitter_ms < 60.0:
                    status_text, status_col = "Schwankend", "#FFFF00"
                else:
                    status_text, status_col = "Kritisch", "#FF3333"

                if now - arrivals[-1] > 1.0:
                    current_fps = 0.0
                    status_text, status_col = "Offline", "#555555"

            self.fps_history[cam].append(current_fps)
            self.jitter_history[cam].append(jitter_ms)

            t_list = list(self.time_data)[-len(self.fps_history[cam]):]
            self.fps_curves[cam].setData(t_list, list(self.fps_history[cam]))
            self.jitter_curves[cam].setData(t_list, list(self.jitter_history[cam]))

            item = QTreeWidgetItem([
                f"📷 {cam}", 
                f"{current_fps:.1f}", 
                f"{cli_ms:.1f} ms", 
                f"{net_ms:.1f} ms", 
                f"{srv_ms:.1f} ms", 
                f"{e2e_ms:.1f} ms"
            ])
            
            item.setForeground(0, QBrush(QColor("#ffffff")))
            item.setForeground(1, QBrush(QColor("#00FFFF")))
            item.setForeground(5, QBrush(QColor("#00FF96")))

            item.setFont(5, QFont("Arial", 10, QFont.Weight.Bold))

            QTreeWidgetItem(item, ["  ↳ Davon KI-Inferenz (Edge)", "-", f"{inf_ms:.1f} ms", "-", "-", "-"])
            
            self.tree_latency.addTopLevelItem(item)

        throughput_str = " | ".join([f"{cam}: {count:,}" for cam, count in self.camera_packet_count.items()])
        self.lbl_through.setText(f"Paket-Durchsatz:\n{throughput_str if throughput_str else 'Warte auf Daten...'}")

        self.historical_ghosts += epipolar_ghosts
        valid_persons = [gp for gp in tracker.global_persons if now - gp.last_update <= 2.0]
        valid_count = len(valid_persons)
        self.historical_valid += valid_count

        self.lbl_ghosts.setText(f"Epipolar-Geister Blockiert:\n{self.historical_ghosts:,}")
        self.lbl_valid.setText(f"Bestätigte Fusionen:\n{self.historical_valid:,} bestätigt")

        self.pie_filter.set_data({"Valide Bestätigungen": (self.historical_valid, "#00FF96"),
                                  "Geister blockiert": (self.historical_ghosts, "#FF5555")})

        pie_cams_data = {}
        for idx, (cam, count) in enumerate(self.camera_packet_count.items()):
            pie_cams_data[cam] = (count, self.CAM_COLORS[idx % len(self.CAM_COLORS)])
        self.pie_cams.set_data(pie_cams_data)

        self.error_data.append(real_avg_err)
        self.lbl_ray_dev.setText(f"Ø Epipolar-Abweichung:\n{real_avg_err:.2f} cm")
        self.curve_error.setData(t_list_full, list(self.error_data))

        server_tri_fps = 0.0
        cam_fps_data = {}
        try:
            parent_dash = self.parent()
            if parent_dash and hasattr(parent_dash, "view_master") and parent_dash.view_master:
                stats_fps = parent_dash.view_master.renderer.get_filter_stats()
                server_tri_fps = float(stats_fps.get("server_triangulation_fps", 0.0))
                cam_lats = stats_fps.get("cam_latencies", {})
                for cname, lat in cam_lats.items():
                    cam_fps_data[cname] = {"client_fps": float(lat.get("client_fps", 0.0))}
        except Exception:
            pass

        self.server_tri_fps_history.append(server_tri_fps)
        t_list_fps = list(self.time_data)[-len(self.server_tri_fps_history):]
        if self.server_tri_fps_curve is not None:
            self.server_tri_fps_curve.setData(t_list_fps, list(self.server_tri_fps_history))
        self.lbl_server_tri_fps.setText(f"Server Tri-FPS:\n{server_tri_fps:.1f}")

        for idx_c, (cname, fdata) in enumerate(cam_fps_data.items()):
            if cname not in self.client_fps_history:
                self.client_fps_history[cname] = deque(maxlen=self.history_size)
                color = self.CAM_COLORS[idx_c % len(self.CAM_COLORS)]
                self.client_fps_curves[cname] = self.plot_fps_compare.plot(
                    pen=pg.mkPen(color=color, width=2), name=f"{cname} (Client)")
            self.client_fps_history[cname].append(fdata["client_fps"])
            t_c = list(self.time_data)[-len(self.client_fps_history[cname]):]
            self.client_fps_curves[cname].setData(t_c, list(self.client_fps_history[cname]))

        self.tree_fps.clear()
        server_row = QTreeWidgetItem(["Server (Triangulation)", "-", "-", f"{server_tri_fps:.1f}", ""])
        server_row.setForeground(3, QBrush(QColor("#00FF96")))
        server_row.setFont(3, QFont("Arial", 10, QFont.Weight.Bold))
        self.tree_fps.addTopLevelItem(server_row)
        for cname, fdata in cam_fps_data.items():
            cfps = fdata["client_fps"]
            status, scol = ("Gut", "#00FF96") if cfps >= 25.0 else (("OK", "#FFFF00") if cfps >= 15.0 else (("Niedrig", "#FF5555") if cfps > 0 else ("Offline", "#555555")))
            row = QTreeWidgetItem([f"📷 {cname}", f"{cfps:.1f}", "-", f"{server_tri_fps:.1f}", status])
            row.setForeground(1, QBrush(QColor("#00FFFF")))
            row.setForeground(4, QBrush(QColor(scol)))
            self.tree_fps.addTopLevelItem(row)

        visited_keys = set()

        def get_or_create_item(key, parent, text, bg_color=None):
            visited_keys.add(key)
            if key not in self.tree_items:
                item = QTreeWidgetItem([text, "", "", ""])
                if parent is None:
                    self.tree.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                item.setExpanded(True)
                self.tree_items[key] = item
            else:
                item = self.tree_items[key]
                item.setText(0, text)
            if bg_color:
                for i in range(4): item.setBackground(i, QBrush(QColor(bg_color)))
            return self.tree_items[key]

        for gp in valid_persons:
            p_key = f"p_{gp.id}"
            p_item = get_or_create_item(p_key, None, f"👤 Person ID {gp.id}", "#15241b")

            skel = getattr(gp, 'skeleton_3d', {})
            pos_text = f"X:{gp.pos[0]:.0f} Y:{gp.pos[1]:.0f} Z:{gp.pos[2]:.0f}" if np.any(gp.pos) else "Berechne..."
            p_item.setText(1, pos_text)
            p_item.setText(2, f"✅ Verifiziert")
            p_item.setText(3, f"Update: {(now - gp.last_update):.2f}s")
            p_item.setForeground(2, QBrush(QColor("#00FF96")))

            for cam_name, obs in gp.client_observations.items():
                if now - obs.last_seen > 3.0: continue
                c_key = f"{p_key}_{cam_name}"
                c_item = get_or_create_item(c_key, p_item, f"📷 {cam_name}", "#1e1e2e")

                packets = self.camera_packet_count.get(cam_name, 0)
                c_item.setText(3, f"Pakete: {packets:,}")
                c_item.setForeground(2, QBrush(QColor("#00FFFF")))

                kps = obs.raw_data.get("keypoints", [])
                if cam_name not in self.joint_conf_sum:
                    self.joint_conf_sum[cam_name] = {i: 0.0 for i in range(17)}
                    self.joint_conf_count[cam_name] = {i: 0 for i in range(17)}

                for kp in kps:
                    j_id, conf = kp.get("id"), kp.get("c", 0.0)
                    if conf > 0.0:
                        self.joint_conf_sum[cam_name][j_id] += conf
                        self.joint_conf_count[cam_name][j_id] += 1

                    avg_conf = 0.0
                    if self.joint_conf_count[cam_name][j_id] > 0:
                        avg_conf = self.joint_conf_sum[cam_name][j_id] / self.joint_conf_count[cam_name][j_id]

                    j_name = self.KP_NAMES.get(j_id, f"Gelenk {j_id}")
                    j_key = f"{c_key}_j{j_id}"
                    j_item = get_or_create_item(j_key, c_item, f"  🦴 {j_name}")

                    j_3d_text = "-"
                    if j_id in skel and isinstance(skel[j_id], dict):
                        j_3d_text = f"{skel[j_id].get('x', 0):.1f}, {skel[j_id].get('y', 0):.1f}, {skel[j_id].get('z', 0):.1f}"
                    elif j_id in skel and isinstance(skel[j_id], np.ndarray):
                        j_3d_text = f"{skel[j_id][0]:.1f}, {skel[j_id][1]:.1f}, {skel[j_id][2]:.1f}"

                    j_item.setText(1, j_3d_text)
                    j_item.setText(2, f"Ø {avg_conf * 100:.1f}%")
                    j_item.setText(3, f"Gesehen: {self.joint_conf_count[cam_name][j_id]}x")

                    conf_color = "#00FF00" if avg_conf > 0.8 else ("#FFFF00" if avg_conf > 0.5 else "#FF0000")
                    j_item.setForeground(2, QBrush(QColor(conf_color)))

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