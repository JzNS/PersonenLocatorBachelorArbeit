import os
import time
import cv2
import numpy as np
from typing import Any, Dict, Tuple
import copy
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QComboBox, QVBoxLayout, QLabel, QHBoxLayout,
    QPushButton, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSlot
from client.utils.ConfigManager import ConfigManager


class SettingsDialog(QDialog):

    def __init__(self, parent: Any, camera_name: str, settings: Dict[str, Any]) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Einstellungen: {camera_name}")
        self.setMinimumWidth(650)

        self.original_settings = copy.deepcopy(settings)
        self.old_resolution = settings.get("resolution", [1920, 1080])

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.inputs: Dict[str, Tuple[QLineEdit, str]] = {}

        self.device_info = "UNKNOWN"
        if hasattr(parent, 'worker') and hasattr(parent.worker, 'detector'):
            self.device_info = getattr(parent.worker.detector, 'actual_device', self.device_info)
        elif hasattr(parent, 'ai_thread') and hasattr(parent.ai_thread, 'detector'):
            self.device_info = getattr(parent.ai_thread.detector, 'actual_device', self.device_info)

        bg_color = "#004422" if "NVIDIA" in self.device_info else "#223344"
        text_color = "#00FF96" if "NVIDIA" in self.device_info else "#AADDFF"

        self.lbl_device = QLabel(f"Aktive Hardware: {self.device_info}")
        self.lbl_device.setStyleSheet(
            f"color: {text_color}; background-color: {bg_color}; padding: 6px; border-radius: 4px; font-weight: bold;")
        form.addRow("KI-Beschleuniger", self.lbl_device)

        self.combo_type = QComboBox()
        self.combo_type.addItems(["Standard", "Top-Down"])
        self.combo_type.setCurrentText(settings.get("camera_type", "Standard"))
        form.addRow("🎥 Kamera-Modus", self.combo_type)

        self.combo_model = QComboBox()
        self.available_models = ["yolo26n-pose.onnx", "yolo26s-pose.onnx", "yolo26m-pose.onnx", "yolo26l-pose.onnx"]
        self.combo_model.addItems(self.available_models)

        current_model = settings.get("model_path", "yolo26n-pose.onnx")
        if os.path.basename(current_model) in self.available_models:
            self.combo_model.setCurrentText(os.path.basename(current_model))

        form.addRow("🧠 KI-Modell", self.combo_model)

        self.btn_auto_calib = QPushButton("🔬 PROFILER STARTEN (Real-World Stresstest)")
        self.btn_auto_calib.setStyleSheet(
            "background-color: #0055AA; color: white; font-weight: bold; border-radius: 4px; padding: 4px 8px;")
        self.btn_auto_calib.clicked.connect(self._run_auto_calibration)

        form.addRow("Hardware Profiler", self.btn_auto_calib)

        all_configs = ConfigManager.load_camera_config()
        global_data = all_configs.get("Camera_ALL", {})

        room_dims = global_data.get("room_dimensions", {"width": 320.0, "height": 250.0, "depth": 470.0})

        self.inp_room_w = QLineEdit(str(room_dims.get("width", 320.0)))
        self.inp_room_h = QLineEdit(str(room_dims.get("height", 250.0)))
        self.inp_room_d = QLineEdit(str(room_dims.get("depth", 470.0)))

        val_room = QDoubleValidator(10.0, 10000.0, 1)
        val_room.setNotation(QDoubleValidator.Notation.StandardNotation)

        self.combo_profile = QComboBox()

        lens_data = global_data.get("lens_profiles", {})

        if isinstance(lens_data, dict) and lens_data:
            profiles = list(lens_data.keys())
        elif isinstance(lens_data, list) and lens_data:
            profiles = lens_data
        else:
            profiles = ["default"]

        self.combo_profile.addItems(profiles)

        current_profile = settings.get("active_lens_profile", "default")
        if current_profile not in profiles:
            self.combo_profile.addItem(current_profile)

        self.combo_profile.setCurrentText(current_profile)
        form.addRow("Objektiv-Profil", self.combo_profile)

        fields_config = [
            ("Kamera Index", "camera_index", "int"),
            ("Ziel FPS (Kamera-Limit)", "target_fps", "int"),
            ("Zoom Faktor", "zoom", "float"),
            ("Rotation", "rotation", "int"),
        ]

        for label, key, dtype in fields_config:
            val = settings.get(key, 0)
            if key == "target_fps":
                cam_fps = 30
                try:
                    cap_obj = None
                    if hasattr(parent, 'worker') and hasattr(parent.worker, 'cap'):
                        cap_obj = parent.worker.cap
                    elif hasattr(parent, 'ai_thread') and hasattr(parent.ai_thread, 'cap'):
                        cap_obj = parent.ai_thread.cap

                    if cap_obj is not None and cap_obj.isOpened():
                        fps = cap_obj.get(cv2.CAP_PROP_FPS)
                        if fps > 0: cam_fps = int(fps)
                except Exception:
                    pass
                val = cam_fps

            widget = QLineEdit(str(val))
            if dtype == "int":
                widget.setValidator(QIntValidator(0, 999))
            else:
                validator = QDoubleValidator(0.1, 10.0, 2)
                validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                widget.setValidator(validator)

            if key == "target_fps":
                widget.setReadOnly(True)
                widget.setStyleSheet("background-color: #2A2A2A; color: #888888; border: 1px solid #444;")

            form.addRow(label, widget)
            self.inputs[key] = (widget, dtype)

        res_options = ["640x480", "1280x720", "1920x1080", "2560x1440"]
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

    def _get_gpu_tdp(self) -> float:
        gpu_tdps = {
            "1080 TI": 250, "1080": 180, "1070": 150, "1060": 120,
            "2080": 215, "2070": 175, "2060": 160,
            "3090": 350, "3080": 320, "3070": 220, "3060": 170, "3050": 130,
            "4090": 450, "4080": 320, "4070": 200, "4060": 115
        }
        device_upper = self.device_info.upper()
        for name, tdp in gpu_tdps.items():
            if name in device_upper: return float(tdp)
        return 200.0

    def _estimate_vram(self, model_name: str, cam_w: int, cam_h: int) -> int:
        base_vram = {"n-": 180, "s-": 350, "m-": 650, "l-": 1200}
        vram_mb = 200
        for key, val in base_vram.items():
            if key in model_name:
                vram_mb = val
                break
        tensor_mb = (cam_w * cam_h * 3 * 4) / (1024 * 1024)
        vram_mb += int(tensor_mb * 5)
        return vram_mb

    @pyqtSlot()
    def _run_auto_calibration(self) -> None:
        parent_widget = self.parent()
        original_detector = None
        worker = None

        if hasattr(parent_widget, 'worker') and hasattr(parent_widget.worker, 'detector'):
            worker = parent_widget.worker
            original_detector = worker.detector
        elif hasattr(parent_widget, 'ai_thread') and hasattr(parent_widget.ai_thread, 'detector'):
            worker = parent_widget.ai_thread
            original_detector = worker.detector

        if not original_detector:
            QMessageBox.warning(self, "Abbruch", "Konnte das YOLO-Modul nicht finden.")
            return

        target_fps_str = self.inputs["target_fps"][0].text().replace(',', '.')
        try:
            target_fps = int(float(target_fps_str)) if target_fps_str else 30
        except ValueError:
            target_fps = 30

        res_str = self.combo_res.currentText()
        try:
            cam_w, cam_h = [int(x) for x in res_str.split('x')]
        except ValueError:
            cam_w, cam_h = 1920, 1080

        smart_target_ms = 1000.0 / target_fps
        max_tdp = self._get_gpu_tdp()
        idle_tdp = 35.0

        test_frame = None
        if worker and hasattr(worker, '_latest_frame') and worker._latest_frame is not None:
            test_frame = worker._latest_frame.copy()

        if test_frame is None:
            test_frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        progress = QMessageBox(self)
        progress.setWindowTitle("Full-Pipeline Hardware Profiling")
        progress.setText(
            f"🔬 Kamera-FPS: {target_fps} | Auflösung: {cam_w}x{cam_h}\n\n"
            f"Messe native Zero-Copy Latenz (GPU-Inference + CPU 3D-Mathematik)...\n\n"
            f"⚠️ WICHTIG: Stellen Sie sich jetzt vor die Kamera, damit die Skelett-Berechnungen die CPU realistisch belasten!\n\n"
            f"Bitte warten."
        )
        progress.setStandardButtons(QMessageBox.StandardButton.NoButton)
        progress.show()
        QApplication.processEvents()

        all_results = []
        detector_class = original_detector.__class__
        base_dir = os.path.join("client", "config", "graka")

        try:
            for model_name in self.available_models:
                model_path = os.path.join(base_dir, model_name)
                if not os.path.exists(model_path): continue

                temp_detector = detector_class(model_onnx_path=model_path)
                tier = 4 if "l-" in model_name else (3 if "m-" in model_name else (2 if "s-" in model_name else 1))

                for _ in range(5):
                    temp_detector.detect_persons(test_frame, use_tracking=False)

                durations = []
                for _ in range(20):
                    st = time.time()
                    dets = temp_detector.detect_persons(test_frame, use_tracking=False)

                    if worker and hasattr(worker, '_process_3d_logic') and dets:
                        gui_dets = copy.deepcopy(dets)
                        try:
                            worker._process_3d_logic(gui_dets, test_frame)
                        except Exception:
                            pass

                    durations.append((time.time() - st) * 1000.0)

                overhead_ms = 5.0
                avg_ms = float(np.mean(durations)) + overhead_ms
                max_ms = float(np.max(durations)) + overhead_ms
                min_ms = float(np.min(durations)) + overhead_ms

                avg_raw_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                useful_fps = min(avg_raw_fps, target_fps)

                load_pct = min(100.0, (avg_ms / smart_target_ms) * 100.0)
                watts = idle_tdp + ((load_pct / 100.0) * (max_tdp - idle_tdp))
                vram_usage = self._estimate_vram(model_name, cam_w, cam_h)

                all_results.append({
                    "model": model_name,
                    "tier": tier,
                    "avg_ms": avg_ms,
                    "useful_fps": useful_fps,
                    "raw_avg_fps": avg_raw_fps,
                    "min_fps": 1000.0 / max_ms if max_ms > 0 else 0,
                    "max_fps": 1000.0 / min_ms if min_ms > 0 else 0,
                    "load_pct": load_pct,
                    "watts": watts,
                    "vram": vram_usage
                })

                del temp_detector

            progress.accept()

            if not all_results:
                QMessageBox.warning(self, "Fehler", "Keine Modelle zum Testen gefunden!")
                return

            playable = [r for r in all_results if r["useful_fps"] >= (target_fps * 0.95)]
            if not playable: playable = all_results

            eco_winner = min(playable, key=lambda x: x["watts"])
            acc_winner = max(playable, key=lambda x: (x["tier"], -x["watts"]))
            fps_winner = min(playable, key=lambda x: x["avg_ms"])

            best_watts = min(r["watts"] for r in playable)
            best_ms = min(r["avg_ms"] for r in playable)

            for r in playable:
                norm_tier = r["tier"] / 4.0
                score_prec = norm_tier

                score_eco = best_watts / r["watts"]
                score_speed = best_ms / r["avg_ms"]

                r["sweet_score"] = (score_prec * 3.0) + (score_eco * 1.0) + (score_speed * 0.1)

            sweet_winner = max(playable, key=lambda x: x["sweet_score"])

            self._show_profile_selection(sweet_winner, fps_winner, acc_winner, eco_winner, target_fps)

        except Exception as e:
            progress.accept()
            QMessageBox.critical(self, "Fehler", f"Fehler im Profiler:\n{e}")

    def _show_profile_selection(self, sweet_data, fps_data, acc_data, eco_data, target_fps):
        dlg = QDialog(self)
        dlg.setWindowTitle("Real-World Hardware Analyse (Zero-Copy)")
        dlg.setMinimumWidth(700)
        vbox = QVBoxLayout(dlg)

        lbl_info = QLabel(
            f"✅ Profiling abgeschlossen! System nutzt native 1080p -> 640x640 Hardware-Skalierung.\n"
            f"Finde das beste Modell für deine GPU:")
        lbl_info.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 5px; color: #DDD;")
        vbox.addWidget(lbl_info)

        btn_style = """
            QPushButton { text-align: left; padding: 10px; border-radius: 6px; font-family: Consolas; font-size: 12px; line-height: 1.4; }
            QPushButton:hover { background-color: #333; }
        """

        def build_text(title, data):
            score_text = f" | Score: {data.get('sweet_score', 0):.1f}" if 'sweet_score' in data else ""
            return (f"{title}\n"
                    f"Modell: {data['model']}{score_text}\n"
                    f"Echte FPS: Ø {data['useful_fps']:.0f} (Puffer bis {data['raw_avg_fps']:.0f}) | Latenz: Ø {data['avg_ms']:.1f} ms\n"
                    f"Effizienz: ~{data['watts']:.0f} Watt (~{data['load_pct']:.0f}% Last) | VRAM: ~{data['vram']} MB")

        btn_sweet = QPushButton(
            build_text("👑 DAS BESTE VOM BESTEN (Fokus auf Präzision & moderatem Strom)", sweet_data))
        btn_sweet.setStyleSheet(btn_style + "background-color: #2D1B36; color: #FFD700; border: 2px solid #FFD700;")

        btn_sweet.clicked.connect(lambda: self._apply_profile(dlg, sweet_data['model'], sweet_data['useful_fps']))

        btn_acc = QPushButton(build_text("🎯 MAX PRÄZISION (Dickstes Modell, das gerade noch flüssig läuft)", acc_data))
        btn_acc.setStyleSheet(btn_style + "background-color: #2A1A10; color: #FF9600; border: 1px solid #FF9600;")

        btn_acc.clicked.connect(lambda: self._apply_profile(dlg, acc_data['model'], acc_data['useful_fps']))

        btn_fps = QPushButton(build_text("🚀 MAX PERFORMANCE (Geringste Latenz, Strom egal)", fps_data))
        btn_fps.setStyleSheet(btn_style + "background-color: #1A251A; color: #00FF96; border: 1px solid #00FF96;")

        btn_fps.clicked.connect(lambda: self._apply_profile(dlg, fps_data['model'], fps_data['useful_fps']))

        btn_eco = QPushButton(build_text("🍃 MAX ECO (Erreicht 30 FPS mit absolutem Minimum an Strom)", eco_data))
        btn_eco.setStyleSheet(btn_style + "background-color: #10202A; color: #00AADD; border: 1px solid #00AADD;")

        btn_eco.clicked.connect(lambda: self._apply_profile(dlg, eco_data['model'], eco_data['useful_fps']))

        vbox.addWidget(btn_sweet)
        vbox.addWidget(btn_acc)
        vbox.addWidget(btn_fps)
        vbox.addWidget(btn_eco)

        dlg.exec()

    def _apply_profile(self, dialog: QDialog, model_name: str, best_fps: float):
            self.combo_model.setCurrentText(model_name)
            optimal_fps = min(30, int(best_fps) + 2)
            self.inputs["target_fps"][0].setText(str(optimal_fps))

            dialog.accept()

    def get_data(self) -> Dict[str, Any]:
        results = {}
        for key, (widget, dtype) in self.inputs.items():
            raw_text = widget.text().replace(',', '.')
            if not raw_text: raw_text = "0"
            try:
                results[key] = int(float(raw_text)) if dtype == "int" else float(raw_text)
            except ValueError:
                results[key] = 0 if dtype == "int" else 1.0

        res_str = self.combo_res.currentText()
        new_res = [int(x) for x in res_str.split('x')]
        results["resolution"] = new_res

        results["render_capacity"] = 100

        results["active_lens_profile"] = self.combo_profile.currentText()
        results["camera_type"] = self.combo_type.currentText()
        results["model_path"] = f"client/config/graka/{self.combo_model.currentText()}"

        results["room_dimensions"] = {
            "width": float(self.inp_room_w.text().replace(',', '.') or 320.0),
            "height": float(self.inp_room_h.text().replace(',', '.') or 250.0),
            "depth": float(self.inp_room_d.text().replace(',', '.') or 470.0)
        }

        if new_res != self.old_resolution:
            scale_x = new_res[0] / max(1, self.old_resolution[0])
            scale_y = new_res[1] / max(1, self.old_resolution[1])

            if "custom_rectangles" in self.original_settings:
                rects = copy.deepcopy(self.original_settings["custom_rectangles"])
                for rect in rects:
                    for corner in rect.get("corners", []):
                        if "px" in corner and "py" in corner:
                            corner["px"] = int(corner["px"] * scale_x)
                            corner["py"] = int(corner["py"] * scale_y)
                results["custom_rectangles"] = rects

            if "pixel_points" in self.original_settings:
                pts = copy.deepcopy(self.original_settings["pixel_points"])
                new_pts = []
                for pt in pts:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        new_pts.append([int(pt[0] * scale_x), int(pt[1] * scale_y)])
                    elif isinstance(pt, dict):
                        pt["px"] = int(pt.get("px", 0) * scale_x)
                        pt["py"] = int(pt.get("py", 0) * scale_y)
                        new_pts.append(pt)
                results["pixel_points"] = new_pts

        return results