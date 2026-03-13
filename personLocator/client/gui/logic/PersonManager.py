import logging
import time
import numpy as np
from collections import deque
from typing import Dict, Optional, List


class PersonManager:
    """
    Verwaltet den Zustand von Personen.
    Zwei Aufgaben:
    1. Live-Glättung für UI (EMA Filter, Farben).
    2. Statistisches Lernen von Körpermaßen (Breite, Bein, Höhe) für Server-Sync.
    """

    def __init__(self, alpha_up: float = 0.6, alpha_down: float = 0.02):
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.history = {}

        self.color_state = {}

        self.manual_heights = {}

        self.stats_history: Dict[str, Dict[int, deque]] = {
            "width": {}, "leg": {}, "height": {}
        }
        self.local_bests: Dict[str, Dict[int, float]] = {
            "width": {}, "leg": {}, "height": {}
        }
        self.server_stats: Dict[str, Dict[int, float]] = {
            "width": {}, "leg": {}, "height": {}
        }
        self.last_sync_times = {}



    def _update_learning_metric(self, track_id: int, metric_type: str, value: float):
        """Sammelt Werte und berechnet den Durchschnitt der oberen 20% (Robustes Maximum)."""
        if value <= 0: return
        if track_id not in self.stats_history[metric_type]:
            self.stats_history[metric_type][track_id] = deque(maxlen=200)

        self.stats_history[metric_type][track_id].append(value)

        measurements = list(self.stats_history[metric_type][track_id])
        if len(measurements) > 5:
            sorted_vals = sorted(measurements, reverse=True)
            count = len(sorted_vals)
            start_idx = int(count * 0.02)
            end_idx = int(count * 0.22)
            if end_idx <= start_idx: end_idx = start_idx + 1
            top_vals = sorted_vals[start_idx:end_idx]

            if top_vals:
                avg_best = sum(top_vals) / len(top_vals)
                self.local_bests[metric_type][track_id] = avg_best

    def _get_reference_value(self, track_id: int, metric_type: str, fallback: float) -> float:
        """Gibt die beste verfügbare Schätzung zurück: Manuell > Server > Lokal gelernt > Fallback."""
        if track_id in self.server_stats[metric_type]:
            return self.server_stats[metric_type][track_id]
        if track_id in self.local_bests[metric_type]:
            return self.local_bests[metric_type][track_id]
        return fallback

    # =========================================================
    # PUBLIC API
    # =========================================================

    def update_width_measurement(self, p_id: int, val: float):
        if 20 < val < 80: self._update_learning_metric(p_id, "width", val)

    def update_leg_measurement(self, p_id: int, val: float):
        if 40 < val < 130: self._update_learning_metric(p_id, "leg", val)

    def update_height_measurement(self, p_id: int, val: float):
        if 100 < val < 280: self._update_learning_metric(p_id, "height", val)

    def set_server_stats(self, p_id: int, stats: dict):
        if "width" in stats: self.server_stats["width"][p_id] = stats["width"]
        if "leg" in stats: self.server_stats["leg"][p_id] = stats["leg"]
        if "height" in stats: self.server_stats["height"][p_id] = stats["height"]

    def set_manual_height(self, track_id: int, height: float):
        self.manual_heights[track_id] = float(height)

    def get_ref_width(self, p_id: int) -> float:
        return self._get_reference_value(p_id, "width", 45.0)

    def get_ref_leg(self, p_id: int) -> float:
        return self._get_reference_value(p_id, "leg", 90.0)

    def get_ref_height(self, p_id: int) -> float:
        return self._get_reference_value(p_id, "height", 175.0)

    def get_manual_height(self, track_id: int):
        return self.manual_heights.get(track_id)

    def is_ready_for_sync(self, p_id: int) -> bool:
        has_w = len(self.stats_history["width"].get(p_id, [])) >= 50
        has_h = len(self.stats_history["height"].get(p_id, [])) >= 40
        return has_w and has_h

    def should_send_update(self, p_id: int) -> bool:
        if not self.is_ready_for_sync(p_id): return False
        last_time = self.last_sync_times.get(p_id, 0)
        if (time.time() - last_time) > 240: return True
        return False

    def get_learned_stats_package(self, p_id: int) -> dict:
        return {
            "width": self.local_bests["width"].get(p_id, 0.0),
            "leg": self.local_bests["leg"].get(p_id, 0.0),
            "height": self.local_bests["height"].get(p_id, 0.0),
            "samples": len(self.stats_history["width"].get(p_id, []))
        }

    def mark_sent(self, p_id: int):
        self.last_sync_times[p_id] = time.time()

    def update_height_ema(self, track_id: int, raw_height: float, status: str) -> float:
        """Glättet die Höhe mit einem EMA-Filter. Manuelle Werte und Serverwerte haben Priorität."""
        if track_id in self.manual_heights: return self.manual_heights[track_id]
        if track_id in self.server_stats["height"]: return self.server_stats["height"][track_id]

        if raw_height is None: return self.history.get(track_id, 0.0)
        try:
            val = float(raw_height)
        except ValueError:
            return self.history.get(track_id, 0.0)

        if val < 50 or val > 280: return self.history.get(track_id, val)

        if track_id not in self.history:
            self.history[track_id] = val
            return val

        old_val = self.history[track_id]
        alpha = self.alpha_up if val > old_val else self.alpha_down
        if abs(val - old_val) > 30: alpha = 0.5
        new_val = (val * alpha) + (old_val * (1.0 - alpha))
        self.history[track_id] = new_val
        return new_val


    def update_learned_stat(self, track_id: int, stat_type: str, value: float) -> float:
        """Sammelt Werte und berechnet den Durchschnitt der oberen 10% (Robustes Maximum)."""
        if value <= 0:
            return self.local_bests[stat_type].get(track_id, 0.0)

        if track_id not in self.stats_history[stat_type]:
            self.stats_history[stat_type][track_id] = deque(maxlen=100)

        self.stats_history[stat_type][track_id].append(value)
        data = list(self.stats_history[stat_type][track_id])

        if len(data) < 5:
            best_val = max(data)
        else:
            data.sort(reverse=True)
            top_10_percent_count = max(1, int(len(data) * 0.10))
            top_values = data[:top_10_percent_count]

            best_val = sum(top_values) / len(top_values)

        self.local_bests[stat_type][track_id] = best_val
        return best_val



    def force_color_sync(self, track_id: int):
            """
            Kopiert das beste existierende 3D-Profil in den General-Speicher.
            Verhindert das 'Springen' der Farben beim Wechsel in den Performance-Modus.
            """
            if track_id not in self.color_state:
                return

            best_source = None
            for key in ['Front', 'Back', 'Side']:
                if self.color_state[track_id].get(key):
                    best_source = self.color_state[track_id][key]
                    break

            if best_source:
                self.color_state[track_id]['General'] = {
                    jid: np.copy(val) for jid, val in best_source.items()
                }
                logging.info(f"PersonManager: Farben für ID {track_id} von 3D nach General synchronisiert.")

    def update_colors(self, track_id: int, new_colors: dict, orientation: str) -> dict:
            """
            Aktualisiert Farben für ALLE Ansichten inkl. Performance (General).
            """
            if track_id not in self.color_state:
                self.color_state[track_id] = {'Front': {}, 'Side': {}, 'Back': {}, 'General': {}}

            view_key = None
            if orientation:
                if "Front" in orientation:
                    view_key = 'Front'
                elif "Back" in orientation or "Rücken" in orientation:
                    view_key = 'Back'
                elif "Side" in orientation or "Schräg" in orientation or "Profil" in orientation:
                    view_key = 'Side'
                elif "General" in orientation:
                    view_key = 'General'

            # 3. Lernen
            if view_key:
                if view_key not in self.color_state[track_id]:
                    self.color_state[track_id][view_key] = {}

                current_view_map = self.color_state[track_id][view_key]
                COLOR_ALPHA = 0.05

                for joint_id, hex_color in new_colors.items():
                    if not hex_color: continue
                    target_rgb = self._hex_to_rgb(hex_color)

                    if joint_id not in current_view_map:
                        current_view_map[joint_id] = target_rgb
                    else:
                        old_rgb = current_view_map[joint_id]
                        current_view_map[joint_id] = (old_rgb * (1.0 - COLOR_ALPHA)) + (target_rgb * COLOR_ALPHA)

            # 4. Output generieren
            detailed_output = {'Front': {}, 'Side': {}, 'Back': {}, 'General': {}}
            for v_key in detailed_output.keys():
                if v_key in self.color_state[track_id]:
                    for j_id, rgb in self.color_state[track_id][v_key].items():
                        detailed_output[v_key][j_id] = self._rgb_to_hex(rgb)

            display_map = {}
            priority_order = [view_key, 'General', 'Front', 'Back', 'Side']
            chosen_rgb_map = {}

            for key in priority_order:
                if key and self.color_state[track_id].get(key):
                    chosen_rgb_map = self.color_state[track_id][key]
                    break

            if not chosen_rgb_map:
                display_map = new_colors
            else:
                for j_id, rgb in chosen_rgb_map.items():
                    display_map[j_id] = self._rgb_to_hex(rgb)

            return {
                "display": display_map,
                "detailed": detailed_output
            }

    def _hex_to_rgb(self, hex_str: str) -> np.ndarray:
        h = hex_str.lstrip('#')
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)

    def _rgb_to_hex(self, rgb: np.ndarray) -> str:
        r, g, b = rgb.astype(int)
        return f"#{r:02x}{g:02x}{b:02x}"