import cv2
import numpy as np

from client.gui.logic.label.math.CalibrationMath import CalibrationMath
from client.gui.logic.label.math.GeometryMath import GeometryMath
from client.gui.logic.label.render.CalibrationRenderer import CalibrationRenderer, RenderColors


class RendererDashboard:
    """Kümmert sich exklusiv um das Live-Statistik-Panel am unteren Bildschirmrand."""

    @staticmethod
    def render_live_distortion_graph(pixel_points: list, world_points: list, custom_rectangles: list,
                                     room_dims: dict, person_results: list, width: int, height: int,
                                     yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
        """Generiert ein dreiteiliges Diagnose-Dashboard zur Echtzeitanalyse
        der Kalibrierungsqualität.
        Es visualisiert eine dynamische Boden-Heatmap inklusive Personen-Radar,
        einen Scatterplot zur Messung des optischen Pixel-Drifts sowie
        eine statistische Auswertung der räumlichen Projektionsfehler in Zentimetern."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # PnP-Pose für die Berechnungen holen
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles)
        if not pose:
            CalibrationRenderer.draw_hud_text(canvas, "Warte auf PnP-Kalibrierung...", (width // 2 - 100, height // 2),
                                              RenderColors.YELLOW, 0.6, 1)
            return canvas

        rvec, tvec, K, dist_coeffs = pose

        # Trennlinien
        sep1_x, sep2_x = 360, 680
        cv2.line(canvas, (sep1_x, 10), (sep1_x, height - 10), RenderColors.GRAY_DARK, 1, cv2.LINE_AA)
        cv2.line(canvas, (sep2_x, 10), (sep2_x, height - 10), RenderColors.GRAY_DARK, 1, cv2.LINE_AA)

        room_w = float(room_dims.get("width", 600))
        room_d = float(room_dims.get("depth", 800))

        # ==========================================================
        # PANEL 1 (LINKS): RAUM-HEATMAP & LIVE-PERSONEN (RADAR)
        # ==========================================================
        CalibrationRenderer.draw_hud_text(canvas, "Boden-Auflösung & Live-Radar", (20, 25), RenderColors.CYAN, 0.5, 1)

        off_x, off_y = 30, 50
        w_map, h_map = 300, 150

        cv2.rectangle(canvas, (off_x, off_y), (off_x + w_map, off_y + h_map), (15, 25, 15), -1)
        cv2.rectangle(canvas, (off_x, off_y), (off_x + w_map, off_y + h_map), (50, 80, 50), 1)

        steps = 8
        for i in range(steps + 1):
            for j in range(steps + 1):
                wx, wz = (i / steps) * room_w, (j / steps) * room_d
                p3d_start = np.array([wx, 0.0, wz], dtype=np.float32)
                p2d, _ = cv2.projectPoints(np.array([p3d_start]), rvec, tvec, K, dist_coeffs)
                px, py = int(p2d[0][0][0]), int(p2d[0][0][1])

                p3d_alt = GeometryMath.project_2d_to_3d(px + 1, py, pixel_points, world_points, custom_rectangles,
                                                        target_y=0.0)
                map_x, map_y = int(off_x + (wx / room_w) * w_map), int(off_y + (wz / room_d) * h_map)

                if p3d_alt is not None:
                    res = np.linalg.norm(p3d_alt - p3d_start)
                    color = (0, 200, 0) if res < 3.0 else ((0, 200, 200) if res < 7.0 else (0, 0, 200))
                    cv2.circle(canvas, (map_x, map_y), 3, color, -1, cv2.LINE_AA)
                    if i % 3 == 0 and j % 3 == 0:
                        cv2.putText(canvas, f"{res:.0f}", (map_x + 4, map_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                                    RenderColors.GRAY_LIGHT, 1, cv2.LINE_AA)

        if person_results:
            for p in person_results:
                pos = p.get('pos')
                if pos is not None:
                    px_map, pz_map = int(off_x + (pos[0] / room_w) * w_map), int(off_y + (pos[2] / room_d) * h_map)
                    if off_x <= px_map <= off_x + w_map and off_y <= pz_map <= off_y + h_map:
                        color = RenderColors.PERSON_STAND if "Stehend" in p.get('status',
                                                                                '') else RenderColors.PERSON_SIT
                        cv2.circle(canvas, (px_map, pz_map), 6, color, -1, cv2.LINE_AA)
                        cv2.circle(canvas, (px_map, pz_map), 8, RenderColors.WHITE, 1, cv2.LINE_AA)
                        cv2.putText(canvas, f"P{p['id']}", (px_map - 8, pz_map + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    RenderColors.WHITE, 1, cv2.LINE_AA)

        # ==========================================================
        # PANEL 2 (MITTE): OPTISCHER PIXEL-DRIFT (ZIELSCHEIBE x2)
        # ==========================================================
        cx_p, cy_p = (sep1_x + sep2_x) // 2, height // 2 + 10
        CalibrationRenderer.draw_hud_text(canvas, "Optischer Drift (x2)", (sep1_x + 20, 25), RenderColors.CYAN, 0.5, 1)

        for r in [25, 50, 75]:
            cv2.circle(canvas, (cx_p, cy_p), r, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx_p - 90, cy_p), (cx_p + 90, cy_p), (40, 40, 40), 1)
        cv2.line(canvas, (cx_p, cy_p - 90), (cx_p, cy_p + 90), (40, 40, 40), 1)

        def is_in_panel2(x, y):
            return (sep1_x + 5) < x < (sep2_x - 5) and 5 < y < (height - 5)

        if custom_rectangles:
            for rect in custom_rectangles:
                if not rect.get("is_active", True):
                    continue

                color = CalibrationRenderer._get_rectangle_color(rect.get("type"))
                for c in rect.get("corners", []):
                    if c.get("px") is None or c.get("py") is None: continue

                    p3d = np.array([[c["x"], c["y"], c["z"]]], dtype=np.float32)
                    p2d_proj, _ = cv2.projectPoints(p3d, rvec, tvec, K, dist_coeffs)

                    dx, dy = c["px"] - p2d_proj[0][0][0], c["py"] - p2d_proj[0][0][1]

                    draw_x, draw_y = int(cx_p + dx * 2), int(cy_p + dy * 2)
                    if is_in_panel2(draw_x, draw_y):
                        cv2.circle(canvas, (draw_x, draw_y), 3, color, -1, cv2.LINE_AA)

        if person_results:
            for p in person_results:
                bbox = p.get('bbox', [0, 0, 0, 0])
                detect_px, detect_py = (bbox[0] + bbox[2]) / 2, bbox[3]

                p3d_foot = np.array([[p['pos'][0], 0.0, p['pos'][2]]], dtype=np.float32)
                p2d_proj, _ = cv2.projectPoints(p3d_foot, rvec, tvec, K, dist_coeffs)

                dx, dy = detect_px - p2d_proj[0][0][0], detect_py - p2d_proj[0][0][1]

                draw_x, draw_y = int(cx_p + dx * 2), int(cy_p + dy * 2)
                if is_in_panel2(draw_x, draw_y):
                    color = RenderColors.PERSON_STAND if "Stehend" in p.get('status', '') else RenderColors.PERSON_SIT
                    cv2.drawMarker(canvas, (draw_x, draw_y), color, cv2.MARKER_TILTED_CROSS, 10, 2)
                    cv2.putText(canvas, f"P{p['id']}", (draw_x + 8, draw_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                                cv2.LINE_AA)

        # ==========================================================
        # PANEL 3 (RECHTS): MATRIX-FEHLER (STATISTIK)
        # ==========================================================
        hx = sep2_x + 15
        CalibrationRenderer.draw_hud_text(canvas, "Matrix-Fehler (Statistik)", (hx, 25), RenderColors.CYAN, 0.5, 1)

        errors = []
        valid_rects = []

        if custom_rectangles:
            for rect in custom_rectangles:
                if not rect.get("is_active", True):
                    continue

                c = rect["corners"][0]
                if c.get("px") is not None:
                    ideal_3d = np.array([c["x"], c["y"], c["z"]])
                    stats = CalibrationMath.evaluate_raycast_precision(c["px"], c["py"], ideal_3d, pixel_points,
                                                                       world_points, custom_rectangles)

                    if stats["status"] == "OK":
                        errors.append(stats["error_cm"])
                        drift_label = "Hinten" if stats["real_3d"][2] > c["z"] else "Vorne"
                        valid_rects.append((rect, stats["error_cm"], drift_label))

        if errors:
            mean_err = np.mean(errors)
            median_err = np.median(errors)
            max_err = np.max(errors)

            cv2.putText(canvas, f"Durchschnitt: {mean_err:5.1f} cm", (hx, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        RenderColors.WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Median:       {median_err:5.1f} cm", (hx, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        RenderColors.WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Maximum:      {max_err:5.1f} cm", (hx, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        RenderColors.YELLOW, 1, cv2.LINE_AA)

            cv2.line(canvas, (hx, 105), (width - 15, 105), RenderColors.GRAY_DARK, 1, cv2.LINE_AA)

            for i, (rect, err, drift) in enumerate(valid_rects[:5]):
                y_off = 125 + (i * 20)
                color = CalibrationRenderer._get_rectangle_color(rect.get("type"))
                cv2.putText(canvas, f"ID {rect['display_id']}: {err:4.1f} cm [{drift}]", (hx, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        return canvas