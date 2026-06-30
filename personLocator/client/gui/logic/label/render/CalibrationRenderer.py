import cv2
import numpy as np


class RenderColors:
    """Zentrale Definition aller Farben für einheitliches Design."""
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY_LIGHT = (200, 200, 200)
    GRAY_MID = (150, 150, 150)
    GRAY_DARK = (80, 80, 80)
    SIGHTLINE = (200, 255, 255)
    PERSON_STAND = (0, 255, 127)
    PERSON_SIT = (0, 165, 255)
    MATRIX_GREEN = (0, 120, 0)


class CalibrationRenderer:
    """Basis-Class für das Zeichnen (Farben, Linien, HUD)."""

    SKELETON_LINKS = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
        (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)
    ]

    @staticmethod
    def draw_hud_text(img: np.ndarray, text: str, pos: tuple[int, int], color: tuple, scale: float = 0.6,
                      thickness: int = 2) -> None:
        """Zeichnet einen formatierten Text mit scharfem Kontrast
        und einem, abdunkelnden Hintergrund-Rechteck
        auf das übergebene Bild, ideal für klare HUD-Elemente."""
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (pos[0] - 6, pos[1] - h - 8), (pos[0] + w + 6, pos[1] + 6), RenderColors.BLACK, -1)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _get_rectangle_color(rect_type: str) -> tuple:
        """Ermittelt und liefert den exakten RGB-Farbwert
        (als Tuple) aus den vordefinierten RenderColors
        basierend auf der übergebenen Kategorie des Rechtecktyps."""
        if rect_type == "Typ 1": return RenderColors.CYAN
        if rect_type == "Typ 2": return RenderColors.MAGENTA
        return RenderColors.YELLOW

    @staticmethod
    def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=8):
        """Zeichnet eine präzise gestrichelte Linie zwischen zwei
        Koordinaten durch iterative Vektoraddition entlang der
        berechneten euklidischen Distanz."""
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist < 1: return
        vec = (np.array(pt2) - np.array(pt1)) / dist
        for i in range(0, int(dist), dash_length * 2):
            start = tuple((np.array(pt1) + vec * i).astype(int))
            end = tuple((np.array(pt1) + vec * min(i + dash_length, dist)).astype(int))
            cv2.line(img, start, end, color, thickness, cv2.LINE_AA)