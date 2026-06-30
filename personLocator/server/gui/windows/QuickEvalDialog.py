"""
QuickEvalDialog — Schnellauswertung nach jedem REC-STOP.
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QGridLayout,
    QPushButton, QSplitter, QFileDialog, QScrollArea,
)
from PyQt6.QtCore import Qt

_STYLE = {
    "figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#45475a",   "axes.labelcolor": "#cdd6f4",
    "axes.grid": True,             "grid.color": "#313244",
    "grid.linestyle": "--",        "text.color": "#cdd6f4",
    "xtick.color": "#a6accd",      "ytick.color": "#a6accd",
    "legend.facecolor": "#1a1a2e", "legend.edgecolor": "#45475a",
}
COLORS = ["#00FFFF","#FF00FF","#FFFF00","#00FF96","#FFA500","#FF5555"]
NUMERIC_COLS = [
    "inference_ms","network_ms","server_ms","e2e_ms","client_fps",
    "server_triangulation_fps","repro_error_px","epipolar_error_avg",
    "loc_error_cm","loc_rmse_cm","health_index","id_switches_delta",
    "error_hungarian_delta","error_greedy_delta","kalman_smoothing_cm_delta",
    "ik_correction_cm_delta","ik_resizes_delta","epipolar_ghosts_delta",
    "camera_count","pos_x","pos_y","pos_z",
]
METRIC_CONFIG: dict = {
    "inference_ms":              {"label":"Inferenzzeit",         "unit":"ms",   "higher_is_better":False, "thresholds":[(20,"good"),(50,"ok"),(9999,"bad")],   "normal":"< 20 ms",    "tip_ok":"Aufloesung leicht reduzieren moeglich",      "tip_bad":"GPU ausgelastet — Aufloesung / Modell pruefen"},
    "network_ms":                {"label":"Netzwerklatenz",       "unit":"ms",   "higher_is_better":False, "thresholds":[(15,"good"),(40,"ok"),(9999,"bad")],   "normal":"< 15 ms",    "tip_ok":"WLAN-Distanz zum AP ggf. reduzieren",        "tip_bad":"Netz ueberlastet — Ethernet oder AP-Naehe"},
    "server_ms":                 {"label":"Server-Verarbeitung",  "unit":"ms",   "higher_is_better":False, "thresholds":[(25,"good"),(60,"ok"),(9999,"bad")],   "normal":"< 25 ms",    "tip_ok":"Bei mehreren Kameras normal",                 "tip_bad":"Triangulationsalgo wechseln (WLS statt LM)"},
    "e2e_ms":                    {"label":"Ende-zu-Ende",         "unit":"ms",   "higher_is_better":False, "thresholds":[(80,"good"),(200,"ok"),(9999,"bad")],  "normal":"< 80 ms",    "tip_ok":"Summe knapp akzeptabel",                     "tip_bad":"Groessten Anteil oben identifizieren"},
    "client_fps":                {"label":"Client FPS",           "unit":"fps",  "higher_is_better":True,  "thresholds":[(20,"good"),(12,"ok"),(0,"bad")],      "normal":"> 20 fps",   "tip_ok":"Ausreichend, Puffer gering",                 "tip_bad":"GPU-Auslastung / Aufloesung reduzieren"},
    "server_triangulation_fps":  {"label":"Server Tri-FPS",       "unit":"fps",  "higher_is_better":True,  "thresholds":[(25,"good"),(15,"ok"),(0,"bad")],      "normal":"> 25 fps",   "tip_ok":"Bei mehr Kameras erwartet",                  "tip_bad":"WLS waehlen oder Kamera-Anzahl reduzieren"},
    "repro_error_px":            {"label":"Reprojektionsfehler",  "unit":"px",   "higher_is_better":False, "thresholds":[(0.5,"good"),(1.5,"ok"),(99,"bad")],   "normal":"< 0.5 px",   "tip_ok":"Kalibrierung verwendbar, verbesserbar",      "tip_bad":"Mehr Schachbrett-Bilder aufnehmen"},
    "epipolar_error_avg":        {"label":"Epipolar-Fehler",      "unit":"cm",   "higher_is_better":False, "thresholds":[(3,"good"),(8,"ok"),(99,"bad")],       "normal":"< 3 cm",     "tip_ok":"Kameramatrizen leicht ungenau",               "tip_bad":"Kalibrierung oder Extrinsik pruefen"},
    "health_index":              {"label":"Health-Index",         "unit":"%",    "higher_is_better":True,  "thresholds":[(90,"good"),(70,"ok"),(0,"bad")],      "normal":"> 90 %",     "tip_ok":"Leichte Probleme aktiv — Filter-Tab pruefen", "tip_bad":"System in schlechtem Zustand"},
    "loc_error_cm":              {"label":"Lokalis.-Fehler",      "unit":"cm",   "higher_is_better":False, "thresholds":[(5,"good"),(15,"ok"),(99,"bad")],      "normal":"< 5 cm",     "tip_ok":"Fuer 1-Kamera normal — mehr Kameras helfen", "tip_bad":"Ground-Truth pruefen oder mehr Kameras"},
    "id_switches_delta":         {"label":"ID-Switches/Frame",    "unit":"/f",   "higher_is_better":False, "thresholds":[(0.01,"good"),(0.1,"ok"),(99,"bad")],  "normal":"< 0.01/f",   "tip_ok":"Gelegentliche Verwechslung bei Verdeckung",  "tip_bad":"Hungarian waehlen, Epipolar-Parameter pruefen"},
    "kalman_smoothing_cm_delta": {"label":"Filter-Glaettung",     "unit":"cm/f", "higher_is_better":False, "thresholds":[(0.5,"good"),(2.0,"ok"),(99,"bad")],   "normal":"< 0.5 cm/f", "tip_ok":"Filter arbeitet, Rauschen noch spuerbar",    "tip_bad":"Filter-Parameter oder Kameraqualitaet pruefen"},
    "ik_correction_cm_delta":    {"label":"IK-Korrektur",         "unit":"cm/f", "higher_is_better":False, "thresholds":[(0.3,"good"),(1.0,"ok"),(99,"bad")],   "normal":"< 0.3 cm/f", "tip_ok":"IK korrigiert, Proportionen leicht falsch",  "tip_bad":"Knochen-Laengenverhaeltnisse pruefen"},
}
STATUS_COLORS = {"good":"#a6e3a1","ok":"#f9e2af","bad":"#f38ba8"}
STATUS_ICONS  = {"good":"OK","ok":"~~","bad":"!!"}
STATUS_LABELS = {"good":"Gut","ok":"Akzeptabel","bad":"Verbesserungsbedarf"}
BLOCK_CONFIG: dict = {
    "Block_1": {"metrics":["inference_ms","client_fps"],                                              "plot":"hist_by_camera",   "plot_col":"inference_ms", "title":"Block 1 - Inferenzlatenz"},
    "Block_2": {"metrics":["inference_ms","network_ms","server_ms","e2e_ms"],                         "plot":"stacked_bar",                                  "title":"Block 2 - Ende-zu-Ende-Latenz"},
    "Block_3": {"metrics":["server_ms","server_triangulation_fps","e2e_ms"],                          "plot":"line_vs_cameras",                              "title":"Block 3 - Skalierbarkeit"},
    "Block_4": {"metrics":["repro_error_px","epipolar_error_avg"],                                    "plot":"single_values",                                "title":"Block 4 - Kalibrierungsqualitaet"},
    "Block_5": {"metrics":["loc_error_cm","server_triangulation_fps","repro_error_px"],               "plot":"compare_triang",                              "title":"Block 5 - Lokalisierungsfehler"},
    "Block_6": {"metrics":["loc_error_cm"],                                                           "plot":"heatmap_2d",                                   "title":"Block 6 - Fehlerverteilung im Raum"},
    "Block_7": {"metrics":["kalman_smoothing_cm_delta","ik_correction_cm_delta"],                     "plot":"bar_by_filter",                                "title":"Block 7 - Filtervergleich"},
    "Block_8": {"metrics":["id_switches_delta","error_hungarian_delta","error_greedy_delta"],          "plot":"timeseries_switches",                          "title":"Block 8 - ID-Switches"},
    "Block_9": {"metrics":["health_index","server_triangulation_fps"],                                "plot":"timeseries_health",                            "title":"Block 9 - Systemstabilitaet"},
}

def _get_status(metric: str, value: float) -> str:
    cfg = METRIC_CONFIG.get(metric)
    if cfg is None or math.isnan(value): return "ok"
    higher = cfg["higher_is_better"]
    for threshold, status in cfg["thresholds"]:
        if higher:
            if value >= threshold: return status
        else:
            if value <= threshold: return status
    return cfg["thresholds"][-1][1]


class QuickEvalDialog(QDialog):
    def __init__(self, session_csv: str, block_name: str, parent=None):
        super().__init__(parent)
        self.session_csv = session_csv
        self.block_name  = block_name
        self.block_key   = next((k for k in BLOCK_CONFIG if block_name.startswith(k)), "Block_2")
        self.block_cfg   = BLOCK_CONFIG[self.block_key]
        self._last_fig: Optional[plt.Figure] = None
        self.setWindowTitle(f"Schnellauswertung -- {block_name}")
        self.resize(1280, 740)
        self.setStyleSheet("background: #121212; color: #cdd6f4;")
        self.df = self._load_csv()
        self._build_ui()

    def _load_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.session_csv)
            for col in NUMERIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            print(f"QuickEval load error: {e}")
            return pd.DataFrame()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8); layout.setContentsMargins(12,12,12,12)
        n = len(self.df)
        title = QLabel(f"  {self.block_cfg['title']}  --  {n:,} Frames aufgezeichnet")
        title.setStyleSheet("font-size:16px;font-weight:bold;color:#00FFFF;margin-bottom:4px;")
        layout.addWidget(title)
        if self.df.empty:
            layout.addWidget(QLabel("Keine Daten in der Session-Datei."))
            layout.addWidget(self._close_btn()); return
        spl = QSplitter(Qt.Orientation.Horizontal)
        spl.setStyleSheet("QSplitter::handle{background:#313244;width:3px;}")
        spl.addWidget(self._metrics_panel())
        spl.addWidget(self._plot_panel())
        spl.setSizes([430, 830])
        layout.addWidget(spl, stretch=1)
        lbl = QLabel(self._summary_text())
        lbl.setWordWrap(True)
        lbl.setStyleSheet("background:#1e1e2e;color:#cdd6f4;font-size:13px;"
                          "padding:10px 14px;border-radius:6px;border:1px solid #313244;")
        layout.addWidget(lbl)
        row = QHBoxLayout()
        btn_s = QPushButton("  Diagramm speichern")
        btn_s.setStyleSheet("background:#313244;color:#cdd6f4;padding:8px 18px;"
                             "font-size:13px;border-radius:4px;border:1px solid #45475a;")
        btn_s.clicked.connect(self._save_plot)
        btn_e = QPushButton("  Daten exportieren")
        btn_e.setStyleSheet("background:#1a2e1a;color:#a6e3a1;padding:8px 18px;"
                             "font-size:13px;border-radius:4px;border:1px solid #45475a;")
        btn_e.clicked.connect(self._export_data)
        row.addWidget(btn_s); row.addWidget(btn_e); row.addStretch(); row.addWidget(self._close_btn())
        layout.addLayout(row)

    def _close_btn(self):
        b = QPushButton("  Schliessen")
        b.setStyleSheet("background:#3d1515;color:#FF5555;padding:8px 18px;"
                         "font-size:13px;border-radius:4px;border:1px solid #45475a;")
        b.clicked.connect(self.accept); return b

    def _metrics_panel(self) -> QWidget:
        panel = QWidget(); panel.setStyleSheet("background:#1a1a2e;")
        v = QVBoxLayout(panel); v.setContentsMargins(10,10,10,10); v.setSpacing(6)
        v.addWidget(QLabel("Metriken -- Bewertung"))
        sc = QScrollArea(); sc.setWidgetResizable(True)
        sc.setStyleSheet("QScrollArea{border:none;background:#1a1a2e;}")
        inner = QWidget(); inner.setStyleSheet("background:#1a1a2e;")
        grid = QGridLayout(inner); grid.setSpacing(4); grid.setContentsMargins(0,0,0,0)
        for ci, h in enumerate(["Metrik","Mittelwert","Median","p95","Normal","Status"]):
            lbl = QLabel(h)
            lbl.setStyleSheet("font-weight:bold;color:#00FFFF;font-size:11px;"
                               "padding:4px 6px;background:#1e1e2e;border-radius:3px;")
            grid.addWidget(lbl, 0, ci)
        triang_modes = []
        if self.block_key == "Block_5" and "triangulation_mode" in self.df.columns:
            triang_modes = sorted(self.df["triangulation_mode"].dropna().unique().tolist())

        for ri, metric in enumerate(self.block_cfg.get("metrics",[]), start=1):
            if metric not in self.df.columns: continue
            s = self.df[metric].dropna()
            if s.empty: continue
            mean_v = float(s.mean()); p50 = float(s.median()); p95 = float(s.quantile(0.95))
            status = _get_status(metric, mean_v)
            color  = STATUS_COLORS[status]
            cfg    = METRIC_CONFIG.get(metric, {})
            unit   = cfg.get("unit",""); normal = cfg.get("normal","--")
            bg = "#1a1a2e" if ri%2==0 else "#16161e"
            cs = f"background:{bg};padding:5px 6px;font-size:12px;border-radius:3px;"
            cells = [
                (cfg.get("label", metric), "#cdd6f4", False),
                (f"{mean_v:.2f} {unit}",   color,     True),
                (f"{p50:.2f}",             "#a6accd",  False),
                (f"{p95:.2f}",             "#585b70",  False),
                (normal,                   "#89b4fa",  False),
                (f"{STATUS_ICONS[status]} {STATUS_LABELS[status]}", color, True),
            ]
            for ci, (txt, col, bold) in enumerate(cells):
                l = QLabel(txt)
                l.setStyleSheet(f"color:{col};font-weight:{'bold' if bold else 'normal'};{cs}")
                l.setWordWrap(True)
                grid.addWidget(l, ri, ci)

            if triang_modes and metric in ("loc_error_cm", "server_triangulation_fps"):
                for mode in triang_modes:
                    ri += 1
                    sub = self.df[self.df["triangulation_mode"] == mode][metric].dropna()
                    if sub.empty: continue
                    mv = float(sub.mean()); st = _get_status(metric, mv)
                    cl = STATUS_COLORS[st]
                    bg2 = "#0d1f2d"
                    cs2 = f"background:{bg2};padding:4px 6px;font-size:11px;border-radius:3px;"
                    mode_cells = [
                        (f"  ↳ {mode}", "#89b4fa", False),
                        (f"{mv:.2f} {unit}", cl, True),
                        (f"{float(sub.median()):.2f}", "#a6accd", False),
                        (f"{float(sub.quantile(0.95)):.2f}", "#585b70", False),
                        ("", "#89b4fa", False),
                        (f"{STATUS_ICONS[st]} {STATUS_LABELS[st]}", cl, True),
                    ]
                    for ci, (txt, col, bold) in enumerate(mode_cells):
                        l = QLabel(txt)
                        l.setStyleSheet(f"color:{col};font-weight:{'bold' if bold else 'normal'};{cs2}")
                        grid.addWidget(l, ri, ci)
        sc.setWidget(inner); v.addWidget(sc, stretch=1)
        return panel

    def _plot_panel(self) -> QWidget:
        panel = QWidget(); panel.setStyleSheet("background:#1e1e2e;")
        v = QVBoxLayout(panel); v.setContentsMargins(8,8,8,8)
        try:
            with plt.rc_context(_STYLE):
                fig = self._make_fig()
            if fig:
                self._last_fig = fig
                canvas = FigureCanvas(fig)
                canvas.setStyleSheet("background:#1e1e2e;")
                v.addWidget(canvas); return panel
        except Exception as e:
            print(f"QuickEval plot error: {e}")
        lbl = QLabel("Keine Plot-Daten verfuegbar.")
        lbl.setStyleSheet("color:#585b70;font-size:14px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lbl); return panel

    def _make_fig(self) -> Optional[plt.Figure]:
        df = self.df; pt = self.block_cfg.get("plot","")

        if pt == "hist_by_camera":
            col = self.block_cfg.get("plot_col","inference_ms")
            if col not in df.columns: return None
            fig, ax = plt.subplots(figsize=(8,4.5))
            ax.set_title(f"Verteilung: {METRIC_CONFIG.get(col,{}).get('label',col)}", fontsize=12)
            for i, (cam, grp) in enumerate(df.groupby("camera")):
                ax.hist(grp[col].dropna(), bins=40, alpha=0.75,
                        color=COLORS[i%len(COLORS)], label=str(cam), edgecolor="none")
            cfg = METRIC_CONFIG.get(col,{})
            for thresh, st in cfg.get("thresholds",[]):
                if thresh < 9000:
                    ax.axvline(thresh, color=STATUS_COLORS[st], ls="--", lw=1.2, alpha=0.7,
                               label=f"{STATUS_LABELS[st]}: {thresh} {cfg.get('unit','')}")
            ax.set_xlabel(cfg.get("label",col)); ax.set_ylabel("Haeufigkeit"); ax.legend(fontsize=9)
            fig.tight_layout(); return fig

        elif pt == "stacked_bar":
            cols = [c for c in ["inference_ms","network_ms","server_ms"] if c in df.columns]
            if not cols or "camera" not in df.columns: return None
            means = df.groupby("camera")[cols].mean()
            x = np.arange(len(means))
            fig, ax = plt.subplots(figsize=(8,4.5))
            ax.set_title("Latenz-Aufschluesselung pro Kamera", fontsize=12)
            bottom = np.zeros(len(means))
            for col, lbl, clr in zip(cols, ["Inferenz","Netzwerk","Server"], ["#00FFFF","#FF00FF","#FFFF00"]):
                vals = means[col].values
                ax.bar(x, vals, 0.55, bottom=bottom, label=lbl, color=clr, alpha=0.85)
                for xi,(v,b) in enumerate(zip(vals,bottom)):
                    if v>1: ax.text(xi,b+v/2,f"{v:.0f}",ha="center",va="center",fontsize=9,color="#11111b",fontweight="bold")
                bottom += vals
            ax.set_xticks(x); ax.set_xticklabels(means.index, rotation=20, ha="right")
            ax.set_ylabel("Zeit (ms)"); ax.legend(fontsize=9)
            fig.tight_layout(); return fig

        elif pt == "line_vs_cameras":
            if "camera_count" not in df.columns or "server_ms" not in df.columns: return None
            grp = df.groupby("camera_count")["server_ms"].agg(["mean","std"]).reset_index()
            fig, ax = plt.subplots(figsize=(8,4.5))
            ax.set_title("Server-Latenz vs. Kameraanzahl", fontsize=12)
            ax.errorbar(grp["camera_count"], grp["mean"], yerr=grp["std"],
                        fmt="o-", color="#00FFFF", lw=2, ms=8, capsize=5, ecolor="#45475a")
            ax.set_xlabel("Aktive Kameras"); ax.set_ylabel("Server-Latenz (ms)")
            fig.tight_layout(); return fig

        elif pt == "single_values":
            ms = [m for m in self.block_cfg.get("metrics",[]) if m in df.columns]
            if not ms: return None
            means = [float(df[m].mean()) for m in ms]
            labels = [METRIC_CONFIG.get(m,{}).get("label",m) for m in ms]
            colors = [STATUS_COLORS[_get_status(m,v)] for m,v in zip(ms,means)]
            fig, ax = plt.subplots(figsize=(8,4))
            ax.set_title("Kalibrierungsqualitaet", fontsize=12)
            bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.4)
            for bar,val in zip(bars,means):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=11, color="#cdd6f4")
            ax.set_ylabel("Fehler"); fig.tight_layout(); return fig

        elif pt == "scatter_positions":
            if not all(c in df.columns for c in ["pos_x","pos_z"]): return None
            valid = df[["pos_x","pos_z"]].dropna()
            if valid.empty: return None
            fig, ax = plt.subplots(figsize=(8,5))
            ax.set_title("Gemessene Positionen (Draufsicht)", fontsize=12)
            ax.scatter(valid["pos_x"], valid["pos_z"], s=12, alpha=0.4, color="#00FFFF")
            ax.set_xlabel("X (cm)"); ax.set_ylabel("Z (cm)")
            fig.tight_layout(); return fig

        elif pt == "heatmap_2d":
            if not all(c in df.columns for c in ["pos_x","pos_z"]): return None
            has_err = "loc_error_cm" in df.columns
            cols_needed = ["pos_x","pos_z"] + (["loc_error_cm"] if has_err else [])
            valid = df[cols_needed].dropna()
            if valid.empty: return None
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Block 6 - Raeumliche Fehlerverteilung (Draufsicht)", fontsize=12)
            ax = axes[0]
            if has_err:
                sc = ax.scatter(valid["pos_x"], valid["pos_z"],
                                c=valid["loc_error_cm"], s=8, alpha=0.5,
                                cmap="RdYlGn_r", vmin=0, vmax=30)
                cb = fig.colorbar(sc, ax=ax)
                cb.set_label("Lokalis.-Fehler (cm)")
                ax.set_title("Fehler pro Position")
                mean_err = float(valid["loc_error_cm"].mean())
                ax.set_xlabel(f"X (cm)  |  Mittl. Fehler: {mean_err:.1f} cm")
            else:
                ax.scatter(valid["pos_x"], valid["pos_z"], s=8, alpha=0.4, color="#00FFFF")
                ax.set_title("Gemessene Positionen (kein Fehler-GT)")
                ax.set_xlabel("X (cm)")
            ax.set_ylabel("Z (cm)")
            ax2 = axes[1]
            hb = ax2.hexbin(valid["pos_x"], valid["pos_z"],
                            gridsize=20, cmap="plasma", mincnt=1)
            fig.colorbar(hb, ax=ax2, label="Frames (Abdeckung)")
            ax2.set_title("Raumabdeckung")
            ax2.set_xlabel("X (cm)"); ax2.set_ylabel("Z (cm)")
            fig.tight_layout(); return fig

        elif pt == "bar_by_filter":
            if "smoothing_mode" not in df.columns: return None
            ms = [m for m in ["kalman_smoothing_cm_delta","ik_correction_cm_delta"] if m in df.columns]
            if not ms: return None
            grp = df.groupby("smoothing_mode")[ms].mean()
            x = np.arange(len(grp)); w = 0.35
            fig, ax = plt.subplots(figsize=(8,4.5))
            ax.set_title("Filtervergleich -- Glaettungstiefe", fontsize=12)
            for i,(col,clr) in enumerate(zip(ms,["#00FF96","#DDAAFF"])):
                ax.bar(x+i*w, grp[col].values, w, label=col.replace("_delta",""), color=clr, alpha=0.85)
            ax.set_xticks(x+w/2); ax.set_xticklabels(grp.index)
            ax.set_ylabel("Korrektur (cm/Frame)"); ax.legend(fontsize=9)
            fig.tight_layout(); return fig

        elif pt == "timeseries_switches":
            if "id_switches_delta" not in df.columns: return None
            sub = df.sort_values("system_time") if "system_time" in df.columns else df.copy()
            t = np.arange(len(sub)); total = int(sub["id_switches_delta"].sum())
            fig, ax = plt.subplots(figsize=(8,4))
            ax.set_title(f"ID-Switches ueber Zeit  (Gesamt: {total})", fontsize=12)
            ax.fill_between(t, sub["id_switches_delta"], step="mid", color="#FF5555", alpha=0.75, label="Switches")
            if "error_hungarian_delta" in sub.columns:
                ax.plot(t, sub["error_hungarian_delta"].rolling(10,min_periods=1).mean(),
                        color="#00FF96", lw=1.2, alpha=0.7, label="Hungarian (geglaett.)")
            ax.set_xlabel("Frame"); ax.set_ylabel("Switches / Frame"); ax.legend(fontsize=9)
            fig.tight_layout(); return fig

        elif pt == "timeseries_health":
            if "health_index" not in df.columns: return None
            sub = df.sort_values("system_time") if "system_time" in df.columns else df.copy()
            t = np.arange(len(sub)); win = min(30, max(1, len(sub)//10))
            has_fps = "server_triangulation_fps" in sub.columns
            fig, axes = plt.subplots(2 if has_fps else 1, 1, figsize=(8,5), sharex=True)
            if not has_fps: axes = [axes]
            fig.suptitle("Systemstabilitaet", fontsize=12)
            axes[0].plot(t, sub["health_index"].rolling(win,min_periods=1).mean(), color="#00FF96", lw=1.5)
            axes[0].axhline(90, color="#FFAA00", ls="--", alpha=0.6, label="Ziel: >= 90 %")
            axes[0].set_ylabel("Health-Index (%)"); axes[0].set_ylim(0,105); axes[0].legend(fontsize=9)
            if has_fps:
                axes[1].plot(t, sub["server_triangulation_fps"].rolling(win,min_periods=1).mean(), color="#00FFFF", lw=1.5)
                axes[1].set_ylabel("Tri-FPS")
            axes[-1].set_xlabel("Frame")
            fig.tight_layout(); return fig
        elif pt == "compare_triang":
            if "loc_error_cm" not in df.columns: return None
            has_mode = "triangulation_mode" in df.columns
            modes = sorted(df["triangulation_mode"].dropna().unique()) if has_mode else ["alle"]
            if not has_mode:
                df = df.copy(); df["triangulation_mode"] = "alle"
            box_colors = ["#00FFFF","#FF00FF","#FFFF00","#00FF96"]
            has_fps = "server_triangulation_fps" in df.columns
            ncols = 3 if has_fps else 2
            fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4.5))
            fig.suptitle("Block 5 - Lokalisierungsfehler & Server-FPS: WLS vs LM", fontsize=12)
            ax = axes[0]
            data_err = [df[df["triangulation_mode"]==m]["loc_error_cm"].dropna().values for m in modes]
            bp = ax.boxplot(data_err, labels=modes, patch_artist=True, notch=False,
                            medianprops={"color":"#11111b","linewidth":2})
            for patch, clr in zip(bp["boxes"], box_colors):
                patch.set_facecolor(clr); patch.set_alpha(0.75)
            for i, (mode, data) in enumerate(zip(modes, data_err)):
                if len(data) > 0:
                    ax.text(i+1, float(np.mean(data)),
                            f"  MW\n  {float(np.mean(data)):.1f} cm",
                            ha="left", va="center", fontsize=9, color="#cdd6f4")
            ax.set_ylabel("Lokalisierungsfehler (cm)")
            ax.set_title("Genauigkeit")
            ax.axhline(5,  color="#a6e3a1", ls="--", lw=1, alpha=0.7, label="Gut: < 5 cm")
            ax.axhline(15, color="#f38ba8", ls="--", lw=1, alpha=0.7, label="Schlecht: > 15 cm")
            ax.legend(fontsize=8)
            if has_fps:
                ax2 = axes[1]
                data_fps = [df[df["triangulation_mode"]==m]["server_triangulation_fps"].dropna().values
                            for m in modes]
                bp2 = ax2.boxplot(data_fps, labels=modes, patch_artist=True, notch=False,
                                  medianprops={"color":"#11111b","linewidth":2})
                for patch, clr in zip(bp2["boxes"], box_colors):
                    patch.set_facecolor(clr); patch.set_alpha(0.75)
                for i, (mode, data) in enumerate(zip(modes, data_fps)):
                    if len(data) > 0:
                        ax2.text(i+1, float(np.mean(data)),
                                 f"  MW\n  {float(np.mean(data)):.1f} fps",
                                 ha="left", va="center", fontsize=9, color="#cdd6f4")
                ax2.set_ylabel("Server Triangulations-FPS")
                ax2.set_title("Rechenaufwand")
                ax2.axhline(25, color="#a6e3a1", ls="--", lw=1, alpha=0.7, label="Gut: > 25 fps")
                ax2.axhline(15, color="#f38ba8", ls="--", lw=1, alpha=0.7, label="Schlecht: < 15 fps")
                ax2.legend(fontsize=8)
            ax3 = axes[2 if has_fps else 1]
            ax3.set_title("Gemessene Positionen (Draufsicht)")
            for i, mode in enumerate(modes):
                sub = df[df["triangulation_mode"]==mode][["pos_x","pos_z"]].dropna()
                if not sub.empty:
                    ax3.scatter(sub["pos_x"], sub["pos_z"], s=10, alpha=0.35,
                                color=box_colors[i % len(box_colors)], label=mode)
            ax3.set_xlabel("X (cm)"); ax3.set_ylabel("Z (cm)")
            ax3.legend(fontsize=9)
            fig.tight_layout(); return fig
        return None

    def _summary_text(self) -> str:
        good, ok_m, bad, tips = [], [], [], []
        for m in self.block_cfg.get("metrics",[]):
            if m not in self.df.columns: continue
            s = self.df[m].dropna()
            if s.empty: continue
            mean_v = float(s.mean()); status = _get_status(m, mean_v)
            cfg = METRIC_CONFIG.get(m,{}); label = cfg.get("label",m)
            unit = cfg.get("unit",""); normal = cfg.get("normal","")
            entry = f"{label}: {mean_v:.2f} {unit} (Normal: {normal})"
            if status == "good": good.append(entry)
            elif status == "ok":
                ok_m.append(entry)
                if cfg.get("tip_ok"): tips.append(f"  {label}: {cfg['tip_ok']}")
            else:
                bad.append(entry)
                if cfg.get("tip_bad"): tips.append(f"  {label}: {cfg['tip_bad']}")
        lines = [f"{len(self.df):,} Frames  |  OK: {len(good)}  Akzeptabel: {len(ok_m)}  Probleme: {len(bad)}"]
        if good:  lines.append("GUT:    " + "  |  ".join(good))
        if ok_m:  lines.append("OK:     " + "  |  ".join(ok_m))
        if bad:   lines.append("FEHLER: " + "  |  ".join(bad))
        if tips:  lines.append("TIPPS:  " + "  |  ".join(tips))
        if not good and not ok_m and not bad:
            lines.append("Keine relevanten Metriken gefunden.")
        return "\n".join(lines)

    def _export_data(self):
        import json, csv as _csv, os
        default_name = f"eval_summary_{self.block_name}"
        path, sel = QFileDialog.getSaveFileName(
            self, "Zusammenfassung exportieren", default_name,
            "JSON (*.json);;CSV (*.csv)")
        if not path:
            return
        if not (path.endswith(".json") or path.endswith(".csv")):
            path += ".json" if "JSON" in sel else ".csv"

        summary = {
            "block": self.block_name,
            "session_csv": self.session_csv,
            "frame_count": len(self.df),
            "metrics": {},
        }
        rows_csv = []
        for m in self.block_cfg.get("metrics", []):
            if m not in self.df.columns:
                continue
            s = self.df[m].dropna()
            if s.empty:
                continue
            mean_v = float(s.mean())
            status = _get_status(m, mean_v)
            cfg = METRIC_CONFIG.get(m, {})
            entry = {
                "label":  cfg.get("label", m),
                "unit":   cfg.get("unit", ""),
                "mean":   round(mean_v, 3),
                "median": round(float(s.median()), 3),
                "p95":    round(float(s.quantile(0.95)), 3),
                "min":    round(float(s.min()), 3),
                "max":    round(float(s.max()), 3),
                "std":    round(float(s.std()), 3),
                "normal": cfg.get("normal", ""),
                "status": status,
                "tip":    cfg.get("tip_bad" if status == "bad" else "tip_ok", "") if status != "good" else "",
            }
            summary["metrics"][m] = entry
            rows_csv.append([m] + [entry[k] for k in
                ["label", "unit", "mean", "median", "p95", "min", "max", "std", "normal", "status", "tip"]])
        try:
            if path.endswith(".csv"):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = _csv.writer(f)
                    w.writerow(["metric","label","unit","mean","median","p95","min","max","std","normal","status","tip"])
                    w.writerows(rows_csv)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
            if self._last_fig is not None:
                plot_path = os.path.splitext(path)[0] + ".png"
                self._last_fig.savefig(plot_path, dpi=150, bbox_inches="tight",
                                       facecolor=self._last_fig.get_facecolor())
        except Exception as e:
            print(f"Export error: {e}")

    def _save_plot(self):
        if self._last_fig is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Diagramm speichern", f"eval_{self.block_name}.png",
            "PNG (*.png);;PDF (*.pdf)")
        if path:
            self._last_fig.savefig(path, dpi=150, bbox_inches="tight",
                                   facecolor=self._last_fig.get_facecolor())
