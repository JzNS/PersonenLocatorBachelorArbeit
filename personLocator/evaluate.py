
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import seaborn as sns

LOG_PATH = Path("logs/eval_log.csv")
OUT_DIR  = Path("logs/plots")

# Eigene Ground-Truth-Messpunkte eintragen: {"Name": (x_cm, z_cm)}
GROUND_TRUTH: dict[str, tuple[float, float]] = {
    "P1": (100.0, 100.0),
    "P2": (300.0, 100.0),
    "P3": (500.0, 100.0),
    "P4": (100.0, 400.0),
    "P5": (300.0, 400.0),
    "P6": (500.0, 400.0),
}

ROOM_IMAGE_PATH: Optional[str] = None

_STYLE = {
    "figure.facecolor": "#121212",
    "axes.facecolor":   "#1e1e2e",
    "axes.edgecolor":   "#45475a",
    "axes.labelcolor":  "#cdd6f4",
    "axes.grid": True,
    "grid.color":       "#313244",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "xtick.color":      "#cdd6f4",
    "ytick.color":      "#cdd6f4",
    "text.color":       "#cdd6f4",
    "legend.facecolor": "#1e1e2e",
    "legend.edgecolor": "#45475a",
}
plt.rcParams.update(_STYLE)
sns.set_theme(style="darkgrid", rc=_STYLE)
COLORS = ["#00FFFF", "#FF00FF", "#FFFF00", "#00FF96", "#FFA500", "#FF5555"]


NUMERIC_COLS = [
    "t1","t2","t3","t4",
    "inference_ms","network_ms","server_ms","e2e_ms",
    "client_fps","server_triangulation_fps",
    "repro_error_px","epipolar_error_avg","loc_error_cm","loc_rmse_cm",
    "health_index","pos_x","pos_y","pos_z",
    "epipolar_ghosts_delta",
    "id_switches_delta","error_hungarian_delta","error_greedy_delta",
    "ik_resizes_delta","ik_correction_cm_delta",
    "kalman_blocked_delta","kalman_smoothing_cm_delta",
]


def _load(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_all(log_dir: Path) -> pd.DataFrame:
    """Laedt alle eval_log_Block_*.csv aus log_dir und haengt sie zusammen.
    Gibt einen leeren DataFrame zurueck wenn keine Dateien gefunden.
    """
    files = sorted(log_dir.glob("eval_log_Block_*.csv"))
    if not files:
        # Fallback: veraltete eval_log.csv
        fallback = log_dir / "eval_log.csv"
        if fallback.exists():
            files = [fallback]
    if not files:
        return pd.DataFrame()
    print(f"  Gefundene Log-Dateien ({len(files)}):")
    parts = []
    for p in files:
        try:
            part = _load(p)
            print(f"    {p.name}: {len(part):,} Zeilen")
            parts.append(part)
        except Exception as e:
            print(f"    {p.name}: Fehler beim Lesen - {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _block(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df[df["test_block"].str.startswith(prefix, na=False)].copy()


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  --> {path}")
    plt.close(fig)


def _describe(df: pd.DataFrame, col: str, label: str = "") -> None:
    s = df[col].dropna()
    tag = label or col
    if s.empty:
        print(f"  {tag}: keine Daten"); return
    print(f"  {tag:25s}  n={len(s):>6,}  mu={s.mean():>8.2f}  "
          f"sigma={s.std():>7.2f}  p50={s.quantile(.50):>8.2f}  "
          f"p95={s.quantile(.95):>8.2f}  max={s.max():>8.2f}")


def block1_inference_latency(df: pd.DataFrame) -> None:
    print("\n[Block 1] Inferenzlatenz")
    sub = _block(df, "Block_1")
    if sub.empty: print("  Keine Daten."); return
    _describe(sub, "inference_ms")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Block 1 - KI-Inferenzlatenz pro Kamera", color="#cdd6f4", fontsize=14)
    for idx, (cam, grp) in enumerate(sub.groupby("camera")):
        ax.hist(grp["inference_ms"].dropna(), bins=50, alpha=0.75,
                color=COLORS[idx % len(COLORS)], label=str(cam), edgecolor="none")
    ax.set_xlabel("Inferenzzeit (ms)"); ax.set_ylabel("Haeufigkeit"); ax.legend()
    _save(fig, "block1_inference_latency")


def block2_network_latency(df: pd.DataFrame) -> None:
    print("\n[Block 2] Netzwerklatenz")
    sub = _block(df, "Block_2")
    if sub.empty: print("  Keine Daten."); return
    means = sub.groupby("camera")[["inference_ms","network_ms","server_ms"]].mean()
    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(max(6, len(means)*2), 6))
    ax.set_title("Block 2 - Latenz-Aufschluesselung (gestapelt)", color="#cdd6f4", fontsize=14)
    bottom = np.zeros(len(means))
    for col, label, color in [
        ("inference_ms","KI-Inferenz (t2-t1)","#00FFFF"),
        ("network_ms","Netzwerk   (t3-t2)","#FF00FF"),
        ("server_ms","Server     (t4-t3)","#FFFF00"),
    ]:
        vals = means[col].values
        ax.bar(x, vals, 0.5, bottom=bottom, label=label, color=color, alpha=0.85)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 1:
                ax.text(xi, b + v/2, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, color="#121212", fontweight="bold")
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(means.index, rotation=20, ha="right")
    ax.set_ylabel("Zeit (ms)"); ax.legend()
    _save(fig, "block2_network_latency_stacked")
    for col in ["inference_ms","network_ms","server_ms","e2e_ms"]:
        _describe(sub, col)


def block3_multi_camera(df: pd.DataFrame) -> None:
    print("\n[Block 3] Multi-Kamera")
    sub = _block(df, "Block_3")
    if sub.empty: print("  Keine Daten."); return
    sub["n_cameras"] = sub.groupby("frame_id")["camera"].transform("nunique")
    pivot = sub.groupby("n_cameras")[["server_triangulation_fps","e2e_ms"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Block 3 - Multi-Kamera-Skalierung", color="#cdd6f4", fontsize=14)
    axes[0].plot(pivot.index, pivot["server_triangulation_fps"], "o-", color="#00FF96", lw=2)
    axes[0].set_xlabel("Anzahl Kameras"); axes[0].set_ylabel("Server Tri-FPS")
    axes[1].plot(pivot.index, pivot["e2e_ms"], "o-", color="#FFAA00", lw=2)
    axes[1].set_xlabel("Anzahl Kameras"); axes[1].set_ylabel("E2E-Latenz (ms)")
    _save(fig, "block3_multi_camera")


def block5_localization(df: pd.DataFrame) -> None:
    print("\n[Block 5] Lokalisierungsfehler")
    sub = _block(df, "Block_5").dropna(subset=["pos_x","pos_z"])
    if sub.empty:
        print("  Keine pos_x/pos_z Werte.")
        if "loc_error_cm" in df.columns:
            _describe(_block(df,"Block_5"), "loc_error_cm")
        return
    if GROUND_TRUTH:
        gt = np.array(list(GROUND_TRUTH.values()))
        meas = sub[["pos_x","pos_z"]].values
        dists = np.min(np.linalg.norm(meas[:,None,:] - gt[None,:,:], axis=2), axis=1)
        rmse = float(np.sqrt(np.mean(dists**2)))
        mae  = float(np.mean(dists))
        print(f"  RMSE: {rmse:.2f} cm   MAE: {mae:.2f} cm   max: {dists.max():.2f} cm")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title(f"Block 5 - Lokalisierungsfehler  (RMSE={rmse:.1f} cm)",
                     color="#cdd6f4", fontsize=14)
        ax.hist(dists, bins=40, color="#00FFFF", alpha=0.8, edgecolor="none")
        ax.axvline(rmse, color="#FF5555", lw=2, linestyle="--", label=f"RMSE={rmse:.1f} cm")
        ax.axvline(mae,  color="#FFAA00", lw=1.5, linestyle=":", label=f"MAE={mae:.1f} cm")
        ax.set_xlabel("Fehler (cm)"); ax.set_ylabel("Haeufigkeit"); ax.legend()
        _save(fig, "block5_localization_rmse")
    else:
        _describe(sub, "loc_error_cm")


def block6_heatmap(df: pd.DataFrame) -> None:
    print("\n[Block 6] Heatmap")
    sub = _block(df, "Block_6").dropna(subset=["pos_x","pos_z"])
    if sub.empty: print("  Keine pos_x/pos_z Werte."); return
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Block 6 - Positionsheatmap (Bodenflaeche)", color="#cdd6f4", fontsize=14)
    if ROOM_IMAGE_PATH and os.path.isfile(ROOM_IMAGE_PATH):
        img = mpimg.imread(ROOM_IMAGE_PATH)
        ax.imshow(img, aspect="auto",
                  extent=[sub["pos_x"].min(), sub["pos_x"].max(),
                          sub["pos_z"].min(), sub["pos_z"].max()],
                  alpha=0.25, zorder=0)
    hb = ax.hexbin(sub["pos_x"], sub["pos_z"], gridsize=40, cmap="plasma", mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, label="Haeufigkeit")
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Z (cm)")
    for name, (gx, gz) in GROUND_TRUTH.items():
        ax.plot(gx, gz, "r+", markersize=14, markeredgewidth=2)
        ax.annotate(name, (gx, gz), color="#FF5555", fontsize=9,
                    xytext=(5, 5), textcoords="offset points")
    _save(fig, "block6_heatmap")



def block_modes_comparison(df: pd.DataFrame) -> None:
    print("\n[Algorithmen-Vergleich] Filter & Triangulation")
    mode_cols = ["smoothing_mode", "triangulation_mode", "tracking_mode", "ik_mode"]
    available = [c for c in mode_cols if c in df.columns]
    if not available:
        print("  Keine Modus-Spalten gefunden."); return

    for mode_col in available:
        print(f"  --- {mode_col} ---")
        for mode_val, grp in df.groupby(mode_col):
            n = len(grp)
            e2e    = grp["e2e_ms"].mean()            if "e2e_ms"            in grp.columns else float("nan")
            repro  = grp["repro_error_px"].mean()    if "repro_error_px"    in grp.columns else float("nan")
            health = grp["health_index"].mean()      if "health_index"      in grp.columns else float("nan")
            smooth = grp["kalman_smoothing_cm_delta"].mean() if "kalman_smoothing_cm_delta" in grp.columns else float("nan")
            print(f"    {mode_val:20s}  n={n:>6,}  e2e={e2e:>7.1f}ms  "
                  f"repro={repro:>6.3f}px  health={health:>5.1f}%  smooth={smooth:>7.3f}cm/f")

    if "ik_mode" in df.columns and "ik_correction_cm_delta" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("IK-Korrekturen (cm/Frame) nach IK-Modus", color="#cdd6f4", fontsize=13)
        for idx, (mode_val, grp) in enumerate(df.groupby("ik_mode")):
            ax.hist(grp["ik_correction_cm_delta"].dropna(), bins=50, alpha=0.7,
                    color=COLORS[idx % len(COLORS)], label=str(mode_val), edgecolor="none")
        ax.set_xlabel("IK Korrektur (cm)"); ax.set_ylabel("Haeufigkeit"); ax.legend()
        _save(fig, "modes_ik_correction_histogram")

    if "triangulation_mode" in df.columns and "repro_error_px" in df.columns:
        modes = df["triangulation_mode"].dropna().unique()
        if len(modes) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title("Reprojektionsfehler nach Triangulations-Modus", color="#cdd6f4", fontsize=13)
            for idx, m in enumerate(sorted(modes)):
                d = df[df["triangulation_mode"] == m]["repro_error_px"].dropna()
                ax.hist(d, bins=40, alpha=0.7, color=COLORS[idx % len(COLORS)], label=m, edgecolor="none")
            ax.set_xlabel("Reprojektionsfehler (px)"); ax.set_ylabel("Haeufigkeit"); ax.legend()
            _save(fig, "modes_reprojection_by_triangulation")



def block8_id_switches(df: pd.DataFrame) -> None:
    print("\n[Block 8] ID-Switches")
    sub = _block(df, "Block_8").sort_values("system_time")
    if sub.empty: print("  Keine Daten."); return
    # Bevorzuge vorberechnetes Delta-Feld; fallback auf diff()
    if "id_switches_delta" in sub.columns:
        sub["delta"] = sub["id_switches_delta"].fillna(0)
    elif "id_switches" in sub.columns:
        sub["delta"] = sub["id_switches"].diff().clip(lower=0).fillna(0)
    else:
        print("  Keine id_switches Spalte."); return
    total = int(sub["delta"].sum())
    dur = sub["system_time"].iloc[-1] - sub["system_time"].iloc[0] + 1e-9
    print(f"  Gesamt: {total}   Rate: {total/dur:.2f}/s")
    t_rel = sub["system_time"] - sub["system_time"].iloc[0]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_title(f"Block 8 - ID-Switches ueber Zeit  (Gesamt: {total})", color="#cdd6f4", fontsize=14)
    ax.fill_between(t_rel, sub["delta"], step="mid", color="#FF5555", alpha=0.75)
    ax.set_xlabel("Zeit (s)"); ax.set_ylabel("Switches pro Frame")
    _save(fig, "block8_id_switches")


def block9_stability(df: pd.DataFrame) -> None:
    print("\n[Block 9] Systemstabilitaet")
    sub = _block(df, "Block_9").sort_values("system_time")
    if sub.empty: print("  Keine Daten."); return
    _describe(sub, "health_index")
    _describe(sub, "server_triangulation_fps")
    t_rel = sub["system_time"] - sub["system_time"].iloc[0]
    win = 30
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Block 9 - Systemstabilitaet", color="#cdd6f4", fontsize=14)
    axes[0].plot(t_rel, sub["health_index"].rolling(win, min_periods=1).mean(),
                 color="#00FF96", lw=1.5, label="Health-Index (geglaettet)")
    axes[0].axhline(90, color="#FFAA00", linestyle="--", alpha=0.6, label="Ziel >= 90 %")
    axes[0].set_ylabel("Health-Index (%)"); axes[0].set_ylim(0, 105); axes[0].legend()
    axes[1].plot(t_rel, sub["server_triangulation_fps"].rolling(win, min_periods=1).mean(),
                 color="#00FFFF", lw=1.5)
    axes[1].set_ylabel("Server Tri-FPS"); axes[1].set_xlabel("Zeit (s)")
    _save(fig, "block9_stability")


def summary_table(df: pd.DataFrame) -> None:
    print("\n[Gesamt-Uebersicht]")
    cols = ["inference_ms","network_ms","server_ms","e2e_ms",
            "health_index","repro_error_px","server_triangulation_fps"]
    avail = [c for c in cols if c in df.columns]
    if not avail: return
    stats = df.groupby("test_block")[avail].agg(["mean","std","count"])
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 160)
    print(stats.to_string())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "summary_table.csv"
    stats.to_csv(out); print(f"  --> {out}")


ALL_BLOCKS: dict = {
    "Block_1": block1_inference_latency,
    "Block_2": block2_network_latency,
    "Block_3": block3_multi_camera,
    "Block_5": block5_localization,
    "Block_6": block6_heatmap,
    "Block_8": block8_id_switches,
    "modes":   block_modes_comparison,
    "Block_9": block9_stability,
}

