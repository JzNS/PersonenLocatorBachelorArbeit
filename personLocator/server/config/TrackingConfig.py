from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass(slots=True)
class TrackingConfig:
    """Zentrale Konfiguration für alle Schwellenwerte der 3D-Tracking-Pipeline."""

    max_epipolar_error_cm:    float = 35.0
    epipolar_ignore_threshold: float = 1000.0
    color_match_threshold:    float = 0.4
    min_joint_match_ratio:    float = 0.30

    max_3d_dist:  float = 150.0
    alpha_pos:    float = 0.70
    beta_col:     float = 0.30

    outlier_recovery_time: float = 0.2

    # IK Ketten: (parent, child, abs_min_cm, abs_max_cm, bone_name)
    ik_chain_strict: list[tuple[int, int, float, float, str]] = field(default_factory=lambda: [
        (5,  7,  25.0, 38.0, "arm_l_up"),   (7,  9,  20.0, 34.0, "arm_l_down"),
        (6,  8,  25.0, 38.0, "arm_r_up"),   (8,  10, 20.0, 34.0, "arm_r_down"),
        (11, 13, 38.0, 54.0, "leg_l_up"),   (13, 15, 35.0, 50.0, "leg_l_down"),
        (12, 14, 38.0, 54.0, "leg_r_up"),   (14, 16, 35.0, 50.0, "leg_r_down"),
    ])

    # Maximaler Abstand vom Massezentrum pro Gelenk (cm)
    max_com_reach: dict[int, float] = field(default_factory=lambda: {
        0: 80.0,  1: 80.0,  2: 80.0,  3: 80.0,  4: 80.0,
        5: 60.0,  6: 60.0,  7: 90.0,  8: 90.0,  9: 120.0, 10: 120.0,
        11: 60.0, 12: 60.0,
        13: 140.0, 14: 140.0,
        15: 190.0, 16: 190.0,
    })

    # Maximale Frame-zu-Frame-Distanz pro Gelenk (cm)
    max_frame_delta: dict[int, float] = field(default_factory=lambda: {
        0: 20.0,  1: 20.0,  2: 20.0,  3: 20.0,  4: 20.0,
        5: 20.0,  6: 20.0,  11: 20.0, 12: 20.0,
        7: 40.0,  8: 40.0,  13: 40.0, 14: 40.0,
        9: 80.0,  10: 80.0, 15: 80.0, 16: 80.0,
    })