
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import CM_TO_M, MPA_TO_PA


@dataclass(frozen=True)
class Segment:
    idx: int
    name: str
    length_m: float
    radius_m: float
    thickness_m: float
    young_modulus_pa: float
    windkessel_r1_pa_s_m3: Optional[float]
    windkessel_r2_pa_s_m3: Optional[float]
    windkessel_c_m3_pa: Optional[float]
    angle_deg: float


# Table S1 values from https://doi.org/10.1109/TBME.2025.3584979

# R1/R2 are stored in units of 1e9 Pa.s/m^3, C in units of 1e-10 m^3/Pa.
_RAW_SEGMENTS = [
    (1, "Ascending aorta", 8.32, 1.70, 0.163, 0.88, None, None, None, 53),
    (2, "Aortic arch I", 2.70, 1.59, 0.126, 0.88, None, None, None, -53),
    (3, "Brachiocephalic", 3.15, 0.73, 0.080, 0.88, None, None, None, 45),
    (4, "R. subclavian I", 3.15, 0.48, 0.067, 0.88, None, None, None, 45),
    (5, "R. carotid", 15.93, 0.44, 0.063, 0.88, None, None, None, 0),
    (6, "R. vertebral", 12.15, 0.23, 0.045, 1.76, 1.659, 7.581, 1.134, 0),
    (7, "R. subclavian II", 35.82, 0.37, 0.067, 0.88, None, None, None, 155),
    (8, "R. radius", 19.80, 0.18, 0.043, 1.76, 3.865, 4.988, 1.287, 180),
    (9, "R. ulnar I", 6.03, 0.25, 0.046, 1.76, None, None, None, 180),
    (10, "Aortic arch II", 3.60, 1.49, 0.115, 0.88, None, None, None, -71),
    (11, "L. carotid", 18.72, 0.44, 0.063, 0.88, None, None, None, 0),
    (12, "Thoracic aorta I", 4.95, 1.36, 0.110, 0.88, None, None, None, 180),
    (13, "Thoracic aorta II", 9.45, 1.17, 0.110, 0.88, None, None, None, 180),
    (14, "Intercostals", 6.57, 0.35, 0.049, 0.88, 0.544, 1.643, 4.878, 90),
    (15, "L. subclavian I", 3.15, 0.48, 0.066, 0.88, None, None, None, -45),
    (16, "L. vertebral", 12.15, 0.23, 0.045, 1.76, 1.659, 7.581, 1.134, 0),
    (17, "L. subclavian II", 35.82, 0.37, 0.067, 0.88, None, None, None, 205),
    (18, "L. ulnar I", 6.03, 0.25, 0.046, 1.76, None, None, None, 180),
    (19, "L. radius", 19.80, 0.18, 0.043, 1.76, 3.865, 4.988, 1.287, 180),
    (20, "Celiac I", 1.80, 0.37, 0.064, 0.88, None, None, None, 90),
    (21, "Celiac II", 1.80, 0.32, 0.064, 0.88, None, None, None, 90),
    (22, "Hepatic", 5.85, 0.30, 0.049, 0.88, 0.762, 4.755, 1.872, 90),
    (23, "Splenic", 5.22, 0.19, 0.054, 0.88, 2.336, 6.248, 1.251, 90),
    (24, "Gastric", 4.95, 0.23, 0.045, 0.88, 1.102, 2.608, 2.925, 90),
    (25, "Abdominal aorta I", 4.77, 1.01, 0.090, 0.88, None, None, None, 180),
    (26, "Sup. mesenteric", 4.50, 0.43, 0.069, 0.88, 0.326, 1.129, 7.29, 90),
    (27, "Abdominal aorta II", 1.35, 0.95, 0.080, 0.88, None, None, None, 180),
    (28, "R. renal", 2.70, 0.32, 0.053, 0.88, 0.549, 1.262, 6.003, 90),
    (29, "Abdominal aorta III", 1.35, 0.92, 0.080, 0.88, None, None, None, 180),
    (30, "L. renal", 2.70, 0.32, 0.053, 0.88, 0.549, 1.262, 6.003, 90),
    (31, "Abdominal aorta IV", 11.25, 0.82, 0.075, 0.88, None, None, None, 180),
    (32, "Inf. mesenteric", 3.42, 0.22, 0.043, 0.88, 1.461, 9.003, 0.99, 90),
    (33, "Abdominal aorta V", 7.20, 0.68, 0.065, 0.88, None, None, None, 180),
    (34, "R. com. iliac", 5.22, 0.44, 0.060, 0.88, None, None, None, 155),
    (35, "R. ext. iliac", 13.05, 0.39, 0.053, 1.76, None, None, None, 162),
    (36, "R. int. iliac", 4.05, 0.23, 0.040, 3.52, 1.741, 6.876, 1.224, 180),
    (37, "R. deep femoral", 10.17, 0.23, 0.047, 1.76, 1.357, 3.917, 2.034, 180),
    (38, "R. femoral", 39.87, 0.34, 0.050, 1.76, None, None, None, 180),
    (39, "R. ext. carotid", 15.93, 0.23, 0.042, 1.76, 1.423, 6.400, 1.332, 0),
    (40, "L. int. carotid", 15.84, 0.33, 0.045, 1.76, 0.688, 6.914, 1.332, 0),
    (41, "R. post. tibial", 30.96, 0.21, 0.045, 3.52, 2.693, 8.916, 0.918, 180),
    (42, "R. ant. tibial", 28.98, 0.29, 0.039, 3.52, 1.077, 4.110, 2.034, 180),
    (43, "R. interosseous", 6.30, 0.12, 0.028, 3.52, 10.725, 115.331, 0.081, 180),
    (44, "R. ulnar II", 15.30, 0.22, 0.046, 1.76, 2.753, 5.766, 1.287, 0),
    (45, "L. ulnar II", 15.30, 0.22, 0.046, 1.76, 2.753, 5.766, 1.287, 0),
    (46, "L. interosseous", 6.30, 0.12, 0.028, 3.52, 10.725, 115.331, 0.081, 180),
    (47, "R. int. carotid", 15.84, 0.33, 0.045, 1.76, 0.688, 6.914, 1.332, 0),
    (48, "L. ext. carotid", 15.93, 0.23, 0.042, 1.76, 1.423, 6.400, 1.332, 0),
    (49, "L. com. iliac", 5.22, 0.44, 0.060, 0.88, None, None, None, 205),
    (50, "L. ext. iliac", 13.05, 0.39, 0.053, 1.76, None, None, None, 198),
    (51, "L. int. iliac", 4.05, 0.23, 0.040, 3.52, 1.741, 6.876, 1.224, 180),
    (52, "L. deep femoral", 10.17, 0.23, 0.047, 1.76, 1.357, 3.917, 2.034, 180),
    (53, "L. femoral", 39.87, 0.34, 0.050, 1.76, None, None, None, 180),
    (54, "L. post. tibial", 28.98, 0.21, 0.045, 3.52, 2.693, 8.916, 0.918, 180),
    (55, "L. ant. tibial", 30.96, 0.29, 0.039, 3.52, 1.077, 4.110, 2.034, 180),
]


# The single linked list of the arterial tree from https://doi.org/10.1142/S0219519411004587
TREE: Dict[int, List[int]] = {
    1: [2, 3],
    2: [10, 11],
    3: [4, 5],
    4: [6, 7],
    5: [39, 47],
    6: [],
    7: [8, 9],
    8: [],
    9: [43, 44],
    10: [12, 15],
    11: [40, 48],
    12: [13, 14],
    13: [20, 25],
    14: [],
    15: [16, 17],
    16: [],
    17: [18, 19],
    18: [45, 46],
    19: [],
    20: [21, 22],
    21: [24, 23],
    22: [],
    23: [],
    24: [],
    25: [26, 27],
    26: [],
    27: [29, 30],
    28: [],
    29: [28, 31],
    30: [],
    31: [32, 33],
    32: [],
    33: [34, 49],
    34: [35, 36],
    35: [37, 38],
    36: [],
    37: [],
    38: [41, 42],
    39: [],
    40: [],
    41: [],
    42: [],
    43: [],
    44: [],
    45: [],
    46: [],
    47: [],
    48: [],
    49: [50, 51],
    50: [52, 53],
    51: [],
    52: [],
    53: [54, 55],
    54: [],
    55: [],
}


def load_segments() -> List[Segment]:
    out: List[Segment] = []
    for row in _RAW_SEGMENTS:
        idx, name, L_cm, r_cm, h_cm, E_MPa, r1_giga, r2_giga, c_1e10, angle_deg = row
        out.append(
            Segment(
                idx=idx,
                name=name,
                length_m=L_cm * CM_TO_M,
                radius_m=r_cm * CM_TO_M,
                thickness_m=h_cm * CM_TO_M,
                young_modulus_pa=E_MPa * MPA_TO_PA,
                windkessel_r1_pa_s_m3=None if r1_giga is None else r1_giga * 1e9,
                windkessel_r2_pa_s_m3=None if r2_giga is None else r2_giga * 1e9,
                windkessel_c_m3_pa=None if c_1e10 is None else c_1e10 * 1e-10,
                angle_deg=angle_deg,
            )
        )
    return out


SEGMENTS: List[Segment] = load_segments()
SEGMENT_MAP: Dict[int, Segment] = {seg.idx: seg for seg in SEGMENTS}


def topology_dataframe() -> pd.DataFrame:
    rows = []
    for seg in SEGMENTS:
        rows.append(
            {
                "idx": seg.idx,
                "name": seg.name,
                "children": TREE[seg.idx],
                "length_m": seg.length_m,
                "radius_m": seg.radius_m,
                "thickness_m": seg.thickness_m,
                "young_modulus_pa": seg.young_modulus_pa,
                "windkessel_r1_pa_s_m3": seg.windkessel_r1_pa_s_m3,
                "windkessel_r2_pa_s_m3": seg.windkessel_r2_pa_s_m3,
                "windkessel_c_m3_pa": seg.windkessel_c_m3_pa,
                "angle_deg": seg.angle_deg,
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def parent_map() -> Dict[int, Optional[int]]:
    parents: Dict[int, Optional[int]] = {1: None}
    for parent, children in TREE.items():
        for child in children:
            parents[child] = parent
    return parents


@lru_cache(maxsize=None)
def path_from_root(idx: int) -> List[int]:
    parents = parent_map()
    path = [idx]
    cur = idx
    while parents[cur] is not None:
        cur = parents[cur]
        path.append(cur)
    path.reverse()
    return path


def is_terminal(idx: int) -> bool:
    return len(TREE[idx]) == 0
