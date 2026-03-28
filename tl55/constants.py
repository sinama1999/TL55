from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Physical constants used by the arterial tree and BCG modules
BLOOD_DENSITY = 1.05e3          # kg / m^3
BLOOD_VISCOSITY = 0.0035        # Pa.s
POISSON_RATIO = 0.5
VISCOELASTIC_K = 1.0            # dimensionless wall viscoelastic parameter
PHI_SCALE = 1.0                 # nominal viscoelastic phase scaling
AORTIC_VALVE_RESISTANCE = 0.008 # mmHg.s.mL^-1
MMHG_TO_PA = 133.29
ML_TO_M3 = 1e-6
CM_TO_M = 1e-2
MPA_TO_PA = 1e6

# Fixed solver sampling settings
DD = 4
F_MAX = 32 * DD                # 128 Hz
FS = 2 * F_MAX                 # 256 Hz

# Nominal operating point
NOMINAL_HR_BPM = 75.0
NOMINAL_SV_ML = 60.0

# Segment indices for midpoint pressure output
MIDDLE_SEGMENTS = [1, 2, 10, 12, 13, 25, 27, 31, 33, 34, 35, 38]

# Left-ventricle / BCG constants
LV_END_SYSTOLIC_ELASTANCE = 2.0   # mmHg / mL
LV_ZERO_VOLUME_ML = -11.0         # mL
LV_REST_PRESSURE_MMHG = 8.0       # mmHg
LV_AXIS_ANGLE_DEG = 45.0          # degrees from head-to-foot axis
BCG_LOWPASS_HZ = 25.0             # Hz

