from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat

from .constants import (
    AORTIC_VALVE_RESISTANCE,
    FS,
    LV_END_SYSTOLIC_ELASTANCE,
    LV_REST_PRESSURE_MMHG,
    LV_ZERO_VOLUME_ML,
)
from .input_flow import infer_lvet


@dataclass(frozen=True)
class LVResult:
    """Approximated left-ventricular pressure, flow, and volume over one beat."""
    volume_mL: np.ndarray
    pressure_mmHg: np.ndarray
    aortic_flow_mL_s: np.ndarray
    mitral_flow_mL_s: np.ndarray


def load_mitral_flow(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the nominal mitral inflow waveform from a MATLAB file."""
    mat = loadmat(path)
    if "MV_Q" not in mat:
        raise KeyError(f"Expected variable 'MV_Q' in {path}, but found keys {list(mat.keys())}")
    arr = np.asarray(mat["MV_Q"], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected 'MV_Q' with shape (N, 2), got {arr.shape}")
    return arr[:, 0], arr[:, 1]


def _stretch_mitral_fill_time(time_s: np.ndarray, flow_mL_s: np.ndarray, fill_time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Stretch the nominal mitral inflow waveform to the requested filling duration."""
    shifted_time = (time_s - time_s[0]) * fill_time_s / (time_s[-1] - time_s[0])
    return shifted_time, flow_mL_s


def _resample_and_scale_flow(time_s: np.ndarray, flow_mL_s: np.ndarray, target_sv_mL: float, fs_hz: float) -> np.ndarray:
    """Resample a single flow waveform to the solver sampling rate and scale its area."""
    xi = np.arange(time_s.min(), time_s.max() + 1.0 / fs_hz, 1.0 / fs_hz)
    yi = CubicSpline(time_s, flow_mL_s, bc_type="not-a-knot")(xi)
    yi[-1] = 0.0

    current_sv = np.trapz(yi, xi)
    if current_sv == 0:
        raise ZeroDivisionError("Mitral inflow has zero area, cannot scale to target stroke volume.")
    return target_sv_mL / current_sv * yi


def _rotate_to_diastole(reference_pressure_mmHg: np.ndarray, hr_bpm: float, signal: np.ndarray, direction: int, fs_hz: float) -> np.ndarray:
    """Rotate a beat so that diastole begins at the start of the array, or reverse that shift."""
    lvet_s = infer_lvet(hr_bpm)
    min_pressure_idx = int(np.argmin(reference_pressure_mmHg))
    start_idx = min_pressure_idx + int(round(lvet_s * fs_hz)) - 1
    start_idx = max(0, min(start_idx, signal.size - 1))

    if direction == 1:
        return np.concatenate([signal[start_idx:], signal[:start_idx]])
    if direction == -1:
        if start_idx == 0:
            return signal.copy()
        return np.concatenate([signal[-start_idx:], signal[:-start_idx]])
    raise ValueError("direction must be 1 or -1")


def _left_ventricular_volume(
    aortic_pressure_mmHg: np.ndarray,
    aortic_flow_mL_s: np.ndarray,
    mitral_flow_mL_s: np.ndarray,
    end_systolic_volume_mL: float,
    hr_bpm: float,
    fs_hz: float,
) -> np.ndarray:
    """Compute left-ventricular volume from mitral inflow and aortic outflow."""
    aortic_flow_diastole_first = _rotate_to_diastole(aortic_pressure_mmHg, hr_bpm, aortic_flow_mL_s, direction=1, fs_hz=fs_hz)
    mitral_flow_padded = np.zeros_like(aortic_flow_diastole_first)
    mitral_flow_padded[: min(len(mitral_flow_mL_s), len(mitral_flow_padded))] = mitral_flow_mL_s[: len(mitral_flow_padded)]

    outflow_integral = np.cumsum(aortic_flow_diastole_first / fs_hz)
    inflow_integral = np.cumsum(mitral_flow_padded / fs_hz)
    volume_diastole_first = end_systolic_volume_mL + inflow_integral - outflow_integral
    return _rotate_to_diastole(aortic_pressure_mmHg, hr_bpm, volume_diastole_first, direction=-1, fs_hz=fs_hz)


def approximate_left_ventricle(
    root_pressure_mmHg: np.ndarray,
    root_flow_mL_s: np.ndarray,
    sv_mL: float,
    hr_bpm: float,
    mv_q_path: str | Path = "data/MV_Q_padded_2.mat",
    fs_hz: float = FS,
) -> LVResult:
    """Approximate left-ventricular pressure and volume over one trimmed beat."""
    lvet_s = infer_lvet(hr_bpm)
    isovolumic_contraction_s = (49.0 - 0.15 * hr_bpm) / 1000.0
    isovolumic_relaxation_s = (113.0 - 0.27 * hr_bpm) / 1000.0
    systolic_interval_s = lvet_s + isovolumic_contraction_s
    cardiac_period_s = 60.0 / hr_bpm

    time_s = np.arange(root_pressure_mmHg.size) / fs_hz
    min_pressure_idx = int(np.argmin(root_pressure_mmHg))

    mitral_time_s, mitral_flow_nominal_mL_s = load_mitral_flow(mv_q_path)
    fill_time_s = cardiac_period_s - systolic_interval_s - isovolumic_relaxation_s
    mitral_time_s, mitral_flow_nominal_mL_s = _stretch_mitral_fill_time(
        mitral_time_s,
        mitral_flow_nominal_mL_s,
        fill_time_s=fill_time_s,
    )
    mitral_flow_mL_s = _resample_and_scale_flow(
        mitral_time_s,
        mitral_flow_nominal_mL_s,
        target_sv_mL=sv_mL,
        fs_hz=fs_hz,
    )

    lv_pressure_mmHg = root_pressure_mmHg + AORTIC_VALVE_RESISTANCE * root_flow_mL_s
    lv_pressure_mmHg[:min_pressure_idx] = LV_REST_PRESSURE_MMHG

    n_lvet = int(round(lvet_s * fs_hz))
    n_isct = int(round(isovolumic_contraction_s * fs_hz))
    n_ivrt = int(round(isovolumic_relaxation_s * fs_hz))
    end_systolic_idx = min(min_pressure_idx + n_lvet - 1, lv_pressure_mmHg.size - 1)
    lv_pressure_mmHg[end_systolic_idx + 1 :] = LV_REST_PRESSURE_MMHG

    for i in range(1, n_isct + 1):
        idx = min_pressure_idx - i
        if idx >= 0:
            lv_pressure_mmHg[idx] = lv_pressure_mmHg[min_pressure_idx] - i / n_isct * (lv_pressure_mmHg[min_pressure_idx] - LV_REST_PRESSURE_MMHG)

    for i in range(1, n_ivrt + 1):
        idx = end_systolic_idx + i
        if idx < lv_pressure_mmHg.size:
            lv_pressure_mmHg[idx] = lv_pressure_mmHg[end_systolic_idx] - i / n_ivrt * (lv_pressure_mmHg[end_systolic_idx] - LV_REST_PRESSURE_MMHG)

    end_systolic_volume_mL = lv_pressure_mmHg[end_systolic_idx] / LV_END_SYSTOLIC_ELASTANCE + LV_ZERO_VOLUME_ML
    lv_volume_mL = _left_ventricular_volume(
        aortic_pressure_mmHg=root_pressure_mmHg,
        aortic_flow_mL_s=root_flow_mL_s,
        mitral_flow_mL_s=mitral_flow_mL_s,
        end_systolic_volume_mL=end_systolic_volume_mL,
        hr_bpm=hr_bpm,
        fs_hz=fs_hz,
    )

    mitral_start_idx = min(min_pressure_idx + n_lvet + n_ivrt - 1, time_s.size - 1)
    mitral_start_time_s = time_s[mitral_start_idx]
    mitral_time_full_s = np.arange(mitral_flow_mL_s.size) / fs_hz + mitral_start_time_s
    wrapped_flow = np.zeros_like(root_flow_mL_s)
    wrapped_indices = np.mod(np.round(mitral_time_full_s * fs_hz).astype(int), root_flow_mL_s.size)
    wrapped_flow[wrapped_indices] = mitral_flow_mL_s[: wrapped_indices.size]

    return LVResult(
        volume_mL=lv_volume_mL,
        pressure_mmHg=lv_pressure_mmHg,
        aortic_flow_mL_s=root_flow_mL_s,
        mitral_flow_mL_s=wrapped_flow,
    )
