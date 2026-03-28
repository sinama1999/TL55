from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .constants import BCG_LOWPASS_HZ, BLOOD_DENSITY, FS, LV_AXIS_ANGLE_DEG, ML_TO_M3
from .data import parent_map
from .left_ventricle import approximate_left_ventricle
from .solver import TL55Result, solve_model


@dataclass(frozen=True)
class BCGResult:
    """BCG waveform and its time axis for one trimmed beat."""
    bcg_force_N: np.ndarray
    time_s: np.ndarray


def _effective_segment_table(waveforms: TL55Result):
    return waveforms.effective_segments.set_index("idx")


def _segment_inlet_flows_mL_s(waveforms: TL55Result) -> np.ndarray:
    """Compute segment inlet flows from parent outlet flows and area ratios."""
    seg_table = _effective_segment_table(waveforms)
    parents = parent_map()
    flow_inlet_mL_s = np.zeros_like(waveforms.flow_outlet_mL_s)
    flow_inlet_mL_s[0, :] = waveforms.root_flow_mL_s

    for segment_idx in range(2, 56):
        parent_idx = parents[segment_idx]
        if parent_idx is None:
            raise ValueError(f"Segment {segment_idx} has no parent in the arterial tree.")
        radius_child_m = float(seg_table.loc[segment_idx, "radius_m"])
        radius_parent_m = float(seg_table.loc[parent_idx, "radius_m"])
        area_ratio = (radius_child_m / radius_parent_m) ** 2
        flow_inlet_mL_s[segment_idx - 1, :] = waveforms.flow_outlet_mL_s[parent_idx - 1, :] * area_ratio
    return flow_inlet_mL_s


def _arterial_momentum_derivative_N(waveforms: TL55Result) -> np.ndarray:
    """Compute d/dt of blood momentum summed over all arterial segments."""
    seg_table = _effective_segment_table(waveforms)
    flow_inlet_mL_s = _segment_inlet_flows_mL_s(waveforms)
    flow_mean_m3_s = 0.5 * (flow_inlet_mL_s + waveforms.flow_outlet_mL_s) * ML_TO_M3

    lengths_m = seg_table["length_m"].to_numpy(dtype=float)[:, None]
    axial_projection = np.cos(np.deg2rad(seg_table["angle_deg"].to_numpy(dtype=float)))[:, None]

    momentum_kg_m_s = BLOOD_DENSITY * lengths_m * flow_mean_m3_s * axial_projection
    return FS * np.diff(momentum_kg_m_s, axis=1).sum(axis=0)


def _left_ventricular_momentum_derivative_N(waveforms: TL55Result, mv_q_path: str | Path) -> np.ndarray:
    """Compute d/dt of the left-ventricular blood momentum term."""
    seg_table = _effective_segment_table(waveforms)
    lv = approximate_left_ventricle(
        root_pressure_mmHg=waveforms.root_pressure_mmHg,
        root_flow_mL_s=waveforms.root_flow_mL_s,
        sv_mL=float(waveforms.controls["sv_mL"]),
        hr_bpm=float(waveforms.controls["hr_bpm"]),
        mv_q_path=mv_q_path,
        fs_hz=FS,
    )

    root_radius_m = float(seg_table.loc[1, "radius_m"])
    aortic_area_m2 = np.pi * root_radius_m**2
    lv_axis_projection = np.cos(np.deg2rad(LV_AXIS_ANGLE_DEG))

    momentum_kg_m_s = (
        BLOOD_DENSITY
        * (lv.volume_mL - np.min(lv.volume_mL))
        * ML_TO_M3
        * lv_axis_projection
        * (lv.aortic_flow_mL_s * ML_TO_M3)
        / aortic_area_m2
    )
    return FS * np.diff(momentum_kg_m_s)


def _lowpass_bcg(signal_N: np.ndarray, cutoff_hz: float = BCG_LOWPASS_HZ, fs_hz: float = FS) -> np.ndarray:
    """Low-pass filter the BCG waveform for smooth display and analysis."""
    sos = butter(4, cutoff_hz, btype="low", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, signal_N)


def compute_bcg_from_waveforms(waveforms: TL55Result, mv_q_path: str | Path = "data/MV_Q_padded_2.mat") -> BCGResult:
    """Compute the BCG waveform from arterial and left-ventricular momentum terms."""
    arterial_force_N = _arterial_momentum_derivative_N(waveforms)
    lv_force_N = _left_ventricular_momentum_derivative_N(waveforms, mv_q_path=mv_q_path)

    total_body_force_N = -(arterial_force_N + lv_force_N)
    bcg_force_N = _lowpass_bcg(total_body_force_N[1:])
    time_s = waveforms.time_s[2:]
    return BCGResult(bcg_force_N=bcg_force_N, time_s=time_s)


def generate_bcg(
    sv_rel: float = 1.0,
    hr_rel: float = 1.0,
    tpr_rel: float = 1.0,
    e_rel: float = 1.0,
    sl: float = 0.0,
    pad_nodes: tuple[int, ...] = (),
    l_pad: float = 1.0,
    q_input_path: str | Path = "data/Q_inputwave2.mat",
    mv_q_path: str | Path = "data/MV_Q_padded_2.mat",
) -> BCGResult:
    """Generate the BCG waveform for one set of hemodynamic control parameters."""
    waveforms = solve_model(
        q_input_path=q_input_path,
        sv_rel=sv_rel,
        hr_rel=hr_rel,
        tpr_rel=tpr_rel,
        e_rel=e_rel,
        sl=sl,
        pad_nodes=tuple(pad_nodes),
        l_pad=l_pad,
    )
    return compute_bcg_from_waveforms(waveforms, mv_q_path=mv_q_path)
