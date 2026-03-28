from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .constants import (
    DD,
    FS,
    MMHG_TO_PA,
    MIDDLE_SEGMENTS,
)
from .data import TREE, path_from_root
from .impedance import PADConfig, ResolvedSegment, solve_impedance_tree
from .input_flow import PreparedFlow, nominal_controls_from_relatives, prepare_flow


@dataclass
class TL55Result:
    """Outputs returned by the arterial tree solver."""
    pressure_outlet_mmHg: np.ndarray
    pressure_mid_mmHg: np.ndarray
    flow_outlet_mL_s: np.ndarray
    root_pressure_mmHg: np.ndarray
    root_flow_mL_s: np.ndarray
    time_s: np.ndarray
    time_all_s: np.ndarray
    freq_hz: np.ndarray
    z_input_root: np.ndarray
    transmission_outlet: np.ndarray
    transmission_mid: np.ndarray
    effective_segments: pd.DataFrame
    controls: dict


def _real_ifft_from_positive_spectrum(positive_spectrum: np.ndarray, d_f0: int) -> np.ndarray:
    """
    Reconstruct a real-valued waveform from its positive-frequency spectrum.

    The negative-frequency terms are added by Hermitian symmetry so that the
    inverse FFT yields a real signal in the time domain.
    """
    if positive_spectrum.ndim == 1:
        full_spectrum = np.concatenate(
            [positive_spectrum, np.array([0.0 + 0.0j]), np.conj(positive_spectrum[1:][::-1])]
        )
        return np.real(np.fft.ifft(full_spectrum * (2 * d_f0)))

    full_spectrum = np.concatenate(
        [
            positive_spectrum,
            np.zeros((positive_spectrum.shape[0], 1), dtype=complex),
            np.conj(positive_spectrum[:, 1:][:, ::-1]),
        ],
        axis=1,
    )
    return np.real(np.fft.ifft(full_spectrum * (2 * d_f0), axis=1))


def _build_transmission_matrices(tree_state) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cumulative pressure transmission ratios from the root to each segment.

    transmission_outlet gives inlet-to-outlet transmission for all 55 segments.
    transmission_mid gives inlet-to-midpoint transmission for the selected
    segments listed in MIDDLE_SEGMENTS.
    """
    n_freq = len(next(iter(tree_state.n_outlet.values())))
    transmission_outlet = np.ones((55, n_freq), dtype=complex)
    transmission_mid = np.ones((len(MIDDLE_SEGMENTS), n_freq), dtype=complex)

    # Multiply the segment-level transmission ratios along the path from the
    # root to each segment outlet.
    for segment_idx in range(1, 56):
        for path_idx in path_from_root(segment_idx):
            transmission_outlet[segment_idx - 1, :] *= tree_state.n_outlet[path_idx]

    # For midpoint pressure, use outlet transmission for upstream segments and
    # midpoint transmission only for the final segment in the path.
    for row_idx, segment_idx in enumerate(MIDDLE_SEGMENTS):
        segment_path = path_from_root(segment_idx)
        for path_pos, path_idx in enumerate(segment_path):
            if path_pos < len(segment_path) - 1:
                transmission_mid[row_idx, :] *= tree_state.n_outlet[path_idx]
            else:
                transmission_mid[row_idx, :] *= tree_state.n_mid[path_idx]

    return transmission_outlet, transmission_mid


def _trim_to_one_cycle(
    pressure_outlet_mmHg: np.ndarray,
    pressure_mid_mmHg: np.ndarray,
    flow_outlet_mL_s: np.ndarray,
    root_pressure_mmHg_all: np.ndarray,
    root_flow_mL_s_all: np.ndarray,
    hr_bpm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract one representative beat from the reconstructed periodic waveform.

    The trimming point is chosen near a local minimum in the root-to-distal
    pressure waveform so that the returned beat starts near end diastole.
    """
    samples_per_beat = int(round((60.0 / hr_bpm) / (1.0 / FS)))

    # Search in a later beat so low-HR cases still have enough signal length.
    startpoint_mat = 20 * samples_per_beat - 10
    start_idx = startpoint_mat - 1

    search_window = pressure_outlet_mmHg[0, start_idx : start_idx + samples_per_beat + 1]
    idx_min_mat = int(np.argmin(search_window)) + 1
    low1_mat = idx_min_mat + startpoint_mat

    shift = int(round(samples_per_beat / 10.0))
    trim_start_mat = low1_mat - shift
    trim_end_mat = low1_mat + samples_per_beat - shift
    trim_idx = np.arange(trim_start_mat - 1, trim_end_mat)

    time_s = trim_idx / FS
    return (
        pressure_outlet_mmHg[:, trim_idx],
        pressure_mid_mmHg[:, trim_idx],
        flow_outlet_mL_s[:, trim_idx],
        root_pressure_mmHg_all[trim_idx],
        root_flow_mL_s_all[trim_idx],
        time_s,
    )


def _effective_segments_dataframe(effective_segments: Dict[int, ResolvedSegment]) -> pd.DataFrame:
    """
    Convert the segment dictionary into a table for inspection and export.
    """
    rows = []
    for segment_idx in range(1, 56):
        seg = effective_segments[segment_idx]
        rows.append(
            {
                "idx": seg.idx,
                "name": seg.name,
                "length_m": seg.length_m,
                "radius_m": seg.radius_m,
                "thickness_m": seg.thickness_m,
                "young_modulus_pa": seg.young_modulus_pa,
                "windkessel_r1_pa_s_m3": seg.windkessel_r1_pa_s_m3,
                "windkessel_r2_pa_s_m3": seg.windkessel_r2_pa_s_m3,
                "windkessel_c_m3_pa": seg.windkessel_c_m3_pa,
                "angle_deg": seg.angle_deg,
                "children": TREE[segment_idx],
            }
        )
    return pd.DataFrame(rows)


def solve_model(
    q_input_path: str | Path,
    sv_rel: float = 1.0,
    hr_rel: float = 1.0,
    tpr_rel: float = 1.0,
    e_rel: float = 1.0,
    sl: float = 0.0,
    pad_nodes: tuple[int, ...] = (),
    l_pad: float = 1.0,
) -> TL55Result:
    """
    Solve the arterial tree model for one set of control parameters.

    Main steps:
    1. Convert relative controls to physical HR and SV
    2. Build the frequency grid
    3. Solve the tree input impedances and transmission ratios
    4. Prepare the aortic inflow waveform and compute its spectrum
    5. Reconstruct pressure and flow waveforms in all segments
    6. Trim the long periodic signal to one representative beat
    """
    stroke_volume_mL, heart_rate_bpm = nominal_controls_from_relatives(
        sv_rel=sv_rel,
        hr_rel=hr_rel,
    )

    # Set the frequency resolution. The signal length is increased at lower HR
    # so enough repeated beats are available before trimming.
    f0 = 32 * DD
    d_f0_base = 512 * DD
    base_signal_duration_s = d_f0_base / f0

    heart_period_s = 60.0 / heart_rate_bpm
    beats_in_base_signal = base_signal_duration_s / heart_period_s
    target_beats = 40
    scale_k = math.ceil(target_beats / beats_in_base_signal)

    d_f0 = int(d_f0_base * scale_k)
    signal_duration_factor = base_signal_duration_s * scale_k
    n_fft = int(signal_duration_factor * d_f0)

    freq_hz = np.arange(d_f0, dtype=float) * (f0 / d_f0)
    freq_hz[0] = np.finfo(float).eps * f0
    omega_rad_s = 2.0 * np.pi * freq_hz

    # Solve the frequency-domain arterial tree.
    pad_config = PADConfig(sl=sl, pad_nodes=tuple(pad_nodes), l_pad=l_pad)
    tree_state = solve_impedance_tree(
        omega_rad_s=omega_rad_s,
        e_rel=e_rel,
        tpr_rel=tpr_rel,
        pad=pad_config,
    )

    # Prepare the input aortic flow waveform at the requested HR and SV.
    prepared_flow: PreparedFlow = prepare_flow(
        q_input_path,
        hr_bpm=heart_rate_bpm,
        sv_mL=stroke_volume_mL,
        samples_per_second=d_f0,
        repeat=40,
        n_fft=n_fft,
    )

    # Root pressure is obtained from root flow times root input impedance.
    root_flow_spectrum = prepared_flow.positive_spectrum_m3_s
    root_pressure_spectrum = root_flow_spectrum * tree_state.z_input[1]

    # Propagate the root pressure spectrum to each segment outlet and midpoint.
    transmission_outlet, transmission_mid = _build_transmission_matrices(tree_state)
    outlet_pressure_spectrum = transmission_outlet * root_pressure_spectrum[np.newaxis, :]
    mid_pressure_spectrum = transmission_mid * root_pressure_spectrum[np.newaxis, :]
    outlet_flow_spectrum = outlet_pressure_spectrum / np.vstack(
        [tree_state.z_input[i] for i in range(1, 56)]
    )

    # Convert spectra back to time-domain waveforms.
    pressure_outlet_pa = _real_ifft_from_positive_spectrum(outlet_pressure_spectrum, d_f0)
    pressure_mid_pa = _real_ifft_from_positive_spectrum(mid_pressure_spectrum, d_f0)
    root_pressure_pa = _real_ifft_from_positive_spectrum(root_pressure_spectrum, d_f0)
    flow_outlet_m3_s = _real_ifft_from_positive_spectrum(outlet_flow_spectrum, d_f0)
    root_flow_m3_s = _real_ifft_from_positive_spectrum(root_flow_spectrum, d_f0)

    # Convert to more convenient physical units.
    pressure_outlet_mmHg = pressure_outlet_pa / MMHG_TO_PA
    pressure_mid_mmHg = pressure_mid_pa / MMHG_TO_PA
    root_pressure_mmHg = root_pressure_pa / MMHG_TO_PA
    flow_outlet_mL_s = flow_outlet_m3_s * 1e6
    root_flow_mL_s = root_flow_m3_s * 1e6

    time_all_s = np.arange(pressure_outlet_mmHg.shape[1]) / FS

    # Keep one representative beat for the returned outputs.
    (
        pressure_outlet_trim,
        pressure_mid_trim,
        flow_outlet_trim,
        root_pressure_trim,
        root_flow_trim,
        time_trim,
    ) = _trim_to_one_cycle(
        pressure_outlet_mmHg=pressure_outlet_mmHg,
        pressure_mid_mmHg=pressure_mid_mmHg,
        flow_outlet_mL_s=flow_outlet_mL_s,
        root_pressure_mmHg_all=root_pressure_mmHg,
        root_flow_mL_s_all=root_flow_mL_s,
        hr_bpm=heart_rate_bpm,
    )

    return TL55Result(
        pressure_outlet_mmHg=pressure_outlet_trim,
        pressure_mid_mmHg=pressure_mid_trim,
        flow_outlet_mL_s=flow_outlet_trim,
        root_pressure_mmHg=root_pressure_trim,
        root_flow_mL_s=root_flow_trim,
        time_s=time_trim,
        time_all_s=time_all_s,
        freq_hz=freq_hz,
        z_input_root=tree_state.z_input[1],
        transmission_outlet=transmission_outlet,
        transmission_mid=transmission_mid,
        effective_segments=_effective_segments_dataframe(tree_state.effective_segments),
        controls={
            "sv_rel": sv_rel,
            "hr_rel": hr_rel,
            "tpr_rel": tpr_rel,
            "e_rel": e_rel,
            "sv_mL": stroke_volume_mL,
            "hr_bpm": heart_rate_bpm,
            "sl": sl,
            "pad_nodes": tuple(pad_nodes),
            "l_pad": l_pad,
        },
    )