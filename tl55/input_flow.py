from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat

from .constants import ML_TO_M3, NOMINAL_HR_BPM, NOMINAL_SV_ML


@dataclass(frozen=True)
class PreparedFlow:
    """
    Container for the processed aortic inflow waveform.

    Attributes
    ----------
    raw_time_s
        Original time vector loaded from file.
    raw_flow_mL_s
        Original aortic flow waveform loaded from file, in mL/s.
    warped_time_s
        Time vector after adjusting the input waveform to the requested heart rate
        and resampling it to a uniform grid.
    tiled_flow_mL_s
        Resampled single-beat inflow waveform repeated multiple times so that a
        longer periodic signal is available for frequency-domain reconstruction.
    positive_spectrum_m3_s
        Positive-frequency FFT coefficients of the repeated inflow waveform,
        expressed in m^3/s.
    """
    raw_time_s: np.ndarray
    raw_flow_mL_s: np.ndarray
    warped_time_s: np.ndarray
    tiled_flow_mL_s: np.ndarray
    positive_spectrum_m3_s: np.ndarray


def load_aortic_flow(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the nominal aortic inflow waveform from a .mat file.

    The file is expected to contain a two-column array named 'f':
    column 1 is time in seconds and column 2 is flow in mL/s.

    Returns
    -------
    t
        Time vector in seconds.
    q
        Aortic flow waveform in mL/s.
    """
    mat = loadmat(path)
    if "f" not in mat:
        raise KeyError(f"Expected variable 'f' in {path}, but found keys {list(mat.keys())}")
    arr = np.asarray(mat["f"], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected 'f' with shape (N, 2), got {arr.shape}")
    t = arr[:, 0]
    q = arr[:, 1]
    return t, q


def infer_lvet(hr_bpm: float) -> float:
    """
    Estimate left ventricular ejection time from heart rate.

    This gives a simple heart-rate-dependent systolic duration used when
    stretching the nominal inflow waveform to a new heart period.

    Parameters
    ----------
    hr_bpm
        Heart rate in beats per minute.

    Returns
    -------
    lvet_s
        Estimated left ventricular ejection time in seconds.
    """
    return -0.0017 * hr_bpm + 0.413


def stretch_flow_to_hr(
    time_s: np.ndarray,
    flow_mL_s: np.ndarray,
    hr_bpm: float,
    lvet_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp the nominal inflow waveform to match a requested heart rate.

    The waveform is divided into two parts:
    1. systole, which is scaled to match the target ejection time
    2. diastole, which is scaled to fill the remainder of the heart period

    Parameters
    ----------
    time_s
        Original time vector for one beat.
    flow_mL_s
        Original aortic flow waveform for one beat.
    hr_bpm
        Requested heart rate in beats per minute.
    lvet_s
        Optional left ventricular ejection time. If None, it is estimated
        from heart rate.

    Returns
    -------
    warped_time_s
        Time vector after heart-rate-dependent warping.
    flow_mL_s
        Flow waveform values, unchanged in amplitude at this stage.
    """
    if lvet_s is None:
        lvet_s = infer_lvet(hr_bpm)

    time_s = np.asarray(time_s, dtype=float).copy()
    flow_mL_s = np.asarray(flow_mL_s, dtype=float).copy()

    # The nominal input waveform contains a systolic portion followed by
    # a diastolic portion. The split location is fixed by the input waveform
    # definition.
    t0 = time_s[54]
    systole = slice(0, 55)
    diastole = slice(55, None)

    # Compress or stretch systole to the requested ejection time.
    time_s[systole] = lvet_s / t0 * time_s[systole]

    # Compress or stretch diastole so that the full beat matches the
    # requested heart period.
    period_s = 60.0 / hr_bpm
    time_s[diastole] = (period_s - lvet_s) / (time_s[-1] - t0) * (time_s[diastole] - t0) + lvet_s
    return time_s, flow_mL_s


def resample_scale_and_tile_flow(
    time_s: np.ndarray,
    flow_mL_s: np.ndarray,
    target_sv_mL: float,
    samples_per_second: int,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample one warped inflow beat, scale it to the target stroke volume,
    and repeat it to create a longer periodic signal.

    Steps
    -----
    1. Interpolate the warped beat onto a uniform time grid.
    2. Scale the waveform so that its time integral equals the requested
       stroke volume.
    3. Tile the beat multiple times to create a longer signal for FFT-based
       frequency-domain analysis.

    Parameters
    ----------
    time_s
        Beat time vector after HR warping.
    flow_mL_s
        Beat flow waveform after HR warping.
    target_sv_mL
        Desired stroke volume in mL.
    samples_per_second
        Uniform resampling density in samples per second.
    repeat
        Number of times to repeat the beat.

    Returns
    -------
    xi
        Uniform time vector for one beat.
    yi
        Repeated flow waveform in mL/s.
    """
    # Build a uniformly sampled version of the warped beat.
    xi = np.arange(time_s.min(), time_s.max() + 1.0 / samples_per_second, 1.0 / samples_per_second)
    spline = CubicSpline(time_s, flow_mL_s, bc_type="not-a-knot")
    yi = spline(xi)

    # Scale the beat so that the integral of flow over one beat equals the
    # requested stroke volume.
    current_sv = np.trapz(yi, xi)
    if current_sv == 0:
        raise ZeroDivisionError("Input flow has zero area, cannot scale to target SV.")
    yi = target_sv_mL / current_sv * yi

    # Repeat the beat to form a longer periodic signal before taking the FFT.
    yi = np.tile(yi, repeat)
    return xi, yi


def prepare_flow(
    path: str | Path,
    hr_bpm: float,
    sv_mL: float,
    samples_per_second: int,
    repeat: int,
    n_fft: int,
    lvet_s: float | None = None,
) -> PreparedFlow:
    """
    Prepare the aortic inflow waveform for the arterial tree solver.

    This function performs the full preprocessing pipeline used before the
    impedance-based arterial simulation:
    1. load the nominal aortic inflow waveform
    2. warp it to the requested heart rate
    3. scale it to the requested stroke volume
    4. repeat it to create a long periodic signal
    5. compute its positive-frequency Fourier coefficients

    Parameters
    ----------
    path
        Path to the .mat file containing the nominal inflow waveform.
    hr_bpm
        Requested heart rate in beats per minute.
    sv_mL
        Requested stroke volume in mL.
    samples_per_second
        Sampling density used to resample the beat.
    repeat
        Number of repeated beats in the constructed long waveform.
    n_fft
        FFT length used to compute the frequency-domain representation.
    lvet_s
        Optional left ventricular ejection time in seconds.

    Returns
    -------
    PreparedFlow
        Object containing the original waveform, processed waveform, and
        positive-frequency spectrum.
    """
    raw_t, raw_q = load_aortic_flow(path)
    warped_t, warped_q = stretch_flow_to_hr(raw_t, raw_q, hr_bpm=hr_bpm, lvet_s=lvet_s)
    resampled_t, tiled_q = resample_scale_and_tile_flow(
        warped_t,
        warped_q,
        target_sv_mL=sv_mL,
        samples_per_second=samples_per_second,
        repeat=repeat,
    )

    # Convert the repeated inflow from mL/s to m^3/s before computing the
    # spectrum used by the transmission-line solver.
    q_m3_s = tiled_q * ML_TO_M3
    q_fft = np.fft.fft(q_m3_s, n=n_fft) / n_fft

    # Keep only the positive-frequency terms because the solver constructs the
    # full real-valued time-domain waveform by Hermitian symmetry later.
    q_fft_pos = q_fft[:samples_per_second]

    return PreparedFlow(
        raw_time_s=raw_t,
        raw_flow_mL_s=raw_q,
        warped_time_s=resampled_t,
        tiled_flow_mL_s=tiled_q,
        positive_spectrum_m3_s=q_fft_pos,
    )


def nominal_controls_from_relatives(sv_rel: float, hr_rel: float) -> tuple[float, float]:
    """
    Convert relative stroke volume and heart rate inputs into physical values.

    Parameters
    ----------
    sv_rel
        Stroke volume scaling relative to the nominal value.
    hr_rel
        Heart rate scaling relative to the nominal value.

    Returns
    -------
    sv_mL
        Stroke volume in mL.
    hr_bpm
        Heart rate in beats per minute.
    """
    return NOMINAL_SV_ML * sv_rel, NOMINAL_HR_BPM * hr_rel