from __future__ import annotations

from pathlib import Path

from .bcg import BCGResult, generate_bcg
from .solver import TL55Result, solve_model


def generate_waveforms(
    sv_rel: float = 1.0,
    hr_rel: float = 1.0,
    tpr_rel: float = 1.0,
    e_rel: float = 1.0,
    sl: float = 0.0,
    pad_nodes: tuple[int, ...] = (),
    l_pad: float = 1.0,
    q_input_path: str | Path = "data/Q_inputwave2.mat",
) -> TL55Result:
    """Generate arterial pressure and flow waveforms for the 55-segment tree."""
    return solve_model(
        q_input_path=q_input_path,
        sv_rel=sv_rel,
        hr_rel=hr_rel,
        tpr_rel=tpr_rel,
        e_rel=e_rel,
        sl=sl,
        pad_nodes=tuple(pad_nodes),
        l_pad=l_pad,
    )


__all__ = ["generate_waveforms", "generate_bcg", "TL55Result", "BCGResult"]
