"""
Impedance solver for the 55-segment arterial tree.

This module converts nominal arterial segment properties into frequency-domain
hemodynamic quantities used by the waveform solver. Its main job is to compute,
for each arterial segment and each frequency:

- characteristic impedance z0
- propagation constant gamma
- input impedance seen at the segment inlet
- pressure transmission ratio from the segment inlet to its outlet
- pressure transmission ratio from the segment inlet to its midpoint

The arterial tree is solved recursively from the terminal branches back to the
root. Terminal branches use 3-element Windkessel loads. Nonterminal branches
use the parallel combination of their daughter input impedances.

This module also supports piecewise modeling of peripheral artery disease (PAD)
by replacing part or all of a segment with a narrowed lesion region and then
recomputing its impedance and transmission properties.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np
from scipy.special import jv

from .constants import BLOOD_DENSITY, BLOOD_VISCOSITY, PHI_SCALE, POISSON_RATIO, VISCOELASTIC_K
from .data import SEGMENT_MAP, TREE, is_terminal


@dataclass(frozen=True)
class PADConfig:
    """
    Configuration for localized peripheral artery disease.

    Parameters
    ----------
    sl
        Fractional lumen narrowing. For example, 0.5 means a 50% radius reduction
        in the diseased portion of the segment.
    pad_nodes
        Segment indices where PAD should be applied.
    l_pad
        Fraction of the segment length occupied by the lesion.
        l_pad = 1.0 means the whole segment is diseased.
    """
    sl: float = 0.0
    pad_nodes: tuple[int, ...] = ()
    l_pad: float = 1.0

    def applies_to(self, idx: int) -> bool:
        """Return True if PAD is active and should be applied to this segment."""
        return self.sl > 0.0 and idx in set(self.pad_nodes)


@dataclass(frozen=True)
class ResolvedSegment:
    """
    Physical parameters actually used for one arterial segment in the solve.

    This dataclass stores the geometry and material properties after applying
    user-level scaling such as arterial stiffness scaling (e_rel) and total
    peripheral resistance scaling (tpr_rel).
    """
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


@dataclass
class ImpedanceState:
    """
    Container for all frequency-domain quantities produced by the recursive solve.

    Each dictionary is keyed by segment index. Arrays are frequency indexed.
    """
    z0: Dict[int, np.ndarray]
    gamma: Dict[int, np.ndarray]
    z_input: Dict[int, np.ndarray]
    n_outlet: Dict[int, np.ndarray]
    n_mid: Dict[int, np.ndarray]
    effective_segments: Dict[int, ResolvedSegment]


def resolve_segments(e_rel: float = 1.0, tpr_rel: float = 1.0) -> Dict[int, ResolvedSegment]:
    """
    Create the effective segment set used in the simulation.

    Young's modulus is scaled by e_rel to represent global arterial stiffness
    changes. Terminal Windkessel resistances are scaled by tpr_rel to represent
    global peripheral resistance changes. Segment geometry is otherwise left at
    its nominal values unless PAD is later applied.
    """
    resolved: Dict[int, ResolvedSegment] = {}
    for idx, seg in SEGMENT_MAP.items():
        resolved[idx] = ResolvedSegment(
            idx=seg.idx,
            name=seg.name,
            length_m=seg.length_m,
            radius_m=seg.radius_m,
            thickness_m=seg.thickness_m,
            young_modulus_pa=seg.young_modulus_pa * e_rel,
            windkessel_r1_pa_s_m3=None if seg.windkessel_r1_pa_s_m3 is None else seg.windkessel_r1_pa_s_m3 * tpr_rel,
            windkessel_r2_pa_s_m3=None if seg.windkessel_r2_pa_s_m3 is None else seg.windkessel_r2_pa_s_m3 * tpr_rel,
            windkessel_c_m3_pa=seg.windkessel_c_m3_pa,
            angle_deg=seg.angle_deg,
        )
    return resolved


def characteristic_impedance(
    length_m: float,
    radius_m: float,
    thickness_m: float,
    young_modulus_pa: float,
    omega_rad_s: np.ndarray,
    phi_scale: float = PHI_SCALE,
    k_viscoelastic: float = VISCOELASTIC_K,
    blood_density: float = BLOOD_DENSITY,
    blood_viscosity: float = BLOOD_VISCOSITY,
    poisson_ratio: float = POISSON_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eq. (7) and Eq. (8) from https://doi.org/10.1109/TBME.2025.3584979.
    """
    c0 = np.sqrt(young_modulus_pa * thickness_m / (blood_density * 2.0 * radius_m))
    alpha = radius_m * np.sqrt(omega_rad_s * blood_density / blood_viscosity)
    phi = phi_scale * (10.0 * thickness_m / (2.0 * radius_m)) * (np.pi / 180.0) * (1.0 - np.exp(-k_viscoelastic * omega_rad_s))

    j15 = (1j) ** 1.5
    z = alpha * j15
    F10 = (2.0 * jv(1, z)) / (z * jv(0, z))
    F10 = np.asarray(F10, dtype=complex).copy()
    F10[0] = F10[0] + np.finfo(float).eps

    z0 = (
        (blood_density * c0 / np.sqrt(1.0 - poisson_ratio**2))
        * np.power(1.0 - F10, -0.5)
        * (np.cos(phi / 2.0) + 1j * np.sin(phi / 2.0))
        / (np.pi * radius_m**2)
    )
    gamma = (
        1j
        * omega_rad_s
        * np.sqrt(1.0 - poisson_ratio**2)
        / c0
        * np.power(1.0 - F10, -0.5)
        * (np.cos(phi / 2.0) - 1j * np.sin(phi / 2.0))
    )
    return z0, gamma


def terminal_load(seg: ResolvedSegment, omega_rad_s: np.ndarray) -> np.ndarray:
    """
    Compute the terminal 3-element Windkessel load for a terminal branch.

    The load is modeled as a proximal resistance in series with a parallel
    combination of distal resistance and compliance.
    """
    if seg.windkessel_r1_pa_s_m3 is None or seg.windkessel_r2_pa_s_m3 is None or seg.windkessel_c_m3_pa is None:
        raise ValueError(f"Segment {seg.idx} is missing terminal Windkessel parameters.")
    return seg.windkessel_r1_pa_s_m3 + 1.0 / (1.0 / seg.windkessel_r2_pa_s_m3 + 1j * omega_rad_s * seg.windkessel_c_m3_pa)


def reflection_coefficient(z_load: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """
    Compute the pressure-wave reflection coefficient at a segment outlet.

    Reflection is zero when the downstream load matches the segment's
    characteristic impedance and grows as mismatch increases.
    """
    return (z_load - z0) / (z_load + z0)


def propagate_uniform_piece(
    z_load: np.ndarray,
    length_m: float,
    radius_m: float,
    thickness_m: float,
    young_modulus_pa: float,
    omega_rad_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate a downstream load through one uniform vessel piece.

    Given the impedance seen at the outlet of this piece, compute:
    - input impedance at the inlet
    - pressure transmission ratio from inlet to outlet
    - characteristic impedance
    - propagation constant
    """
    z0, gamma = characteristic_impedance(
        length_m=length_m,
        radius_m=radius_m,
        thickness_m=thickness_m,
        young_modulus_pa=young_modulus_pa,
        omega_rad_s=omega_rad_s,
    )
    gamma_ref = reflection_coefficient(z_load, z0)
    z_in = z0 * (1.0 + gamma_ref * np.exp(-2.0 * length_m * gamma)) / (1.0 - gamma_ref * np.exp(-2.0 * length_m * gamma))
    n_out = (1.0 + gamma_ref) / (np.exp(gamma * length_m) + gamma_ref * np.exp(-gamma * length_m))
    return z_in, n_out, z0, gamma


def _split_piece(piece, cut_len):
    """
    Split one vessel piece into proximal and distal parts at a specified length.

    This is used to build partial-lesion models and to evaluate midpoint
    transmission in piecewise segments.
    """
    kind, L, r, h = piece
    if cut_len <= 0:
        return None, piece
    if cut_len >= L:
        return piece, None
    left = (kind, cut_len, r, h)
    right = (kind, L - cut_len, r, h)
    return left, right


def _build_pad_pieces(seg: ResolvedSegment, sl: float, l_pad: float):
    """
    Construct a piecewise representation of a diseased segment.

    If l_pad == 1, the whole segment is treated as stenotic.
    Otherwise, the segment is represented as healthy-lesion-healthy.
    The lesion radius is reduced according to sl, and wall thickness is adjusted
    to preserve the chosen lesion geometry model.
    """
    r0 = seg.radius_m
    h0 = seg.thickness_m
    l = seg.length_m
    r_pad = (1.0 - sl) * r0
    h_pad = h0 + (r0 - r_pad)

    if l_pad >= 1.0:
        pieces = [("lesion", l, r_pad, h_pad)]
        eff_r = r_pad
        eff_h = h_pad
    else:
        l_h = (1.0 - l_pad) / 2.0 * l
        l_d = l * l_pad
        pieces = [
            ("healthy", l_h, r0, h0),
            ("lesion", l_d, r_pad, h_pad),
            ("healthy", l_h, r0, h0),
        ]
        eff_r = r0 * (1.0 - l_pad) + r_pad * l_pad
        eff_h = h0 * (1.0 - l_pad) + h_pad * l_pad
    return pieces, eff_r, eff_h


def _backprop_ratio(pieces_inlet_to_outlet, downstream_load, omega_rad_s, young_modulus_pa):
    """
    Propagate a downstream load backward through a sequence of vessel pieces.

    This helper is used for piecewise PAD segments. It returns the impedance seen
    at the inlet of the first piece and the cumulative pressure transmission
    ratio across all pieces.
    """
    z_running = downstream_load.copy()
    n_total = np.ones_like(z_running, dtype=complex)
    for _, L, r, h in reversed(pieces_inlet_to_outlet):
        z_running, n_piece, _, _ = propagate_uniform_piece(
            z_load=z_running,
            length_m=L,
            radius_m=r,
            thickness_m=h,
            young_modulus_pa=young_modulus_pa,
            omega_rad_s=omega_rad_s,
        )
        n_total *= n_piece
    return z_running, n_total


def _split_pieces_at_midpoint(pieces, midpoint_distance):
    """
    Split a piecewise segment into proximal and distal groups at the physical midpoint.

    This is used to compute the pressure transmission ratio from the segment inlet
    to the midpoint, which is needed for selected arterial locations.
    """
    prox = []
    dist = 0.0
    rem = midpoint_distance
    for piece in pieces:
        kind, L, r, h = piece
        if rem <= 0:
            break
        if rem >= L:
            prox.append(piece)
            rem -= L
            dist += L
        else:
            left, _ = _split_piece(piece, rem)
            if left is not None:
                prox.append(left)
            rem = 0.0
            dist += left[1] if left is not None else 0.0
            break

    # distal pieces start at midpoint
    dist = 0.0
    distal = []
    remaining_from_start = midpoint_distance
    for piece in pieces:
        kind, L, r, h = piece
        if remaining_from_start >= L:
            remaining_from_start -= L
            continue
        if remaining_from_start > 0:
            _, right = _split_piece(piece, remaining_from_start)
            if right is not None:
                distal.append(right)
            remaining_from_start = 0.0
        else:
            distal.append(piece)
    return prox, distal


def apply_pad_piecewise(
    seg: ResolvedSegment,
    downstream_load: np.ndarray,
    omega_rad_s: np.ndarray,
    sl: float,
    l_pad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ResolvedSegment]:
    """
    Solve one segment using a piecewise PAD model.

    The segment is divided into healthy and stenotic sub-pieces, then the
    downstream load is propagated backward through those pieces. This yields:

    - segment input impedance
    - inlet-to-outlet pressure transmission
    - inlet-to-midpoint pressure transmission
    - an updated effective segment geometry for reporting
    """
    z0_nominal, gamma_nominal = characteristic_impedance(
        length_m=seg.length_m,
        radius_m=seg.radius_m,
        thickness_m=seg.thickness_m,
        young_modulus_pa=seg.young_modulus_pa,
        omega_rad_s=omega_rad_s,
    )

    pieces, eff_r, eff_h = _build_pad_pieces(seg, sl=sl, l_pad=l_pad)
    z_in, n_out = _backprop_ratio(
        pieces_inlet_to_outlet=pieces,
        downstream_load=downstream_load,
        omega_rad_s=omega_rad_s,
        young_modulus_pa=seg.young_modulus_pa,
    )

    midpoint_distance = seg.length_m / 2.0
    prox_pieces, distal_pieces = _split_pieces_at_midpoint(pieces, midpoint_distance)
    z_mid, _ = _backprop_ratio(
        pieces_inlet_to_outlet=distal_pieces,
        downstream_load=downstream_load,
        omega_rad_s=omega_rad_s,
        young_modulus_pa=seg.young_modulus_pa,
    )
    _, n_mid = _backprop_ratio(
        pieces_inlet_to_outlet=prox_pieces,
        downstream_load=z_mid,
        omega_rad_s=omega_rad_s,
        young_modulus_pa=seg.young_modulus_pa,
    )

    eff_seg = replace(seg, radius_m=eff_r, thickness_m=eff_h)
    return z_in, n_out, n_mid, z0_nominal, gamma_nominal, eff_seg


def solve_subtree(
    idx: int,
    omega_rad_s: np.ndarray,
    segments: Dict[int, ResolvedSegment],
    pad: PADConfig,
    state: ImpedanceState,
) -> np.ndarray:
    """
    Recursively solve one subtree of the arterial network.

    For a terminal segment, the downstream load is its Windkessel load.
    For a branching segment, the downstream load is the parallel combination
    of daughter input impedances. Once the downstream load is known, this
    function computes the segment input impedance and pressure transmission
    ratios, then stores them in the running state object.
    """
    seg = segments[idx]

    z0_nominal, gamma_nominal = characteristic_impedance(
        length_m=seg.length_m,
        radius_m=seg.radius_m,
        thickness_m=seg.thickness_m,
        young_modulus_pa=seg.young_modulus_pa,
        omega_rad_s=omega_rad_s,
    )
    state.z0[idx] = z0_nominal
    state.gamma[idx] = gamma_nominal
    state.effective_segments[idx] = seg

    children = TREE[idx]
    if is_terminal(idx):
        z_load = terminal_load(seg, omega_rad_s)
    else:
        child_impedances = [solve_subtree(ch, omega_rad_s, segments, pad, state) for ch in children]
        z_load = 1.0 / np.sum([1.0 / z for z in child_impedances], axis=0)

    if pad.applies_to(idx):
        z_in, n_out, n_mid, _, _, eff_seg = apply_pad_piecewise(
            seg=seg,
            downstream_load=z_load,
            omega_rad_s=omega_rad_s,
            sl=pad.sl,
            l_pad=pad.l_pad,
        )
        state.z_input[idx] = z_in
        state.n_outlet[idx] = n_out
        state.n_mid[idx] = n_mid
        state.effective_segments[idx] = eff_seg
        return z_in

    gamma_ref = reflection_coefficient(z_load, z0_nominal)
    z_in = z0_nominal * (1.0 + gamma_ref * np.exp(-2.0 * seg.length_m * gamma_nominal)) / (1.0 - gamma_ref * np.exp(-2.0 * seg.length_m * gamma_nominal))
    n_out = (1.0 + gamma_ref) / (np.exp(gamma_nominal * seg.length_m) + gamma_ref * np.exp(-gamma_nominal * seg.length_m))
    n_mid = (1.0 + gamma_ref) / (np.exp(gamma_nominal * seg.length_m / 2.0) + gamma_ref * np.exp(-gamma_nominal * seg.length_m / 2.0))

    state.z_input[idx] = z_in
    state.n_outlet[idx] = n_out
    state.n_mid[idx] = n_mid
    return z_in


def solve_impedance_tree(
    omega_rad_s: np.ndarray,
    e_rel: float = 1.0,
    tpr_rel: float = 1.0,
    pad: PADConfig | None = None,
) -> ImpedanceState:
    """
    Solve the full arterial tree in the frequency domain.

    This is the main entry point for the impedance module. It resolves the
    effective segment parameters, initializes the storage object, and starts
    the recursive solve at the root segment.
    """
    if pad is None:
        pad = PADConfig()

    segments = resolve_segments(e_rel=e_rel, tpr_rel=tpr_rel)
    state = ImpedanceState(z0={}, gamma={}, z_input={}, n_outlet={}, n_mid={}, effective_segments={})
    solve_subtree(idx=1, omega_rad_s=omega_rad_s, segments=segments, pad=pad, state=state)
    return state
