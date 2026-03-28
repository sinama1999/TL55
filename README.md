# TL55

Python package for generating arterial pressure and flow waveforms with a 55-segment systemic arterial transmission-line model from an aortic inflow waveform.

The package solves the arterial tree in the frequency domain, reconstructs time-domain waveforms, and returns pressure and flow at the outlet of all 55 arterial segments together with midpoint pressure for selected segments.

## Citation

If you use this code, please cite the following references:

- S. M. Shahrbabak, A. Mousavi, R. Mukkamala, and J.-O. Hahn, "In Silico Investigation of a Mathematical Model Relating the Ballistocardiogram to Aortic Blood Pressure," *IEEE Transactions on Biomedical Engineering*, vol. 73, no. 1, pp. 462-471, Jan. 2026, doi: 10.1109/TBME.2025.3584979.

- W. He, H. Xiao, and X. Liu, "Numerical Simulation of Human Systemic Arterial Hemodynamics Based on a Transmission Line Model and Recursive Algorithm," *Journal of Mechanics in Medicine and Biology*, vol. 12, no. 1, 2012.

Use of this code in published work should acknowledge these references.


## What this package does

Given:
- an aortic inflow waveform
- a heart rate
- a stroke volume
- relative arterial stiffness
- relative peripheral resistance
- optional peripheral artery disease settings

the package computes:
- pressure waveforms at the outlet of all 55 arterial segments
- flow waveforms at the outlet of all 55 arterial segments
- pressure waveforms at the midpoint of selected segments
- pressure and flow at the root of the arterial tree
- the effective segment properties used in the solve

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

Typical dependencies are:
- numpy
- scipy
- pandas

## Package layout

- `tl55/api.py`  
  User-facing entry point

- `tl55/data.py`  
  Nominal arterial geometry, topology, and terminal loads

- `tl55/constants.py`  
  Physical constants and nominal operating-point values

- `tl55/input_flow.py`  
  Aortic inflow loading, heart-rate warping, stroke-volume scaling, repetition, and FFT preparation

- `tl55/impedance.py`  
  Segment impedance formulas, recursive tree solve, and optional stenosis handling

- `tl55/solver.py`  
  Frequency-domain solve, waveform reconstruction, and beat trimming

## Basic use

```python
from tl55 import generate_waveforms

result = generate_waveforms(
    sv_rel=1.0,
    hr_rel=1.0,
    tpr_rel=1.0,
    e_rel=1.0,
    q_input_path="data/Q_inputwave2.mat",
)
```

This returns a `TL55Result` object containing the simulated waveforms and metadata.

## Inputs

### `q_input_path`
Path to a `.mat` file containing the nominal aortic inflow waveform.

The file must contain a variable named `f` with two columns:
- column 1: time in seconds
- column 2: flow in mL/s

### `sv_rel`
Relative stroke volume scaling.  
Nominal stroke volume is 60 mL.

Examples:
- `sv_rel=1.0` gives 60 mL
- `sv_rel=0.5` gives 30 mL
- `sv_rel=1.5` gives 90 mL

### `hr_rel`
Relative heart rate scaling.  
Nominal heart rate is 75 bpm.

Examples:
- `hr_rel=1.0` gives 75 bpm
- `hr_rel=0.8` gives 60 bpm
- `hr_rel=1.2` gives 90 bpm

### `tpr_rel`
Relative scaling applied to terminal resistances of the arterial tree.  
This changes the downstream vascular load.

### `e_rel`
Relative scaling applied to Young's modulus of the arterial segments.  
This changes arterial stiffness globally.

### `sl`
Stenosis severity fraction.  
If a segment is diseased, its lesion radius is set to:

```python
r_lesion = (1 - sl) * r_nominal
```

Examples:
- `sl=0.0` means no stenosis
- `sl=0.5` means 50% radius reduction in the diseased region

### `pad_nodes`
Tuple of segment indices where stenosis should be applied.

Example:
```python
pad_nodes=(33,)
```

### `l_pad`
Fraction of the segment length occupied by the stenosis.

Examples:
- `l_pad=1.0` means the whole segment is diseased
- `l_pad=0.5` means half the segment length is diseased

## Example 1: nominal simulation

```python
from tl55 import generate_waveforms

result = generate_waveforms(
    sv_rel=1.0,
    hr_rel=1.0,
    tpr_rel=1.0,
    e_rel=1.0,
    q_input_path="data/Q_inputwave2.mat",
)

print(result.pressure_outlet_mmHg.shape)
print(result.flow_outlet_mL_s.shape)
print(result.time_s[0], result.time_s[-1])
```

Typical interpretation:
- `pressure_outlet_mmHg[i, :]` is the outlet pressure waveform of segment `i+1`
- `flow_outlet_mL_s[i, :]` is the outlet flow waveform of segment `i+1`

## Example 2: altered physiology

```python
from tl55 import generate_waveforms

result = generate_waveforms(
    sv_rel=0.8,
    hr_rel=1.2,
    tpr_rel=1.1,
    e_rel=0.9,
    q_input_path="data/Q_inputwave2.mat",
)
```

This example uses:
- lower stroke volume
- higher heart rate
- higher peripheral resistance
- lower arterial stiffness

## Example 3: add stenosis

```python
from tl55 import generate_waveforms

result = generate_waveforms(
    sv_rel=1.0,
    hr_rel=1.0,
    tpr_rel=1.0,
    e_rel=1.0,
    sl=0.5,
    pad_nodes=(33,),
    l_pad=1.0,
    q_input_path="data/Q_inputwave2.mat",
)
```

This applies a 50% radius reduction to segment 33 over the full segment length.

## Returned fields

### `pressure_outlet_mmHg`
Pressure waveforms at the outlet of all 55 arterial segments, in mmHg.  
Shape: `(55, n_samples)`

### `pressure_mid_mmHg`
Pressure waveforms at the midpoint of selected segments, in mmHg.  
Shape: `(n_mid_segments, n_samples)`

### `flow_outlet_mL_s`
Flow waveforms at the outlet of all 55 arterial segments, in mL/s.  
Shape: `(55, n_samples)`

### `root_pressure_mmHg`
Pressure waveform at the root of the arterial tree, in mmHg.  
Shape: `(n_samples,)`

### `root_flow_mL_s`
Flow waveform at the root of the arterial tree, in mL/s.  
Shape: `(n_samples,)`

### `time_s`
Time vector associated with the returned trimmed beat.

This axis preserves the sample times of the beat as extracted from the longer reconstructed periodic signal, so it does not usually start at 0 s. It is mainly useful for traceability and internal consistency with the trimming step. For display, it is often clearer to shift it to a beat-local axis with:

```python
t_plot = result.time_s - result.time_s[0]
```

### `time_all_s`
Time vector for the full reconstructed periodic waveform before the final single-beat trimming step.

### `freq_hz`
Frequency grid used in the frequency-domain solve.

### `z_input_root`
Input impedance at the root of the arterial tree as a function of frequency.

### `transmission_outlet`
Pressure transmission ratio from the root to the outlet of each segment.

### `transmission_mid`
Pressure transmission ratio from the root to the midpoint of each selected segment.

### `effective_segments`
`pandas.DataFrame` listing the segment properties actually used in the solve.

This includes:
- segment index
- segment name
- length
- radius
- wall thickness
- Young's modulus
- terminal Windkessel parameters where applicable
- angle
- child segments

### `controls`
Dictionary summarizing the simulation settings used for that run.

## Plotting an example waveform

```python
import matplotlib.pyplot as plt
from tl55 import generate_waveforms

result = generate_waveforms(
    sv_rel=1.0,
    hr_rel=1.0,
    tpr_rel=1.0,
    e_rel=1.0,
    q_input_path="data/Q_inputwave2.mat",
)

segment_idx = 0  # segment 1
t_plot = result.time_s - result.time_s[0]

plt.figure()
plt.plot(t_plot, result.pressure_outlet_mmHg[segment_idx, :])
plt.xlabel("Time within beat [s]")
plt.ylabel("Pressure [mmHg]")
plt.title("Outlet pressure, segment 1")
plt.show()
```

## Inspecting the segment table

```python
print(result.effective_segments.head())
```

This is useful for confirming:
- which segment properties were used
- which segments are terminal
- where stenosis was applied
- the effective geometry after stenosis

## Modeling notes

- The aortic inflow waveform is first warped to the requested heart rate.
- Stroke volume is imposed by scaling the area under the inflow waveform.
- Terminal branches use 3-element Windkessel loads.
- The arterial tree is solved recursively in the frequency domain from the terminal branches back to the root.
- Pressure and flow waveforms are reconstructed in the time domain by inverse FFT.
- A single representative beat is returned for convenience, while the full reconstructed signal is also available through `time_all_s`.

## Typical workflow

1. Choose a nominal aortic inflow waveform file
2. Set `sv_rel`, `hr_rel`, `tpr_rel`, and `e_rel`
3. Optionally add stenosis with `sl`, `pad_nodes`, and `l_pad`
4. Run `generate_waveforms(...)`
5. Extract and plot the segment waveforms of interest
6. Inspect `effective_segments` to confirm the simulated arterial properties

## Minimal complete example

```python
from tl55 import generate_waveforms
import matplotlib.pyplot as plt

result = generate_waveforms(
    sv_rel=0.9,
    hr_rel=1.1,
    tpr_rel=1.0,
    e_rel=1.0,
    q_input_path="data/Q_inputwave2.mat",
)

plt.figure()
plt.plot(result.time_s, result.root_pressure_mmHg)
plt.xlabel("Time [s]")
plt.ylabel("Root pressure [mmHg]")
plt.title("Root pressure waveform")
plt.show()
```
