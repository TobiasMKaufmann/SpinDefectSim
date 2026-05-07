# SpinDefectSim — Spin-Defect Sensing Simulation Library

A Python library for simulating optically detected magnetic resonance (ODMR) experiments
with spin-defect centres, herein referred to as **SpinDefectSim**. Includes a general
spin Hamiltonian for arbitrary defect species, inhomogeneous ensemble modelling, AC
sensing protocols, screened Coulomb electrostatics, magnetometry and electrometry from
2-D field distributions.

> **Physical models:** all formulae, assumptions, and literature references are documented
> in [`docs/theory.md`](docs/theory.md).

---

## Installation

```bash
pip install numpy scipy matplotlib
# clone or copy SpinDefectSim/ into your project, then:
import sys; sys.path.insert(0, "/path/to/parent")
import SpinDefectSim
```

---

## Supported defect types

The library ships with the following built-in presets, selectable by name string.
All can be overridden per-parameter.

| Name       | Host     | Spin  | D₀ (GHz) | d⊥ (Hz V⁻¹m) | Notes                                 |
|------------|----------|-------|-----------|----------------|---------------------------------------|
| `vb_minus` | hBN      | 1     | 3.46      | 0.35           | **Default.** VB⁻, Gottscholl 2020    |
| `nv_minus` | diamond  | 1     | 2.87      | 0.17           | NV⁻ centre, Dolde 2011               |
| `v_sic`    | 4H-SiC   | 1     | 1.28      | 0.10           | V2 silicon vacancy                    |
| `p1`       | diamond  | 1/2   | 0         | 0              | P1 N centre, Zeeman-only              |
| `cr_gaN`   | GaN      | 3/2   | 1.80      | 0              | Cr⁴⁺ impurity, approximate           |

Custom defect types can be created directly:

```python
from SpinDefectSim.spin.defects import DefectType
my_defect = DefectType(name="my_centre", spin=1, D0_Hz=1.5e9, d_perp=0.2)
```

---

## Spin Hamiltonian

For a spin-S defect with quantization axis z′, the Hamiltonian (H/h, in Hz) is:

```
H = D₀·(Sz′² − S(S+1)/3·I)                       ← ZFS axial    (S ≥ 1)
  + E₀·(Sx′² − Sy′²)                              ← ZFS strain   (S ≥ 1)
  + d∥·Ez′·(Sz′² − S(S+1)/3·I)                    ← E axial      (S ≥ 1)
  + d⊥·[Ey′·(Sx′²−Sy′²) + Ex′·{Sx′,Sy′}]         ← E transverse (S ≥ 1)
  + γₑ·(Bx′·Sx′ + By′·Sy′ + Bz′·Sz′)             ← electron Zeeman (all S)
  + Σₖ S·Aₖ·Iₖ                                    ← hyperfine    (nuclear spins)
  + Σₖ γₙ,ₖ·(Bx′·Ix,ₖ + By′·Iy,ₖ + Bz′·Iz,ₖ)    ← nuclear Zeeman
  + Σₖ Pₖ·(Iz,ₖ² − Iₖ(Iₖ+1)/3·I)                ← quadrupole   (Iₖ ≥ 1)
```

The electron-only Hamiltonian (top five terms) is built by `odmr_hamiltonian_Hz`.
The full electron + nuclear Hamiltonian (all terms) is built by
`full_hyperfine_hamiltonian_Hz`, which assembles the tensor-product Hilbert space
H_e ⊗ H_n1 ⊗ H_n2 ⊗ … automatically.

Applied B and E vectors are supplied in the **lab frame** and automatically
rotated into the defect's local frame via the quantization axis before H is built.

ODMR observables come from transitions out of the |ms=0⟩-like eigenstate (the
one with highest overlap with the Sz eigenstate at index `ms0_index`).

---

## Nuclear spins and hyperfine coupling

`NuclearSpin` encapsulates a single nuclear spin with its hyperfine tensor,
gyromagnetic ratio, and (optionally) quadrupole coupling.  All built-in defect
presets already carry their natural nuclear environments.

```python
from SpinDefectSim.spin.nuclear import (
    NuclearSpin, axial_A_tensor, isotropic_A_tensor,
    GAMMA_14N, GAMMA_15N, GAMMA_13C, GAMMA_11B,
)
from SpinDefectSim.spin.hamiltonian import (
    SpinParams, full_hyperfine_hamiltonian_Hz, odmr_transitions_Hz,
)

# ── NV⁻ on-site ¹⁴N (I = 1) ──────────────────────────────────────────────
N14_NV = NuclearSpin(
    spin=1,
    A_tensor_Hz=axial_A_tensor(A_zz_Hz=-2.14e6, A_perp_Hz=-2.70e6),
    gamma_Hz_T=GAMMA_14N,
    label="14N",
    quadrupole_Hz=-4.95e6,
)

# ── VB⁻ three equivalent ¹⁴N neighbours (in-plane, axial A ≈ 0) ──────────
N14_VB = NuclearSpin(
    spin=1,
    A_tensor_Hz=axial_A_tensor(A_zz_Hz=0, A_perp_Hz=47.8e6),
    gamma_Hz_T=GAMMA_14N,
    label="14N_VB",
)

# ── Build the full electron + nuclear Hamiltonian ─────────────────────────
sp = SpinParams(spin=1, D0_Hz=2.87e9)    # NV⁻ ground state
H  = full_hyperfine_hamiltonian_Hz(sp, E_vec_lab=[0, 0, 0], nuclear_spins=[N14_NV])
print(H.shape)   # → (9, 9)  [3 electron × 3 nuclear]

# Extract ODMR transitions (hyperfine-split lines)
freqs = odmr_transitions_Hz(H, electron_dim=3, ms0_basis_index=1)
print([f"{f/1e6:.2f} MHz" for f in freqs])

# ── ¹³C nearest-neighbour in diamond ─────────────────────────────────────
C13 = NuclearSpin(
    spin=0.5,
    A_tensor_Hz=isotropic_A_tensor(13.0e6),   # Fermi contact ~ 13 MHz
    gamma_Hz_T=GAMMA_13C,
    label="13C",
)

# ── Use the preset NV⁻ defect (includes ¹⁴N automatically) ───────────────
from SpinDefectSim.spin.hamiltonian import SpinDefect
sd_nv = SpinDefect("nv_minus", B_mT=3.0)
H_full = sd_nv.full_hamiltonian()     # includes on-site ¹⁴N
freqs_hf = sd_nv.hyperfine_transitions()   # hyperfine-split ODMR lines
```

Isotope gyromagnetic ratios shipped with the library:

| Constant        | Isotope | I    | γₙ (MHz/T) |
|-----------------|---------|------|------------|
| `GAMMA_14N`     | ¹⁴N     | 1    | +3.077     |
| `GAMMA_15N`     | ¹⁵N     | 1/2  | −4.316     |
| `GAMMA_11B`     | ¹¹B     | 3/2  | +13.660    |
| `GAMMA_10B`     | ¹⁰B     | 3    | +4.575     |
| `GAMMA_13C`     | ¹³C     | 1/2  | +10.708    |
| `GAMMA_29Si`    | ²⁹Si    | 1/2  | −5.319     |

---

## CW ODMR contrast — rate model

`RateModel` solves the optical / ISC rate equations in steady state to predict
the CW ODMR contrast $C = (\text{PL}_\text{off} - \text{PL}_\text{on}) / \text{PL}_\text{off}$
for any spin-$S$ defect.

```python
from SpinDefectSim.spin.rates import (
    RateModel, RateParams,
    NV_RATES, VB_RATES, VSIC_RATES, P1_RATES,
)

# ── Inspect pre-built contrasts ──────────────────────────────────────────
for name, rp, ms0 in [("NV⁻",  NV_RATES,   1),
                      ("VB⁻",  VB_RATES,   1),
                      ("V_SiC",VSIC_RATES, 1),
                      ("P1",   P1_RATES,   0)]:
    model = RateModel(rp, ms0_index=ms0)
    print(f"{name:6s}  C = {model.contrast()*100:.1f} %")
# NV⁻    C = 22.9 %
# VB⁻    C =  2.0 %
# V_SiC  C = 15.1 %
# P1     C =  0.0 %   ← no ISC path

# ── Custom defect ────────────────────────────────────────────────────────
my_rates = RateParams(
    spin=1,
    k_optical=20e6,
    k_rad=100e6,
    k_isc_excited=[50e6, 5e6, 50e6],   # ms = +1, 0, -1
    k_from_shelving=[0.0, 10e6, 0.0],  # spin-polarising return to ms = 0
)
model = RateModel(my_rates, ms0_index=1)
print(f"contrast = {model.contrast()*100:.1f} %")

# ── Photoluminescence vs. MW power ──────────────────────────────────────
import numpy as np
k_mw_range = np.logspace(3, 9, 200)    # MW Rabi rate (Hz)
pl_vals = [model.pl(k_mw) for k_mw in k_mw_range]   # normalised PL
```

The rate model is integrated into `Defaults.get_contrast()`: if `contrast=None`
(the default), it auto-computes $C$ from the defect's `rate_params`.

---

## Quick start — single defect

```python
import sys; sys.path.insert(0, "/app")
import numpy as np
from SpinDefectSim.spin.hamiltonian import SpinDefect

# VB⁻ in hBN (default) — transition frequencies
sd = SpinDefect(B_mT=1.5)
f1, f2 = sd.transition_frequencies()           # (Hz)

# NV⁻ in diamond, B along z-axis
sd_nv = SpinDefect("nv_minus", B_vec_mT=[0, 0, 3.0])

# --- Run a sensing protocol on a single defect ---

# E-field sensing: signal has E turned on, reference has E = 0
exp = sd_nv.to_experiment(E_vec_Vpm=[1e7, 0, 0], sensing="E")
tau, *_, dS_peak = exp.echo_static()

# B-field sensing: signal has extra stray B, reference has none
exp = sd_nv.to_experiment(B_extra_T=[0, 0, 5e-5], sensing="B")
f_axis = np.linspace(2.8e9, 2.95e9, 2000)
pl_w, pl_no, dpl = exp.cw_odmr(f_axis)

# Both E and B contribute to the contrast simultaneously
exp = sd_nv.to_experiment(
    E_vec_Vpm=[1e7, 0, 0],
    B_extra_T=[0, 0, 5e-5],
    sensing="both",
)

# Spin-1/2 P1 centre
sd_p1 = SpinDefect("p1", B_mT=10.0)
(f_single,) = sd_p1.transition_frequencies()   # one transition

# Fully custom spin-3/2
sd_custom = SpinDefect(spin=1.5, D0_Hz=2.0e9, B_mT=2.0)

# Specific quantization axis (e.g. NV axis tilted 45° in xz-plane)
sd_tilt = SpinDefect("nv_minus", B_mT=3.0,
                     quantization_axis=[np.sin(np.pi/4), 0, np.cos(np.pi/4)])
```

---

## Defect-type defaults for an experiment

```python
from SpinDefectSim.base.params import Defaults

# Use Defaults.for_defect() to load Hamiltonian params from a preset
d = Defaults.for_defect("nv_minus", B_mT=5.0, T2echo=10e-6)
sp = d.to_spin_params()      # SpinParams with NV⁻ D₀, d⊥, spin=1, ms0_index=1

# Or keep VB⁻ defaults and just change a field
d = Defaults(B_mT=3.0, T2echo=20e-6)
```
---

## CW ODMR spectrum

```python
import numpy as np
from SpinDefectSim.spin.spectra import ensemble_transitions_from_Efields, ensemble_odmr_spectrum

d  = Defaults()
sp = d.to_spin_params()
f  = np.linspace(3.35e9, 3.60e9, 2000)

# Inhomogeneous ensemble — random E-field disorder
E_fields = np.random.default_rng(0).normal(0, 5e4, (500, 3))
E_fields[:, 2] = 0                           # in-plane disorder
transitions = ensemble_transitions_from_Efields(E_fields, sp)

fwhm = 1.0 / (np.pi * d.T2star)
pl   = ensemble_odmr_spectrum(f, transitions, fwhm, d.contrast)
```

---

## Hahn-echo and Ramsey lock-in

```python
from SpinDefectSim.spin.echo import lock_in_difference_echo, lock_in_difference_ramsey

tr_signal = ensemble_transitions_from_Efields(E_fields, sp)
tr_ref    = ensemble_transitions_from_Efields(np.zeros_like(E_fields), sp)

# Hahn-echo
tau = np.linspace(0, 3 * d.T2echo, 600)
S_w, S_n, dS = lock_in_difference_echo(tr_signal, tr_ref, tau, d.T2echo)

# Ramsey FID
tau_r = np.linspace(0, 3 * d.T2star, 400)
S_w_r, S_n_r, dS_r, _ = lock_in_difference_ramsey(tr_signal, tr_ref, tau_r, d.T2star)
```

---

## Ensemble sensing workflow

The `sensing` parameter of `to_experiment()` selects which fields contribute
to the contrast between signal and reference branches:

| `sensing` | Signal branch uses | Reference branch uses |
|-----------|--------------------|-----------------------|
| `"E"`     | E_fields + bias B  | bias B only, E = 0 |
| `"B"`     | stray B, E = 0     | bias B only, B_extra = 0 |
| `"both"`  | E_fields + stray B | bias B only, both zero |

The bias B (from `Defaults.B_mT`) is always active in both branches.

```python
from SpinDefectSim.analysis.ensemble import DefectEnsemble
from scipy.constants import e as e_charge

d   = Defaults(T2echo=10e-6, B_mT=1.5)
ens = DefectEnsemble(N_def=300, defaults=d)
ens.generate_defects(seed=42)

# ── E-field sensing ───────────────────────────────────────────────
rng  = np.random.default_rng(7)
xyzq = np.column_stack([
    rng.uniform(-d.R_patch, d.R_patch, (20, 2)),
    np.zeros(20),
    rng.choice([-1, 1], 20) * e_charge,
])
ens.compute_efields(E0_gate=(0, 0, 1e4), disorder_xyzq=xyzq)
exp_E = ens.to_experiment(sensing="E")
tau, Sw, Sn, dS, tau_opt, dS_peak = exp_E.echo_static()
print(f"E-sensing: Peak ΔS = {dS_peak:.4f} at τ = {tau_opt*1e6:.2f} µs")

# ── B-field sensing (stray field from a magnetic sample) ──────────
from SpinDefectSim.magnetometry import SquareGeometry
geom = SquareGeometry(side=500e-9, n_boundary_pts=200)
ens.compute_bfields(magnetization=lambda x, y: 1e-4, geometry=geom)
exp_B = ens.to_experiment(sensing="B")
print("B-sensing dS_peak:", exp_B.echo_static()[-1])

# ── Both fields contribute simultaneously ─────────────────────────
exp_both = ens.to_experiment(sensing="both")
print("Combined dS_peak:", exp_both.echo_static()[-1])
```

---

## Random quantization axes (powder average)

When defects have random orientations (e.g. in powdered samples or disordered
films), generate uniformly distributed quantization axes:

```python
ens = DefectEnsemble(N_def=500, defaults=d)
ens.generate_defects(seed=0, quantization_axis="random")     # isotropic powder
ens.compute_efields(disorder_xyzq=xyzq)
exp = ens.to_experiment()

# Or a specific tilt shared by all defects:
ens.generate_defects(seed=0, quantization_axis=[0, 0, 1])    # all along z
ens.generate_defects(seed=0, quantization_axis=[1, 0, 0])    # all along x
```

Axes can also be set after placement:
```python
ens.set_quantization_axis("random", seed=99)
```

---

## AC sensing sequences and SNR

```python
from SpinDefectSim.sensing.sequences import RamseySequence, HahnEchoSequence, XY8Sequence
from SpinDefectSim.sensing.snr import snr, n_avg_for_threshold

tau_ac = 5e-6          # sensing τ for a 100 kHz AC field

for name, seq in [("Ramsey", RamseySequence()),
                  ("Hahn-echo", HahnEchoSequence()),
                  ("XY8", XY8Sequence())]:
    r = seq.repetition_rate(tau_ac)
    print(f"{name}: {r/1e3:.1f} kHz rep rate")

N_thresh = n_avg_for_threshold(0.05, snr_target=5.0,
                               contrast=d.contrast, n_photons=d.n_photons)
print(f"Averages for SNR=5: {N_thresh:.0f}")
```

---

## Screened Coulomb E-fields

Three screening models for the Coulomb interaction between disorder charges
and defects:

| `screening_model` | Description                                     |
|-------------------|-------------------------------------------------|
| `None`            | Bare Coulomb 1/r²                               |
| `"yukawa"`        | Yukawa: exp(−r/λ)/r² with `lambda_screen`       |
| `"dual_gate"`     | Image-charge sum enforcing V=0 at ±d_gate       |

```python
from SpinDefectSim.electrometry.efield import E_disorder_point_charges

E_vec = E_disorder_point_charges(
    obs_xyz=[0, 0, 0.34e-9],
    charges_xyzq=xyzq,
    epsilon_eff=7.0,
    screening_model="dual_gate",
    d_gate=15e-9,
    n_images=30,
)
```

---

## Importing E-fields from external solvers (FEM / FDTD / COMSOL)

Any E-field source — COMSOL, Lumerical, meep, FEniCS, or a custom simulation —
can be fed directly into the ensemble.  Choose whichever path matches your data format.

### From a regular grid (2-D or 3-D)

The most common case: your solver exports field components on a Cartesian mesh.

```python
# 2-D slice (e.g. COMSOL "Export > Data" at fixed z)
# xs : (Nx,) array of x-coordinates in metres
# ys : (Ny,) array of y-coordinates in metres
# Ex2d, Ey2d, Ez2d : (Nx, Ny) arrays in V/m
ens.efields_from_grid(Ex2d, Ey2d, Ez2d, xs, ys)

# 3-D volumetric (e.g. FDTD output) — evaluated at z = z_defect
ens.efields_from_grid(Ex3d, Ey3d, Ez3d, xs, ys, zs, z_defect=0.34e-9)

# Nearest-neighbour instead of bilinear — useful for very coarse grids
ens.efields_from_grid(Ex2d, Ey2d, Ez2d, xs, ys, method="nearest")
```

The `add=True` flag superimposes the grid field on top of an already-computed
disorder/gate contribution:

```python
ens.compute_efields(disorder_xyzq=charges)      # analytic disorder
ens.efields_from_grid(Ex_fdtd, Ey_fdtd, Ez_fdtd, xs, ys, add=True)   # + FDTD gate
```

Out-of-bounds defects are filled with zero by default (`fill_value=0.0`).
Set `bounds_error=True` to raise an error instead.

### From a callable (most flexible)

Wrap any interpolator, closed-form expression, or file-reading function:

```python
# Uniform field — trivial example
ens.efields_from_callable(lambda xyz: [1e4, 0, 0])

# Wrapping scipy RegularGridInterpolator built from solver output
from scipy.interpolate import RegularGridInterpolator

interp_x = RegularGridInterpolator((xs, ys), Ex2d)
interp_y = RegularGridInterpolator((xs, ys), Ey2d)
interp_z = RegularGridInterpolator((xs, ys), Ez2d)

def my_field(xyz):
    pt = [[xyz[0], xyz[1]]]
    return [float(interp_x(pt)), float(interp_y(pt)), float(interp_z(pt))]

ens.efields_from_callable(my_field)

# add=True works here too
ens.compute_efields(disorder_xyzq=charges)
ens.efields_from_callable(my_field, add=True)
```

The callable receives `xyz` as a length-3 numpy array `[x, y, z_defect]` in metres
and must return `[Ex, Ey, Ez]` in V/m.

---

## Parameter sweeps

```python
from SpinDefectSim.analysis.sweep import ParameterSweep
import pandas as pd

ps = ParameterSweep(N_def=200, seed=0)

def run(n_charges):
    ens = ps.make_ensemble()
    rng2 = np.random.default_rng(1)
    n = int(n_charges)
    xyzq2 = np.column_stack([
        rng2.uniform(-d.R_patch, d.R_patch, (n, 2)),
        np.zeros(n),
        rng2.choice([-1, 1], n) * e_charge,
    ])
    ens.compute_efields(disorder_xyzq=xyzq2)
    _, _, _, _, tau_opt, dS_peak = ens.to_experiment().echo_static()
    return dict(n_charges=n, dS_peak=dS_peak, tau_opt_us=tau_opt * 1e6)

results = ps.sweep(run, n_charges=[5, 10, 20, 40])
df = pd.DataFrame(results)
```

---

## Electrometry

Two complementary workflows for E-field sensing:

### Scan maps (single observation point swept over 2-D grid)

Use `ElectrometryExperiment` to compute ODMR frequency maps across a sample
surface — the E-field analogue of `MagnetometryExperiment`.

```python
import numpy as np
from SpinDefectSim.electrometry import ElectrometryExperiment
from SpinDefectSim.base.params import Defaults
from scipy.constants import e as e_charge

# Disorder charges at known positions
charges = np.array([
    [ 50e-9,  0.0, 0.0,  e_charge],
    [-50e-9,  0.0, 0.0, -e_charge],
])

exp = ElectrometryExperiment(charges, Defaults(), z_defect=0.34e-9)

# Point-by-point E-field
E_vec = exp.E_field(0.0, 0.0)                  # (3,) V/m
f1, f2 = exp.transition_frequencies(0.0, 0.0)  # Hz

# 2-D maps
x_arr = np.linspace(-300e-9, 300e-9, 80)
Ez_map = exp.E_z_map(x_arr, x_arr)             # (80, 80) V/m
df_map = exp.frequency_shift_map(x_arr, x_arr) # (80, 80) Hz

# Gate-only experiment (no discrete charges)
exp_gate = ElectrometryExperiment(
    charges_xyzq=None,
    defaults=Defaults(),
    z_defect=0.34e-9,
    E0_gate=(0, 0, 1e4),              # uniform z-gate (V/m)
    epsilon_eff=4.0,
    screening_model="dual_gate",
    d_gate=15e-9,
)
```

### Ensemble sensing with E-field disorder

Use `DefectEnsemble.compute_efields()` to compute the screened-Coulomb E-field
at each defect position and feed it into the echo / Ramsey contrast:

```python
ens.compute_efields(
    E0_gate=(0, 0, 1e4),    # background gate bias
    disorder_xyzq=charges,  # discrete disorder charges
)
exp = ens.to_experiment(sensing="E")
tau, *_, dS_peak = exp.echo_static()

# Or import E fields directly from a FEM simulation:
ens.efields_from_grid(Ex2d, Ey2d, Ez2d, xs, ys)
exp = ens.to_experiment(sensing="E")

# Or from a callable:
ens.efields_from_callable(lambda xyz: my_comsol_interp(xyz[0], xyz[1]))
```

---

## Magnetometry

Two complementary workflows for B-field sensing:

### Scan maps (single observation point swept over 2-D grid)

Use `MagnetometryExperiment` to compute ODMR frequency maps across a sample
surface — the standard NV scanning-magnetometry workflow.

```python
from SpinDefectSim.magnetometry import SquareGeometry, MagnetometryExperiment

geom    = SquareGeometry(side=1e-6, n_boundary_pts=300)
mag_exp = MagnetometryExperiment(
    geom,
    magnetization=lambda x, y: 1e3 * np.exp(-(x**2 + y**2) / (250e-9)**2),
    defaults=Defaults(),
    z_defect=50e-9,
    bias_B_T=[0, 0, 1e-3],
)

x_arr   = np.linspace(-800e-9, 800e-9, 50)
Bz_map  = mag_exp.B_z_map(x_arr, x_arr)          # (50, 50) array
f1, f2  = mag_exp.transition_frequencies(0.0, 0.0)
```

### Ensemble sensing with stray B

Use `DefectEnsemble.compute_bfields()` to compute the Biot-Savart stray B at
each defect position and feed it into the echo / Ramsey contrast:

```python
ens.compute_bfields(
    magnetization=lambda x, y: 1e-3,   # uniform 1 mA
    geometry=geom,
    n_pts=80,
)
exp = ens.to_experiment(sensing="B")
tau, *_, dS_peak = exp.echo_static()

# Or import B fields directly from a micromagnetic simulation:
ens.bfields_from_grid(Bx2d, By2d, Bz2d, xs, ys)
exp = ens.to_experiment(sensing="B")

# Or from a callable (mumax3 wrap, OOMMF output, etc.):
ens.bfields_from_callable(lambda xyz: my_mumax_interp(xyz[0], xyz[1]))
```

---

## SensingExperiment API

`SensingExperiment` is the central object for computing observables.  It is
normally obtained from `DefectEnsemble.to_experiment()`, but can be constructed
directly.

| Method               | Returns                                        |
|----------------------|------------------------------------------------|
| `cw_odmr(f_axis)`    | `pl_with, pl_no, dpl`                          |
| `ramsey()`           | `tau, S_w, S_n, dS, tau_opt, dS_peak`          |
| `echo_static()`      | `tau, S_w, S_n, dS, tau_opt, dS_peak`          |
| `echo_odmr_lockIn()` | `f_axis, dpl`                                  |
| `snr(tau_s, N_avg)`  | scalar SNR                                     |
| `n_avg_to_detect()`  | N averages for specified SNR at optimal τ      |

---

## Key API reference

### `Defaults`

```python
Defaults(
    defect_type="vb_minus",  # spin species
    D0_Hz=3.46e9, E0_Hz=50e6, d_perp=0.35, B_mT=1.5,
    T2star=50e-9, T2echo=10e-6,
    contrast=0.02, n_photons=500,
    N_def=1200, R_patch=200e-9,
    z_defect=0.34e-9, epsilon_eff=7.0,
    screening_model="dual_gate", d_gate=15e-9,
)
Defaults.for_defect("nv_minus", B_mT=3.0)   # classmethod
```

### `SpinDefect`

```python
SpinDefect(defect_type="vb_minus", *, spin=None, B_mT=None, B_vec_mT=None,
           quantization_axis=None, D0_Hz=None, E0_Hz=None, d_perp=None,
           d_parallel=None, defaults=None)

# Single-defect sensing — returns a full SensingExperiment
exp = defect.to_experiment(
    E_vec_Vpm=(0, 0, 0),   # lab-frame E-field for signal branch (V/m)
    B_extra_T=(0, 0, 0),   # lab-frame stray B for signal branch (T)
    sensing="E" | "B" | "both",
)
```

### `DefectEnsemble`

```python
ens = DefectEnsemble(N_def, R_patch, defaults)
ens.generate_defects(seed, quantization_axis=None|"random"|[x,y,z]|(N,3))
ens.generate_defects_gaussian(beam_waist_m, seed)
ens.set_defects(positions)
ens.set_quantization_axis(spec, seed)

# E-field sources
ens.compute_efields(E0_gate, gate_grad, disorder_xyzq, verbose)
ens.efields_from_grid(Ex, Ey, Ez, x_coords, y_coords, [z_coords], *, add=False)
ens.efields_from_callable(E_func, *, add=False)
ens.set_efields(E_fields)                   # inject (N, 3) V/m directly

# B-field sources (stray field signal)
ens.compute_bfields(magnetization, geometry, *, n_pts, add=False)
ens.bfields_from_grid(Bx, By, Bz, x_coords, y_coords, [z_coords], *, add=False)
ens.bfields_from_callable(B_func, *, add=False)
ens.set_bfields(B_fields)                   # inject (N, 3) T directly

# Run experiment — sensing selects which fields create the contrast
exp = ens.to_experiment(B_mT=None, sensing="E" | "B" | "both")
```

### `DefectType`

```python
from SpinDefectSim.spin.defects import DefectType, get_defect, list_defects
dt = get_defect("nv_minus")       # retrieve preset
list_defects()                    # print table of all presets
dt = DefectType(name="custom", spin=1, D0_Hz=2e9, d_perp=0.1)
```

---

## Package structure

```
SpinDefectSim/
├── base/
│   ├── params.py          Defaults dataclass, PhysicalParams base, mixins
│   └── mixins.py          PlottingMixin, SerializationMixin, SweepMixin
├── spin/
│   ├── defects.py         DefectType class and built-in presets
│   ├── matrices.py        Spin operators for any spin S
│   ├── hamiltonian.py     odmr_hamiltonian_Hz(), full_hyperfine_hamiltonian_Hz(), SpinDefect
│   ├── nuclear.py         NuclearSpin dataclass, isotope γ constants, A-tensor helpers
│   ├── rates.py           RateModel, RateParams, CW ODMR contrast from ISC equations
│   ├── spectra.py         CW lineshapes, ensemble spectrum builders
│   └── echo.py            Hahn-echo, Ramsey lock-in signals
├── coulomb/
│   └── kernels.py         Screened Coulomb kernels (bare, Yukawa, dual-gate)
├── sensing/
│   ├── protocols.py       SensingExperiment — CW, Ramsey, Hahn-echo
│   ├── sequences.py       RamseySequence, HahnEchoSequence, XY8Sequence
│   └── snr.py             Shot-noise SNR utilities
├── electrometry/          ← E-field sources and E-field imaging
│   ├── efield.py          ElectricFieldBuilder, E_gate_bias, E_disorder_point_charges
│   └── electrometry.py    ElectrometryExperiment — charges → E → ODMR maps
├── magnetometry/          ← B-field sources and B-field imaging
│   ├── geometry.py        SquareGeometry, DiskGeometry sample shapes
│   ├── bfield.py          Biot-Savart stray B-field calculations
│   └── magnetometry.py    MagnetometryExperiment — M(x,y) → B → ODMR maps
└── analysis/
    ├── ensemble.py        DefectEnsemble — positions, E-fields, B-fields, axes
    └── sweep.py           ParameterSweep wrapper
```
