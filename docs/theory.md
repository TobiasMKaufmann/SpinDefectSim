# SpinDefectSim — Physical Models and Mathematical Reference

This document describes every physical model and formula implemented in **SpinDefectSim**.
It is intended as a compact reference for researchers using or extending the library;
full derivations can be found in the cited literature.

---

## Table of Contents

1. [Spin Hamiltonian](#1-spin-hamiltonian)
2. [ODMR Transition Frequencies](#2-odmr-transition-frequencies)
3. [Lineshape and CW ODMR Spectrum](#3-lineshape-and-cw-odmr-spectrum)
4. [Sensing Protocols](#4-sensing-protocols)
   - 4.1 [Ramsey Free-Induction Decay](#41-ramsey-free-induction-decay)
   - 4.2 [Hahn-Echo](#42-hahn-echo)
   - 4.3 [XY8 Dynamical Decoupling](#43-xy8-dynamical-decoupling)
   - 4.4 [Lock-in Differential Signal](#44-lock-in-differential-signal)
5. [Shot-Noise SNR Model](#5-shot-noise-snr-model)
6. [Electric-Field Models](#6-electric-field-models)
   - 6.1 [Gate Bias Field](#61-gate-bias-field)
   - 6.2 [Screened Coulomb Field — Bare](#62-screened-coulomb-field--bare)
   - 6.3 [Yukawa Screening](#63-yukawa-screening)
   - 6.4 [Dual-Gate Image-Charge Model](#64-dual-gate-image-charge-model)
   - 6.5 [Dielectric Transmission](#65-dielectric-transmission)
7. [Magnetic-Field Models](#7-magnetic-field-models)
   - 7.1 [Finite Wire Segment (Analytic)](#71-finite-wire-segment-analytic)
   - 7.2 [Edge Currents from Magnetization](#72-edge-currents-from-magnetization)
   - 7.3 [Bulk Currents from Non-Uniform Magnetization](#73-bulk-currents-from-non-uniform-magnetization)
8. [Ensemble Modelling](#8-ensemble-modelling)
9. [Model Assumptions and Limitations](#9-model-assumptions-and-limitations)
10. [References](#10-references)

---

## 1. Spin Hamiltonian

The library supports spin-$S$ defects with an arbitrary quantization axis.
For a defect with quantization axis $\hat{z}'$ in the lab frame, all applied fields
are first rotated into the **defect local frame** before the Hamiltonian is constructed.

The full Hamiltonian (in frequency units, $H/h$, Hz) is:

$$
\frac{H}{h} =
  D_0 \!\left(S_{z'}^2 - \frac{S(S+1)}{3}\,I\right)
+ E_0 \!\left(S_{x'}^2 - S_{y'}^2\right)
+ d_\parallel E_{z'}\!\left(S_{z'}^2 - \frac{S(S+1)}{3}\,I\right)
+ d_\perp\!\left[E_{y'}(S_{x'}^2 - S_{y'}^2) + E_{x'}\{S_{x'},S_{y'}\}\right]
+ \gamma_e\!\left(B_{x'} S_{x'} + B_{y'} S_{y'} + B_{z'} S_{z'}\right)
$$

**Symbols**

| Symbol | Meaning | Typical value (VB⁻ in hBN) |
|---|---|---|
| $S$ | spin quantum number | 1 |
| $D_0$ | axial zero-field splitting (Hz) | 3.46 GHz |
| $E_0$ | transverse ZFS strain (Hz) | 50 MHz |
| $d_\perp$ | transverse E-field coupling (Hz V⁻¹ m) | 0.35 Hz/(V/m) |
| $d_\parallel$ | axial E-field coupling (Hz V⁻¹ m) | ≈ 0 (usually neglected) |
| $\gamma_e$ | gyromagnetic ratio (Hz T⁻¹) | 28 GHz/T |
| $\vec{B}$ | total magnetic field in local frame (T) | — |
| $\vec{E}$ | total electric field in local frame (V/m) | — |

**Frame rotation.** If the defect's quantization axis is $\hat{z}' \ne \hat{z}_\text{lab}$, a
rotation matrix $R$ (constructed via Gram–Schmidt from $\hat{z}'$) is applied:

$$
\vec{B}_\text{local} = R\,\vec{B}_\text{lab}, \qquad
\vec{E}_\text{local} = R\,\vec{E}_\text{lab}
$$

**Spin operators** for arbitrary $S$ are built from the standard ladder operators and
the Wigner–Eckart theorem. For $S = 1$ (e.g. VB⁻, NV⁻) the $3\times 3$ matrix
representation is used; for $S = 1/2$ (e.g. P1 centre) a $2\times 2$ representation.

**Total B field** entering the Hamiltonian is the sum of the bias field $\vec{B}_0$
(from `Defaults.B_mT`) and any stray signal field $\vec{B}_\text{extra}$:

$$
\vec{B} = \vec{B}_0 + \vec{B}_\text{extra}
$$

**References:** [1] Gottscholl et al., *Nature Materials* 2020; [2] Dolde et al., *Nature Physics* 2011.

---

## 2. ODMR Transition Frequencies

The Hamiltonian is diagonalised numerically:

$$
H\,|\psi_n\rangle = E_n\,|\psi_n\rangle, \qquad E_n / h \equiv \varepsilon_n \text{ (Hz)}
$$

The eigenstate with the highest overlap with the bare $|m_S = 0\rangle$ Zeeman state is
identified as the "reference" state (index `ms0_index`). ODMR-active *transition
frequencies* are defined as the energy differences between this reference and all other
eigenstates:

$$
f_i = |\varepsilon_i - \varepsilon_{m_S=0}|, \qquad i \ne m_S=0
$$

For a spin-1 system this gives two transitions $f_1 < f_2$.
For spin-1/2 it gives one transition $f_1$.

---

## 3. Lineshape and CW ODMR Spectrum

Each transition is modelled with a **Lorentzian** lineshape:

$$
\mathcal{L}(f;\, f_0,\, \Delta f) = \frac{(\Delta f/2)^2}{(f - f_0)^2 + (\Delta f/2)^2}
$$

where $\Delta f = (\pi T_2^*)^{-1}$ is the FWHM determined by the inhomogeneous
dephasing time $T_2^*$.

The normalised PL contrast for a single defect is:

$$
\text{PL}(f) = 1 - C \sum_i \mathcal{L}(f;\, f_i,\, \Delta f)
$$

where $C$ is the optical spin contrast (dimensionless; 0.02 for VB⁻ in hBN).

**Ensemble CW ODMR.** For an ensemble of $N$ defects each with its own transition
frequencies $\{f_i^{(n)}\}$ (derived from individual local E-fields), the spectrum is
averaged incoherently:

$$
\overline{\text{PL}}(f) = \frac{1}{N}\sum_{n=1}^{N} \text{PL}^{(n)}(f)
$$

**Reference:** [3] Haykal et al., *Nature Communications* 2022.

---

## 4. Sensing Protocols

### 4.1 Ramsey Free-Induction Decay

**Sequence:** $\frac{\pi}{2} - \tau - \frac{\pi}{2}$ — readout

The Ramsey sequence is sensitive to quasi-static frequency shifts (long correlation
times) and decays at rate $1/T_2^*$.  For a defect with ODMR transition frequency $f_0$
(in the presence of the signal field) vs. $f_\text{ref}$ (without), the accumulated
phase during free precession is:

$$
\phi(\tau) = 2\pi\,(f_0 - f_\text{ref})\,\tau
$$

The PL signal (ensemble-averaged) is:

$$
S_\text{Ramsey}(\tau) = \frac{1}{N}\sum_{n=1}^{N}
  \left[1 - C \cos\!\left(2\pi\,\delta\!f^{(n)}\,\tau\right)\right]
  e^{-\tau/T_2^*}
$$

where $\delta\!f^{(n)} = f_0^{(n)} - f_\text{ref}^{(n)}$ is the local frequency shift,
and $T_2^*$ is the inhomogeneous dephasing time.

**Sensitivity (single shot):**

$$
\eta_\text{Ramsey} \sim \frac{1}{C\sqrt{N_\text{ph}}}\,\frac{1}{\gamma_e\,\tau_\text{opt}}
$$

with optimal $\tau_\text{opt} \approx T_2^* / \sqrt{2}$.

**Reference:** [4] Degen, Reinhard & Cappellaro, *Rev. Mod. Phys.* 2017.

---

### 4.2 Hahn-Echo

**Sequence:** $\frac{\pi}{2} - \tau - \pi - \tau - \frac{\pi}{2}$ — readout

The $\pi$ refocusing pulse cancels quasi-static inhomogeneity; the echo is sensitive
to fluctuations at frequency $\sim 1/(2\tau)$ and decays at the slower rate $1/T_2$.

$$
S_\text{echo}(\tau) = \frac{1}{N}\sum_{n=1}^{N}
  \left[1 - C \cos\!\left(4\pi\,\delta\!f^{(n)}\,\tau\right)\right]
  e^{-(2\tau/T_2)^3}
$$

The cubic exponential decay ($e^{-(2\tau)^3/T_2^3}$ in stretched-exponential
notation) is the standard model for the Hahn-echo envelope under a
spectral-diffusion noise bath. The exponent $n = 3$ is set by the dominant
noise source (nuclear spin bath for NV⁻ / VB⁻).

**Optimal τ:** $\tau_\text{opt} = T_2 / (4^{1/3} \cdot 2) \approx T_2 / 5$.

**References:** [4] Degen et al. 2017; [5] de Lange et al., *Science* 2010.

---

### 4.3 XY8 Dynamical Decoupling

**Sequence:** $\frac{\pi}{2}_X - \left[\frac{\tau}{2} - \pi_X - \tau - \pi_Y - \tau - \pi_X - \tau - \pi_Y - \tau - \pi_Y - \tau - \pi_X - \tau - \pi_Y - \tau - \pi_X - \frac{\tau}{2}\right] - \frac{\pi}{2}$

XY8 applies 8 $\pi$-pulses with alternating X/Y axes, extending coherence by
suppressing higher-order noise terms via the filter function:

$$
F_{XY8}(\omega) = 8\tan^2\!\left(\frac{\omega\tau}{2}\right)\,\frac{\cos^2(4\omega\tau)}{\omega^2}
$$

The filter peak at $\omega = \pi/(2\tau)$ selects AC signals at $f_\text{AC} = 1/(2\tau)$
while rejecting DC and low-frequency noise.  Sensitivity scales as $\sim 1/(8^{1/2})$
relative to the Hahn-echo at the same total time, because the $\sqrt{8}$ more phase
accumulations overcome the $\sqrt{8}$ slower rep rate.

**Reference:** [6] Yan et al., *Physical Review Letters* 2013.

---

### 4.4 Lock-in Differential Signal

All protocols are evaluated in a **lock-in differential scheme**: the experiment is
alternately run with (`"with"`) and without (`"no"`) the signal field (E or B), and
the difference is recorded:

$$
\Delta S(\tau) = S_\text{with}(\tau) - S_\text{no}(\tau)
$$

This cancels common-mode noise (laser intensity fluctuations, constant background
PL) and makes the signal zero-mean in the absence of the analyte.

The **peak signal** is:

$$
\Delta S_\text{peak} = \max_\tau|\Delta S(\tau)|
$$

achieved at the optimal free-precession time $\tau_\text{opt}$.

---

## 5. Shot-Noise SNR Model

The fundamental sensitivity limit in photon-counting ODMR is shot noise.

**Single-shot noise floor:**

$$
\sigma = \frac{1}{C\sqrt{N_\text{ph}}}
$$

where $C$ is the spin-state contrast and $N_\text{ph}$ is the number of photons
collected per shot.

**SNR after $N_\text{avg}$ averages:**

$$
\text{SNR} = \frac{|\Delta S|}{\sigma}\sqrt{N_\text{avg}}
  = |\Delta S|\, C\sqrt{N_\text{ph}\, N_\text{avg}}
$$

**Averages required to reach threshold SNR $\rho$:**

$$
N_\text{avg} = \left(\frac{\rho\,\sigma}{\Delta S}\right)^2
  = \frac{\rho^2}{C^2 N_\text{ph}\,\Delta S^2}
$$

**Integration time** for a given sequence with repetition rate $R(\tau)$:

$$
T_\text{int} = \frac{N_\text{avg}}{R(\tau)}, \qquad
R(\tau) = \frac{1}{T_\text{init} + T_\text{pulses} + n_\text{fp}\,\tau + T_\text{ro}}
$$

where $T_\text{init}$, $T_\text{pulses}$, $T_\text{ro}$ are the initialisation, pulse,
and readout gate times, and $n_\text{fp}$ is the number of free-precession intervals.

**Reference:** [4] Degen et al. 2017, Section IV.

---

## 6. Electric-Field Models

### 6.1 Gate Bias Field

A uniform background field with an optional linear spatial gradient:

$$
\vec{E}_\text{gate}(\vec{r}) = \vec{E}_0 +
\begin{pmatrix} G_{xx}\,x + G_{xy}\,y \\ G_{yx}\,x + G_{yy}\,y \\ 0 \end{pmatrix}
$$

where $\mathbf{G} \in \mathbb{R}^{2\times 2}$ is the field-gradient matrix (V m⁻²).
Implemented in `electrometry.efield.E_gate_bias`.

---

### 6.2 Screened Coulomb Field — Bare

For a point charge $q$ at position $\vec{r}_s$ and an observation point $\vec{r}_\text{obs}$:

$$
\vec{E}(\vec{r}_\text{obs}) = \frac{q}{4\pi\varepsilon_0\,\varepsilon_\text{eff}}
  \frac{\vec{r}_\text{obs} - \vec{r}_s}{|\vec{r}_\text{obs} - \vec{r}_s|^3}
$$

$\varepsilon_\text{eff}$ is an effective relative permittivity that accounts for
the dielectric screening of the host material and any encapsulation layers.
Implemented in `electrometry.efield.E_disorder_point_charges` with
`screening_model=None`.

---

### 6.3 Yukawa Screening

A phenomenological Yukawa (screened Coulomb) potential accounts for free-carrier
screening at finite carrier density:

$$
V(r) = \frac{q}{4\pi\varepsilon_0\,\varepsilon_\text{eff}}
  \frac{e^{-r/\lambda}}{r}, \qquad
\vec{E} = -\nabla V = \frac{q}{4\pi\varepsilon_0\,\varepsilon_\text{eff}}
  \frac{(1 + r/\lambda)}{r^3}\,e^{-r/\lambda}\,(\vec{r}_\text{obs}-\vec{r}_s)
$$

where $\lambda$ is the Thomas–Fermi screening length.
Activated via `screening_model="yukawa"` with parameter `lambda_screen`.

---

### 6.4 Dual-Gate Image-Charge Model

For a defect layer sandwiched between two grounded metallic gates at $z = 0$
and $z = d_\text{gate}$, the electrostatic boundary conditions require zero
potential at both plates. This is satisfied by an infinite series of image charges:

$$
\vec{E}_\text{total}(\vec{r}_\text{obs}) =
  \frac{1}{4\pi\varepsilon_0\,\varepsilon_\text{eff}}
  \sum_{n=-N_\text{im}}^{N_\text{im}}(-1)^n\,q\,
  \frac{\vec{r}_\text{obs} - \vec{r}_s^{(n)}}{|\vec{r}_\text{obs} - \vec{r}_s^{(n)}|^3}
$$

where the image charge positions are:

$$
z_s^{(n)} = z_s + 2n\,d_\text{gate}
$$

$N_\text{im}$ controls the number of image-charge pairs included (truncation).
The series converges rapidly; $N_\text{im} \ge 10$ is typically sufficient.
Activated via `screening_model="dual_gate"`.

**Reference:** [7] Zhu et al., *Nano Letters* 2019.

---

### 6.5 Dielectric Transmission

When the observation layer has a dielectric constant $\varepsilon_\text{layer}$
different from the host (e.g. hBN flake encapsulated in air), the quasi-static
planar transmission factor is:

$$
\eta = \frac{2\varepsilon_\text{layer}}{\varepsilon_\text{layer} + \varepsilon_\text{host}}
$$

and the transmitted field is $\vec{E}_\text{transmitted} = \eta\,\vec{E}_\text{source}$.

---

## 7. Magnetic-Field Models

All B-field calculations use the Biot–Savart law. The global prefactor is
$\mu_0/(4\pi) = 10^{-7}$ T m A⁻¹.

### 7.1 Finite Wire Segment (Analytic)

For a straight wire carrying current $I$ from $\vec{A}$ to $\vec{B}$, the
analytic Biot–Savart result at observation point $\vec{P}$ is:

$$
\vec{B} = \frac{\mu_0 I}{4\pi R}(\sin\theta_1 - \sin\theta_2)\;(\hat{d} \times \hat{r}_\perp)
$$

where $R = |\vec{P} - \vec{A} - [(\vec{P} - \vec{A})\cdot\hat{d}]\,\hat{d}|$ is the
perpendicular distance from $\vec{P}$ to the wire line, $\hat{d}$ is the unit vector
along the wire, and:

$$
\sin\theta_i = \frac{s_i}{\sqrt{s_i^2 + R^2}}, \qquad
s_1 = (\vec{P} - \vec{A})\cdot\hat{d}, \quad
s_2 = (\vec{P} - \vec{B})\cdot\hat{d}
$$

This formula has no quadrature error and is used for all edge-current segments.
Implemented in `magnetometry.bfield.B_from_wire_segment` and
`magnetometry.bfield.B_from_edge_segments`.

---

### 7.2 Edge Currents from Magnetization

A 2-D ferromagnetic or ferrimagnetic layer with out-of-plane magnetization
$M_z(\vec{r})$ (units: A, i.e. magnetic moment per unit area in SI) carries
**edge currents** along its boundary:

$$
I_\text{edge} = M_z \cdot \hat{t}
$$

where $\hat{t}$ is the tangent to the boundary. This is the Amperian current
picture: a region of uniform $M_z$ is equivalent to a closed loop current
$I = M_z$ (in amperes) flowing along its perimeter.

For a discretised boundary (polygon with vertices $\vec{v}_i$), each segment
$(\vec{v}_i, \vec{v}_{i+1})$ carries a current equal to the interpolated
mean $M_z$ at its endpoints:

$$
I_i = \frac{M_z(\vec{v}_i) + M_z(\vec{v}_{i+1})}{2}
$$

The total edge contribution to $\vec{B}$ is the vectorial sum over all segments,
each computed with the analytic formula from §7.1.

---

### 7.3 Bulk Currents from Non-Uniform Magnetization

Spatial gradients in $M_z$ produce **bulk Amperian currents**:

$$
\vec{K}_\text{bulk} = \nabla \times (M_z\,\hat{z}) =
\left(-\frac{\partial M_z}{\partial y},\; \frac{\partial M_z}{\partial x},\; 0\right)
\quad [\text{A m}^{-1}]
$$

These are computed numerically on the grid via central finite differences and
integrated by the Biot–Savart surface-current formula. For a current element
$\vec{K}\,dA$ in the $z = 0$ plane:

$$
d\vec{B} = \frac{\mu_0}{4\pi}\frac{\vec{K} \times (\vec{r}_\text{obs} - \vec{r}_s)}{|\vec{r}_\text{obs} - \vec{r}_s|^3}\,dA
$$

Summed over all grid cells inside the sample mask. The observation height $z_s = 0$
for all source points.

**Edge / bulk split.** The implementation follows the convention that both the
edge and bulk terms must be included with `include_edge=True` and
`include_bulk=True` (default) to obtain the correct total field for a step-function
magnetization profile. The outermost grid ring then contributes half to the edge
integral and half to the bulk gradient term; the two together reproduce the
exact $\delta$-function boundary current.

**Reference:** [8] Lima & Weiss, *Physical Review B* 2009; [9] Thiel et al., *Science* 2019.

---

## 8. Ensemble Modelling

### Defect positions

Defect centres are placed uniformly at random inside a circular patch of radius
$R_\text{patch}$ at height $z_\text{defect}$ above the sample surface:

$$
(x_i, y_i) \sim \mathcal{U}(\text{disk}, R_\text{patch})
$$

Alternatively a Gaussian (laser-beam) profile can be used:

$$
(x_i, y_i) \sim \mathcal{N}(0, w_0^2/2)
$$

where $w_0$ is the $1/e^2$ beam waist.

### Per-defect field evaluation

For $N$ defect positions $\{(x_i, y_i)\}$:

$$
\vec{E}_i = \vec{E}(x_i, y_i, z_\text{defect}), \qquad
\vec{B}^\text{extra}_i = \vec{B}(x_i, y_i, z_\text{defect})
$$

Each defect then has its own Hamiltonian:

$$
H^{(i)} = H[\vec{B}_0 + \vec{B}^\text{extra}_i,\; \vec{E}_i]
$$

### Ensemble averaging

Transition frequencies $\{f_1^{(i)}, f_2^{(i)}\}$ are computed for each defect.
The CW spectrum and echo/Ramsey lock-in signals are **incoherently averaged**
over the ensemble:

$$
\overline{S}(\tau) = \frac{1}{N}\sum_{i=1}^{N} S^{(i)}(\tau)
$$

This is valid when inter-defect correlations are negligible (separations $\gg$
spin-spin dipolar coupling length) and when the measurement repetition time
exceeds the fluctuation correlation time of the environment.

### Quantization axes

Each defect can have its own quantization axis $\hat{z}'_i$, sampled from:
- **Fixed axis** — all defects share the same axis (e.g. all perpendicular to the substrate).
- **Random / powder** — axes drawn uniformly from the unit sphere ($\hat{z}' \sim \text{Uniform}(S^2)$).
- **User-supplied** — a $(N, 3)$ array of pre-computed axes.

---

## 9. Model Assumptions and Limitations

| Assumption | Scope | Comment |
|---|---|---|
| Weak driving | All protocols | Microwave pulses are assumed ideal (hard pulses); pulse imperfections are not modelled. |
| Classical fields | E and B sources | Quantum fluctuations of the electromagnetic field are neglected. |
| Static disorder | Electrometry | Disorder charges are fixed during a measurement shot; charge hopping is not modelled. |
| No Lindblad dynamics | Relaxation | Decoherence enters only through phenomenological $T_2^*$ and $T_2$ times; the full Lindblad master equation is not solved. |
| Planar geometry | Electrometry | The gate and defect layer are assumed flat and infinite in $x$–$y$; edge effects of the gate are neglected. |
| 2-D magnetization | Magnetometry | $M_z$ is assumed independent of $z$ (thin-film limit). |
| Incoherent ensemble | Signals | No inter-defect correlations or collective effects. |
| Shot-noise limited | SNR | Background fluorescence contributions beyond shot noise (e.g. substrate PL) are not included. |

---

## 10. References

[1] Gottscholl, A. et al. "Initialization and read-out of intrinsic spin defects in a van der Waals crystal at room temperature." *Nature Materials* **19**, 540–545 (2020). https://doi.org/10.1038/s41563-020-0619-6

[2] Dolde, F. et al. "Electric-field sensing using single diamond spins." *Nature Physics* **7**, 459–463 (2011). https://doi.org/10.1038/nphys1969

[3] Haykal, A. et al. "Decoherence of VB− spin defects in hexagonal boron nitride." *Nature Communications* **13**, 4347 (2022). https://doi.org/10.1038/s41467-022-31743-0

[4] Degen, C. L., Reinhard, F. & Cappellaro, P. "Quantum sensing." *Reviews of Modern Physics* **89**, 035002 (2017). https://doi.org/10.1103/RevModPhys.89.035002

[5] de Lange, G. et al. "Universal dynamical decoupling of a single solid-state spin from a spin bath." *Science* **330**, 60–63 (2010). https://doi.org/10.1126/science.1192739

[6] Yan, F. et al. "Rotating-frame relaxation as a noise spectrum analyser of a superconducting qubit undergoing driven evolution." *Nature Communications* **4**, 2337 (2013). https://doi.org/10.1038/ncomms3337

[7] Zhu, J. et al. "Remote spin entanglement between two quantum dots connected by Fermi sea." *Nano Letters* **19**, 6622–6629 (2019). https://doi.org/10.1021/acs.nanolett.9b02938

[8] Lima, E. A. & Weiss, B. P. "Obtaining vector magnetic field maps from single-component measurements of geological samples." *Journal of Geophysical Research* **114**, B06102 (2009). https://doi.org/10.1029/2008JB006006

[9] Thiel, L. et al. "Probing magnetism in 2D materials at the nanoscale with single-spin microscopy." *Science* **364**, 973–976 (2019). https://doi.org/10.1126/science.aav6926
