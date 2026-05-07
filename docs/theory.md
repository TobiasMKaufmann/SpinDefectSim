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
10. [Nuclear-Spin Hyperfine Interaction](#10-nuclear-spin-hyperfine-interaction)
    - 10.1 [Hyperfine Hamiltonian](#101-hyperfine-hamiltonian)
    - 10.2 [Nuclear Quadrupole Coupling](#102-nuclear-quadrupole-coupling)
    - 10.3 [Nuclear Zeeman Term](#103-nuclear-zeeman-term)
    - 10.4 [Tensor-Product Hilbert Space](#104-tensor-product-hilbert-space)
    - 10.5 [Isotope Constants](#105-isotope-constants)
11. [Rate-Equation Model for CW Contrast](#11-rate-equation-model-for-cw-contrast)
    - 11.1 [Level Structure](#111-level-structure)
    - 11.2 [Rate Matrix and Steady State](#112-rate-matrix-and-steady-state)
    - 11.3 [CW ODMR Contrast](#113-cw-odmr-contrast)
    - 11.4 [Pre-built Defect Parameters](#114-pre-built-defect-parameters)
12. [References](#12-references)

---

## 1. Spin Hamiltonian

The library supports spin-$S$ defects with an arbitrary quantization axis.
For a defect with quantization axis $\hat{z}'$ in the lab frame, all applied fields
are first rotated into the **defect local frame** before the Hamiltonian is constructed.

The full Hamiltonian (in frequency units, $H/h$, Hz) is:

$$
\frac{H}{h} =
  D_0 \left(S_{z'}^2 - \frac{S(S+1)}{3}\,I\right)
+ E_0 \left(S_{x'}^2 - S_{y'}^2\right)
+ d_\parallel E_{z'}\left(S_{z'}^2 - \frac{S(S+1)}{3}\,I\right)
+ d_\perp\left[E_{y'}(S_{x'}^2 - S_{y'}^2) + E_{x'}\{S_{x'},S_{y'}\}\right]
+ \gamma_e\left(B_{x'} S_{x'} + B_{y'} S_{y'} + B_{z'} S_{z'}\right)
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
  \left[1 - C \cos\left(2\pi\,\delta f^{(n)}\,\tau\right)\right]
  e^{-\tau/T_2^*}
$$

where $\delta f^{(n)} = f_0^{(n)} - f_\text{ref}^{(n)}$ is the local frequency shift,
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
  \cos\left(2\pi\,\delta f^{(n)}\,\tau\right)
  e^{-\tau/T_2}
$$

The decay envelope $e^{-\tau/T_2}$ is a phenomenological simple exponential
parameterised by the coherence time $T_2$.  (The physically exact envelope
under a nuclear-spin-bath noise spectral density is a stretched exponential
$e^{-(2\tau/T_2)^3}$; the library uses the simpler form for efficiency.)

**Optimal τ:** $\tau_\text{opt} = T_2$, the maximum of the signal envelope
$\tau\,e^{-\tau/T_2}$.

**References:** [4] Degen et al. 2017; [5] de Lange et al., *Science* 2010.

---

### 4.3 XY8 Dynamical Decoupling

**Sequence:** $\frac{\pi}{2}_X - \left[\frac{\tau}{2} - \pi_X - \tau - \pi_Y - \tau - \pi_X - \tau - \pi_Y - \tau - \pi_Y - \tau - \pi_X - \tau - \pi_Y - \tau - \pi_X - \frac{\tau}{2}\right] - \frac{\pi}{2}$

XY8 applies 8 $\pi$-pulses with alternating X/Y axes, extending coherence by
suppressing higher-order noise terms via the filter function:

$$
F_{XY8}(\omega) = 8\tan^2\left(\frac{\omega\tau}{2}\right)\,\frac{\cos^2(4\omega\tau)}{\omega^2}
$$

The filter peak at $\omega = \pi/(2\tau)$ selects AC signals at $f_\text{AC} = 1/(2\tau)$
while rejecting DC and low-frequency noise.  Sensitivity scales as $\sim 1/(8^{1/2})$
relative to the Hahn-echo at the same total time, because the $\sqrt{8}$ more phase
accumulations overcome the $\sqrt{8}$ slower rep rate.

**Reference:** [6] Yan et al., *Nature Communications* 2013.

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

**Reference:** [7] Jackson, J. D. *Classical Electrodynamics*, 3rd ed. (Wiley, 1999), §2.9.

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
\left(\frac{\partial M_z}{\partial y},\; -\frac{\partial M_z}{\partial x},\; 0\right)
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
- **Random / powder** — axes drawn uniformly from the unit sphere ($\hat{z}' \sim \mathrm{Uniform}(\mathbb{S}^2)$).
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
| Diagonal hyperfine tensor | Hyperfine | The hyperfine tensor $A$ is supplied in the defect local frame; off-diagonal elements are supported but typically assumed to vanish for axially symmetric sites. |
| Single metastable state | Rate model | A single collective shelving state pools all ISC population; multiple distinct singlets are not resolved. |

---

## 10. Nuclear-Spin Hyperfine Interaction

### 10.1 Hyperfine Hamiltonian

Coupling between the electron spin $\vec{S}$ and a nuclear spin $\vec{I}_k$ is
described by the hyperfine interaction:

$$
\frac{H_\text{hf}}{h} = \vec{S} \cdot A_k \cdot \vec{I}_k
  = \sum_{i,j} A_k^{ij}\, S_i \otimes I_{kj}
$$

where $A_k$ is the $3\times 3$ hyperfine coupling tensor (Hz) of nucleus $k$ in the
defect local frame ($z' =$ electron spin quantization axis).

For an **axially symmetric** site the tensor is diagonal:

$$
A_k = \operatorname{diag}(A_\perp,\, A_\perp,\, A_\parallel)
$$

where $A_\parallel = A_{zz}$ is the longitudinal component (along the defect
quantization axis) and $A_\perp = A_{xx} = A_{yy}$ is the transverse component.
Built with `axial_A_tensor(A_zz_Hz, A_perp_Hz)`.

For an **isotropic** (Fermi contact) coupling:

$$
A_k = A_\text{iso}\,\mathbf{1}_3
$$

Built with `isotropic_A_tensor(A_iso_Hz)`.

Typical values:

| Defect | Nucleus | $A_\parallel$ (MHz) | $A_\perp$ (MHz) |
|--------|---------|---------------------|-----------------|
| NV⁻ (diamond) | ¹⁴N on-site | −2.14 | −2.70 |
| VB⁻ (hBN)     | ¹⁴N ×3 in-plane | ≈0 | +47.8 |

---

### 10.2 Nuclear Quadrupole Coupling

For nuclei with spin $I \ge 1$, the non-spherical nuclear charge distribution
couples to the electric field gradient at the nuclear site:

$$
\frac{H_Q}{h} = P_k\left(I_{z,k}^2 - \frac{I_k(I_k+1)}{3}\,\mathbf{1}\right)
$$

where $P_k$ (Hz) is the quadrupole coupling constant.
For NV⁻ ¹⁴N: $P \approx -4.95$ MHz.

**Reference:** [10] Felton et al., *Physical Review B* 2009.

---

### 10.3 Nuclear Zeeman Term

The nuclear spin also precesses in the applied magnetic field:

$$
\frac{H_{nZ}}{h} = \gamma_{n,k}\,\bigl(B_{x'} I_{x,k} + B_{y'} I_{y,k} + B_{z'} I_{z,k}\bigr)
$$

where $\gamma_{n,k}$ (Hz T⁻¹) is the nuclear gyromagnetic ratio for isotope $k$.
At typical laboratory fields ($\lesssim 10$ mT) the nuclear Zeeman splitting
($\lesssim 100$ kHz for ¹⁴N) is much smaller than the hyperfine splitting and
acts as a small perturbation.

---

### 10.4 Tensor-Product Hilbert Space

The full electron + nuclear Hamiltonian is assembled in the tensor-product space:

$$
\mathcal{H} = \mathcal{H}_e \otimes \mathcal{H}_{n_1} \otimes \mathcal{H}_{n_2} \otimes \cdots
$$

with total dimension $(2S+1)\prod_k(2I_k+1)$.  Each term in $H$ is embedded as a
tensor-product operator acting on its own subspace and as the identity on all others:

$$
\frac{H}{h} =
  \underbrace{H_e \otimes \mathbf{1}_{n_1} \otimes \cdots}_{\text{electron}}
+ \sum_k\underbrace{\mathbf{1}_e \otimes \cdots \otimes H_{n_k} \otimes \cdots}_{\text{nuclear }k}
+ \sum_k \sum_{i,j} A_k^{ij}\,S_i \otimes I_{kj}
$$

Implemented in `full_hyperfine_hamiltonian_Hz`.

---

### 10.5 Isotope Constants

Gyromagnetic ratios (Hz T⁻¹) shipped with the library:

| Constant | Isotope | $I$ | $\gamma_n$ (MHz T⁻¹) |
|----------|---------|-----|----------------------|
| `GAMMA_14N`  | ¹⁴N  | 1   | +3.077  |
| `GAMMA_15N`  | ¹⁵N  | 1/2 | −4.316  |
| `GAMMA_11B`  | ¹¹B  | 3/2 | +13.660 |
| `GAMMA_10B`  | ¹⁰B  | 3   | +4.575  |
| `GAMMA_13C`  | ¹³C  | 1/2 | +10.708 |
| `GAMMA_29Si` | ²⁹Si | 1/2 | −5.319  |

**Reference:** [11] Harris et al. (IUPAC recommendations), *Pure and Applied Chemistry* 2001.

---

## 11. Rate-Equation Model for CW Contrast

### 11.1 Level Structure

For a spin-$S$ defect with $N = 2S+1$ magnetic sublevels, the level structure
has three sub-manifolds:

$$
\underbrace{|g, +S\rangle, \ldots, |g, -S\rangle}_{N\text{ ground states}}
\quad
\underbrace{|e, +S\rangle, \ldots, |e, -S\rangle}_{N\text{ excited states}}
\quad
\underbrace{|\text{shelf}\rangle}_{1\text{ metastable state (if ISC active)}}
$$

Level indices: ground $= 0\ldots N-1$, excited $= N\ldots 2N-1$, shelving $= 2N$.

---

### 11.2 Rate Matrix and Steady State

The population vector $\vec{P}$ evolves as $\dot{\vec{P}} = R\,\vec{P}$, where the
rate matrix $R$ encodes the following processes:

| Process | Rate | Comment |
|---------|------|---------|
| Laser excitation | $k_\text{opt}$ | Spin-blind; same for all $m_S$ |
| Radiative decay | $k_\text{rad}$ | Spin-preserving: $|e, m_S\rangle \to |g, m_S\rangle$ |
| ISC to shelving | $k_\text{ISC}(m_S)$ | Spin-selective; faster for $|m_S|>0$ |
| Return from shelf | $k_\text{shelf}(m_S)$ | Spin-polarising: predominant return to $m_S=0$ |
| Non-radiative | $k_\text{nr}$ | Spin-blind; contributes to lifetime, not contrast |

Steady state is the null vector of $R$ normalised to $\sum_i P_i = 1$:

$$
R\,\vec{P}_\text{ss} = 0, \qquad \sum_i P_i = 1
$$

---

### 11.3 CW ODMR Contrast

PL in the **reference** state (spin polarised, far from MW resonance):

$$
\text{PL}_\text{off} = k_\text{rad} \sum_{n=N}^{2N-1} P_n^\text{(ss,off)}
$$

PL when a **saturating MW** drives transition $(m_{S,0} \leftrightarrow m_{S,1})$:

$$
\text{PL}_\text{on} = k_\text{rad} \sum_{n=N}^{2N-1} P_n^\text{(ss,on)}
$$

CW ODMR contrast:

$$
C = \frac{\text{PL}_\text{off} - \text{PL}_\text{on}}{\text{PL}_\text{off}}
$$

The sign convention is positive for dips (PL decreases on resonance, as for NV⁻ and VB⁻).

---

### 11.4 Pre-built Defect Parameters

| Defect | Spin | Host | $k_\text{rad}$ (MHz) | $\tau$ | Predicted $C$ |
|--------|------|------|---------------------|-------|---------------|
| NV⁻   | 1    | diamond | 77  | 13 ns  | ≈23 % |
| VB⁻   | 1    | hBN     | 294 | 3.4 ns | ≈2 %  |
| V_SiC | 1    | 4H-SiC  | 167 | 6 ns   | ≈15 % |
| P1    | 1/2  | diamond | 100 | —      | 0 %  (no ISC) |
| Cr/GaN| 3/2  | GaN     | 100 | —      | 0 %  (ISC unknown) |

Values from: Tetienne et al., NJP 14, 103033 (2012) [NV⁻]; Haykal et al.,
npj Quantum Inf. 8, 16 (2022) [VB⁻]; Christle et al., Nat. Mater. 14, 160 (2015) [V_SiC].

The contrast is auto-computed via `Defaults.get_contrast()` when `contrast=None`.

**References:** [12], [13], [14], [15].

---

## 12. References

[1] Gottscholl, A. et al. "Initialization and read-out of intrinsic spin defects in a van der Waals crystal at room temperature." *Nature Materials* **19**, 540–545 (2020). https://doi.org/10.1038/s41563-020-0619-6

[2] Dolde, F. et al. "Electric-field sensing using single diamond spins." *Nature Physics* **7**, 459–463 (2011). https://doi.org/10.1038/nphys1969

[3] Haykal, A. et al. "Decoherence of VB− spin defects in hexagonal boron nitride." *Nature Communications* **13**, 4347 (2022). https://doi.org/10.1038/s41467-022-31743-0

[4] Degen, C. L., Reinhard, F. & Cappellaro, P. "Quantum sensing." *Reviews of Modern Physics* **89**, 035002 (2017). https://doi.org/10.1103/RevModPhys.89.035002

[5] de Lange, G. et al. "Universal dynamical decoupling of a single solid-state spin from a spin bath." *Science* **330**, 60–63 (2010). https://doi.org/10.1126/science.1192739

[6] Yan, F. et al. "Rotating-frame relaxation as a noise spectrum analyser of a superconducting qubit undergoing driven evolution." *Nature Communications* **4**, 2337 (2013). https://doi.org/10.1038/ncomms3337

[7] Jackson, J. D. *Classical Electrodynamics*, 3rd ed. (Wiley, 1999), §2.9 — Method of Images for parallel conducting planes.

[8] Lima, E. A. & Weiss, B. P. "Obtaining vector magnetic field maps from single-component measurements of geological samples." *Journal of Geophysical Research* **114**, B06102 (2009). https://doi.org/10.1029/2008JB006006

[9] Thiel, L. et al. "Probing magnetism in 2D materials at the nanoscale with single-spin microscopy." *Science* **364**, 973–976 (2019). https://doi.org/10.1126/science.aav6926

[10] Felton, S. et al. "Hyperfine interaction in the ground state of the negatively charged nitrogen vacancy center in diamond." *Physical Review B* **79**, 075203 (2009). https://doi.org/10.1103/PhysRevB.79.075203

[11] Harris, R. K. et al. "NMR nomenclature: nuclear spin properties and conventions for chemical shifts." *Pure and Applied Chemistry* **73**, 1795–1818 (2001). https://doi.org/10.1351/pac200173111795

[12] Tetienne, J.-P. et al. "Magnetic-field-dependent photodynamics of single NV defects in diamond." *New Journal of Physics* **14**, 103033 (2012). https://doi.org/10.1088/1367-2630/14/10/103033

[13] Christle, D. J. et al. "Isolated electron spins in silicon carbide with millisecond coherence times." *Nature Materials* **14**, 160–163 (2015). https://doi.org/10.1038/nmat4144

[14] Robledo, L. et al. "Spin dynamics in the optical cycle of single nitrogen-vacancy centres in diamond." *New Journal of Physics* **13**, 025013 (2011). https://doi.org/10.1088/1367-2630/13/2/025013

[15] Haykal, A. et al. "Decoherence properties and optical lifetime of VB⁻ spin defects in hexagonal boron nitride." *npj Quantum Information* **8**, 16 (2022). https://doi.org/10.1038/s41534-022-00528-0
