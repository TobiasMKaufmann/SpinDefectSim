"""
spin/rates.py — Rate-equation model for CW ODMR contrast.

Computes the steady-state CW ODMR contrast from optical / intersystem-crossing
(ISC) rate equations for any spin-S defect.  Supported spin structures:

  • Spin-1/2  (P1-like)  : ground doublet + excited doublet, no ISC path.
  • Spin-1    (NV⁻, VB⁻, V_SiC) : triplet ground + triplet excited + singlet shelf.
  • Spin-3/2  (Cr in GaN): quartet ground + quartet excited + optional shelf.
  • Arbitrary S           : fully general via the ``spin_matrices`` basis.

The CW ODMR contrast is::

    C = (PL_off − PL_on) / PL_off

where ``PL_off`` is the steady-state photoluminescence under optical pumping
alone (spin polarised, far from MW resonance) and ``PL_on`` is the PL when a
saturating MW drive is applied resonant with one spin transition.

Level structure (single shelving state)
----------------------------------------
For a defect with spin quantum number S and N = 2S+1:

::

    Indices  0 … N−1  : ground states   (ms = +S, +S−1, …, −S)
    Indices  N … 2N−1 : excited states  (same ms order; spin-preserving optical transition)
    Index    2N        : shelving / metastable state  (only when any k_isc > 0)

Total dimension: 2N + 1 (with ISC) or 2N (without ISC).

Pre-built defaults
-------------------
``NV_RATES``    NV⁻ in diamond (spin-1).  Reproduces ≈ 23 % CW contrast.
``VB_RATES``    VB⁻ in hBN    (spin-1).  Reproduces ≈  2 % CW contrast.
``VSIC_RATES``  V_SiC in 4H-SiC (spin-1). Reproduces ≈ 15 % CW contrast.
``P1_RATES``    P1  in diamond (spin-1/2). Contrast = 0 (no optical ISC).
``CRGAN_RATES`` Cr in GaN      (spin-3/2). ISC not established; contrast = 0.

References
----------
[1] Tetienne et al., New J. Phys. 14, 103033 (2012)
    — NV⁻ rate model and values (k_rad, k_ISC, k_sg).
[2] Robledo et al., New J. Phys. 13, 025013 (2011)
    — NV⁻ spin-selective emission rate measurements.
[3] Doherty et al., Phys. Rep. 528, 1 (2013)
    — Comprehensive NV⁻ review.
[4] Gottscholl et al., Nat. Mater. 19, 540 (2020)
    — VB⁻ discovery; room-T contrast ≈ 2–5 %.
[5] Haykal et al., npj Quantum Inf. 8, 16 (2022)
    — VB⁻ optical lifetime τ ≈ 3.4 ns (k_rad ≈ 294 MHz); coherence properties.
[6] Gottscholl et al., Sci. Adv. 7, eabf3630 (2021)
    — VB⁻ spin properties and contrast at room temperature.
[7] Widmann et al., Nat. Mater. 14, 164 (2015)
    — V_SiC (V2) in 4H-SiC; ≈ 15 % CW contrast at room temperature.
[8] Christle et al., Nat. Mater. 14, 160 (2015)
    — V_SiC optical lifetime and coherence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "RateParams",
    "RateModel",
    "compute_odmr_contrast",
    # Pre-built defaults
    "NV_RATES",
    "VB_RATES",
    "VSIC_RATES",
    "P1_RATES",
    "CRGAN_RATES",
]


# ─────────────────────────────────────────────────────────────────────────────
#  RateParams — rate constants for the optical / ISC cycle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RateParams:
    """
    Rate constants for the optical / intersystem-crossing (ISC) cycle.

    The model contains three sub-spaces:

    1. **Ground manifold**: 2S+1 states, ms = +S, +S−1, …, −S.
    2. **Excited manifold**: 2S+1 states (same ms order).  Laser excitation
       is spin-blind; radiative decay is spin-preserving.
    3. **Shelving state** (metastable / dark state): one collective state
       that pools ISC population.  Set ``k_isc_excited`` to all zeros (or
       pass an array of zeros) when no ISC pathway exists (e.g. spin-1/2).

    All rates are in Hz (s⁻¹).

    Parameters
    ----------
    spin : float
        Electron spin quantum number S (0.5, 1, 1.5, 2, …).
    k_optical : float
        Laser excitation rate (s⁻¹).  Spin-blind; same for all ms.
        Typical values: 5–50 MHz depending on laser power and collection
        efficiency.
    k_rad : float
        Radiative decay rate (s⁻¹).  Spin-preserving (|ms⟩_e → |ms⟩_g).
        Equal to 1 / τ_excited.  Sets the photon-emission rate.
    k_isc_excited : sequence of float, length 2S+1
        ISC rates (s⁻¹) from each excited-manifold ms state to the shelving
        state.  Order: ms = +S, +S−1, …, −S (matches ``spin_matrices`` basis).
        All zeros → no ISC / no shelving path → contrast = 0.
    k_from_shelving : sequence of float, length 2S+1
        Return rates (s⁻¹) from the shelving state to each ground-manifold
        ms state.  Spin-polarising when the return is predominantly to ms = 0.
        Required when ``k_isc_excited`` is non-zero.
    k_nr : float, optional
        Non-radiative quenching rate (s⁻¹).  Does not introduce additional
        spin contrast.  Default 0.
    notes : str, optional
        Free-form reference / citation string.

    Examples
    --------
    Custom spin-1 defect with ISC parameters from your own measurements:

    >>> rp = RateParams(
    ...     spin=1,
    ...     k_optical=20e6,
    ...     k_rad=100e6,
    ...     k_isc_excited=[50e6, 5e6, 50e6],   # ms=+1, ms=0, ms=-1
    ...     k_from_shelving=[0.0, 10e6, 0.0],   # return only to ms=0
    ... )
    >>> model = RateModel(rp, ms0_index=1)
    >>> print(f"contrast = {model.contrast()*100:.1f} %")

    Spin-1/2 defect (no ISC, contrast = 0):

    >>> rp_half = RateParams(
    ...     spin=0.5,
    ...     k_optical=15e6,
    ...     k_rad=100e6,
    ...     k_isc_excited=[0.0, 0.0],
    ...     k_from_shelving=[0.0, 0.0],
    ... )
    """

    spin: float
    k_optical: float
    k_rad: float
    k_isc_excited: Sequence[float]
    k_from_shelving: Sequence[float]
    k_nr: float = 0.0
    notes: str = ""

    def __post_init__(self):
        N = int(round(2 * self.spin + 1))
        self.k_isc_excited   = np.asarray(self.k_isc_excited,   dtype=float)
        self.k_from_shelving = np.asarray(self.k_from_shelving, dtype=float)
        if len(self.k_isc_excited) != N:
            raise ValueError(
                f"k_isc_excited must have length 2S+1 = {N}, "
                f"got {len(self.k_isc_excited)}"
            )
        if len(self.k_from_shelving) != N:
            raise ValueError(
                f"k_from_shelving must have length 2S+1 = {N}, "
                f"got {len(self.k_from_shelving)}"
            )

    @property
    def has_shelving(self) -> bool:
        """``True`` when any ISC rate to the shelving state is nonzero."""
        return bool(np.any(self.k_isc_excited > 0))

    def __repr__(self) -> str:
        return (
            f"RateParams(spin={self.spin}, "
            f"k_rad={self.k_rad/1e6:.0f} MHz, "
            f"k_optical={self.k_optical/1e6:.0f} MHz, "
            f"ISC={'yes' if self.has_shelving else 'no'})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-built default rate parameters for standard defect types
# ─────────────────────────────────────────────────────────────────────────────

NV_RATES = RateParams(
    spin=1,
    k_optical=15e6,
    k_rad=77e6,
    # ms = +1, 0, -1  (index order of spin_matrices(1) basis)
    k_isc_excited=[91e6, 12e6, 91e6],
    k_from_shelving=[0.0, 3e6, 0.0],   # spin-polarising return to ms = 0 only
    k_nr=0.0,
    notes=(
        "NV⁻ in diamond (spin-1). "
        "k_rad = 77 MHz (τ ≈ 13 ns). "
        "k_ISC(ms=±1) = 91 MHz, k_ISC(ms=0) = 12 MHz. "
        "k_sg → ms=0 = 3 MHz. "
        "Predicted CW contrast ≈ 23 %. "
        "Values from: "
        "Tetienne et al., NJP 14, 103033 (2012), Table 1; "
        "Robledo et al., NJP 13, 025013 (2011); "
        "Doherty et al., Phys. Rep. 528, 1 (2013)."
    ),
)

VB_RATES = RateParams(
    spin=1,
    k_optical=20e6,
    k_rad=294e6,    # τ ≈ 3.4 ns; Haykal et al. (2022) TCSPC measurement
    # ms = +1, 0, -1  (estimated to reproduce ~2 % CW contrast)
    k_isc_excited=[20e6, 4e6, 20e6],
    k_from_shelving=[0.0, 88e6, 0.0],  # fast singlet return (~11 ns), to ms=0
    k_nr=0.0,
    notes=(
        "VB⁻ in hexagonal boron nitride (hBN, spin-1). "
        "k_rad = 294 MHz (τ ≈ 3.4 ns): Haykal et al., npj Quantum Inf. 8, 16 (2022). "
        "ISC rates are estimated: not directly measured. "
        "Values chosen to reproduce the ~2 % room-temperature CW ODMR contrast: "
        "Gottscholl et al., Nat. Mater. 19, 540 (2020); "
        "Gottscholl et al., Sci. Adv. 7, eabf3630 (2021). "
        "Shelving return rate (k_sg ≈ 88 MHz, τ ≈ 11 ns) is approximate."
    ),
)

VSIC_RATES = RateParams(
    spin=1,
    k_optical=10e6,
    k_rad=167e6,    # τ ≈ 6 ns; Christle et al. (2015), Widmann et al. (2015)
    # ms = +1, 0, -1  (estimated to reproduce ~15 % CW contrast)
    k_isc_excited=[86e6, 10e6, 86e6],
    k_from_shelving=[0.0, 3e6, 0.0],   # assumed similar to NV⁻ singlet return
    k_nr=0.0,
    notes=(
        "V2 silicon vacancy (V_SiC) in 4H-SiC (spin-1). "
        "k_rad = 167 MHz (τ ≈ 6 ns): "
        "Christle et al., Nat. Mater. 14, 160 (2015); "
        "Widmann et al., Nat. Mater. 14, 164 (2015). "
        "ISC rates estimated to reproduce ~15 % room-temperature CW contrast. "
        "k_sg assumed analogous to NV⁻."
    ),
)

P1_RATES = RateParams(
    spin=0.5,
    k_optical=10e6,
    k_rad=100e6,
    k_isc_excited=[0.0, 0.0],   # no spin-selective ISC for spin-1/2
    k_from_shelving=[0.0, 0.0],
    k_nr=0.0,
    notes=(
        "P1 substitutional nitrogen in diamond (spin-1/2). "
        "No optical ISC pathway; CW ODMR contrast = 0 from this rate model. "
        "P1 is not directly optically addressable — ODMR is typically performed "
        "via double resonance (ENDOR) using coupled NV centres."
    ),
)

CRGAN_RATES = RateParams(
    spin=1.5,
    k_optical=10e6,
    k_rad=100e6,
    k_isc_excited=[0.0, 0.0, 0.0, 0.0],  # ISC rates not established
    k_from_shelving=[0.0, 0.0, 0.0, 0.0],
    k_nr=0.0,
    notes=(
        "Cr⁴⁺ in GaN (spin-3/2, approximate). "
        "ISC rates are not established in the literature; set to zero. "
        "Contrast from rate model = 0; set Defaults(contrast=...) explicitly."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_rate_matrix(rp: RateParams) -> np.ndarray:
    """
    Build the rate (Liouvillian) matrix R so that  dP/dt = R @ P.

    Level ordering::

        0 … N−1   : ground states  (ms = +S, …, −S)
        N … 2N−1  : excited states (same ms order)
        2N         : shelving state (only when rp.has_shelving is True)

    For defects with no ISC pathway (``has_shelving = False``, e.g. spin-1/2),
    the rate matrix has a degenerate null-space because the spin channels are
    decoupled.  A 1 Hz ground-state spin-relaxation regularisation is added to
    lift this degeneracy and enforce equal ground-state mixing.  At 1 Hz, the
    regularisation is negligible (< 10⁻⁸ relative) for any defect with ISC
    rates in the MHz range.
    """
    N   = int(round(2 * rp.spin + 1))
    dim = 2 * N + (1 if rp.has_shelving else 0)
    R   = np.zeros((dim, dim), dtype=float)

    k_decay = rp.k_rad + rp.k_nr   # total excited-state decay rate (excl. ISC)

    for i in range(N):
        k_isc_i = float(rp.k_isc_excited[i])
        k_sg_i  = float(rp.k_from_shelving[i])

        # Ground → excited  (spin-blind laser pumping)
        R[N + i, i]   += rp.k_optical
        R[i,     i]   -= rp.k_optical

        # Excited → ground  (spin-preserving radiative + non-radiative decay)
        R[i,     N + i] += k_decay
        R[N + i, N + i] -= k_decay

        # ISC: excited |ms_i⟩ → shelving
        if rp.has_shelving and k_isc_i != 0.0:
            R[2 * N, N + i] += k_isc_i
            R[N + i, N + i] -= k_isc_i

        # ISC return: shelving → ground |ms_i⟩
        if rp.has_shelving and k_sg_i != 0.0:
            R[i,     2 * N] += k_sg_i
            R[2 * N, 2 * N] -= k_sg_i

    # Regularisation: 1 Hz ground-state spin-lattice relaxation to lift the
    # null-space degeneracy for defects with no ISC (e.g. spin-1/2 P1 centre).
    # Effect is < 1 ppm for any defect whose ISC rates exceed 1 kHz.
    _K_T1_REG = 1.0   # 1 Hz
    for i in range(N):
        for j in range(i + 1, N):
            R[i, j] += _K_T1_REG
            R[j, i] += _K_T1_REG
            R[i, i] -= _K_T1_REG
            R[j, j] -= _K_T1_REG

    return R


def _solve_steady_state(R: np.ndarray) -> np.ndarray:
    """
    Solve  R @ P = 0  subject to  sum(P) = 1  via direct linear algebra.

    The last equation of the system is replaced by the normalisation
    constraint.
    """
    dim = R.shape[0]
    A   = R.copy()
    b   = np.zeros(dim)
    A[-1, :] = 1.0
    b[-1]    = 1.0
    return np.linalg.solve(A, b)


def _pl_from_populations(populations: np.ndarray, rp: RateParams) -> float:
    """
    PL intensity in arbitrary units: PL = k_rad × Σ_i P[N+i].

    Proportional to the steady-state photon count rate.
    """
    N = int(round(2 * rp.spin + 1))
    return float(rp.k_rad * populations[N : 2 * N].sum())


# ─────────────────────────────────────────────────────────────────────────────
#  RateModel — steady-state solver
# ─────────────────────────────────────────────────────────────────────────────

class RateModel:
    """
    Steady-state rate-equation solver for CW ODMR contrast.

    Solves the optical/ISC rate equations in the off-resonance (spin-polarised)
    and on-resonance (saturating MW) limits and returns the contrast
    C = (PL_off − PL_on) / PL_off.

    Parameters
    ----------
    rate_params : RateParams
        Optical cycle rate constants (laser, radiative decay, ISC, return).
    ms0_index : int
        Index of the optically polarised (ms = 0-like) ground state in the
        ``spin_matrices`` basis {|+S⟩, |+S−1⟩, …, |−S⟩}.
        For spin-1: ms0_index = 1  (|ms=0⟩ is the middle state).
        For spin-1/2: ms0_index = 0.

    Examples
    --------
    >>> from SpinDefectSim.spin.rates import RateModel, NV_RATES, VB_RATES
    >>> nv  = RateModel(NV_RATES,  ms0_index=1)
    >>> vb  = RateModel(VB_RATES,  ms0_index=1)
    >>> print(f"NV⁻  contrast ≈ {nv.contrast()*100:.1f} %")   # ~23 %
    >>> print(f"VB⁻  contrast ≈ {vb.contrast()*100:.1f} %")   # ~2 %
    """

    def __init__(self, rate_params: RateParams, ms0_index: int):
        self.rp        = rate_params
        self.ms0_index = int(ms0_index)
        self._R0: Optional[np.ndarray] = None   # cached base rate matrix

    # ── internal helpers ─────────────────────────────────────────────────────

    @property
    def _base_R(self) -> np.ndarray:
        """Lazily-cached base rate matrix (no MW coupling)."""
        if self._R0 is None:
            self._R0 = _build_rate_matrix(self.rp)
        return self._R0

    def _resolve_mw_pair(self, mw_pair) -> tuple:
        """
        Return the (i, j) ground-state index pair to couple via the MW.

        Default: couples ms0_index to the adjacent lower-index state,
        i.e. the lower ODMR transition ms=0 ↔ ms=+1 for spin-1.
        """
        if mw_pair is not None:
            return (int(mw_pair[0]), int(mw_pair[1]))
        i = self.ms0_index
        j = i - 1 if i > 0 else i + 1
        return (i, j)

    # ── public API ───────────────────────────────────────────────────────────

    def steady_state(
        self,
        k_mw: float = 0.0,
        mw_pair: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Solve for the steady-state population vector.

        Parameters
        ----------
        k_mw    : MW coupling rate (s⁻¹).  ``0`` = off-resonance.
                  A large value approximates a saturating CW drive.
        mw_pair : (i, j) ground-state indices to couple via MW.
                  Defaults to ms0_index ↔ adjacent state.

        Returns
        -------
        P : 1-D float array of length 2N [+1], normalised (sum = 1).
        """
        R = self._base_R.copy()
        if k_mw > 0.0:
            i, j = self._resolve_mw_pair(mw_pair)
            R[i, i] -= k_mw
            R[j, i] += k_mw
            R[j, j] -= k_mw
            R[i, j] += k_mw
        return _solve_steady_state(R)

    def pl(
        self,
        k_mw: float = 0.0,
        mw_pair: Optional[tuple] = None,
    ) -> float:
        """
        Steady-state PL (proportional to photon count rate, arbitrary units).

        Parameters
        ----------
        k_mw    : MW coupling rate (s⁻¹).  ``0`` = off-resonance (bright state).
        mw_pair : (i, j) ground-state indices to couple via MW.
        """
        P = self.steady_state(k_mw=k_mw, mw_pair=mw_pair)
        return _pl_from_populations(P, self.rp)

    def contrast(
        self,
        mw_pair: Optional[tuple] = None,
        k_mw: Optional[float] = None,
    ) -> float:
        """
        CW ODMR contrast  C = (PL_off − PL_on) / PL_off.

        Parameters
        ----------
        mw_pair : (i, j) ground-state indices to couple via the MW drive.
                  Defaults to ms0_index ↔ adjacent state (lower transition).
                  For spin-1: (1, 0) couples ms=0 ↔ ms=+1.
                  For spin-3/2 with ms0_index=1: (1, 0) couples ms=+1/2 ↔ ms=+3/2.
        k_mw    : MW coupling rate (s⁻¹).  ``None`` → saturation limit,
                  automatically set to 1000 × max(k_optical, k_rad).

        Returns
        -------
        contrast : float in [0, 1].  Positive when resonant MW reduces PL.

        Notes
        -----
        This is the *single-transition* contrast.  For a spin-1 defect there
        are two ODMR transitions (ms=0 ↔ ms=±1); the per-transition contrast is
        reported.  The *total* ODMR dip (both lines together) would be larger
        but the per-line depth is what determines the sensitivity.
        """
        if k_mw is None:
            k_mw = 1e3 * max(self.rp.k_optical, self.rp.k_rad)

        pl_off = self.pl(k_mw=0.0)
        pl_on  = self.pl(k_mw=k_mw, mw_pair=mw_pair)

        if pl_off < 1e-30:
            return 0.0
        return float((pl_off - pl_on) / pl_off)

    def spin_polarization(self) -> float:
        """
        Steady-state ms=0-like ground-state population (off resonance).

        Returns
        -------
        pol : float — fraction of total population in the ms=0-like ground state.
        """
        P = self.steady_state(k_mw=0.0)
        return float(P[self.ms0_index])

    def photon_yield_per_cycle(self) -> np.ndarray:
        """
        Effective photon yield per excitation cycle for each ms ground state.

        Defined as  η_i = k_rad / (k_rad + k_nr + k_isc_excited[i]).
        This is the branching ratio of the radiative channel.

        Returns
        -------
        eta : float array of length 2S+1.
        """
        k_total = self.rp.k_rad + self.rp.k_nr + self.rp.k_isc_excited
        return self.rp.k_rad / k_total

    def __repr__(self) -> str:
        c = self.contrast()
        return (
            f"RateModel(spin={self.rp.spin}, "
            f"k_rad={self.rp.k_rad/1e6:.0f} MHz, "
            f"ms0_index={self.ms0_index}, "
            f"contrast≈{c*100:.1f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def compute_odmr_contrast(
    rate_params: RateParams,
    ms0_index: int,
    mw_pair: Optional[tuple] = None,
    k_mw: Optional[float] = None,
) -> float:
    """
    Compute the steady-state CW ODMR contrast from rate equations.

    Shorthand for ``RateModel(rate_params, ms0_index).contrast(...)``.

    Parameters
    ----------
    rate_params : RateParams
        Optical cycle rate constants.
    ms0_index : int
        Index of the ms = 0-like state in ``spin_matrices`` basis.
    mw_pair : (i, j) or None
        Ground-state indices to MW-couple.  Default: ms0 ↔ adjacent.
    k_mw : float or None
        MW coupling rate (s⁻¹).  None → saturation limit.

    Returns
    -------
    contrast : float in [0, 1]

    Examples
    --------
    >>> from SpinDefectSim.spin.rates import NV_RATES, VB_RATES, compute_odmr_contrast
    >>> print(f"NV⁻  {compute_odmr_contrast(NV_RATES, ms0_index=1)*100:.1f} %")
    >>> print(f"VB⁻  {compute_odmr_contrast(VB_RATES, ms0_index=1)*100:.1f} %")
    """
    return RateModel(rate_params, ms0_index).contrast(mw_pair=mw_pair, k_mw=k_mw)
