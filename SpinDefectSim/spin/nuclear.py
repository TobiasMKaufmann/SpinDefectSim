"""
spin/nuclear.py — Nuclear spin dataclass, standard isotope constants, and
hyperfine tensor helpers.

The hyperfine contribution to the spin Hamiltonian is (in Hz)::

    H_hf / h = S · A · I

where S is the electron spin operator vector, A is the 3×3 hyperfine coupling
tensor in the defect's local frame (z = electron spin quantization axis), and
I is the nuclear spin operator vector.  Multiple nuclei contribute additively.

For spins with I ≥ 1 a nuclear electric quadrupole term can be included::

    H_Q / h = P · (Iz² − I(I+1)/3 · 𝟙)

Convenience constructors
------------------------
axial_A_tensor(A_zz, A_perp)  →  diag([A_perp, A_perp, A_zz])
isotropic_A_tensor(A_iso)     →  A_iso * I₃

Standard isotope gyromagnetic ratios
-------------------------------------
GAMMA_14N   ¹⁴N   I = 1      +3.077 MHz/T
GAMMA_15N   ¹⁵N   I = 1/2    −4.316 MHz/T  (negative)
GAMMA_11B   ¹¹B   I = 3/2   +13.660 MHz/T
GAMMA_10B   ¹⁰B   I = 3      +4.575 MHz/T
GAMMA_13C   ¹³C   I = 1/2   +10.708 MHz/T
GAMMA_29Si  ²⁹Si  I = 1/2    −5.319 MHz/T  (negative)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


__all__ = [
    "NuclearSpin",
    "axial_A_tensor",
    "isotropic_A_tensor",
    # isotope gyromagnetic ratios (Hz/T)
    "GAMMA_14N",
    "GAMMA_15N",
    "GAMMA_11B",
    "GAMMA_10B",
    "GAMMA_13C",
    "GAMMA_29Si",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Nuclear gyromagnetic ratios  (Hz/T)
# ─────────────────────────────────────────────────────────────────────────────

GAMMA_14N:  float =  3.077e6    # ¹⁴N   I = 1
GAMMA_15N:  float = -4.316e6   # ¹⁵N   I = 1/2   (negative)
GAMMA_11B:  float = 13.660e6   # ¹¹B   I = 3/2
GAMMA_10B:  float =  4.575e6   # ¹⁰B   I = 3
GAMMA_13C:  float = 10.708e6   # ¹³C   I = 1/2
GAMMA_29Si: float = -5.319e6   # ²⁹Si  I = 1/2   (negative)


# ─────────────────────────────────────────────────────────────────────────────
#  Tensor builders
# ─────────────────────────────────────────────────────────────────────────────

def axial_A_tensor(A_zz_Hz: float, A_perp_Hz: float) -> np.ndarray:
    """
    Axially symmetric hyperfine tensor in diagonal form.

    Parameters
    ----------
    A_zz_Hz   : longitudinal component A∥ (Hz), along the defect z′-axis.
    A_perp_Hz : transverse component A⊥ (Hz), A_xx = A_yy = A_perp_Hz.

    Returns
    -------
    A : (3, 3) float64 ndarray  —  diag([A_perp, A_perp, A_zz]).
    """
    return np.diag([float(A_perp_Hz), float(A_perp_Hz), float(A_zz_Hz)])


def isotropic_A_tensor(A_iso_Hz: float) -> np.ndarray:
    """
    Isotropic (Fermi contact) hyperfine tensor A_iso * I₃.

    Parameters
    ----------
    A_iso_Hz : isotropic coupling constant (Hz).

    Returns
    -------
    A : (3, 3) float64 ndarray.
    """
    return float(A_iso_Hz) * np.eye(3)


# ─────────────────────────────────────────────────────────────────────────────
#  NuclearSpin dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NuclearSpin:
    """
    A single nuclear spin coupled to the electron spin via a hyperfine tensor.

    The hyperfine Hamiltonian for this nucleus (in Hz) is::

        H_hf / h = S · A · I

    and, for I ≥ 1, the nuclear electric quadrupole contribution is::

        H_Q / h = P · (Iz² − I(I+1)/3 · 𝟙)

    where all quantities are in the defect's local frame (z = electron spin
    quantization axis).

    Attributes
    ----------
    spin          : nuclear spin quantum number I (0.5, 1, 1.5, …).
    A_tensor_Hz   : (3, 3) hyperfine tensor in Hz in the defect local frame.
                    Build with :func:`axial_A_tensor` or
                    :func:`isotropic_A_tensor`, or supply a full matrix for
                    lower-symmetry sites.
    gamma_Hz_T    : nuclear gyromagnetic ratio γ_n in Hz/T.
                    Use the module-level GAMMA_* constants for standard isotopes.
    label         : human-readable isotope label, e.g. ``"14N"``, ``"11B"``.
    quadrupole_Hz : nuclear electric quadrupole coupling P (Hz) for I ≥ 1.
                    Zero for I = 1/2. NV ¹⁴N: P ≈ −4.95 MHz.

    Examples
    --------
    NV⁻ on-site ¹⁴N::

        from SpinDefectSim.spin.nuclear import NuclearSpin, axial_A_tensor, GAMMA_14N
        N14_NV = NuclearSpin(
            spin=1,
            A_tensor_Hz=axial_A_tensor(A_zz_Hz=-2.14e6, A_perp_Hz=-2.70e6),
            gamma_Hz_T=GAMMA_14N,
            label="14N",
            quadrupole_Hz=-4.95e6,
        )

    Custom ¹³C nearest-neighbour in diamond::

        C13 = NuclearSpin(
            spin=0.5,
            A_tensor_Hz=isotropic_A_tensor(13.0e6),  # ~ 13 MHz Fermi contact
            gamma_Hz_T=GAMMA_13C,
            label="13C",
        )
    """

    spin: float
    A_tensor_Hz: np.ndarray       # shape (3, 3) in defect local frame
    gamma_Hz_T: float
    label: str = ""
    quadrupole_Hz: float = 0.0

    def __post_init__(self) -> None:
        self.A_tensor_Hz = np.asarray(self.A_tensor_Hz, dtype=float)
        if self.A_tensor_Hz.shape != (3, 3):
            raise ValueError(
                f"NuclearSpin.A_tensor_Hz must be shape (3, 3), "
                f"got {self.A_tensor_Hz.shape}"
            )
        if abs(2 * self.spin - round(2 * self.spin)) > 1e-9:
            raise ValueError(
                f"NuclearSpin.spin must be integer or half-integer, got {self.spin}"
            )

    def __repr__(self) -> str:
        diag = np.diag(self.A_tensor_Hz) / 1e6
        return (
            f"NuclearSpin(label={self.label!r}, I={self.spin}, "
            f"A_diag=[{diag[0]:.3f}, {diag[1]:.3f}, {diag[2]:.3f}] MHz, "
            f"P={self.quadrupole_Hz / 1e6:.3f} MHz)"
        )
