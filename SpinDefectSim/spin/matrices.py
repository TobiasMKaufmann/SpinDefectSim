"""
spin/matrices.py — Spin operator matrices for arbitrary spin quantum number.

All matrices are complex128 and dimensionless (units of ℏ).
"""
from __future__ import annotations

import numpy as np


def spin_matrices(S: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spin operators Sx, Sy, Sz and identity I for arbitrary spin quantum number S.

    The basis is ordered as {|+S⟩, |+S−1⟩, …, |−S⟩} (descending Sz eigenvalues).
    Constructed via the standard raising/lowering operator algebra:

        S⁺|m⟩ = √[S(S+1) − m(m+1)] |m+1⟩,   Sx = (S⁺+S⁻)/2,   Sy = (S⁺−S⁻)/(2i)

    Parameters
    ----------
    S : spin quantum number (0.5, 1, 1.5, 2, …)

    Returns
    -------
    Sx, Sy, Sz, I : complex128 arrays of shape (2S+1, 2S+1)

    Examples
    --------
    >>> Sx, Sy, Sz, I = spin_matrices(1)    # spin-1, identical to spin_1_matrices()
    >>> Sx, Sy, Sz, I = spin_matrices(0.5)  # spin-1/2, identical to spin_half_matrices()
    >>> Sx, Sy, Sz, I = spin_matrices(1.5)  # spin-3/2, 4×4 matrices
    """
    S = float(S)
    if abs(2 * S - round(2 * S)) > 1e-9:
        raise ValueError(f"S must be an integer or half-integer, got {S}")
    dim = int(round(2 * S + 1))
    # m_vals[i] = S - i  (descending)
    m_vals = np.array([S - i for i in range(dim)])

    Sz = np.diag(m_vals).astype(complex)
    I  = np.eye(dim, dtype=complex)

    # S⁺[i-1, i] = sqrt(S(S+1) - m_vals[i] * (m_vals[i] + 1))  for i = 1..dim-1
    # (raises |m_vals[i]⟩ → |m_vals[i]+1⟩ = |m_vals[i-1]⟩)
    S_plus = np.zeros((dim, dim), dtype=complex)
    for i in range(1, dim):
        m = m_vals[i]
        S_plus[i - 1, i] = np.sqrt(S * (S + 1) - m * (m + 1))

    S_minus = S_plus.conj().T
    Sx = (S_plus + S_minus) / 2.0
    Sy = (S_plus - S_minus) / (2.0j)
    return Sx, Sy, Sz, I


def spin_1_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spin-1 operators Sx, Sy, Sz and identity I3 in the {|+1⟩, |0⟩, |−1⟩} basis.

    Returns
    -------
    Sx, Sy, Sz, I3 : complex128 arrays of shape (3, 3)
    """
    return spin_matrices(1)


def spin_half_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spin-½ operators Sx, Sy, Sz and identity I2 in the {|↑⟩, |↓⟩} basis.

    Returns
    -------
    Sx, Sy, Sz, I2 : complex128 arrays of shape (2, 2)
    """
    return spin_matrices(0.5)
