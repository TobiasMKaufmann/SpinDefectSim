"""
spin/spectra.py — ODMR lineshapes and ensemble spectrum builders.

Functions
---------
lorentzian                  normalised Lorentzian L(f; f0, fwhm)
PL_model                    single-defect CW PL response
ensemble_transitions_from_Efields   diagonalise H for an array of E-fields
ensemble_odmr_spectrum       contrast-normalised inhomogeneous CW spectrum
"""
from __future__ import annotations

import numpy as np
from .hamiltonian import (
    SpinParams,
    odmr_hamiltonian_Hz,
    _local_frame_rotation,
    _odmr_hamiltonian_local_Hz,
    diagonalize_hamiltonian,
    extract_ms0_like_transitions_Hz,
)

# Backward-compatible alias
vb_spin_hamiltonian_Hz = odmr_hamiltonian_Hz


# ─────────────────────────────────────────────────────────────────────────────
#  Lineshape primitives
# ─────────────────────────────────────────────────────────────────────────────
def lorentzian(f: np.ndarray, f0: float, fwhm: float) -> np.ndarray:
    """
    Normalised Lorentzian with unit peak at f = f0.

    L(f) = (γ/2)² / [(f-f0)² + (γ/2)²]   where γ = fwhm.
    """
    gamma = fwhm / 2.0
    return (gamma ** 2) / ((np.asarray(f) - f0) ** 2 + gamma ** 2)


def PL_model(
    f_MW_Hz: np.ndarray,
    transitions_Hz,
    linewidth_Hz: float,
    contrast: float = 0.02,
) -> np.ndarray:
    """
    Single-defect toy ODMR: PL = 1 − contrast · (sum of Lorentzians) / N_trans.

    Parameters
    ----------
    f_MW_Hz       : microwave drive frequency sweep (Hz)
    transitions_Hz: sequence of transition frequencies (Hz)
    linewidth_Hz  : FWHM (Hz)
    contrast      : fractional PL reduction at resonance

    Returns
    -------
    pl : float array, same shape as f_MW_Hz
    """
    val = sum(lorentzian(f_MW_Hz, f0, linewidth_Hz) for f0 in transitions_Hz)
    return 1.0 - contrast * (val / max(len(list(transitions_Hz)), 1))


# ─────────────────────────────────────────────────────────────────────────────
#  Ensemble helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensemble_transitions_from_Efields(
    E_fields_Vpm: np.ndarray,
    spin_params: SpinParams,
    ms0_basis_index: int = None,
    quantization_axes: np.ndarray = None,
    B_extra_fields: np.ndarray = None,
) -> list:
    """
    Compute ODMR transition frequencies for an array of E-fields.

    Parameters
    ----------
    E_fields_Vpm    : shape (N, 3) array of lab-frame E-fields (V/m)
    spin_params     : SpinParams instance
    ms0_basis_index : basis index for |ms=0⟩ (default: spin_params.ms0_index)
    quantization_axes : (N, 3) array of per-defect quantization axes in the
                        lab frame.  When supplied, B and E are rotated
                        per-defect before the Hamiltonian is built.
                        If None, uses spin_params.quantization_axis for all.
    B_extra_fields  : optional shape (N, 3) array of per-defect stray B-fields
                      in the lab frame (T).  Added on top of sp.B_T + sp.B_extra_T.
                      Pass zeros (N, 3) for the reference branch.

    Returns
    -------
    transitions_list : list of length N, each a float array (Hz)
                       Length per defect: (2S) transitions from |ms=0⟩-like state
    """
    if ms0_basis_index is None:
        ms0_basis_index = getattr(spin_params, 'ms0_index', 1)

    E_fields_Vpm = np.asarray(E_fields_Vpm, dtype=float)
    N = len(E_fields_Vpm)

    # Uniform bias (same for all defects); per-defect stray-B is added per iteration
    B_bias = (np.asarray(spin_params.B_T, float)
              + np.asarray(spin_params.B_extra_T, float))

    if B_extra_fields is not None:
        B_extra_fields = np.asarray(B_extra_fields, dtype=float)
        if B_extra_fields.shape != (N, 3):
            raise ValueError(
                f"B_extra_fields must have shape ({N}, 3); got {B_extra_fields.shape}"
            )

    # Resolve per-defect quantization axes
    if quantization_axes is not None:
        axes = np.asarray(quantization_axes, float)
        norms = np.linalg.norm(axes, axis=1, keepdims=True)
        axes = axes / norms
    else:
        q0 = np.asarray(
            getattr(spin_params, 'quantization_axis', [0., 0., 1.]), float
        )
        q0 = q0 / np.linalg.norm(q0)
        axes = np.broadcast_to(q0, (N, 3))

    out = []
    for i, (E_vec, z_hat) in enumerate(zip(E_fields_Vpm, axes)):
        B_lab_i = B_bias if B_extra_fields is None else B_bias + B_extra_fields[i]
        if np.allclose(z_hat, [0., 0., 1.]):
            B_local, E_local = B_lab_i, E_vec
        else:
            R = _local_frame_rotation(z_hat)
            B_local = R @ B_lab_i
            E_local = R @ E_vec
        H = _odmr_hamiltonian_local_Hz(spin_params, B_local, E_local)
        evals, evecs = diagonalize_hamiltonian(H)
        freqs, _ = extract_ms0_like_transitions_Hz(evals, evecs, ms0_basis_index)
        out.append(freqs)
    return out


def ensemble_odmr_spectrum(
    f_axis: np.ndarray,
    transitions_list: list,
    fwhm: float,
    contrast: float = 0.10,
) -> np.ndarray:
    """
    Inhomogeneous ensemble CW ODMR spectrum.

    Each defect contributes equal weight.  The total lineshape is normalised so
    its peak equals 1, then multiplied by *contrast*.

    Parameters
    ----------
    f_axis           : 1-D frequency array (Hz)
    transitions_list : list length N, each entry shape-(2,) Hz array
    fwhm             : CW linewidth (Hz)  ≈ 1/(π T2*)
    contrast         : peak PL contrast (fraction)

    Returns
    -------
    pl : 1-D float array  (1 = no resonance, dip → 1 − contrast)
    """
    f_axis = np.asarray(f_axis, dtype=float)
    dip = np.zeros_like(f_axis)
    for trans in transitions_list:
        for f0 in trans:
            dip += lorentzian(f_axis, f0, fwhm)
    dip /= max(dip.max(), 1e-12)
    return 1.0 - contrast * dip
