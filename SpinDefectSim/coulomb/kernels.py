"""
coulomb/kernels.py — Screened Coulomb kernels G(ρ) and G(ρ, z).

Three screening models are supported:

  None         bare Coulomb  1/r
  'yukawa'     exp(−r/λ) / r
  'dual_gate'  image-charge sum representing two parallel metallic gates

Public API
----------
G_rho(screening_model, rho, ...)          in-plane kernel
G_rz (screening_model, rho, z_obs, ...)   3-D kernel at height z_obs
"""
from __future__ import annotations

from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Private single-model implementations  (rho only)
# ─────────────────────────────────────────────────────────────────────────────
def _G_rho_coulomb(rho, r_min=1e-12):
    rho = np.maximum(rho, r_min)
    return 1.0 / rho


def _G_rho_yukawa(rho, lambda_screen=5e-9, r_min=1e-12):
    rho = np.maximum(rho, r_min)
    return np.exp(-rho / float(lambda_screen)) / rho


def _G_rho_dual_gate_fast(rho, d_gate=5e-9, n_images=50, r_min=1e-12):
    rho = np.asarray(rho, dtype=float)
    n   = np.arange(-int(n_images), int(n_images) + 1, dtype=int)
    shifts2 = (2.0 * n * float(d_gate)) ** 2
    signs   = (-1.0) ** n
    denom   = np.sqrt(rho[..., None] ** 2 + shifts2[None, :])
    denom   = np.maximum(denom, float(r_min))
    return np.sum(signs[None, :] / denom, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Private single-model implementations  (rho + z)
# ─────────────────────────────────────────────────────────────────────────────
def _G_rz_coulomb(rho, z, r_min=1e-12):
    s = np.maximum(np.sqrt(rho ** 2 + z ** 2), r_min)
    return 1.0 / s


def _G_rz_yukawa(rho, z, lambda_screen=5e-9, r_min=1e-12):
    s = np.maximum(np.sqrt(rho ** 2 + z ** 2), r_min)
    return np.exp(-s / float(lambda_screen)) / s


def _G_rz_dual_gate_fast(rho, z_obs, d_gate=5e-9, n_images=50, r_min=1e-12):
    rho = np.asarray(rho, dtype=float)
    n    = np.arange(-int(n_images), int(n_images) + 1, dtype=int)
    signs = (-1.0) ** n
    dz    = float(z_obs) - 2.0 * n * float(d_gate)
    denom = np.sqrt(rho[..., None] ** 2 + dz[None, :] ** 2)
    denom = np.maximum(denom, float(r_min))
    return np.sum(signs[None, :] / denom, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Public dispatcher functions
# ─────────────────────────────────────────────────────────────────────────────
def G_rho(
    screening_model: Optional[str],
    rho: np.ndarray,
    *,
    lambda_screen: float = 5e-9,
    d_gate: float = 5e-9,
    n_images: int = 50,
    r_min: float = 1e-12,
) -> np.ndarray:
    """
    In-plane screened 2-D Coulomb kernel G(ρ).

    Parameters
    ----------
    screening_model : None | 'yukawa' | 'dual_gate'
    rho             : in-plane distance (m), any broadcastable array
    lambda_screen   : Yukawa screening length (m)
    d_gate          : half-separation between gates (m)
    n_images        : truncation of image sum
    r_min           : regularisation floor (m)

    Returns
    -------
    G : same shape as rho
    """
    if screening_model is None:
        return _G_rho_coulomb(rho, r_min=r_min)
    if screening_model == "yukawa":
        return _G_rho_yukawa(rho, lambda_screen=lambda_screen, r_min=r_min)
    if screening_model == "dual_gate":
        return _G_rho_dual_gate_fast(rho, d_gate=d_gate, n_images=n_images, r_min=r_min)
    raise ValueError(f"Unknown screening_model: {screening_model!r}")


def G_rz(
    screening_model: Optional[str],
    rho: np.ndarray,
    z_obs: float,
    *,
    lambda_screen: float = 5e-9,
    d_gate: float = 5e-9,
    n_images: int = 50,
    r_min: float = 1e-12,
) -> np.ndarray:
    """
    3-D screened Coulomb kernel G(ρ, z_obs) for source in-plane at z=0,
    observer at height z_obs.

    Parameters
    ----------
    screening_model : None | 'yukawa' | 'dual_gate'
    rho             : in-plane separation (m)
    z_obs           : observer height (m)

    Returns
    -------
    G : same shape as rho
    """
    if screening_model is None:
        return _G_rz_coulomb(rho, z_obs, r_min=r_min)
    if screening_model == "yukawa":
        return _G_rz_yukawa(rho, z_obs, lambda_screen=lambda_screen, r_min=r_min)
    if screening_model == "dual_gate":
        return _G_rz_dual_gate_fast(rho, z_obs, d_gate=d_gate, n_images=n_images, r_min=r_min)
    raise ValueError(f"Unknown screening_model: {screening_model!r}")
