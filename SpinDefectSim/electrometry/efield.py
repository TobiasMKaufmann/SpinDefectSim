"""
electrometry/efield.py — ElectricFieldBuilder: composes gate and disorder E-fields.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.constants import epsilon_0

from ..base.params import PhysicalParams, Defaults


def E_gate_bias(
    r_obs_xyz,
    E0=(0.0, 0.0, 0.0),
    grad=None,
) -> np.ndarray:
    """
    Uniform + linear-gradient gate-bias field at observation point.

    Parameters
    ----------
    r_obs_xyz : (x, y, z) in metres
    E0        : (Ex, Ey, Ez) uniform component (V/m)
    grad      : 2×2 [[dEx/dx, dEx/dy], [dEy/dx, dEy/dy]] (V/m²), optional

    Returns
    -------
    E : float64 array shape (3,), V/m
    """
    x, y, z = r_obs_xyz
    E = np.array(E0, dtype=float)
    if grad is not None:
        G = np.asarray(grad, dtype=float).reshape(2, 2)
        E[0] += G[0, 0] * x + G[0, 1] * y
        E[1] += G[1, 0] * x + G[1, 1] * y
    return E

def E_disorder_point_charges(
    r_obs_xyz,
    charges_xyzq: np.ndarray,
    *,
    epsilon_eff: float,
    screening_model,
    lambda_screen: float,
    d_gate: float,
    n_images: int,
    r_min: float = 0.2e-9,
) -> np.ndarray:
    """
    Electric field at r_obs_xyz from static disorder point charges.

    Parameters
    ----------
    charges_xyzq : shape (N_charges, 4) — each row is (x, y, z, q)

    Returns
    -------
    E : shape (3,) V/m
    """
    charges = np.asarray(charges_xyzq, dtype=float)
    if charges.size == 0:
        return np.zeros(3)
    x0, y0, z0 = map(float, r_obs_xyz)
    E = np.zeros(3, dtype=float)

    for xs, ys, zs, q in charges:
        rx, ry = x0 - xs, y0 - ys
        if screening_model in (None,):
            rz = z0 - zs
            r  = max(np.sqrt(rx**2 + ry**2 + rz**2), r_min)
            E += q * np.array([rx, ry, rz]) / r**3
        elif screening_model == "dual_gate":
            n     = np.arange(-int(n_images), int(n_images) + 1, dtype=int)
            signs = (-1.0) ** n
            rz    = z0 - (zs + 2.0 * n * float(d_gate))
            denom = np.maximum(np.sqrt(rx**2 + ry**2 + rz**2), r_min)
            d3    = denom**3
            E += q * np.array([
                np.sum(signs * rx / d3),
                np.sum(signs * ry / d3),
                np.sum(signs * rz / d3),
            ])
        else:
            # Yukawa or unknown — fallback to bare Coulomb
            rz = z0 - zs
            r  = max(np.sqrt(rx**2 + ry**2 + rz**2), r_min)
            E += q * np.array([rx, ry, rz]) / r**3

    pref = 1.0 / (4.0 * np.pi * epsilon_0 * float(epsilon_eff))
    return pref * E


def apply_dielectric_transmission(E: np.ndarray, eps_layer: float, eps_host: float) -> np.ndarray:
    """
    Quasi-static planar-interface transmission: η = 2ε_layer/(ε_layer + ε_host).
    """
    eta = 2.0 * float(eps_layer) / (float(eps_layer) + float(eps_host))
    return eta * np.asarray(E, dtype=float)


class ElectricFieldBuilder(PhysicalParams):
    """
    Composes gate-bias and disorder contributions to the total E-field at a VB⁻ defect.

    Usage
    -----
    >>> builder = ElectricFieldBuilder(defaults=Defaults())
    >>> E_tot, components = builder.total(
    ...     defect_xyz=(0, 0, 0.34e-9),
    ...     E0_gate=(0, 1e4, 0))
    """

    def __init__(self, defaults: Optional[Defaults] = None):
        super().__init__(defaults=defaults)

    def total(
        self,
        defect_xyz,
        *,
        E0_gate=(0.0, 0.0, 0.0),
        gate_grad=None,
        disorder_xyzq=None,
    ):
        """
        Total E-field at a VB⁻ defect from gate bias + disorder charges.

        Parameters
        ----------
        defect_xyz    : (x, y, z) defect position (m)
        E0_gate       : uniform gate field (V/m)
        gate_grad     : 2×2 field-gradient matrix (V/m²)
        disorder_xyzq : (N, 4) static charge array [x, y, z, q]

        Returns
        -------
        E_tot      : shape (3,) V/m
        components : dict with 'E_gate', 'E_dis'
        """
        d   = self.defaults
        obs = np.asarray(defect_xyz, dtype=float)

        E_gate = E_gate_bias(obs, E0=E0_gate, grad=gate_grad)

        if disorder_xyzq is None:
            disorder_xyzq = np.zeros((0, 4))
        E_dis = E_disorder_point_charges(
            obs, disorder_xyzq,
            epsilon_eff=d.epsilon_eff,
            screening_model=d.screening_model,
            lambda_screen=d.lambda_screen,
            d_gate=d.d_gate,
            n_images=d.n_images,
        )

        E_tot = E_gate + E_dis
        if (d.eps_layer is not None) and (d.eps_host is not None):
            E_tot = apply_dielectric_transmission(E_tot, d.eps_layer, d.eps_host)

        return E_tot, dict(E_gate=E_gate, E_dis=E_dis)
