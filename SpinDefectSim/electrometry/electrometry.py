"""
electrometry/electrometry.py — ElectrometryExperiment: charges → E → ODMR.

High-level workflow
-------------------
1. User supplies a set of disorder charges ``charges_xyzq`` (array with columns
   [x, y, z, q] in SI units) and an optional uniform gate field ``E0_gate``.
2. :class:`ElectrometryExperiment` computes the total E-field at any
   observation point (x_obs, y_obs) via the screened-Coulomb model in
   :mod:`SpinDefectSim.sensing.efield`.
3. The E-field is fed into the spin Hamiltonian (with the bias B but zero
   stray B) to produce CW ODMR transition frequencies — standard electrometry.

This class is the direct symmetric counterpart of
:class:`~SpinDefectSim.magnetometry.MagnetometryExperiment`.

Usage example
-------------
>>> import numpy as np
>>> from SpinDefectSim.electrometry import ElectrometryExperiment
>>> from SpinDefectSim.base.params import Defaults
>>> from scipy.constants import e as e_charge

Single charge:

>>> charges = np.array([[50e-9, 0.0, 0.0, e_charge]])
>>> exp = ElectrometryExperiment(charges, Defaults(), z_defect=0.34e-9)
>>> f1, f2 = exp.transition_frequencies(0.0, 0.0)
>>> dfreq_map = exp.frequency_shift_map(
...     np.linspace(-300e-9, 300e-9, 40),
...     np.linspace(-300e-9, 300e-9, 40),
... )

Uniform gate bias (no disorder):

>>> exp_gate = ElectrometryExperiment(
...     charges_xyzq=None,
...     defaults=Defaults(),
...     z_defect=0.34e-9,
...     E0_gate=(0, 0, 1e4),
... )
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from ..base.params import Defaults, PhysicalParams
from ..spin.hamiltonian import (
    SpinParams,
    odmr_hamiltonian_Hz,
    diagonalize_hamiltonian,
    extract_ms0_like_transitions_Hz,
)
from ..spin.spectra import ensemble_odmr_spectrum
from .efield import E_gate_bias, E_disorder_point_charges, apply_dielectric_transmission

__all__ = ["ElectrometryExperiment"]


class ElectrometryExperiment(PhysicalParams):
    """
    Compute ODMR signals at arbitrary observation points from a static
    charge distribution and/or a gate bias field.

    The electric field at each observation point is computed from the
    supplied charge positions using the screened-Coulomb model from
    :mod:`SpinDefectSim.sensing.efield`, then fed into the spin Hamiltonian.
    No magnetic stray field is assumed (pure electrometry regime).

    Parameters
    ----------
    charges_xyzq : ndarray shape (M, 4) *or* None
        Static disorder charges.  Each row is ``[x, y, z, q]`` in SI units
        (metres and coulombs).  Pass ``None`` for gate-only experiments.
    defaults     : Defaults  physical parameter set
    z_defect     : float [m]  height of defects above the charge plane.
                   Defaults to ``defaults.z_defect``.
    E0_gate      : array-like (3,) [V/m]  uniform background gate field.
                   Added on top of the Coulomb contribution.
    gate_grad    : ndarray (2, 2) [V/m²]  optional linear field gradient.
    epsilon_eff  : float  effective relative permittivity (overrides defaults).
    screening_model : str or None  Coulomb screening model
        (``None``, ``"yukawa"``, ``"dual_gate"``).  Overrides defaults.
    lambda_screen : float [m]  Yukawa screening length (overrides defaults).
    d_gate       : float [m]   gate distance for dual-gate model (overrides defaults).
    n_images     : int         number of image charges (overrides defaults).
    bias_B_T     : array-like (3,) [T]  optional external bias field.
                   If None the bias field is taken from ``defaults.B_mT``
                   (applied along x).
    """

    def __init__(
        self,
        charges_xyzq,
        defaults: Optional[Defaults] = None,
        *,
        z_defect: Optional[float] = None,
        E0_gate=(0.0, 0.0, 0.0),
        gate_grad=None,
        epsilon_eff: Optional[float] = None,
        screening_model=None,          # None → use defaults
        lambda_screen: Optional[float] = None,
        d_gate: Optional[float] = None,
        n_images: Optional[int] = None,
        bias_B_T=None,
    ):
        super().__init__(defaults=defaults)
        d = self.defaults

        # Charges
        if charges_xyzq is None:
            self._charges = np.zeros((0, 4), dtype=float)
        else:
            self._charges = np.asarray(charges_xyzq, dtype=float)
            if self._charges.ndim != 2 or self._charges.shape[1] != 4:
                raise ValueError(
                    "charges_xyzq must have shape (M, 4): [x, y, z, q]. "
                    f"Got shape {self._charges.shape}."
                )

        # Height of defects
        self.z_defect: float = float(z_defect if z_defect is not None else d.z_defect)

        # Gate field
        self._E0_gate  = np.asarray(E0_gate, dtype=float).ravel()
        self._gate_grad = gate_grad

        # Coulomb parameters (fall back to defaults if not overridden)
        self._epsilon_eff      = float(epsilon_eff     if epsilon_eff     is not None else d.epsilon_eff)
        self._screening_model  = screening_model  if screening_model  is not None else d.screening_model
        self._lambda_screen    = float(lambda_screen   if lambda_screen   is not None else d.lambda_screen)
        self._d_gate           = float(d_gate          if d_gate          is not None else d.d_gate)
        self._n_images         = int(n_images          if n_images         is not None else d.n_images)

        # Dielectric transmission
        self._eps_layer = d.eps_layer
        self._eps_host  = d.eps_host

        # Bias B field
        if bias_B_T is not None:
            self._bias_B_T = np.asarray(bias_B_T, dtype=float).ravel()
        else:
            B_mag = d.B_mT * 1e-3
            self._bias_B_T = np.array([B_mag, 0.0, 0.0])

        # Spin Hamiltonian base (no E-field; added per observation point)
        self._spin_params_base = SpinParams(
            D0=d.D0_Hz,
            E0=d.E0_Hz,
            d_perp_Hz_per_Vpm=d.d_perp,
            d_parallel_Hz_per_Vpm=d.d_parallel,
            B_T=self._bias_B_T,
            gamma_e_Hz_per_T=d.gamma_e,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Core: E-field at a single observation point
    # ──────────────────────────────────────────────────────────────────────

    def E_field(
        self,
        x_obs: float,
        y_obs: float,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        Total E-field at one observation point.

        Parameters
        ----------
        x_obs, y_obs : float [m]   in-plane position
        z_obs        : float [m]   height (default: self.z_defect)

        Returns
        -------
        E : ndarray shape (3,)  [V/m]
        """
        z = float(z_obs) if z_obs is not None else self.z_defect
        r_obs = np.array([float(x_obs), float(y_obs), z])

        E = E_gate_bias(r_obs, E0=self._E0_gate, grad=self._gate_grad)
        E += E_disorder_point_charges(
            r_obs, self._charges,
            epsilon_eff=self._epsilon_eff,
            screening_model=self._screening_model,
            lambda_screen=self._lambda_screen,
            d_gate=self._d_gate,
            n_images=self._n_images,
        )
        if (self._eps_layer is not None) and (self._eps_host is not None):
            E = apply_dielectric_transmission(E, self._eps_layer, self._eps_host)
        return E

    # ──────────────────────────────────────────────────────────────────────
    #  Transition frequencies at a single point
    # ──────────────────────────────────────────────────────────────────────

    def transition_frequencies(
        self,
        x_obs: float,
        y_obs: float,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        ODMR transition frequencies (Hz) at one observation point.

        The E-field from charges + gate is fed into the spin Hamiltonian;
        the bias B field is included but no stray B.

        Parameters
        ----------
        x_obs, y_obs : float [m]
        z_obs        : float [m]  (default: self.z_defect)

        Returns
        -------
        freqs : ndarray shape (2S,)  [Hz], sorted ascending
        """
        E_vec = self.E_field(x_obs, y_obs, z_obs)
        H = odmr_hamiltonian_Hz(self._spin_params_base, E_vec)
        evals, evecs = diagonalize_hamiltonian(H)
        sp = self._spin_params_base
        freqs, _ = extract_ms0_like_transitions_Hz(evals, evecs, sp.ms0_index)
        return freqs

    # ──────────────────────────────────────────────────────────────────────
    #  Maps over 2-D observation grids
    # ──────────────────────────────────────────────────────────────────────

    def E_field_map(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        E-field on a 2-D observation grid.

        Returns
        -------
        E_map : ndarray shape (Ny, Nx, 3)  [V/m]
        """
        x_arr = np.asarray(x_obs, dtype=float)
        y_arr = np.asarray(y_obs, dtype=float)
        E_map = np.zeros((len(y_arr), len(x_arr), 3), dtype=float)
        for iy, y in enumerate(y_arr):
            for ix, x in enumerate(x_arr):
                E_map[iy, ix] = self.E_field(x, y, z_obs)
        return E_map

    def E_z_map(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        Out-of-plane component E_z on the observation grid.

        Returns
        -------
        Ez : ndarray shape (Ny, Nx)  [V/m]
        """
        return self.E_field_map(x_obs, y_obs, z_obs)[..., 2]

    def transition_frequency_map(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        ODMR transition frequencies on a 2-D observation grid.

        Returns
        -------
        freq_map : ndarray shape (Ny, Nx, 2S)  [Hz]
        """
        x_arr = np.asarray(x_obs, dtype=float)
        y_arr = np.asarray(y_obs, dtype=float)
        n_trans = len(self.transition_frequencies(float(x_arr[0]), float(y_arr[0])))
        freq_map = np.zeros((len(y_arr), len(x_arr), n_trans), dtype=float)
        for iy, y in enumerate(y_arr):
            for ix, x in enumerate(x_arr):
                freq_map[iy, ix] = self.transition_frequencies(x, y, z_obs)
        return freq_map

    # ──────────────────────────────────────────────────────────────────────
    #  CW ODMR spectrum at a single point
    # ──────────────────────────────────────────────────────────────────────

    def odmr_spectrum(
        self,
        f_axis: np.ndarray,
        x_obs: float,
        y_obs: float,
        z_obs: Optional[float] = None,
        *,
        linewidth_Hz: Optional[float] = None,
        contrast: Optional[float] = None,
    ) -> np.ndarray:
        """
        CW ODMR PL spectrum at a single observation point.

        Returns
        -------
        pl : float array same shape as f_axis
        """
        from ..spin.spectra import PL_model
        lw  = linewidth_Hz if linewidth_Hz is not None else (
            1.0 / self.defaults.T2star
        )
        con = contrast if contrast is not None else self.defaults.contrast
        freqs = self.transition_frequencies(x_obs, y_obs, z_obs)
        return PL_model(np.asarray(f_axis, dtype=float), freqs, lw, con)

    # ──────────────────────────────────────────────────────────────────────
    #  Frequency shift map vs a reference point
    # ──────────────────────────────────────────────────────────────────────

    def frequency_shift_map(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        z_obs: Optional[float] = None,
        *,
        reference_xy: Tuple[float, float] = (1e6, 1e6),
        which: int = 0,
    ) -> np.ndarray:
        """
        Map of ODMR frequency shift relative to a reference location.

        Useful for imaging the spatial variation in charge-induced E-field.

        Parameters
        ----------
        x_obs, y_obs  : 1-D arrays [m]
        z_obs         : height [m] (default: self.z_defect)
        reference_xy  : (x, y) reference point where only the bias B contributes.
                        Defaults to (1e6, 1e6) — far from any realistic sample.
        which         : 0 or 1, selects the lower (0) or upper (1) transition.

        Returns
        -------
        delta_f : ndarray shape (Ny, Nx)  [Hz]
        """
        f_ref = self.transition_frequencies(*reference_xy, z_obs)[which]
        freq_map = self.transition_frequency_map(x_obs, y_obs, z_obs)
        return freq_map[..., which] - f_ref

    # ──────────────────────────────────────────────────────────────────────
    #  Repr
    # ──────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        n_q   = len(self._charges)
        bias  = np.linalg.norm(self._bias_B_T) * 1e3
        E0    = np.linalg.norm(self._E0_gate)
        return (
            f"ElectrometryExperiment("
            f"n_charges={n_q}, "
            f"|E0_gate|={E0:.1e} V/m, "
            f"z_defect={self.z_defect*1e9:.1f} nm, "
            f"screening={self._screening_model!r}, "
            f"bias_B={bias:.2f} mT)"
        )
