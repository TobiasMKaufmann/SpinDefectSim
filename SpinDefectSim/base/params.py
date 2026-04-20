"""
base/params.py — Physical constants, experiment defaults, and base dataclass.

All numerical defaults are centralised here so every module and class can
inherit or reference them without magic numbers scattered across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.constants import e, epsilon_0, k as k_B, hbar


# ─────────────────────────────────────────────────────────────────────────────
#  Immutable physical constants (never change these)
# ─────────────────────────────────────────────────────────────────────────────
PHYSICAL = dict(
    e=e,                   # elementary charge (C)
    epsilon_0=epsilon_0,   # vacuum permittivity (F/m)
    k_B=k_B,               # Boltzmann constant (J/K)
    hbar=hbar,             # reduced Planck constant (J·s)
    gamma_e_Hz_per_T=28e9, # electron gyromagnetic ratio (Hz/T)
)


# ─────────────────────────────────────────────────────────────────────────────
#  Single source of truth for all tuneable defaults
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Defaults:
    """
    Single source of truth for all tuneable simulation parameters.

    The defect species is selected via *defect_type* (default ``'vb_minus'``).
    The convenience class method :meth:`for_defect` creates a ``Defaults``
    whose spin Hamiltonian params are pre-loaded from the named defect type.

    Examples
    --------
    >>> d = Defaults()                         # VB⁻ defaults
    >>> d = Defaults.for_defect("nv_minus")   # NV⁻ defaults
    >>> d = Defaults(B_mT=3.0)                # override single param
    >>> sp = d.to_spin_params()
    """

    # --- Defect species ---
    defect_type: str = "vb_minus"  # name from spin.defects registry

    # --- Spin Hamiltonian ---
    D0_Hz: float = 3.46e9          # ZFS axial splitting (Hz)
    E0_Hz: float = 50e6            # intrinsic strain splitting (Hz)
    d_perp: float = 0.35           # transverse E-field coupling (Hz/(V/m))
    d_parallel: float = 0.0        # axial E-field coupling (Hz/(V/m)), usually ignored
    B_mT: float = 1.5              # external B-field magnitude (mT), in-plane (x)
    gamma_e: float = 28e9          # electron gyromagnetic ratio (Hz/T)

    # --- Coherence / detection ---
    T2star: float = 50e-9          # CW / Ramsey dephasing time (s)
    T2echo: float = 10e-6          # Hahn-echo coherence time (s)
    contrast: float = 0.02         # optical spin contrast (fraction)
    n_photons: int = 500           # photons per shot

    # --- Ensemble geometry ---
    N_def: int = 1200              # number of VB⁻ defects in laser beam
    R_laser: float = 500e-9        # beam radius (m)
    R_patch: float = 200e-9        # sensing patch radius (m)

    # --- Geometry / dielectrics ---
    z_defect: float = 0.34e-9      # VB⁻ height above moiré plane (m)
    epsilon_eff: float = 7.0       # effective dielectric (dimensionless)
    eps_layer: Optional[float] = None  # hBN layer dielectric (for transmission correction)
    eps_host: Optional[float] = None   # host dielectric

    # --- Coulomb / screening ---
    screening_model: Optional[str] = "dual_gate"  # None | 'yukawa' | 'dual_gate'
    lambda_screen: float = 10e-9   # Yukawa screening length (m)
    d_gate: float = 15e-9          # dual-gate half-separation (m)
    n_images: int = 30             # image charge sum truncation
    r_min: float = 0.2e-9         # minimum distance regulariser (m)

    @classmethod
    def for_defect(cls, defect_type, **kwargs) -> "Defaults":
        """
        Create a ``Defaults`` with spin Hamiltonian parameters taken from the
        named defect type.  Any additional keyword arguments override those.

        Parameters
        ----------
        defect_type : str or DefectType
            E.g. ``'nv_minus'``, ``'p1'``, ``'v_sic'``.
        **kwargs :
            Override any Defaults field (e.g. ``B_mT=5.0``, ``T2echo=20e-6``).

        Examples
        --------
        >>> d = Defaults.for_defect("nv_minus", B_mT=5.0, T2echo=20e-6)
        """
        from SpinDefectSim.spin.defects import get_defect
        dt = get_defect(defect_type)
        name = defect_type if isinstance(defect_type, str) else dt.name
        return cls(
            defect_type=name,
            D0_Hz=dt.D0_Hz,
            E0_Hz=dt.E0_Hz,
            d_perp=dt.d_perp,
            d_parallel=dt.d_parallel,
            gamma_e=dt.gamma_Hz_T,
            **kwargs,
        )

    def to_spin_params(self):
        """Return a :class:`~SpinDefectSim.spin.hamiltonian.SpinParams` from these defaults."""
        from SpinDefectSim.spin.hamiltonian import SpinParams
        from SpinDefectSim.spin.defects import get_defect
        dt = get_defect(self.defect_type)
        return SpinParams(
            D0=self.D0_Hz,
            E0=self.E0_Hz,
            d_perp_Hz_per_Vpm=self.d_perp,
            d_parallel_Hz_per_Vpm=self.d_parallel,
            B_T=np.array([self.B_mT * 1e-3, 0.0, 0.0]),
            gamma_e_Hz_per_T=self.gamma_e,
            spin=dt.spin,
            ms0_index=dt.ms0_index,
        )

    def coulomb_kwargs(self) -> dict:
        """Return keyword arguments expected by the Coulomb kernel functions."""
        return dict(
            epsilon_eff=self.epsilon_eff,
            screening_model=self.screening_model,
            lambda_screen=self.lambda_screen,
            d_gate=self.d_gate,
            n_images=self.n_images,
            r_min=self.r_min,
        )


# Convenience singleton — use when you just need the defaults unchanged
DEFAULT = Defaults()


# ─────────────────────────────────────────────────────────────────────────────
#  Abstract base for domain objects
# ─────────────────────────────────────────────────────────────────────────────
class PhysicalParams:
    """
    Lightweight base class for all domain objects that carry parameters.

    Provides:
    - ``defaults``  attribute (a :class:`Defaults` instance)
    - ``_resolve``  helper to fall back to defaults for Optional kwargs
    """

    def __init__(self, defaults: Optional[Defaults] = None):
        self.defaults: Defaults = defaults if defaults is not None else Defaults()

    def _resolve(self, value, key: str):
        """Return *value* if it is not None, else ``self.defaults.<key>``."""
        if value is not None:
            return value
        return getattr(self.defaults, key)
