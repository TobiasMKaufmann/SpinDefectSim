"""
magnetometry/magnetometry.py — MagnetometryExperiment: magnetization → B → ODMR.

High-level workflow
-------------------
1. User supplies a :class:`~nv_sensing.magnetometry.geometry.SampleGeometry`
   and a 2-D magnetization M_z (array or callable).
2. :class:`MagnetometryExperiment` computes the stray B field at any
   observation point (x_obs, y_obs, z_defect) via Biot-Savart
   (edge + bulk contributions).
3. The stray B is fed directly into the VB⁻ spin Hamiltonian (zero E-field)
   to produce CW ODMR transition frequencies — standard magnetometry.

Usage example
-------------
>>> import numpy as np
>>> from nv_sensing.magnetometry import SquareGeometry, MagnetometryExperiment
>>> from nv_sensing.base.params import Defaults

>>> geom = SquareGeometry(side=500e-9, n_boundary_pts=300)

Uniform magnetization (scalar field) as a callable:

>>> exp = MagnetometryExperiment(
...     geometry    = geom,
...     magnetization = lambda x, y: 1e-3,   # 1 mA everywhere
...     defaults    = Defaults(),
...     z_defect    = 50e-9,
...     n_pts       = 60,
... )
>>> B = exp.B_field(x_obs=0.0, y_obs=0.0)          # (3,) T
>>> f1, f2 = exp.transition_frequencies(0.0, 0.0)  # Hz

Or pass a pre-computed grid:

>>> xx, yy = geom.make_grid(n_pts=60)
>>> M = np.exp(-(xx**2 + yy**2) / (100e-9)**2) * 1e-3   # Gaussian profile
>>> exp2 = MagnetometryExperiment(geom, M, Defaults(), z_defect=50e-9, n_pts=60)
>>> Bz_img = exp2.B_z_map(
...     np.linspace(-300e-9, 300e-9, 40),
...     np.linspace(-300e-9, 300e-9, 40),
... )
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import numpy as np

from ..base.params import Defaults, PhysicalParams
from ..spin.hamiltonian import (
    SpinParams,
    vb_spin_hamiltonian_Hz,
    diagonalize_hamiltonian,
    extract_ms0_like_transitions_Hz,
)
from ..spin.spectra import ensemble_odmr_spectrum
from .geometry import SampleGeometry
from .bfield import B_from_magnetization_grid, B_from_edge_segments, B_from_bulk_current_density

# Magnetization can be an ndarray or a callable (x, y) → float
MagnetizationType = Union[np.ndarray, Callable[[float, float], float]]

__all__ = ["MagnetometryExperiment"]


class MagnetometryExperiment(PhysicalParams):
    """
    Compute VB⁻ ODMR signals from a 2-D magnetization distribution.

    The magnetization M_z(x, y) is converted to stray B field via Biot-Savart
    (edge currents along the sample boundary + optional bulk ∇×M term).
    No electric-field contribution is assumed (pure magnetometry regime).

    Parameters
    ----------
    geometry      : SampleGeometry  instance (SquareGeometry, DiskGeometry, …)
    magnetization : ndarray shape (n_pts, n_pts)  *or*  callable (x, y) → float
        The out-of-plane magnetic moment density M_z [A] on the sample plane.
        • If an ndarray, it must be sampled on the grid ``geometry.make_grid(n_pts)``.
        • If callable, it is evaluated on-demand on the grid.
    defaults      : Defaults  physical parameter set (D0, E0, d⊥, bias B, …)
    z_defect      : float [m]  height of VB⁻ defects above the sample plane.
                    Defaults to ``defaults.z_defect``.
    n_pts         : int  grid resolution for Biot-Savart integration (per axis).
    include_bulk  : bool  include ∇×M bulk current term (default True).
    include_edge  : bool  include boundary edge-current term (default True).
    bias_B_T      : array-like (3,) [T]  optional external bias field *in addition*
                    to the stray B.  If None the bias field is taken from
                    ``defaults.B_mT`` (applied along x, as in SpinDefect).
    """

    def __init__(
        self,
        geometry: SampleGeometry,
        magnetization: MagnetizationType,
        defaults: Optional[Defaults] = None,
        *,
        z_defect: Optional[float] = None,
        n_pts: int = 100,
        include_bulk: bool = True,
        include_edge: bool = True,
        bias_B_T: Optional[np.ndarray] = None,
    ):
        super().__init__(defaults=defaults)

        self.geometry      = geometry
        self.n_pts         = int(n_pts)
        self.include_bulk  = bool(include_bulk)
        self.include_edge  = bool(include_edge)

        # Height of NV defects
        self.z_defect: float = float(
            z_defect if z_defect is not None else self.defaults.z_defect
        )

        # Build (or cache) the magnetization grid
        # Also remember whether the *original* magnetization was callable:
        # if so we can evaluate it exactly at boundary vertices (no step-function
        # interpolation artefact) and safely erode the bulk mask.
        self._xx, self._yy = geometry.make_grid(n_pts=self.n_pts)
        self._magnetization_callable = callable(magnetization)
        if self._magnetization_callable:
            self._magnetization_fn   = magnetization          # keep original callable
            M = np.vectorize(magnetization)(self._xx, self._yy)
            self.M_grid: np.ndarray = np.asarray(M, dtype=float)
        else:
            self._magnetization_fn   = None
            self.M_grid = np.asarray(magnetization, dtype=float)
            if self.M_grid.shape != self._xx.shape:
                raise ValueError(
                    f"magnetization array shape {self.M_grid.shape} does not "
                    f"match grid shape {self._xx.shape} for n_pts={n_pts}."
                )

        # Bias (external) B field — does NOT include stray B from magnetization
        if bias_B_T is not None:
            self._bias_B_T = np.asarray(bias_B_T, dtype=float).ravel()
        else:
            B_mag = self.defaults.B_mT * 1e-3
            self._bias_B_T = np.array([B_mag, 0.0, 0.0])

        # Spin Hamiltonian parameters (without stray B; it is added per-point)
        self._spin_params_base = SpinParams(
            D0=self.defaults.D0_Hz,
            E0=self.defaults.E0_Hz,
            d_perp_Hz_per_Vpm=self.defaults.d_perp,
            d_parallel_Hz_per_Vpm=self.defaults.d_parallel,
            B_T=self._bias_B_T,
            gamma_e_Hz_per_T=self.defaults.gamma_e,
            B_extra_T=np.zeros(3),
        )

        # Pre-compute edge-current segments (reused for all obs points)
        if self.include_edge:
            # When M was supplied as a callable, evaluate it directly at boundary
            # vertices for exact values (avoids the ~0.5× bilinear artefact that
            # occurs when interpolating a step-function grid to boundary points).
            if self._magnetization_callable:
                _m_for_verts = self._magnetization_fn
            else:
                _m_for_verts = self.M_grid
            M_verts = self.geometry.sample_M_at_vertices(
                _m_for_verts, xx=self._xx, yy=self._yy
            )
            (self._seg_start,
             self._seg_end,
             self._seg_I) = self.geometry.edge_current_segments(M_verts)
        else:
            self._seg_start = self._seg_end = self._seg_I = None

        # Pre-compute bulk current density (reused for all obs points)
        if self.include_bulk:
            Kx, Ky = self.geometry.bulk_current_density(
                self.M_grid, self._xx, self._yy
            )
            bulk_mask = self.geometry.interior_mask(self._xx, self._yy)
            if self.include_edge and self._magnetization_callable:
                # Edge was computed from exact callable values → it fully owns
                # the boundary delta-current.  Erode the bulk mask by one pixel
                # so the boundary ring is not double-counted.
                try:
                    from scipy.ndimage import binary_erosion
                    bulk_mask = binary_erosion(bulk_mask)
                except ImportError:
                    pass
            self._bulk_mask = bulk_mask
            self._Kx = Kx
            self._Ky = Ky
        else:
            self._Kx = self._Ky = self._bulk_mask = None

    # ──────────────────────────────────────────────────────────────────────
    #  Core: B field at a single observation point
    # ──────────────────────────────────────────────────────────────────────

    def B_field(
        self,
        x_obs: float,
        y_obs: float,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        Stray magnetic field (excluding bias) at one observation point.

        Parameters
        ----------
        x_obs, y_obs : float [m]   in-plane position
        z_obs        : float [m]   height (default: self.z_defect)

        Returns
        -------
        B_stray : ndarray shape (3,)  [T]
        """
        z = float(z_obs) if z_obs is not None else self.z_defect
        r_obs = np.array([float(x_obs), float(y_obs), z])

        B = np.zeros(3, dtype=float)

        if self.include_edge and self._seg_I is not None:
            B += B_from_edge_segments(
                self._seg_start, self._seg_end, self._seg_I, r_obs
            )

        if self.include_bulk and self._Kx is not None:
            B += B_from_bulk_current_density(
                self._xx, self._yy,
                self._Kx, self._Ky,
                r_obs,
                mask=self._bulk_mask,
            )

        return B

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
        VB⁻ ODMR transition frequencies (Hz) at one observation point.

        The stray B field is combined with the bias field and fed into the
        spin-1 Hamiltonian at zero E-field.

        Parameters
        ----------
        x_obs, y_obs : float [m]
        z_obs        : float [m]  (default: self.z_defect)

        Returns
        -------
        freqs : ndarray shape (2,)  [Hz], sorted ascending
        """
        import dataclasses
        B_stray = self.B_field(x_obs, y_obs, z_obs)
        sp = dataclasses.replace(self._spin_params_base, B_extra_T=B_stray)
        H = vb_spin_hamiltonian_Hz(sp, np.zeros(3))
        evals, evecs = diagonalize_hamiltonian(H)
        freqs, _ = extract_ms0_like_transitions_Hz(evals, evecs)
        return freqs

    # ──────────────────────────────────────────────────────────────────────
    #  Maps: B field or transition frequencies over a 2-D scan grid
    # ──────────────────────────────────────────────────────────────────────

    def B_field_map(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        Stray B field on a 2-D observation grid.

        Parameters
        ----------
        x_obs : 1-D array of x positions [m]  (length Nx)
        y_obs : 1-D array of y positions [m]  (length Ny)
        z_obs : float [m]  height above sample (default: self.z_defect)

        Returns
        -------
        B_map : ndarray shape (Ny, Nx, 3)   [T]
                ``B_map[iy, ix]`` is the (Bx, By, Bz) vector at
                (x_obs[ix], y_obs[iy]).
        """
        x_arr = np.asarray(x_obs, dtype=float)
        y_arr = np.asarray(y_obs, dtype=float)
        Nx, Ny = len(x_arr), len(y_arr)
        B_map = np.zeros((Ny, Nx, 3), dtype=float)
        for iy, y in enumerate(y_arr):
            for ix, x in enumerate(x_arr):
                B_map[iy, ix] = self.B_field(x, y, z_obs)
        return B_map

    def B_z_map(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convenience: out-of-plane component B_z on the observation grid.

        Returns
        -------
        Bz : ndarray shape (Ny, Nx)  [T]
        """
        return self.B_field_map(x_obs, y_obs, z_obs)[..., 2]

    def transition_frequency_map(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        z_obs: Optional[float] = None,
    ) -> np.ndarray:
        """
        ODMR transition frequencies on a 2-D observation grid.

        Parameters
        ----------
        x_obs : 1-D array [m]  (Nx,)
        y_obs : 1-D array [m]  (Ny,)
        z_obs : float [m]

        Returns
        -------
        freq_map : ndarray shape (Ny, Nx, 2)  [Hz]
                   Two transition frequencies (ascending) per pixel.
        """
        x_arr = np.asarray(x_obs, dtype=float)
        y_arr = np.asarray(y_obs, dtype=float)
        freq_map = np.zeros((len(y_arr), len(x_arr), 2), dtype=float)
        for iy, y in enumerate(y_arr):
            for ix, x in enumerate(x_arr):
                freq_map[iy, ix] = self.transition_frequencies(x, y, z_obs)
        return freq_map

    # ──────────────────────────────────────────────────────────────────────
    #  ODMR spectrum at a single point
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

        Parameters
        ----------
        f_axis      : microwave frequency axis (Hz)
        x_obs, y_obs: observation position [m]
        z_obs       : height [m] (default: self.z_defect)
        linewidth_Hz: FWHM [Hz] (default: 1/T2star from Defaults)
        contrast    : peak contrast fraction (default: Defaults.contrast)

        Returns
        -------
        pl : float array same shape as f_axis (PL normalised to 1 far off resonance)
        """
        from ..spin.spectra import PL_model
        lw  = linewidth_Hz if linewidth_Hz is not None else (
            1.0 / self.defaults.T2star
        )
        con = contrast if contrast is not None else self.defaults.get_contrast()
        freqs = self.transition_frequencies(x_obs, y_obs, z_obs)
        return PL_model(np.asarray(f_axis, dtype=float), freqs, lw, con)

    # ──────────────────────────────────────────────────────────────────────
    #  Convenience: peak frequency shift vs reference point
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

        Useful for imaging the spatial variation in magnetic field strength.

        Parameters
        ----------
        x_obs, y_obs  : 1-D arrays [m]
        z_obs         : height [m] (default: self.z_defect)
        reference_xy  : (x, y) reference point for the unperturbed frequency.
                        Defaults to (1e6, 1e6) — far outside any realistic sample,
                        so only the bias B contributes.
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
        M_max = float(np.abs(self.M_grid).max())
        bias  = np.linalg.norm(self._bias_B_T) * 1e3
        return (
            f"MagnetometryExperiment("
            f"geometry={self.geometry.__class__.__name__}, "
            f"|M|_max={M_max:.3e} A, "
            f"z_defect={self.z_defect*1e9:.1f} nm, "
            f"n_pts={self.n_pts}, "
            f"bias_B={bias:.2f} mT)"
        )
