"""
analysis/ensemble.py — DefectEnsemble: generates defect positions and computes E-fields.

Workflow:
  1. Generate random defect positions in the laser beam.
  2. Compute the E-field at each defect (gate bias + disorder charges).
  3. Hand off to SensingExperiment for protocol calculations.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from ..base.params import PhysicalParams, Defaults
from ..base.mixins import PlottingMixin, SerializationMixin, SweepMixin
from ..electrometry.efield import E_gate_bias, E_disorder_point_charges, apply_dielectric_transmission
from ..sensing.protocols import SensingExperiment


class DefectEnsemble(PhysicalParams, PlottingMixin, SerializationMixin, SweepMixin):
    """
    An ensemble of spin defects with their local E-fields and/or B-fields.

    Construction
    ------------
    >>> ens = DefectEnsemble(N_def=200, defaults=Defaults())
    >>> ens.generate_defects(seed=42)
    >>> ens.compute_efields(E0_gate=(0, 0, 1e4), disorder_xyzq=charges)

    E-field sources (pick one, or combine with add=True)
    ---------------------------------------------------
    compute_efields()        analytic: gate bias + screened point charges
    efields_from_grid()      interpolate from a regular Cartesian mesh
                             (COMSOL / FDTD / FEM export, 2-D or 3-D)
    efields_from_callable()  evaluate any callable E_func(xyz) -> (3,)
    set_efields()            inject a pre-computed (N, 3) array directly

    B-field sources (pick one, or combine with add=True)
    ---------------------------------------------------
    compute_bfields()        Biot-Savart from a 2-D magnetization distribution
    bfields_from_callable()  evaluate any callable B_func(xyz) -> (3,)
    bfields_from_grid()      interpolate from a regular Cartesian mesh
    set_bfields()            inject a pre-computed (N, 3) array directly (T)

    Post-compute
    ------------
    >>> exp = ens.to_experiment(sensing="E")     # E-field sensing only
    >>> exp = ens.to_experiment(sensing="B")     # B-field sensing only
    >>> exp = ens.to_experiment(sensing="both")  # full Hamiltonian contrast
    >>> tau, S_w, S_no, dS, tau_opt, dS_peak = exp.echo_static()
    >>> ens.save("run_gate_1kVpm")
    """

    def __init__(
        self,
        N_def: Optional[int] = None,
        R_patch: Optional[float] = None,
        defaults: Optional[Defaults] = None,
    ):
        super().__init__(defaults=defaults)
        self.N_def   = int(self._resolve(N_def, "N_def"))
        self.R_patch = float(self._resolve(R_patch, "R_patch"))

        # Defect geometry
        self.defect_positions:   Optional[np.ndarray] = None  # (N, 2) metres
        self.quantization_axes:  Optional[np.ndarray] = None  # (N, 3) unit vectors

        # Computed fields
        self.E_fields:       Optional[np.ndarray] = None  # (N, 3) V/m
        self.B_extra_fields: Optional[np.ndarray] = None  # (N, 3) T  — stray B signal

    # ── defect placement ─────────────────────────────────────────────────────
    def generate_defects(
        self,
        seed: int = 0,
        quantization_axis=None,
    ) -> np.ndarray:
        """
        Place *N_def* defects uniformly at random in a circular patch of radius
        *R_patch* and optionally assign quantization axes.

        Parameters
        ----------
        seed : random seed for reproducibility
        quantization_axis : specifies the z′-axis orientation for each defect.

            * ``None``  — no axes stored; ``SpinParams.quantization_axis``
              (default [0,0,1]) will be used for all defects.
            * ``"random"`` — axes drawn uniformly on the unit sphere (seed-derived).
            * array shape (3,) — one axis shared by all defects (e.g. a crystal axis).
            * array shape (N_def, 3) — explicit per-defect axes.

        Returns
        -------
        defect_positions : (N_def, 2) array in metres
        """
        rng   = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, self.N_def)
        r     = self.R_patch * np.sqrt(rng.uniform(0, 1, self.N_def))
        self.defect_positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        self.quantization_axes = self._resolve_quantization_axes(
            quantization_axis, seed=seed + 1000
        )
        return self.defect_positions

    def _resolve_quantization_axes(self, spec, seed: int = 0) -> Optional[np.ndarray]:
        """Convert a quantization_axis spec to an (N, 3) float array or None."""
        N = self.N_def
        if spec is None:
            return None
        if isinstance(spec, str) and spec == "random":
            rng = np.random.default_rng(seed)
            vecs = rng.standard_normal((N, 3))
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs
        arr = np.asarray(spec, float)
        if arr.ndim == 1:
            arr = arr / np.linalg.norm(arr)
            return np.broadcast_to(arr, (N, 3)).copy()
        if arr.shape == (N, 3):
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / norms
        raise ValueError(
            f"quantization_axis must be 'random', shape (3,), or shape ({N}, 3); "
            f"got shape {arr.shape}"
        )

    def generate_defects_gaussian(
        self,
        beam_waist_m: float,
        seed: int = 0,
        n_sigma: float = 3.0,
    ) -> np.ndarray:
        """
        Place *N_def* defects sampled from a 2-D Gaussian intensity profile.

        The Gaussian laser beam has 1/e² intensity radius *beam_waist_m*.
        Positions are drawn from a 2-D isotropic Gaussian with σ = beam_waist_m/√2
        (so the 1/e² *power* density contour lies at r = beam_waist_m) and
        truncated at *n_sigma* × σ to avoid rare far outliers.

        Parameters
        ----------
        beam_waist_m : 1/e² beam intensity radius w (m)
        seed         : random seed for reproducibility
        n_sigma      : truncation radius in units of σ (default 3)

        Returns
        -------
        defect_positions : (N_def, 2) array in metres
        """
        rng   = np.random.default_rng(seed)
        sigma = beam_waist_m / np.sqrt(2.0)   # Gaussian σ for each coordinate
        r_cut = n_sigma * sigma

        # Rejection-sample a 2-D Gaussian truncated at r_cut
        pts = []
        batch = max(self.N_def * 4, 1000)
        while len(pts) < self.N_def:
            x = rng.normal(0.0, sigma, batch)
            y = rng.normal(0.0, sigma, batch)
            mask = x**2 + y**2 <= r_cut**2
            for xi, yi in zip(x[mask], y[mask]):
                pts.append([xi, yi])
                if len(pts) == self.N_def:
                    break

        self.defect_positions = np.array(pts[:self.N_def])
        return self.defect_positions

    def set_defects(self, positions: np.ndarray) -> None:
        """Manually supply defect positions (m), shape (N, 2)."""
        self.defect_positions = np.asarray(positions, dtype=float)
        self.N_def = len(self.defect_positions)

    def set_quantization_axis(self, spec, seed: int = 0) -> None:
        """
        Set per-defect quantization axes after positions have been placed.

        Accepts the same *spec* as ``generate_defects(quantization_axis=...)``:
        ``None``, ``"random"``, a shape-(3,) array, or a shape-(N, 3) array.
        """
        self.quantization_axes = self._resolve_quantization_axes(spec, seed=seed)

    @staticmethod
    def n_defects_from_ppm(
        ppm: float,
        beam_waist_m: float,
        hbn_thickness_m: float,
        *,
        monolayer_B_density_m2: float = 1.84e19,
        hbn_interlayer_m: float = 0.333e-9,
    ) -> int:
        """
        Estimate the number of VB⁻ defects in a Gaussian laser spot.

        Model
        -----
        - hBN has *monolayer_B_density_m2* B-sites per monolayer per m²
          (default 1.84 × 10¹⁹ m⁻², hexagonal lattice a = 2.504 Å).
        - Number of layers ≈ hbn_thickness_m / hbn_interlayer_m.
        - Effective illuminated area ≈ π × beam_waist_m² (1/e² intensity circle).
        - Each B site replaced by VB⁻ with probability *ppm* × 10⁻⁶.

        Parameters
        ----------
        ppm              : defect concentration (parts per million of B atoms)
        beam_waist_m     : 1/e² intensity radius of the laser beam (m)
        hbn_thickness_m  : hBN flake or film thickness (m)

        Returns
        -------
        N : estimated integer number of VB⁻ defects in the beam
        """
        n_layers   = hbn_thickness_m / hbn_interlayer_m
        area       = np.pi * beam_waist_m ** 2
        total_B    = monolayer_B_density_m2 * n_layers * area
        return max(1, int(round(ppm * 1e-6 * total_B)))

    # ── E-field computation ──────────────────────────────────────────────────
    def compute_efields(
        self,
        *,
        E0_gate=(0.0, 0.0, 0.0),
        gate_grad=None,
        disorder_xyzq=None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Compute the E-field at each defect from gate bias + disorder charges.

        Parameters
        ----------
        E0_gate       : (Ex, Ey, Ez) uniform gate field (V/m)
        gate_grad     : 2×2 field-gradient matrix (V/m²) — optional
        disorder_xyzq : (M, 4) static disorder charges [x, y, z, q] (SI units)
        verbose       : print progress

        Returns
        -------
        E_fields : (N_def, 3) float array in V/m
        """
        if self.defect_positions is None:
            self.generate_defects()

        d   = self.defaults
        dis = (np.zeros((0, 4)) if disorder_xyzq is None
               else np.asarray(disorder_xyzq, dtype=float))
        N   = len(self.defect_positions)
        E_arr = np.zeros((N, 3), dtype=float)

        for i, dxy in enumerate(self.defect_positions):
            if verbose and (i % max(1, N // 10) == 0):
                print(f"  defect {i+1}/{N}", flush=True)
            obs = np.array([dxy[0], dxy[1], d.z_defect], dtype=float)
            E   = E_gate_bias(obs, E0=E0_gate, grad=gate_grad)
            E  += E_disorder_point_charges(
                obs, dis,
                epsilon_eff=d.epsilon_eff,
                screening_model=d.screening_model,
                lambda_screen=d.lambda_screen,
                d_gate=d.d_gate,
                n_images=d.n_images,
            )
            if (d.eps_layer is not None) and (d.eps_host is not None):
                E = apply_dielectric_transmission(E, d.eps_layer, d.eps_host)
            E_arr[i] = E

        self.E_fields = E_arr
        return E_arr

    def set_efields(self, E_fields: np.ndarray) -> None:
        """Manually supply per-defect E-fields, shape (N_def, 3) V/m."""
        self.E_fields = np.asarray(E_fields, dtype=float)

    # ── B-field computation ──────────────────────────────────────────────────
    def set_bfields(self, B_fields: np.ndarray) -> None:
        """Manually supply per-defect stray B-fields, shape (N_def, 3) T."""
        self.B_extra_fields = np.asarray(B_fields, dtype=float)

    def compute_bfields(
        self,
        magnetization,
        geometry,
        *,
        n_pts: int = 100,
        include_bulk: bool = True,
        include_edge: bool = True,
        add: bool = False,
    ) -> np.ndarray:
        """
        Compute the stray B field at each defect from a 2-D magnetization
        distribution via Biot-Savart.

        Parameters
        ----------
        magnetization : ndarray (n_pts, n_pts) *or* callable (x, y) → float
            Out-of-plane magnetization M_z [A] on the sample plane.
        geometry      : SampleGeometry instance (SquareGeometry, DiskGeometry, …)
        n_pts         : grid resolution for Biot-Savart integration (per axis)
        include_bulk  : include ∇×M bulk current contribution
        include_edge  : include boundary edge-current contribution
        add           : if True, add to existing B_extra_fields

        Returns
        -------
        B_extra_fields : (N_def, 3) float array in T
        """
        if self.defect_positions is None:
            raise RuntimeError("Place defects first with generate_defects().")
        from ..magnetometry.magnetometry import MagnetometryExperiment
        mag_exp = MagnetometryExperiment(
            geometry=geometry,
            magnetization=magnetization,
            defaults=self.defaults,
            z_defect=self.defaults.z_defect,
            n_pts=n_pts,
            include_bulk=include_bulk,
            include_edge=include_edge,
        )
        N = len(self.defect_positions)
        B_arr = np.zeros((N, 3), dtype=float)
        for i, dxy in enumerate(self.defect_positions):
            B_arr[i] = mag_exp.B_field(float(dxy[0]), float(dxy[1]))
        if add and self.B_extra_fields is not None:
            self.B_extra_fields = self.B_extra_fields + B_arr
        else:
            self.B_extra_fields = B_arr
        return self.B_extra_fields

    def bfields_from_callable(
        self,
        B_func,
        *,
        add: bool = False,
    ) -> np.ndarray:
        """
        Evaluate an arbitrary B-field function at every defect position.

        Parameters
        ----------
        B_func : callable  ``(xyz: array-like of shape (3,)) -> array-like (3,)``
            Called once per defect with ``[x, y, z_defect]`` in metres.
            Must return ``[Bx, By, Bz]`` in tesla.
        add    : if True, add to existing B_extra_fields

        Returns
        -------
        B_extra_fields : (N_def, 3) float array in T

        Examples
        --------
        Uniform z-field (e.g. external coil)::

            ens.bfields_from_callable(lambda xyz: [0, 0, 1e-4])

        Superimpose on Biot-Savart result::

            ens.compute_bfields(magnetization, geometry)
            ens.bfields_from_callable(lambda xyz: [0, 0, 5e-5], add=True)
        """
        if self.defect_positions is None:
            raise RuntimeError("Place defects first with generate_defects().")
        z_def = float(self.defaults.z_defect)
        N = len(self.defect_positions)
        B_arr = np.zeros((N, 3), dtype=float)
        for i, dxy in enumerate(self.defect_positions):
            xyz = np.array([dxy[0], dxy[1], z_def])
            B_arr[i] = np.asarray(B_func(xyz), dtype=float)
        if add and self.B_extra_fields is not None:
            self.B_extra_fields = self.B_extra_fields + B_arr
        else:
            self.B_extra_fields = B_arr
        return self.B_extra_fields

    def bfields_from_grid(
        self,
        Bx: np.ndarray,
        By: np.ndarray,
        Bz: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray = None,
        *,
        z_defect: float = None,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float = 0.0,
        add: bool = False,
    ) -> np.ndarray:
        """
        Interpolate B-field components from a regular grid onto defect positions.

        Mirrors :meth:`efields_from_grid` exactly but for B-fields (T).
        Accepts 2-D (Nx, Ny) or 3-D (Nx, Ny, Nz) component arrays.

        Parameters
        ----------
        Bx, By, Bz : component arrays (T)
        x_coords, y_coords : 1-D coordinate arrays (m)
        z_coords   : 1-D z-axis array (m); required for 3-D grids
        z_defect   : depth for 3-D queries (default: self.defaults.z_defect)
        method     : ``"linear"`` or ``"nearest"``
        bounds_error : raise on out-of-bounds (default False)
        fill_value : value for out-of-bounds positions (default 0.0)
        add        : add to existing B_extra_fields

        Returns
        -------
        B_extra_fields : (N_def, 3) float array in T

        Examples
        --------
        2-D slice (e.g. micromagnetic simulation output at fixed z)::

            ens.bfields_from_grid(Bx2d, By2d, Bz2d, xs, ys)

        3-D volumetric export::

            ens.bfields_from_grid(Bx3d, By3d, Bz3d, xs, ys, zs, z_defect=50e-9)
        """
        from scipy.interpolate import RegularGridInterpolator
        Bx = np.asarray(Bx, dtype=float)
        By = np.asarray(By, dtype=float)
        Bz = np.asarray(Bz, dtype=float)
        if not (Bx.shape == By.shape == Bz.shape):
            raise ValueError(
                f"Bx, By, Bz must have identical shapes; "
                f"got {Bx.shape}, {By.shape}, {Bz.shape}"
            )
        x_coords = np.asarray(x_coords, dtype=float)
        y_coords = np.asarray(y_coords, dtype=float)
        is_3d = Bz.ndim == 3
        if is_3d:
            if z_coords is None:
                raise ValueError("z_coords must be provided for a 3-D grid.")
            z_coords = np.asarray(z_coords, dtype=float)
            points = (x_coords, y_coords, z_coords)
        else:
            if Bz.ndim != 2:
                raise ValueError(
                    f"Bx/By/Bz must be 2-D (Nx, Ny) or 3-D (Nx, Ny, Nz); "
                    f"got shape {Bz.shape}"
                )
            if z_coords is not None:
                raise ValueError(
                    "z_coords should be None for a 2-D grid. "
                    "Pass z_coords only for 3-D volumetric data."
                )
            points = (x_coords, y_coords)
        kw = dict(method=method, bounds_error=bounds_error, fill_value=fill_value)
        interp_x = RegularGridInterpolator(points, Bx, **kw)
        interp_y = RegularGridInterpolator(points, By, **kw)
        interp_z = RegularGridInterpolator(points, Bz, **kw)
        if self.defect_positions is None:
            raise RuntimeError("Place defects first with generate_defects().")
        z_def = float(z_defect if z_defect is not None else self.defaults.z_defect)
        N = len(self.defect_positions)
        B_arr = np.zeros((N, 3), dtype=float)
        if is_3d:
            query = np.column_stack([
                self.defect_positions[:, 0],
                self.defect_positions[:, 1],
                np.full(N, z_def),
            ])
        else:
            query = self.defect_positions
        B_arr[:, 0] = interp_x(query)
        B_arr[:, 1] = interp_y(query)
        B_arr[:, 2] = interp_z(query)
        if add and self.B_extra_fields is not None:
            self.B_extra_fields = self.B_extra_fields + B_arr
        else:
            self.B_extra_fields = B_arr
        return self.B_extra_fields

    def efields_from_callable(
        self,
        E_func,
        *,
        add: bool = False,
    ) -> np.ndarray:
        """
        Evaluate an arbitrary E-field function at every defect position.

        This is the most flexible import path: wrap any external source
        (FEM solver output, FDTD field export, analytic expression) in a
        Python callable and pass it here.

        Parameters
        ----------
        E_func : callable  ``(xyz: array-like of shape (3,)) -> array-like (3,)``
            Called once per defect with ``[x, y, z_defect]`` in metres.
            Must return ``[Ex, Ey, Ez]`` in V/m.
        add    : bool, default False
            If True, *add* to any previously stored E_fields instead of replacing.
            Useful for superimposing a grid-imported field onto the built-in
            disorder/gate contribution.

        Returns
        -------
        E_fields : (N_def, 3) float array in V/m

        Examples
        --------
        Direct analytic function::

            ens.efields_from_callable(lambda xyz: [1e4, 0, 0])  # uniform x-field

        Wrapping a scipy interpolator built from COMSOL/FDTD output::

            from scipy.interpolate import RegularGridInterpolator
            interp_x = RegularGridInterpolator((xs, ys), Ex_2d)
            interp_y = RegularGridInterpolator((xs, ys), Ey_2d)
            interp_z = RegularGridInterpolator((xs, ys), Ez_2d)

            def comsol_field(xyz):
                pt = [[xyz[0], xyz[1]]]
                return [float(interp_x(pt)), float(interp_y(pt)), float(interp_z(pt))]

            ens.efields_from_callable(comsol_field)

        Superimpose on top of disorder charges::

            ens.compute_efields(disorder_xyzq=charges)
            ens.efields_from_callable(comsol_field, add=True)
        """
        if self.defect_positions is None:
            raise RuntimeError("Place defects first with generate_defects().")

        z_def = float(self.defaults.z_defect)
        N     = len(self.defect_positions)
        E_arr = np.zeros((N, 3), dtype=float)

        for i, dxy in enumerate(self.defect_positions):
            xyz = np.array([dxy[0], dxy[1], z_def])
            result = E_func(xyz)
            E_arr[i] = np.asarray(result, dtype=float)

        if add and self.E_fields is not None:
            self.E_fields = self.E_fields + E_arr
        else:
            self.E_fields = E_arr
        return self.E_fields

    def efields_from_grid(
        self,
        Ex: np.ndarray,
        Ey: np.ndarray,
        Ez: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray = None,
        *,
        z_defect: float = None,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float = 0.0,
        add: bool = False,
    ) -> np.ndarray:
        """
        Interpolate E-field components from a regular grid onto defect positions.

        The grid can be 2-D (one z-slice) or 3-D (volumetric).  This is the
        natural import path for FEM (COMSOL, FEniCS), FDTD (Lumerical, meep),
        or any other solver that exports fields on a Cartesian mesh.

        Parameters
        ----------
        Ex, Ey, Ez : ndarray
            Field component arrays.  For a 2-D grid: shape ``(Nx, Ny)``.
            For a 3-D grid: shape ``(Nx, Ny, Nz)``.  All three must have the
            same shape.
        x_coords   : 1-D float array, length Nx.  Grid x-axis in metres.
        y_coords   : 1-D float array, length Ny.  Grid y-axis in metres.
        z_coords   : 1-D float array, length Nz.  Grid z-axis in metres.
            Required for 3-D grids.  Must be ``None`` for 2-D grids.
        z_defect   : float (m).  Depth at which to evaluate a 3-D grid.
            Defaults to ``self.defaults.z_defect``.  Ignored for 2-D grids.
        method     : interpolation method — ``"linear"`` (default) or
            ``"nearest"``.  Passed to
            :class:`scipy.interpolate.RegularGridInterpolator`.
        bounds_error : bool, default False.
            If True, raise an error when a defect lies outside the grid.
            If False, return *fill_value* for out-of-bounds defects.
        fill_value : float, default 0.0.
            Value used for defect positions outside the grid.
        add        : bool, default False.
            If True, add to existing E_fields; otherwise replace.

        Returns
        -------
        E_fields : (N_def, 3) float array in V/m

        Examples
        --------
        2-D grid (single z-slice from a COMSOL export)::

            # xs, ys : 1-D coordinate arrays
            # Ex2d, Ey2d, Ez2d : (Nx, Ny) arrays
            ens.efields_from_grid(Ex2d, Ey2d, Ez2d, xs, ys)

        3-D volumetric grid::

            ens.efields_from_grid(Ex3d, Ey3d, Ez3d, xs, ys, zs,
                                  z_defect=0.34e-9)

        Superimpose a gate field from FDTD on top of disorder charges::

            ens.compute_efields(disorder_xyzq=charges)
            ens.efields_from_grid(Ex_fdtd, Ey_fdtd, Ez_fdtd, xs, ys, add=True)
        """
        from scipy.interpolate import RegularGridInterpolator

        Ex = np.asarray(Ex, dtype=float)
        Ey = np.asarray(Ey, dtype=float)
        Ez = np.asarray(Ez, dtype=float)
        if not (Ex.shape == Ey.shape == Ez.shape):
            raise ValueError(
                f"Ex, Ey, Ez must have identical shapes; "
                f"got {Ex.shape}, {Ey.shape}, {Ez.shape}"
            )

        x_coords = np.asarray(x_coords, dtype=float)
        y_coords = np.asarray(y_coords, dtype=float)

        is_3d = Ez.ndim == 3
        if is_3d:
            if z_coords is None:
                raise ValueError("z_coords must be provided for a 3-D grid.")
            z_coords = np.asarray(z_coords, dtype=float)
            points = (x_coords, y_coords, z_coords)
        else:
            if Ez.ndim != 2:
                raise ValueError(
                    f"Ex/Ey/Ez must be 2-D (Nx, Ny) or 3-D (Nx, Ny, Nz); "
                    f"got shape {Ez.shape}"
                )
            if z_coords is not None:
                raise ValueError(
                    "z_coords should be None for a 2-D grid. "
                    "Pass z_coords only for 3-D volumetric data."
                )
            points = (x_coords, y_coords)

        kw = dict(method=method, bounds_error=bounds_error, fill_value=fill_value)
        interp_x = RegularGridInterpolator(points, Ex, **kw)
        interp_y = RegularGridInterpolator(points, Ey, **kw)
        interp_z = RegularGridInterpolator(points, Ez, **kw)

        if self.defect_positions is None:
            raise RuntimeError("Place defects first with generate_defects().")

        z_def = float(z_defect if z_defect is not None else self.defaults.z_defect)
        N     = len(self.defect_positions)
        E_arr = np.zeros((N, 3), dtype=float)

        if is_3d:
            query = np.column_stack([
                self.defect_positions[:, 0],
                self.defect_positions[:, 1],
                np.full(N, z_def),
            ])
        else:
            query = self.defect_positions  # (N, 2)

        E_arr[:, 0] = interp_x(query)
        E_arr[:, 1] = interp_y(query)
        E_arr[:, 2] = interp_z(query)

        if add and self.E_fields is not None:
            self.E_fields = self.E_fields + E_arr
        else:
            self.E_fields = E_arr
        return self.E_fields

    # ── conversion to SensingExperiment ─────────────────────────────────────
    def to_experiment(
        self,
        B_mT: Optional[float] = None,
        sensing: str = "both",
    ) -> SensingExperiment:
        """
        Convert the cached fields into a :class:`~sensing.protocols.SensingExperiment`.

        The *signal* branch uses whatever fields are requested by *sensing*;
        the *reference* branch uses zero for those same fields (bare defects).
        Both branches always include the same uniform bias B (from ``defaults.B_mT``).

        Parameters
        ----------
        B_mT    : override the bias magnetic field (mT); uses default if None.
        sensing : which field(s) contribute to the sensing contrast.

            ``"E"``    — E-field sensing only.  Requires :attr:`E_fields`.
                         B-field is the same (bias only) in both branches.
            ``"B"``    — B-field sensing only.  Requires :attr:`B_extra_fields`.
                         E-field is zero in both branches.
            ``"both"`` — Full contrast: signal has both E *and* stray B;
                         reference has neither.  At least one of
                         :attr:`E_fields` / :attr:`B_extra_fields` must be set.

        Returns
        -------
        SensingExperiment

        Examples
        --------
        >>> ens.compute_efields(E0_gate=(0, 0, 5e3))
        >>> exp = ens.to_experiment(sensing="E")

        >>> ens.compute_bfields(magnetization, geometry)
        >>> exp = ens.to_experiment(sensing="B")

        >>> # Both contributions active at once
        >>> ens.compute_efields(...)
        >>> ens.compute_bfields(...)
        >>> exp = ens.to_experiment(sensing="both")
        """
        sensing = sensing.lower()
        if sensing not in ("e", "b", "both"):
            raise ValueError(f"sensing must be 'E', 'B', or 'both'; got {sensing!r}")

        import dataclasses
        from ..spin.spectra import ensemble_transitions_from_Efields
        d = self.defaults
        if B_mT is not None:
            d = dataclasses.replace(d, B_mT=float(B_mT))
        sp = d.to_spin_params()
        axes = self.quantization_axes
        N = self.N_def

        # Resolve signal/reference fields for each mode
        use_E = sensing in ("e", "both")
        use_B = sensing in ("b", "both")

        if use_E and self.E_fields is None:
            raise RuntimeError(
                "E-fields not set. Call compute_efields(), efields_from_grid(), "
                "efields_from_callable(), or set_efields() first."
            )
        if use_B and self.B_extra_fields is None:
            raise RuntimeError(
                "B-fields not set. Call compute_bfields(), bfields_from_grid(), "
                "bfields_from_callable(), or set_bfields() first."
            )
        if not use_E and not use_B:
            raise ValueError("sensing must include at least 'E' or 'B'.")

        E_with  = self.E_fields       if use_E else np.zeros((N, 3))
        B_with  = self.B_extra_fields if use_B else np.zeros((N, 3))
        B_no    = np.zeros((N, 3))

        tr_with = ensemble_transitions_from_Efields(
            E_with, sp, quantization_axes=axes, B_extra_fields=B_with
        )
        tr_no = ensemble_transitions_from_Efields(
            np.zeros((N, 3)), sp, quantization_axes=axes, B_extra_fields=B_no
        )

        exp = SensingExperiment(
            sp, sp, E_with, defaults=d,
            quantization_axes=axes,
            B_extra_fields_with=B_with,
            B_extra_fields_no=B_no,
        )
        exp._tr_with = tr_with
        exp._tr_no   = tr_no
        return exp

    # ── serialisation ────────────────────────────────────────────────────────
    def _serializable_arrays(self) -> dict:
        out: dict = {}
        if self.defect_positions is not None:
            out["defect_positions"] = self.defect_positions
        if self.E_fields is not None:
            out["E_fields"] = self.E_fields
        return out

    @classmethod
    def from_npz(cls, path: str, defaults: Optional[Defaults] = None) -> "DefectEnsemble":
        """
        Restore a DefectEnsemble from a previously saved .npz file.

        Example
        -------
        >>> ens = DefectEnsemble.from_npz("run_V0_1meV.npz")
        """
        data = np.load(path, allow_pickle=False)
        ens  = cls(defaults=defaults)
        if "defect_positions" in data:
            ens.set_defects(data["defect_positions"])
        if "E_fields" in data:
            ens.E_fields = data["E_fields"]
        return ens

    def __repr__(self) -> str:
        d = self.defaults
        computed = self.E_fields is not None
        return (
            f"DefectEnsemble("
            f"N={self.N_def}, "
            f"Nqh={self.Nqh_used if self.Nqh_used else d.Nqh}, "
            f"V0={d.V0_meV} meV, "
            f"aM={d.aM*1e9:.1f} nm, "
            f"E-fields={'computed' if computed else 'not yet computed'})"
        )
