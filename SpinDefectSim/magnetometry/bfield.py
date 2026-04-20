"""
magnetometry/bfield.py — Biot-Savart integration for 2-D current distributions.

Two contributions to the stray B field are handled:

1. **Edge currents** — line integrals along boundary segments.
   Each segment is a finite straight wire; the analytic Biot-Savart
   expression for a finite wire is used (no quadrature error).

2. **Bulk currents** — area integral over the interior current density
   K(r) = ∇×M_z.  Discretised on the grid returned by ``make_grid``.

Public API
----------
B_from_wire_segment(r_start, r_end, current_A, r_obs_xyz)
    Analytic Biot-Savart field from a single finite wire segment.

B_from_edge_segments(seg_start, seg_end, seg_current, r_obs_xyz)
    Vectorised sum over all boundary segments.

B_from_bulk_current_density(xx, yy, Kx, Ky, r_obs_xyz, *, mask)
    Numerical area integral (2-D Biot-Savart).

B_from_magnetization_grid(geom, M_grid, r_obs_xyz, *, n_pts,
                           include_bulk, include_edge)
    High-level wrapper: magnetization grid → total B field.

Physical units
--------------
All lengths in metres, currents in amperes, fields in tesla.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import SampleGeometry

__all__ = [
    "B_from_wire_segment",
    "B_from_edge_segments",
    "B_from_bulk_current_density",
    "B_from_magnetization_grid",
]

_MU0_OVER_4PI: float = 1e-7   # μ₀/(4π)  (T·m/A)


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Analytic Biot-Savart for a finite straight wire segment
# ─────────────────────────────────────────────────────────────────────────────

def B_from_wire_segment(
    r_start: np.ndarray,
    r_end: np.ndarray,
    current_A: float,
    r_obs_xyz: np.ndarray,
    *,
    r_min: float = 1e-15,
) -> np.ndarray:
    """
    Analytic Biot-Savart B field from a single finite straight wire segment.

    Uses the standard result for a wire of finite length L carrying current I:

        B = (μ₀ I) / (4π R) × (s₁/√(s₁²+R²) − s₂/√(s₂²+R²)) × (d̂ × r̂⊥)

    where R is the perpendicular distance from the observation point to the
    wire line, and s₁, s₂ are the signed projections of (obs − start) and
    (obs − end) along the wire direction *d̂*.

    Parameters
    ----------
    r_start    : array-like shape (2,) or (3,)  [m]  wire start point
    r_end      : array-like shape (2,) or (3,)  [m]  wire end  point
    current_A  : float [A]  current flowing from r_start → r_end
    r_obs_xyz  : array-like shape (3,)          [m]  observation point
    r_min      : float [m]  regularisation: minimum R (avoids div-by-zero
                             when r_obs lies on the wire)

    Returns
    -------
    B : ndarray shape (3,)  [T]
    """
    A   = _to3d(r_start)
    B_  = _to3d(r_end)
    P   = np.asarray(r_obs_xyz, dtype=float).ravel()
    if P.size != 3:
        raise ValueError("r_obs_xyz must have 3 components.")

    wire_vec = B_ - A                          # d̂ unnormalised
    L        = np.linalg.norm(wire_vec)
    if L < 1e-30:
        return np.zeros(3)

    d_hat = wire_vec / L

    # Perpendicular displacement from wire line to P
    AP       = P - A
    s1       = float(np.dot(AP, d_hat))          # proj of A→P along d̂
    r_perp   = AP - s1 * d_hat                   # perpendicular vector
    R        = np.linalg.norm(r_perp)

    if R < r_min:
        return np.zeros(3)                        # on the wire → B = 0

    r_perp_hat = r_perp / R

    # s₁ = signed dist from A-projection to P,  s₂ = signed dist from B-projection to P
    s2 = s1 - L
    factor = (s1 / np.sqrt(s1 ** 2 + R ** 2)
              - s2 / np.sqrt(s2 ** 2 + R ** 2))

    # Direction: d̂ × r̂⊥
    direction = np.cross(d_hat, r_perp_hat)

    return (_MU0_OVER_4PI * float(current_A) / R * factor) * direction


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Vectorised sum over all boundary segments
# ─────────────────────────────────────────────────────────────────────────────

def B_from_edge_segments(
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    seg_current: np.ndarray,
    r_obs_xyz: np.ndarray,
    *,
    r_min: float = 1e-15,
) -> np.ndarray:
    """
    Total B field from all boundary edge-current segments at one point.

    Parameters
    ----------
    seg_start   : shape (N_seg, 2) or (N_seg, 3)  [m]
    seg_end     : shape (N_seg, 2) or (N_seg, 3)  [m]
    seg_current : shape (N_seg,)  [A]   current per segment
    r_obs_xyz   : shape (3,)      [m]   observation point
    r_min       : regularisation distance

    Returns
    -------
    B : ndarray shape (3,)  [T]
    """
    # Vectorised analytic formula for all segments at once -----------------
    # Pad to 3-D if needed
    A_arr = _to3d_array(np.asarray(seg_start, dtype=float))   # (N, 3)
    B_arr = _to3d_array(np.asarray(seg_end,   dtype=float))   # (N, 3)
    I_arr = np.asarray(seg_current, dtype=float)              # (N,)
    P     = np.asarray(r_obs_xyz,   dtype=float).ravel()      # (3,)

    wire_vecs = B_arr - A_arr                                  # (N, 3)
    L_arr     = np.linalg.norm(wire_vecs, axis=1)              # (N,)
    valid     = L_arr > 1e-30
    d_hat     = np.where(
        valid[:, None],
        wire_vecs / np.maximum(L_arr[:, None], 1e-300),
        np.zeros((len(L_arr), 3)),
    )                                                          # (N, 3)

    AP    = P[None, :] - A_arr                                 # (N, 3)
    s1    = np.einsum("ni,ni->n", AP, d_hat)                   # (N,)
    r_perp = AP - s1[:, None] * d_hat                         # (N, 3)
    R      = np.linalg.norm(r_perp, axis=1)                   # (N,)

    on_wire = R < r_min
    R_safe  = np.where(on_wire, 1.0, R)
    r_perp_hat = r_perp / R_safe[:, None]                     # (N, 3)

    s2   = s1 - L_arr                                          # (N,)
    fact = (s1 / np.sqrt(np.maximum(s1 ** 2 + R ** 2, 1e-300))
            - s2 / np.sqrt(np.maximum(s2 ** 2 + R ** 2, 1e-300)))  # (N,)

    # d̂ × r̂⊥, shape (N, 3)
    direction = np.cross(d_hat, r_perp_hat)

    # Amplitude per segment, shape (N,)
    amplitude = np.where(
        on_wire | ~valid,
        0.0,
        _MU0_OVER_4PI * I_arr / R_safe * fact,
    )

    return np.einsum("n,ni->i", amplitude, direction)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Bulk current area integral (numerical 2-D Biot-Savart)
# ─────────────────────────────────────────────────────────────────────────────

def B_from_bulk_current_density(
    xx: np.ndarray,
    yy: np.ndarray,
    Kx: np.ndarray,
    Ky: np.ndarray,
    r_obs_xyz: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    r_min: float = 1e-15,
) -> np.ndarray:
    """
    B field at *r_obs_xyz* from a 2-D in-plane sheet current K(r) = (Kx, Ky).

    Biot-Savart for a surface current element K dA in the z = 0 plane:

        dB = (μ₀/4π) K × r̂ / |r|² dA

    where r = r_obs − r_source.

    The sum is computed by numericallly integrating over grid cells.

    Parameters
    ----------
    xx, yy  : ndarray shape (ny, nx)   grid coordinates [m]
    Kx, Ky  : ndarray shape (ny, nx)   surface current density [A/m]
               (= ∇×M_z where M_z is in A; integrating K [A/m] over area [m²]
               yields a current element I·dl [A·m] as required by Biot-Savart)
    r_obs_xyz : shape (3,)  [m]
    mask    : optional bool array shape (ny, nx); True = include in integral.
              If None, all cells are used.
    r_min   : regularisation (m)

    Returns
    -------
    B : ndarray shape (3,)  [T]
    """
    xx   = np.asarray(xx,   dtype=float)
    yy   = np.asarray(yy,   dtype=float)
    Kx_  = np.asarray(Kx,   dtype=float)
    Ky_  = np.asarray(Ky,   dtype=float)
    P    = np.asarray(r_obs_xyz, dtype=float).ravel()

    dx   = xx[0, 1] - xx[0, 0]
    dy   = yy[1, 0] - yy[0, 0]
    dA   = abs(dx * dy)

    # Flatten arrays
    x_s  = xx.ravel()
    y_s  = yy.ravel()

    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        x_s   = x_s[m]
        y_s   = y_s[m]
        kx_s  = Kx_.ravel()[m]
        ky_s  = Ky_.ravel()[m]
    else:
        kx_s = Kx_.ravel()
        ky_s = Ky_.ravel()

    # Displacement r_obs − r_source, shape (N,)
    rx = P[0] - x_s
    ry = P[1] - y_s
    rz = P[2]                              # scalar (obs z − 0 plane)

    r2    = np.maximum(rx ** 2 + ry ** 2 + rz ** 2, r_min ** 2)
    r3    = r2 ** 1.5

    # K × r / r³  for in-plane K = (kx, ky, 0) cross r = (rx, ry, rz)
    #  (K × r)_x = ky * rz − 0 * ry = ky * rz
    #  (K × r)_y = 0 * rx − kx * rz = −kx * rz
    #  (K × r)_z = kx * ry − ky * rx

    Bx = np.sum((ky_s * rz            ) / r3) * dA
    By = np.sum((-kx_s * rz           ) / r3) * dA
    Bz = np.sum((kx_s * ry - ky_s * rx) / r3) * dA

    return _MU0_OVER_4PI * np.array([Bx, By, Bz], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  High-level wrapper: magnetization grid → total B
# ─────────────────────────────────────────────────────────────────────────────

def B_from_magnetization_grid(
    geom: SampleGeometry,
    M_grid: np.ndarray,
    r_obs_xyz: np.ndarray,
    *,
    n_pts: int = 100,
    include_bulk: bool = True,
    include_edge: bool = True,
    erode_bulk_boundary: bool = False,
    r_min: float = 1e-15,
) -> np.ndarray:
    """
    Total B field at *r_obs_xyz* from a 2-D magnetization grid.

    Combines edge and bulk contributions:

    * Edge: boundary carries a loop current I = M_z (A) — every segment of
      the closed boundary polygon is part of the same series circuit, so each
      segment carries the local average M_z (A).  The geometry's resampled
      boundary vertices serve as the discretisation.
    * Bulk: interior carries K_bulk = ∇×M_z  (A/m), integrated numerically
      over the bounding box (masked to the sample interior).

    Parameters
    ----------
    geom       : SampleGeometry  (e.g. SquareGeometry, PolygonGeometry)
    M_grid     : ndarray shape (n_pts, n_pts)  M_z values [A] on the grid
                 returned by ``geom.make_grid(n_pts)``.
    r_obs_xyz  : shape (3,)  [m]  observation point
    n_pts      : int  grid resolution (must match M_grid if provided as array)
    include_bulk : bool  include ∇×M bulk term (default True)
    include_edge : bool  include boundary edge-current term (default True)
    erode_bulk_boundary : bool  (default False)
        Controls how the edge / bulk split handles the outermost ring of grid
        cells at the sample boundary.

        ``False`` (default) — **bulk owns the boundary**.
        The full interior mask is used.  Required when *M_grid* is a
        step-function (M inside, 0 outside): bilinear interpolation gives
        vertex values ≈00.5×M, so the edge loop captures ~half the current;
        the boundary-ring gradient in the bulk captures the other half.
        Using include_edge=True + include_bulk=True gives the correct total.

        ``True`` — **edge owns the boundary**.
        The outermost ring of interior cells is eroded from the bulk mask.
        Only correct when the M_grid boundary values are exact (e.g. computed
        from a callable that returns M inside AND outside, so interpolation
        at boundary vertices gives the full M).  Prevents double-counting in
        that case.
    r_min      : float  regularisation distance [m]

    Returns
    -------
    B : ndarray shape (3,)  [T]
    """
    xx, yy = geom.make_grid(n_pts=n_pts)
    M_grid  = np.asarray(M_grid, dtype=float)

    B_total = np.zeros(3, dtype=float)

    # ---- edge contribution ----
    if include_edge:
        M_verts = geom.sample_M_at_vertices(M_grid, xx=xx, yy=yy)
        s_start, s_end, s_I = geom.edge_current_segments(M_verts)
        B_total += B_from_edge_segments(s_start, s_end, s_I, r_obs_xyz,
                                        r_min=r_min)

    # ---- bulk contribution ----
    if include_bulk:
        Kx, Ky = geom.bulk_current_density(M_grid, xx, yy)
        inside  = geom.interior_mask(xx, yy)
        if erode_bulk_boundary and include_edge:
            # Edge has exact boundary values → it fully owns the boundary
            # delta-current.  Remove the outermost ring from bulk to prevent
            # double-counting.
            try:
                from scipy.ndimage import binary_erosion
                inside = binary_erosion(inside)
            except ImportError:
                pass   # scipy unavailable: small double-count near edge
        B_total += B_from_bulk_current_density(xx, yy, Kx, Ky, r_obs_xyz,
                                               mask=inside, r_min=r_min)

    return B_total


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to3d(r) -> np.ndarray:
    """Pad a 2- or 3-component vector to shape (3,) with z = 0."""
    r = np.asarray(r, dtype=float).ravel()
    if r.size == 2:
        return np.array([r[0], r[1], 0.0])
    return r


def _to3d_array(r: np.ndarray) -> np.ndarray:
    """Pad a (N, 2) or (N, 3) array to (N, 3) with z = 0."""
    if r.shape[1] == 2:
        return np.column_stack([r, np.zeros(len(r))])
    return r
