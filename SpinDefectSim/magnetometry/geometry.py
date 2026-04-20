"""
magnetometry/geometry.py — 2-D sample geometries and current extraction.

A *SampleGeometry* encapsulates:

* The boundary polygon (vertices, CCW orientation assumed positive).
* A rectangular bounding grid used for bulk-current integration.
* Factory methods to construct edge-current segments and bulk-current
  grids from a given magnetization distribution.

Physical conventions
--------------------
The sample lies in the z = 0 plane.  Magnetization is always **out-of-plane**:
M(r) = M_z(r) ẑ  (units: A — equivalent to A/m² × sample thickness, or you
can interpret M_z directly as surface magnetic moment per unit area in A).

Edge current (A/m, sheet current density tangent to boundary):

    K_edge = M_z × t̂          t̂ = ẑ × n̂   (CCW tangent)

where n̂ is the outward-pointing boundary normal.

Bulk current (A/m², in-plane sheet current):

    Kx = ∂M_z/∂y,   Ky = −∂M_z/∂x

For *uniform* M_z the bulk term vanishes and only edge currents remain.

Public classes / functions
--------------------------
SampleGeometry     abstract base
PolygonGeometry    arbitrary closed polygon
SquareGeometry     axis-aligned square (convenience)
DiskGeometry       circular disk (approximated by N-gon)
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np


__all__ = [
    "SampleGeometry",
    "PolygonGeometry",
    "SquareGeometry",
    "DiskGeometry",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Abstract base
# ─────────────────────────────────────────────────────────────────────────────
class SampleGeometry(abc.ABC):
    """
    Abstract base class for 2-D sample geometries.

    Subclasses must implement ``boundary_vertices`` (CCW polygon vertices) and
    ``make_grid`` (a regular rectangular grid that covers the interior).
    """

    # ------------------------------------------------------------------
    #  Abstract interface
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def boundary_vertices(self) -> np.ndarray:
        """
        CCW-ordered polygon vertices, shape (N_verts, 2), in metres.
        The polygon is closed: the last edge connects vertex[−1] → vertex[0].
        """

    @abc.abstractmethod
    def make_grid(self, n_pts: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a rectangular grid *inside* the geometry.

        Parameters
        ----------
        n_pts : int  (per dimension)

        Returns
        -------
        xx, yy : ndarray shape (n_pts, n_pts) — x and y coordinate grids.
        """

    # ------------------------------------------------------------------
    #  Edge-current segments from a magnetization sampled at vertices
    # ------------------------------------------------------------------
    def edge_current_segments(
        self,
        M_at_vertices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose the boundary into straight wire segments with a linear
        surface current per unit length K = M_z (A/m).

        For segment i running from vertex[i] to vertex[i+1]:
          - The *average* M_z across the segment is ``(M[i] + M[i+1]) / 2``.
          - The *total current* through the segment is ``K_avg × |seg|`` (A).
          - The current direction is t̂ = (vertex[i+1] − vertex[i]) / |seg|.

        Parameters
        ----------
        M_at_vertices : ndarray shape (N_verts,)
            Out-of-plane magnetization sampled at each boundary vertex (A).

        Returns
        -------
        seg_start  : (N_verts, 2) — start point of each segment (m)
        seg_end    : (N_verts, 2) — end  point of each segment (m)
        seg_current: (N_verts,)   — circulating current for each segment (A)
                     (positive = CCW, matching the magnetization sign convention)

        Physical note
        -------------
        M_z [A] is the surface magnetisation (magnetic moment per unit area,
        A·m²/m² = A).  The boundary of a uniformly magnetised patch behaves
        like a **single closed current loop** carrying current I = M_z [A].
        Every segment of the boundary carries that *same* current in series —
        it is NOT distributed proportionally to segment length.

        For a spatially varying M_z the local loop current is taken as the
        average M_z between the two endpoint vertices, which is the natural
        linear-interpolation estimate consistent with the bulk ∇×M term.
        """
        verts = self.boundary_vertices
        N = len(verts)
        M = np.asarray(M_at_vertices, dtype=float)
        if M.shape != (N,):
            raise ValueError(
                f"M_at_vertices must have shape ({N},), got {M.shape}"
            )

        seg_start   = verts                            # shape (N, 2)
        seg_end     = np.roll(verts, -1, axis=0)       # vertex[i+1] mod N

        # Average M_z over segment (linear interpolation from endpoints)
        M_avg = 0.5 * (M + np.roll(M, -1))            # shape (N,)

        # Loop current = M_z [A].  Do NOT multiply by segment length:
        # the boundary is a series circuit, not a parallel current sheet.
        seg_current = M_avg

        return seg_start, seg_end, seg_current

    # ------------------------------------------------------------------
    #  Evaluate M on the boundary via a callable or grid interpolation
    # ------------------------------------------------------------------
    def sample_M_at_vertices(
        self,
        magnetization,
        *,
        xx: Optional[np.ndarray] = None,
        yy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate M_z at the boundary vertices.

        Parameters
        ----------
        magnetization : callable (x, y) → float  OR  ndarray shape (n, n)
            If callable, evaluated directly.  If an array, bilinear
            interpolation on the grid (xx, yy) is used.
        xx, yy : grids returned by ``make_grid``, required when
            *magnetization* is an array.

        Returns
        -------
        M_verts : ndarray shape (N_verts,)
        """
        verts = self.boundary_vertices
        if callable(magnetization):
            return np.array(
                [float(magnetization(x, y)) for x, y in verts], dtype=float
            )

        # Array interpolation
        if xx is None or yy is None:
            raise ValueError(
                "xx and yy grids must be provided when magnetization is an array."
            )
        from scipy.interpolate import RegularGridInterpolator
        M_arr = np.asarray(magnetization, dtype=float)
        x_1d  = xx[0, :]   # shape (n,)
        y_1d  = yy[:, 0]   # shape (n,)
        interp = RegularGridInterpolator(
            (y_1d, x_1d), M_arr, method="linear", bounds_error=False,
            fill_value=0.0,
        )
        pts = verts[:, ::-1]   # (y, x) order for RegularGridInterpolator
        return interp(pts)

    # ------------------------------------------------------------------
    #  Bulk current density on the interior grid
    # ------------------------------------------------------------------
    def bulk_current_density(
        self,
        M_grid: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute in-plane bulk current density from ∇×M_z.

        For M = M_z(x, y) ẑ:

            Kx(r) = ∂M_z/∂y,    Ky(r) = −∂M_z/∂x

        Gradients are computed with numpy's second-order central finite
        differences.

        Parameters
        ----------
        M_grid : ndarray shape (ny, nx)   — M_z values on the grid
        xx     : ndarray shape (ny, nx)   — x coordinates (from make_grid)
        yy     : ndarray shape (ny, nx)   — y coordinates (from make_grid)

        Returns
        -------
        Kx, Ky : ndarray shape (ny, nx), A/m
        """
        M  = np.asarray(M_grid, dtype=float)
        dx = xx[0, 1] - xx[0, 0]
        dy = yy[1, 0] - yy[0, 0]

        dMdy, dMdx = np.gradient(M, dy, dx)  # note: np.gradient(M, dy, dx)
        Kx = dMdy
        Ky = -dMdx
        return Kx, Ky

    # ------------------------------------------------------------------
    #  Mask: True inside the polygon
    # ------------------------------------------------------------------
    def interior_mask(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        """
        Boolean mask (shape matching xx/yy) that is True for grid points
        inside (or on) the boundary polygon.

        Uses the ray-casting algorithm (matplotlib.path if available,
        otherwise pure-numpy fallback).
        """
        verts = self.boundary_vertices
        pts   = np.column_stack([xx.ravel(), yy.ravel()])
        try:
            from matplotlib.path import Path
            path = Path(np.vstack([verts, verts[0]]))   # close the path
            inside = path.contains_points(pts, radius=1e-15)
        except ImportError:
            inside = _ray_cast_inside(pts, verts)
        return inside.reshape(xx.shape)


# ─────────────────────────────────────────────────────────────────────────────
#  Polygon geometry
# ─────────────────────────────────────────────────────────────────────────────
class PolygonGeometry(SampleGeometry):
    """
    Arbitrary closed polygon sample (vertices listed in CCW order).

    Parameters
    ----------
    vertices : array-like shape (N_verts, 2)   [m]
        Polygon corners in CCW order.  The boundary automatically closes
        (last vertex → first vertex), do not repeat the first vertex.
    n_boundary_pts : int
        Optional: if > 0, the polygon is re-sampled to this many equally
        spaced segment midpoints along its perimeter (denser boundary for
        accuracy of stray-field integration).  Default 0 (use raw vertices).
    """

    def __init__(
        self,
        vertices: np.ndarray,
        n_boundary_pts: int = 0,
    ):
        self._vertices = np.asarray(vertices, dtype=float)
        if n_boundary_pts > 0:
            self._vertices = _resample_polygon(self._vertices, n_boundary_pts)

    @property
    def boundary_vertices(self) -> np.ndarray:
        return self._vertices

    def make_grid(self, n_pts: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Rectangular bounding-box grid; use ``interior_mask`` to restrict."""
        xs = self._vertices[:, 0]
        ys = self._vertices[:, 1]
        x_1d = np.linspace(xs.min(), xs.max(), n_pts)
        y_1d = np.linspace(ys.min(), ys.max(), n_pts)
        return np.meshgrid(x_1d, y_1d)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: axis-aligned square
# ─────────────────────────────────────────────────────────────────────────────
class SquareGeometry(PolygonGeometry):
    """
    Axis-aligned square sample.

    Parameters
    ----------
    side   : float  [m]   side length
    center : (x, y) [m]   centre (default (0, 0))
    n_boundary_pts : int  boundary re-sampling (0 = use 4 corners only;
                          set to ~200 for accurate stray-field integrals)
    """

    def __init__(
        self,
        side: float,
        center: Tuple[float, float] = (0.0, 0.0),
        n_boundary_pts: int = 200,
    ):
        self.side   = float(side)
        self.center = np.asarray(center, dtype=float)
        a = self.side / 2.0
        cx, cy = self.center
        # CCW: bottom-left → bottom-right → top-right → top-left
        verts = np.array([
            [cx - a, cy - a],
            [cx + a, cy - a],
            [cx + a, cy + a],
            [cx - a, cy + a],
        ], dtype=float)
        super().__init__(verts, n_boundary_pts=n_boundary_pts)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: circular disk (N-gon approximation)
# ─────────────────────────────────────────────────────────────────────────────
class DiskGeometry(PolygonGeometry):
    """
    Circular disk approximated by a regular N-gon.

    Parameters
    ----------
    radius : float  [m]    disk radius
    center : (x, y) [m]    centre (default (0, 0))
    n_sides: int           polygon approximation (≥ 32 recommended)
    """

    def __init__(
        self,
        radius: float,
        center: Tuple[float, float] = (0.0, 0.0),
        n_sides: int = 128,
    ):
        self.radius = float(radius)
        self.center = np.asarray(center, dtype=float)
        phi    = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
        verts  = np.column_stack([
            self.center[0] + self.radius * np.cos(phi),
            self.center[1] + self.radius * np.sin(phi),
        ])
        # Pass n_boundary_pts=0 because we already have a smooth polygon
        super().__init__(verts, n_boundary_pts=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resample_polygon(verts: np.ndarray, n_pts: int) -> np.ndarray:
    """
    Re-sample *verts* to *n_pts* equally-spaced points along the perimeter.
    The polygon is closed: last edge connects verts[−1] → verts[0].
    """
    # Close the polygon temporarily
    closed = np.vstack([verts, verts[0]])
    seg_vecs   = np.diff(closed, axis=0)                      # (N, 2)
    seg_lens   = np.linalg.norm(seg_vecs, axis=1)             # (N,)
    perimeter  = seg_lens.sum()
    cum_len    = np.concatenate([[0.0], np.cumsum(seg_lens)])  # (N+1,)

    targets = np.linspace(0.0, perimeter, n_pts, endpoint=False)
    new_pts = np.empty((n_pts, 2), dtype=float)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seg_lens) - 1 and cum_len[j + 1] <= t:
            j += 1
        # Parametric position along segment j
        frac = (t - cum_len[j]) / (seg_lens[j] + 1e-300)
        new_pts[i] = closed[j] + frac * seg_vecs[j]
    return new_pts


def _ray_cast_inside(pts: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Pure-numpy ray-casting: return bool array of shape (N_pts,).
    """
    n = len(polygon)
    inside = np.zeros(len(pts), dtype=bool)
    px, py = pts[:, 0], pts[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        cond = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi
        )
        inside ^= cond
        j = i
    return inside
