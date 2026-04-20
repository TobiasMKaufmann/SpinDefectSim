"""
nv_sensing.magnetometry — Magnetometry from 2-D magnetization distributions.

Given a 2-D magnetic moment density M(r) defined over some sample geometry
(e.g. a square flake), this subpackage computes:

1. Edge currents  K_edge = M_z · t̂  (A/m) along the sample boundary.
2. Bulk currents  K_bulk = ∇×M      (A/m²) from non-uniform magnetization.
3. The stray magnetic field B(r_obs) anywhere above the sample, via
   Biot-Savart integration over edges and bulk.
4. VB⁻ transition frequencies from that B field (pure magnetometry,
   without an additional E field).

Typical usage
-------------
>>> import numpy as np
>>> from nv_sensing.magnetometry import SquareGeometry, MagnetometryExperiment
>>> from nv_sensing.base.params import Defaults

>>> geom = SquareGeometry(side=500e-9)                       # 500 nm square
>>> M_uniform = np.ones((50, 50)) * 1e-3                     # M_z = 1 mA (uniform)
>>> exp = MagnetometryExperiment(geom, M_uniform, Defaults(), z_defect=50e-9)
>>> B_map = exp.B_field_map(x_obs=np.linspace(-300e-9, 300e-9, 30),
...                          y_obs=np.linspace(-300e-9, 300e-9, 30))
>>> freqs = exp.transition_frequencies(x_obs=0.0, y_obs=0.0)

Public API
----------
geometry  : SampleGeometry, PolygonGeometry, SquareGeometry, DiskGeometry
bfield    : B_from_edge_segments, B_from_bulk_current_density,
            B_from_magnetization_grid
magnetometry : MagnetometryExperiment
"""

from .geometry import (
    SampleGeometry,
    PolygonGeometry,
    SquareGeometry,
    DiskGeometry,
)
from .magnetometry import MagnetometryExperiment
from .bfield import (
    B_from_wire_segment,
    B_from_edge_segments,
    B_from_bulk_current_density,
    B_from_magnetization_grid,
)

__all__ = [
    # geometry
    "SampleGeometry",
    "PolygonGeometry",
    "SquareGeometry",
    "DiskGeometry",
    # experiment
    "MagnetometryExperiment",
    # bfield sources
    "B_from_wire_segment",
    "B_from_edge_segments",
    "B_from_bulk_current_density",
    "B_from_magnetization_grid",
]
