"""
electrometry — E-field sensing from 2-D charge distributions.

  electrometry.py  —  :class:`ElectrometryExperiment` wraps a static charge
                       distribution and computes ODMR observables (transition
                       frequencies, CW spectra, frequency-shift maps) over a
                       2-D observation grid — the E-field analogue of
                       :class:`~SpinDefectSim.magnetometry.MagnetometryExperiment`.

Physical picture
----------------
Disorder charges (or a gate electrode generating a non-uniform field) polarise
the spin Hamiltonian via the d⊥ and d∥ coupling terms.  At each observation
point (x_obs, y_obs), the total E-field is computed from the supplied charge
positions using the screened-Coulomb model from :mod:`SpinDefectSim.sensing.efield`,
and the ODMR transition frequencies are returned.

Typical usage
-------------
>>> import numpy as np
>>> from SpinDefectSim.electrometry import ElectrometryExperiment
>>> from SpinDefectSim.base.params import Defaults
>>> from scipy.constants import e as e_charge

>>> # A single disorder charge on-axis
>>> charges = np.array([[0.0, 0.0, 0.0, e_charge]])   # (x, y, z, q)
>>> exp = ElectrometryExperiment(charges, Defaults(), z_defect=0.34e-9)
>>> f1, f2 = exp.transition_frequencies(0.0, 0.0)     # Hz
>>> E_vec  = exp.E_field(0.0, 100e-9)                 # (3,) V/m
>>> dfreq_map = exp.frequency_shift_map(
...     np.linspace(-300e-9, 300e-9, 40),
...     np.linspace(-300e-9, 300e-9, 40),
... )
"""

from .efield import (
    E_gate_bias,
    E_disorder_point_charges,
    apply_dielectric_transmission,
    ElectricFieldBuilder,
)
from .electrometry import ElectrometryExperiment

__all__ = [
    "E_gate_bias",
    "E_disorder_point_charges",
    "apply_dielectric_transmission",
    "ElectricFieldBuilder",
    "ElectrometryExperiment",
]
