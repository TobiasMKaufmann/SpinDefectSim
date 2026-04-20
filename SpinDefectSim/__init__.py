"""
ODMR — spin-defect sensing simulation library.

Package layout
--------------
ODMR/
├── base/          Physical defaults, mixins (plot / save / sweep)
├── spin/          Spin Hamiltonian, ODMR spectra, echo / Ramsey signals, defect types
├── sensing/       E and B field sources, protocols, pulse sequences, SNR
│   ├── efield.py  — gate bias + screened-Coulomb E-fields
│   ├── bfield.py  — Biot-Savart B-fields from 2-D magnetization distributions
│   ├── protocols.py — SensingExperiment (CW ODMR, Ramsey, Hahn-echo, SNR)
│   ├── sequences.py — pulse sequences (Ramsey, Hahn-echo, XY8, …)
│   └── snr.py       — noise floor, SNR, averaging-time estimates
├── electrometry/  E-field imaging over 2-D scan grids (charges → E → ODMR)
│   └── electrometry.py — ElectrometryExperiment
├── magnetometry/  B-field imaging over 2-D scan grids (M → Biot-Savart → ODMR)
│   ├── geometry.py    — sample geometry (SquareGeometry, DiskGeometry, …)
│   └── magnetometry.py — MagnetometryExperiment
└── analysis/      Ensemble builder, parameter sweeps

Quick start — single defect E-field sensing
-------------------------------------------
>>> from SpinDefectSim.spin.hamiltonian import SpinDefect
>>>
>>> defect = SpinDefect("nv_minus", B_mT=3.0)
>>> exp = defect.to_experiment(E_vec_Vpm=[1e4, 0, 0], sensing="E")
>>> tau, *_, dS_peak = exp.echo_static()

Quick start — single defect B-field sensing
-------------------------------------------
>>> defect = SpinDefect("nv_minus", B_mT=3.0)
>>> exp = defect.to_experiment(B_extra_T=[0, 0, 5e-5], sensing="B")
>>> pl_w, pl_no, dpl = exp.cw_odmr(f_axis_Hz)

Quick start — ensemble E-field sensing
--------------------------------------
>>> import numpy as np
>>> from SpinDefectSim.base.params import Defaults
>>> from SpinDefectSim.analysis.ensemble import DefectEnsemble
>>>
>>> ens = DefectEnsemble(N_def=500, defaults=Defaults())
>>> ens.generate_defects(seed=0)
>>> ens.compute_efields(E0_gate=(0, 0, 5e4))
>>> exp = ens.to_experiment(sensing="E")
>>> tau, S_w, S_no, dS, tau_opt, dS_peak = exp.echo_static()

Quick start — ensemble B-field sensing
--------------------------------------
>>> from SpinDefectSim.magnetometry import SquareGeometry
>>>
>>> geom = SquareGeometry(side=500e-9, n_boundary_pts=300)
>>> ens.compute_bfields(magnetization=lambda x, y: 1e-3, geometry=geom)
>>> exp = ens.to_experiment(sensing="B")

Quick start — combined E + B sensing
-------------------------------------
>>> ens.compute_efields(...)
>>> ens.compute_bfields(...)
>>> exp = ens.to_experiment(sensing="both")

Quick start — magnetometry scan maps
-------------------------------------
>>> from SpinDefectSim.magnetometry import SquareGeometry, MagnetometryExperiment
>>> geom = SquareGeometry(side=500e-9, n_boundary_pts=300)
>>> scan = MagnetometryExperiment(geom, magnetization=lambda x, y: 1e-3,
...                               defaults=Defaults(), z_defect=50e-9)
>>> Bz   = scan.B_z_map(np.linspace(-400e-9, 400e-9, 40),
...                      np.linspace(-400e-9, 400e-9, 40))
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nv_sensing")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"
