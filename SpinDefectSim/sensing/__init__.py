"""sensing — sensing protocols, SNR, and pulse sequences.

Field sources live in their respective modules:
  SpinDefectSim.electrometry.efield  — gate bias, screened-Coulomb disorder E-fields
  SpinDefectSim.magnetometry.bfield  — Biot-Savart B-fields from 2-D current distributions

High-level experiment and utilities:

  protocols.py  — SensingExperiment (CW ODMR, Ramsey, Hahn-echo, SNR)
  sequences.py  — pulse sequences (Ramsey, Hahn-echo, XY8, …)
  snr.py        — noise floor, SNR, averaging-time estimates
"""
from .protocols import SensingExperiment
from .snr import noise_floor, snr, n_avg_for_threshold
from .sequences import PulseSequence, RamseySequence, HahnEchoSequence, XY8Sequence

__all__ = [
    "SensingExperiment",
    "noise_floor", "snr", "n_avg_for_threshold",
    "PulseSequence", "RamseySequence", "HahnEchoSequence", "XY8Sequence",
]
