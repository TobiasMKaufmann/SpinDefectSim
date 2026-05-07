"""
sensing/snr.py — Shot-noise floor, SNR, and integration-time estimates.

Functions
---------
noise_floor             single-shot σ_shot = 1/(C √N_ph)
snr                     SNR after N_avg cycles
n_avg_for_threshold     N_avg to reach target SNR
"""
from __future__ import annotations

import numpy as np
from ..base.params import DEFAULT


def noise_floor(contrast: float | None = None, n_photons: float | None = None) -> float:
    """
    Single-shot shot-noise floor: σ = 1 / (C √N_ph).

    Parameters
    ----------
    contrast   : optical spin contrast C (dimensionless fraction)
    n_photons  : photons detected per shot N_ph

    Returns
    -------
    sigma : float
    """
    C  = float(contrast  if contrast  is not None else DEFAULT.get_contrast())
    Np = float(n_photons if n_photons is not None else DEFAULT.n_photons)
    return 1.0 / (C * np.sqrt(Np))


def snr(
    delta_S,
    N_avg: float,
    contrast: float | None = None,
    n_photons: float | None = None,
) -> np.ndarray:
    """
    Signal-to-noise ratio: SNR = |ΔS| √N_avg / σ_shot.

    Parameters
    ----------
    delta_S  : echo / Ramsey lock-in signal ΔS (scalar or array)
    N_avg    : number of ON/OFF gate cycles averaged
    contrast : optical contrast C
    n_photons: photons per shot N_ph

    Returns
    -------
    snr_val : same shape as delta_S
    """
    sigma = noise_floor(contrast, n_photons)
    return np.asarray(delta_S) * np.sqrt(np.asarray(N_avg, dtype=float)) / sigma


def n_avg_for_threshold(
    delta_S,
    snr_target: float = 5.0,
    contrast: float | None = None,
    n_photons: float | None = None,
) -> np.ndarray:
    """
    Number of averaging cycles needed to reach *snr_target*.

    N_avg = (snr_target · σ_shot / |ΔS|)²

    Parameters
    ----------
    delta_S    : lock-in signal ΔS (scalar or array)
    snr_target : target SNR (default 5)

    Returns
    -------
    N_avg : same shape as delta_S
    """
    sigma = noise_floor(contrast, n_photons)
    return (snr_target * sigma / np.asarray(delta_S)) ** 2
