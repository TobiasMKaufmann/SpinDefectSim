"""
sensing/protocols.py — High-level sensing protocol class.

SensingExperiment encapsulates:
  - CW ODMR spectra
  - Ramsey time-domain lock-in signal
  - Hahn-echo time-domain lock-in signal
  - Frequency-domain echo lock-in spectrum
  - SNR / integration-time estimates
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from ..base.params import PhysicalParams, Defaults
from ..base.mixins import PlottingMixin
from ..spin.hamiltonian import SpinParams
from ..spin.spectra import ensemble_transitions_from_Efields, ensemble_odmr_spectrum
from ..spin.echo import (
    ensemble_echo_signal,
    lock_in_difference_ramsey,
    lock_in_difference_echo,
    lock_in_odmr_spectrum,
)
from .snr import noise_floor, snr, n_avg_for_threshold


# ─────────────────────────────────────────────────────────────────────────────
#  SensingExperiment
# ─────────────────────────────────────────────────────────────────────────────
class SensingExperiment(PhysicalParams, PlottingMixin):
    """
    Container for a single sensing experiment (one ensemble, one protocol).

    Pass computed E-field and/or B-field arrays and spin parameters; call a
    protocol method to get theoretical observables.

    Parameters
    ----------
    sp_with          : SpinParams  — spin parameters *with* the signal field
    sp_no            : SpinParams  — reference spin parameters (no signal)
    E_fields_Vpm     : (N, 3) per-defect E-field array for the signal branch (V/m)
    defaults         : Defaults
    quantization_axes : optional (N, 3) per-defect quantization axes
    B_extra_fields_with : optional (N, 3) per-defect stray B (T) for signal branch
    B_extra_fields_no   : optional (N, 3) per-defect stray B (T) for reference branch

    Notes
    -----
    Transitions are computed lazily on first access.  ``DefectEnsemble.to_experiment()``
    pre-injects ``_tr_with`` / ``_tr_no`` to avoid recomputing them.

    Examples
    --------
    >>> exp = SensingExperiment(sp_with=sp, sp_no=sp_ref, E_fields_Vpm=E_arr)
    >>> tau, S_with, S_no, dS, tau_opt, dS_peak = exp.echo_static()
    """

    def __init__(
        self,
        sp_with: SpinParams,
        sp_no: SpinParams,
        E_fields_Vpm: np.ndarray,
        defaults: Optional[Defaults] = None,
        quantization_axes: Optional[np.ndarray] = None,
        B_extra_fields_with: Optional[np.ndarray] = None,
        B_extra_fields_no: Optional[np.ndarray] = None,
    ):
        super().__init__(defaults=defaults)
        self.sp_with            = sp_with
        self.sp_no              = sp_no
        self.E_fields           = np.asarray(E_fields_Vpm, dtype=float)
        self.quantization_axes  = (
            None if quantization_axes is None
            else np.asarray(quantization_axes, dtype=float)
        )
        self.B_extra_fields_with = (
            None if B_extra_fields_with is None
            else np.asarray(B_extra_fields_with, dtype=float)
        )
        self.B_extra_fields_no = (
            None if B_extra_fields_no is None
            else np.asarray(B_extra_fields_no, dtype=float)
        )

        # Cached transition lists (computed on demand)
        self._tr_with: Optional[list] = None
        self._tr_no:   Optional[list] = None

    # ── lazy evaluation ──────────────────────────────────────────────────────
    @property
    def transitions_with(self) -> list:
        if self._tr_with is None:
            self._tr_with = ensemble_transitions_from_Efields(
                self.E_fields, self.sp_with,
                quantization_axes=self.quantization_axes,
                B_extra_fields=self.B_extra_fields_with)
        return self._tr_with

    @property
    def transitions_no(self) -> list:
        if self._tr_no is None:
            N = len(self.E_fields)
            self._tr_no = ensemble_transitions_from_Efields(
                np.zeros_like(self.E_fields), self.sp_no,
                quantization_axes=self.quantization_axes,
                B_extra_fields=self.B_extra_fields_no)
        return self._tr_no

    # ── CW ODMR ─────────────────────────────────────────────────────────────
    def cw_odmr(self, f_axis_Hz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CW ODMR spectra (signal, reference) and their difference.

        Returns
        -------
        pl_with, pl_no, dpl
        """
        d    = self.defaults
        fwhm = 1.0 / (np.pi * d.T2star)
        con  = d.get_contrast()
        pl_with = ensemble_odmr_spectrum(f_axis_Hz, self.transitions_with, fwhm, con)
        pl_no   = ensemble_odmr_spectrum(f_axis_Hz, self.transitions_no,   fwhm, con)
        return pl_with, pl_no, pl_with - pl_no

    # ── Ramsey ───────────────────────────────────────────────────────────────
    def ramsey(
        self,
        tau_range_s: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Ramsey free-induction decay lock-in signal.

        Returns
        -------
        tau_s, S_with, S_no, dS, tau_opt, dS_peak
        """
        d   = self.defaults
        T2s = d.T2star
        if tau_range_s is None:
            tau_range_s = np.linspace(0, 3.0 * T2s, 500)
        tau_s = np.asarray(tau_range_s, dtype=float)
        S_with, S_no, dS, _ = lock_in_difference_ramsey(
            self.transitions_with, self.transitions_no, tau_s, T2s)
        idx = int(np.argmax(np.abs(dS)))
        return tau_s, S_with, S_no, dS, float(tau_s[idx]), float(np.abs(dS[idx]))

    # ── Hahn-echo ────────────────────────────────────────────────────────────
    def echo_static(
        self,
        tau_range_s: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Hahn-echo lock-in signal.

        Returns
        -------
        tau_s, S_with, S_no, dS, tau_opt, dS_peak
        """
        d   = self.defaults
        T2e = d.T2echo
        if tau_range_s is None:
            tau_range_s = np.linspace(0, 3.0 * T2e, 600)
        tau_s = np.asarray(tau_range_s, dtype=float)
        S_with, S_no, dS = lock_in_difference_echo(
            self.transitions_with, self.transitions_no, tau_s, T2e)
        idx = int(np.argmax(np.abs(dS)))
        return tau_s, S_with, S_no, dS, float(tau_s[idx]), float(np.abs(dS[idx]))

    # ── Echo frequency-domain lock-in ────────────────────────────────────────
    def echo_odmr_lockIn(
        self,
        f_axis_Hz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Frequency-domain echo lock-in spectrum ΔPL = PL_with − PL_no.

        Returns
        -------
        dpl, pl_with, pl_no
        """
        return lock_in_odmr_spectrum(
            f_axis_Hz,
            self.transitions_with,
            self.transitions_no,
            T2_s=self.defaults.T2echo,
            contrast=self.defaults.get_contrast(),
        )

    # ── SNR helpers ──────────────────────────────────────────────────────────
    def snr(self, delta_S, N_avg: float) -> np.ndarray:
        """SNR after *N_avg* gate ON/OFF cycles."""
        d = self.defaults
        return snr(delta_S, N_avg, d.get_contrast(), d.n_photons)

    def n_avg_to_detect(self, delta_S, snr_target: float = 5.0) -> np.ndarray:
        """Number of averaging cycles to reach *snr_target*."""
        d = self.defaults
        return n_avg_for_threshold(delta_S, snr_target, d.get_contrast(), d.n_photons)

    def __repr__(self) -> str:
        d = self.defaults
        return (
            f"SensingExperiment("
            f"N_def={len(self.E_fields)}, "
            f"T2*={d.T2star*1e9:.0f} ns, "
            f"T2_echo={d.T2echo*1e6:.0f} µs, "
            f"contrast={d.get_contrast():.3f})"
        )
