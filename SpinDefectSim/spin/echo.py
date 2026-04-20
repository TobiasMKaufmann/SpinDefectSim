"""
spin/echo.py — Hahn-echo and time-domain lock-in signals.

Functions
---------
spin_echo_effective_fwhm        1/(π T2)  Fourier-limited echo linewidth
echo_detected_odmr_spectrum     echo-detected (T2-limited) ensemble spectrum
ensemble_echo_signal            time-domain S(τ) for an ensemble
lock_in_difference_echo         ΔS(τ) = S_with − S_no  (lock-in output)
lock_in_odmr_spectrum           frequency-domain ΔPL = PL_with − PL_no
"""
from __future__ import annotations

import numpy as np
from .spectra import ensemble_odmr_spectrum


# ─────────────────────────────────────────────────────────────────────────────
#  Echo linewidth
# ─────────────────────────────────────────────────────────────────────────────
def spin_echo_effective_fwhm(T2_s: float) -> float:
    """
    Fourier-limited FWHM of the Hahn-echo-detected ESR line.

    After static inhomogeneity is refocused by the π-pulse the lineshape
    is a Lorentzian with FWHM = 1/(π T2).

    Parameters
    ----------
    T2_s : irreversible coherence time (s)

    Returns
    -------
    fwhm : float (Hz)
    """
    return 1.0 / (np.pi * float(T2_s))


# ─────────────────────────────────────────────────────────────────────────────
#  Echo-detected frequency-domain spectrum
# ─────────────────────────────────────────────────────────────────────────────
def echo_detected_odmr_spectrum(
    f_axis: np.ndarray,
    transitions_list: list,
    T2_s: float,
    contrast: float = 0.10,
) -> np.ndarray:
    """
    Echo-detected ensemble ODMR spectrum with T2-limited linewidth.

    Replaces the CW T2* linewidth with the echo T2 linewidth so that
    QH-induced shifts become individually resolved rather than buried in the
    broad inhomogeneous background.

    Parameters
    ----------
    f_axis           : 1-D frequency array (Hz)
    transitions_list : list length N, each shape-(2,) Hz array
    T2_s             : echo coherence time (s) — sets linewidth via 1/(π T2)
    contrast         : peak PL contrast (fraction)

    Returns
    -------
    pl : 1-D float array

    Notes
    -----
    Typical VB⁻ T2 values:
      RT   : T2 ≈ 100 ns → FWHM ≈ 3.2 MHz
      4 K  : T2 ≈ 2 µs  → FWHM ≈ 160 kHz
      1.8 K: T2 ≈ 10 µs → FWHM ≈  32 kHz  (50–300× better than CW)
    """
    fwhm_echo = spin_echo_effective_fwhm(T2_s)
    return ensemble_odmr_spectrum(f_axis, transitions_list, fwhm_echo, contrast)


# ─────────────────────────────────────────────────────────────────────────────
#  Time-domain echo signal
# ─────────────────────────────────────────────────────────────────────────────
def ensemble_echo_signal(
    freqs_list: list,
    tau_range_s: np.ndarray,
    T2_s: float,
    reference_Hz: float | list | None = None,
) -> np.ndarray:
    """
    Ensemble-averaged Hahn-echo amplitude vs free-precession time τ.

    Each defect i contributes:
        A_i(τ) = cos(2π (f_i − f_ref) τ) × exp(−τ / T2)

    Parameters
    ----------
    freqs_list   : list of arrays, each shape (2,) Hz — [f_lower, f_upper]
    tau_range_s  : 1-D array of τ values (s)
    T2_s         : irreversible coherence time (s)
    reference_Hz : reference frequency for phase demodulation.
                   - None   → mean of lower-branch transitions in freqs_list
                   - scalar → broadcast-subtracted from every defect’s freq
                   - list   → pairwise-subtracted per defect; each element may
                              be a scalar Hz value or a transition array
                              (in which case the lower branch t[0] is used)

    Returns
    -------
    signal : 1-D array shape (len(tau_range_s),), values in [−1, 1]
    """
    f_arr = np.array([t[0] for t in freqs_list], dtype=float)
    if reference_Hz is None:
        ref = float(np.mean(f_arr))
    elif np.isscalar(reference_Hz):
        ref = float(reference_Hz)
    else:
        # list/array: one reference per defect → pairwise subtraction
        ref = np.array(
            [t[0] if hasattr(t, "__len__") else float(t) for t in reference_Hz],
            dtype=float,
        )                                                 # (N,)
    delta = f_arr - ref                                   # (N,)
    tau   = np.asarray(tau_range_s, dtype=float)          # (M,)
    phase = 2.0 * np.pi * delta[:, None] * tau[None, :]   # (N, M)
    return np.mean(np.cos(phase), axis=0) * np.exp(-tau / float(T2_s))


# ─────────────────────────────────────────────────────────────────────────────
#  Lock-in / differential signals
# ─────────────────────────────────────────────────────────────────────────────
def lock_in_difference_echo(
    freqs_with: list,
    freqs_no: list,
    tau_range_s: np.ndarray,
    T2_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Modulated lock-in echo signal: ΔS(τ) = S_with(τ) - S_no(τ).

    Models a measurement that synchronously switches the FCI gate ON/OFF
    and demodulates; common-mode drifts cancel, leaving only the QH
    differential contribution.

    Parameters
    ----------
    freqs_with, freqs_no : per-defect transition lists (with/without QH)
    tau_range_s          : τ array (s)
    T2_s                 : echo coherence time (s)

    Returns
    -------
    S_with   : echo amplitude with QH
    S_no     : echo amplitude without QH (reference)
    S_diff   : ΔS = S_with - S_no
    """
    S_with = ensemble_echo_signal(freqs_with, tau_range_s, T2_s, freqs_no)
    S_no   = ensemble_echo_signal(freqs_no,   tau_range_s, T2_s, freqs_no)
    return S_with, S_no, S_with - S_no


def lock_in_odmr_spectrum(
    f_axis: np.ndarray,
    freqs_with: list,
    freqs_no: list,
    T2_s: float,
    contrast: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frequency-domain lock-in / difference spectrum: ΔPL = PL_with - PL_no.

    Uses the echo-limited T2 linewidth, making small QH-induced shifts visible.

    Parameters
    ----------
    f_axis               : 1-D frequency array (Hz)
    freqs_with, freqs_no : per-defect transition frequency lists
    T2_s                 : echo coherence time (s)
    contrast             : PL contrast fraction

    Returns
    -------
    diff_pl : ΔPL (signed)
    pl_with : absolute echo spectrum with QH
    pl_no   : absolute echo spectrum without QH
    """
    pl_with = echo_detected_odmr_spectrum(f_axis, freqs_with, T2_s, contrast)
    pl_no   = echo_detected_odmr_spectrum(f_axis, freqs_no,   T2_s, contrast)
    return pl_with - pl_no, pl_with, pl_no


def lock_in_difference_ramsey(
    freqs_with: list,
    freqs_no: list,
    tau_range_s: np.ndarray,
    T2_s: float,
    f_ref_Hz: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Modulated lock-in Ramsey signal: ΔS(τ) = S_with(τ) - S_no(τ).

    Models a measurement that synchronously switches the FCI gate ON/OFF
    and demodulates; common-mode drifts cancel, leaving only the QH
    differential contribution.

    Parameters
    ----------
    freqs_with, freqs_no : per-defect transition lists (with/without QH)
    tau_range_s          : τ array (s)
    T2_s                 : echo coherence time (s)

    Returns
    -------
    S_with   : Ramsey amplitude with QH
    S_no     : Ramsey amplitude without QH (reference)
    S_diff   : ΔS = S_with - S_no
    f_ref_Hz : common reference frequency
    """
    if f_ref_Hz is None:
        f_ref_Hz = float(np.mean([t[0] for t in freqs_no]))
    S_with = ensemble_echo_signal(freqs_with, tau_range_s, T2_s, f_ref_Hz)
    S_no   = ensemble_echo_signal(freqs_no,   tau_range_s, T2_s, f_ref_Hz)
    return S_with, S_no, S_with - S_no, f_ref_Hz
