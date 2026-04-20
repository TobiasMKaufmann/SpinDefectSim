"""
sensing/sequences.py — Pulse-sequence timing for VB⁻ / NV sensing.

Classes
-------
PulseSequence      base class (abstract)
RamseySequence     π/2 – τ – π/2
HahnEchoSequence   π/2 – τ – π – τ – π/2
XY8Sequence        π/2 – [τ/2 – π × 8 – τ/2] – π/2

Each class stores hardware gate times and exposes:

    total_time(tau_s)          wall-clock time per shot (s)
    repetition_rate(tau_s)     shots per second (Hz)
    n_avg_in_time(T, tau_s)    shots achievable in integration time T (s)

Usage
-----
>>> from nv_sensing.sensing.sequences import HahnEchoSequence
>>> seq = HahnEchoSequence(t_pi_half_s=12e-9, t_pi_s=24e-9)
>>> print(seq.total_time(5e-6))     # total shot duration at τ = 5 µs
>>> print(seq.repetition_rate(5e-6))
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


__all__ = [
    "PulseSequence",
    "RamseySequence",
    "HahnEchoSequence",
    "XY8Sequence",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Abstract base
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PulseSequence(ABC):
    """
    Abstract base class for spin pulse sequences with hardware gate times.

    Default gate times are representative for VB⁻ in hBN::

        π/2 pulse  : t_pi_half_s = 10 ns
        π   pulse  : t_pi_s      = 20 ns
        Readout    : t_readout_s = 300 ns
        Init./reset: t_init_s    = 1 µs

    Sub-classes implement the three abstract methods that describe the
    sequence topology.
    """

    t_pi_half_s: float = 10e-9    # π/2 pulse duration (s)
    t_pi_s:      float = 20e-9    # π   pulse duration (s)
    t_readout_s: float = 300e-9   # optical readout window (s)
    t_init_s:    float = 1.0e-6   # spin-initialisation laser pulse (s)

    # ── abstract topology ────────────────────────────────────────────────────
    @abstractmethod
    def n_pi_half_pulses(self) -> int:
        """Number of π/2 pulses in the sensing block."""

    @abstractmethod
    def n_pi_pulses(self) -> int:
        """Number of π refocusing pulses in the sensing block."""

    @abstractmethod
    def n_free_precession_intervals(self) -> int:
        """
        Number of free-precession intervals of length *τ* in the sensing block.

        For a simple Ramsey this is 1; for a Hahn-echo it is 2 (one each side
        of the π pulse).
        """

    # ── derived timing ───────────────────────────────────────────────────────
    def pulse_time_s(self) -> float:
        """Total time spent inside microwave pulses (s)."""
        return (
            self.n_pi_half_pulses() * self.t_pi_half_s
            + self.n_pi_pulses() * self.t_pi_s
        )

    def total_time(self, tau_s: float | np.ndarray) -> float | np.ndarray:
        """
        Wall-clock time of a single sequence shot (s).

        T_total = t_init + t_pulses + n_fp × τ + t_readout

        Parameters
        ----------
        tau_s : free-precession half-interval τ (s), scalar or array

        Returns
        -------
        T_total : scalar or ndarray (s)
        """
        tau = np.asarray(tau_s, dtype=float)
        T = (
            self.t_init_s
            + self.pulse_time_s()
            + self.n_free_precession_intervals() * tau
            + self.t_readout_s
        )
        # Return a plain float when input was scalar
        return float(T) if T.ndim == 0 else T

    def repetition_rate(self, tau_s: float | np.ndarray) -> float | np.ndarray:
        """
        Shot repetition rate R(τ) = 1 / total_time(τ)  (Hz).

        Parameters
        ----------
        tau_s : τ (s), scalar or array

        Returns
        -------
        R : float or ndarray (Hz)
        """
        return 1.0 / self.total_time(tau_s)

    def n_avg_in_time(
        self,
        T_int_s: float,
        tau_s: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Number of averaging cycles achievable in integration time *T_int_s*.

        N_avg = T_int_s × R(τ)

        Parameters
        ----------
        T_int_s : total integration time (s)
        tau_s   : τ (s), scalar or array

        Returns
        -------
        N_avg : float or ndarray
        """
        return float(T_int_s) * self.repetition_rate(tau_s)

    # ── helpers ──────────────────────────────────────────────────────────────
    def summary(self, tau_s: float) -> dict:
        """
        Return a dict with all timing quantities at a given τ.

        Keys: 'tau_s', 'pulse_time_s', 'total_time_s', 'repetition_rate_Hz',
              'n_pi_half', 'n_pi', 'n_fp_intervals'
        """
        return {
            "sequence":          self.__class__.__name__,
            "tau_s":             float(tau_s),
            "t_pi_half_s":       self.t_pi_half_s,
            "t_pi_s":            self.t_pi_s,
            "t_init_s":          self.t_init_s,
            "t_readout_s":       self.t_readout_s,
            "n_pi_half":         self.n_pi_half_pulses(),
            "n_pi":              self.n_pi_pulses(),
            "n_fp_intervals":    self.n_free_precession_intervals(),
            "pulse_time_s":      self.pulse_time_s(),
            "total_time_s":      self.total_time(tau_s),
            "repetition_rate_Hz": self.repetition_rate(tau_s),
        }

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"t_π/2={self.t_pi_half_s * 1e9:.0f} ns, "
            f"t_π={self.t_pi_s * 1e9:.0f} ns, "
            f"t_ro={self.t_readout_s * 1e9:.0f} ns, "
            f"t_init={self.t_init_s * 1e6:.2f} µs, "
            f"n_π/2={self.n_pi_half_pulses()}, "
            f"n_π={self.n_pi_pulses()}, "
            f"n_fp={self.n_free_precession_intervals()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Concrete sequences
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RamseySequence(PulseSequence):
    """
    Ramsey free-induction decay sequence.

    Topology
    --------
    init – π/2 – [τ] – π/2 – readout

    Total time:  T = t_init + t_π/2 + τ + t_π/2 + t_readout

    This is limited by T₂* (inhomogeneous broadening).
    """

    def n_pi_half_pulses(self) -> int:
        return 2

    def n_pi_pulses(self) -> int:
        return 0

    def n_free_precession_intervals(self) -> int:
        return 1


@dataclass
class HahnEchoSequence(PulseSequence):
    """
    Hahn-echo (CPMG-1) sequence.

    Topology
    --------
    init – π/2 – [τ] – π – [τ] – π/2 – readout

    Total time:  T = t_init + t_π/2 + τ + t_π + τ + t_π/2 + t_readout
                   = t_init + 2·t_π/2 + t_π + 2τ + t_readout

    Refocuses static inhomogeneities; sensitive to fluctuations at ~ 1/(2τ).
    Limited by T₂ (echo coherence time), which is typically ≫ T₂*.
    """

    def n_pi_half_pulses(self) -> int:
        return 2

    def n_pi_pulses(self) -> int:
        return 1

    def n_free_precession_intervals(self) -> int:
        return 2   # one interval on each side of the π refocusing pulse


@dataclass
class XY8Sequence(PulseSequence):
    """
    XY-8 dynamical decoupling sequence.

    Topology (one block)
    --------------------
    init – π/2 – [τ/2 – π_X – τ – π_Y – τ – π_X – τ – π_Y –
                         τ – π_Y – τ – π_X – τ – π_Y – τ/2] – π/2 – readout

    The 8 π-pulses are interleaved with 8 full-τ intervals plus two
    half-τ gaps at the ends, which is equivalent to 9 half-intervals on
    each side → total free-precession = 9τ … but the conventional way to
    parameterise XY-8 is by the *inter-pulse spacing* τ, giving:

        Free precession time = 16τ   (8 echo sub-intervals × 2τ each)

    Total time:  T = t_init + 2·t_π/2 + 8·t_π + 16τ + t_readout

    Suppresses both X and Y noise components to high order.
    """

    def n_pi_half_pulses(self) -> int:
        return 2

    def n_pi_pulses(self) -> int:
        return 8

    def n_free_precession_intervals(self) -> int:
        return 16   # 8 echo intervals × 2 τ each
