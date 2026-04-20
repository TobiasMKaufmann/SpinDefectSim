"""
analysis/sweep.py — ParameterSweep: structured sweeps with result collection.
"""
from __future__ import annotations

from typing import Optional, Callable
import numpy as np

from ..base.params import Defaults
from ..base.mixins import SweepMixin, PlottingMixin, SerializationMixin
from .ensemble import DefectEnsemble


class ParameterSweep(SweepMixin, PlottingMixin, SerializationMixin):
    """
    Run a Cartesian parameter sweep over physical parameters.

    Each grid point calls a user-supplied function that returns a dict of
    scalar results.  Use the :meth:`sweep` engine from SweepMixin directly,
    or the provided convenience recipes.

    Examples
    --------
    >>> from SpinDefectSim.analysis.sweep import ParameterSweep
    >>> import numpy as np
    >>>
    >>> ps = ParameterSweep(N_def=100, seed=0)
    >>>
    >>> # Sweep gate-field amplitude over an ensemble
    >>> def run(E_gate_z):
    ...     ens = ps.make_ensemble()
    ...     ens.compute_efields(E0_gate=(0, 0, E_gate_z))
    ...     exp = ens.to_experiment()
    ...     _, _, _, dS, tau_opt, dS_peak = exp.echo_static()
    ...     return dict(dS_peak=dS_peak, tau_opt_us=tau_opt * 1e6)
    >>>
    >>> results = ps.sweep(run, E_gate_z=np.linspace(0, 1e5, 5))
    >>> import pandas as pd; df = pd.DataFrame(results)
    """

    def __init__(
        self,
        N_def: int = 200,
        seed: int = 0,
        tau_echo_s: Optional[np.ndarray] = None,
        defaults: Optional[Defaults] = None,
    ):
        self.N_def    = N_def
        self.seed     = seed
        self.tau_echo = tau_echo_s
        self.defaults = defaults if defaults is not None else Defaults()
        self._results: list = []

    def make_ensemble(self, **override_defaults) -> DefectEnsemble:
        """
        Build a DefectEnsemble with defects already placed.

        Keyword arguments override individual Defaults fields, e.g.
        ``make_ensemble(T2echo=20e-6)``.
        """
        import dataclasses
        d   = dataclasses.replace(self.defaults, **override_defaults) if override_defaults else self.defaults
        ens = DefectEnsemble(N_def=self.N_def, defaults=d)
        ens.generate_defects(seed=self.seed)
        return ens

    # ── serialisation ────────────────────────────────────────────────────────
    def _serializable_arrays(self) -> dict:
        if not self._results:
            return {}
        keys = [k for k, v in self._results[0].items()
                if isinstance(v, (int, float, np.floating, np.integer))]
        return {k: np.array([r[k] for r in self._results]) for k in keys}
