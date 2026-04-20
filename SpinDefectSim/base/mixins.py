"""
base/mixins.py — Reusable behaviour mixins for all domain classes.

Mixins
------
PlottingMixin       quick_plot() / save_fig() convenience wrappers
SerializationMixin  save() / load() via numpy .npz archives
SweepMixin          sweep() — generic 1-D / 2-D parameter scan
"""
from __future__ import annotations

import os
import time
import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
class PlottingMixin:
    """
    Mixin that adds lightweight matplotlib helpers.

    Classes that inherit this can call ``self.quick_plot(x, y, ...)`` and
    ``self.save_fig(fig, name)`` without repeating boilerplate everywhere.

    The mixin is *display-library agnostic*: matplotlib is imported lazily
    so classes that don't plot never pay the import cost.
    """

    # Subclasses may override this to change the default output directory.
    _fig_output_dir: str = "figures"

    def quick_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        label: str = "",
        ax=None,
        **kwargs,
    ):
        """
        Plot *y* vs *x* on *ax* (or a fresh figure/axes if ax is None).

        Returns (fig, ax).
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(np.asarray(x), np.asarray(y), label=label or None, **kwargs)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if label:
            ax.legend()
        return fig, ax

    def save_fig(self, fig, name: str, *, dpi: int = 150, fmt: str = "png") -> str:
        """
        Save *fig* to ``<_fig_output_dir>/<name>.<fmt>``.

        Creates the output directory if it does not exist.
        Returns the absolute path of the saved file.
        """
        import matplotlib.pyplot as plt

        os.makedirs(self._fig_output_dir, exist_ok=True)
        path = os.path.join(self._fig_output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────
class SerializationMixin:
    """
    Mixin that adds .save() / .load() via numpy .npz archives.

    Each domain object decides which arrays it wants to persist by overriding
    ``_serializable_arrays()`` (return a dict of {name: array}).

    Example
    -------
    >>> class MyObj(SerializationMixin):
    ...     def _serializable_arrays(self):
    ...         return {"E_fields": self.E_fields, "positions": self.positions}
    ...
    >>> obj.save("run_01")
    'run_01.npz'
    >>> obj2 = MyObj.load("run_01.npz")
    """

    def _serializable_arrays(self) -> Dict[str, np.ndarray]:
        """Override in subclasses to declare which arrays to persist."""
        return {}

    def save(self, path: str) -> str:
        """
        Save serializable arrays to *path*.npz.

        Returns the written file path.
        """
        arrays = self._serializable_arrays()
        if not arrays:
            raise NotImplementedError(
                f"{type(self).__name__} does not override _serializable_arrays()"
            )
        np.savez_compressed(path, **arrays)
        fpath = path if path.endswith(".npz") else path + ".npz"
        return fpath

    @classmethod
    def load(cls, path: str) -> Dict[str, np.ndarray]:
        """
        Load a previously saved .npz and return a plain dict of arrays.

        Usage note: Full round-trip reconstruction is class-specific.
        Subclasses should provide a ``from_npz(path)`` classmethod if needed.
        """
        data = np.load(path, allow_pickle=False)
        return dict(data)


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter sweep helper
# ─────────────────────────────────────────────────────────────────────────────
class SweepMixin:
    """
    Mixin that provides a generic 1-D / 2-D parameter sweep engine.

    Usage
    -----
    >>> class MySim(SweepMixin):
    ...     def run_single(self, V0_meV: float, Nqh: int) -> dict:
    ...         # ... compute something ...
    ...         return {"peak_dS": 0.01}
    ...
    >>> sim = MySim()
    >>> results = sim.sweep(
    ...     V0_meV=[0.5, 1.0, 2.0],
    ...     Nqh=[1, 2, 3],
    ...     fn=sim.run_single,
    ...     verbose=True,
    ... )

    The returned ``results`` is a list of dicts, each extended with the swept
    parameter values so you can easily convert to a DataFrame.
    """

    def sweep(
        self,
        fn: Callable[..., Any],
        verbose: bool = False,
        **param_grids: Sequence,
    ) -> List[Dict[str, Any]]:
        """
        Cartesian sweep over all keyword argument grids.

        Parameters
        ----------
        fn         : callable(param_dict) → dict
                     Will be called with each combination of swept parameters
                     as keyword arguments.
        verbose    : print progress lines if True
        **param_grids :
            Each keyword maps a parameter name to a sequence of values.
            For example: ``V0_meV=[0.5, 1.0]``, ``Nqh=[1, 2, 3]``.

        Returns
        -------
        results : list of dicts.  Each dict contains the swept parameter values
                  merged with whatever `fn` returned.
        """
        names: List[str] = list(param_grids.keys())
        grids: List[Sequence] = list(param_grids.values())
        combos = list(itertools.product(*grids))
        n_total = len(combos)
        results: List[Dict[str, Any]] = []

        t0 = time.perf_counter()
        for i, combo in enumerate(combos):
            params = dict(zip(names, combo))
            if verbose:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (n_total - i - 1) if i > 0 else 0.0
                print(
                    f"  [{i+1}/{n_total}]  {params}  "
                    f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                    flush=True,
                )
            out = fn(**params)
            if isinstance(out, dict):
                row = {**params, **out}
            else:
                row = {**params, "_result": out}
            results.append(row)

        if verbose:
            print(f"  Sweep complete in {time.perf_counter()-t0:.2f}s", flush=True)
        return results
