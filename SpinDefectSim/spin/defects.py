"""
spin/defects.py — Spin defect type definitions and built-in presets.

A :class:`DefectType` fully parameterises the spin Hamiltonian for a given
defect species.  Use :func:`get_defect` to retrieve a preset by name, or
instantiate :class:`DefectType` directly for fully custom defects.

Built-in presets
----------------
Name          Host        Spin   D₀ (GHz)   ms₀ idx   Notes
vb_minus      hBN         1      3.46        1         DEFAULT — VB⁻
nv_minus      diamond     1      2.87        1         NV⁻ centre
v_sic         4H-SiC      1      1.28        1         V2 silicon vacancy
p1            diamond     1/2    0           0         P1 nitrogen (Zeeman only)
cr_gaN        GaN         3/2    1.8         1         Cr impurity (approx.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

__all__ = [
    "DefectType",
    "get_defect",
    "list_defects",
    "VB_MINUS",
    "NV_MINUS",
    "V_SIC",
    "P1_CENTER",
    "CR_GAN",
]


@dataclass
class DefectType:
    """
    Physical parameters describing an ODMR-active spin defect species.

    Attributes
    ----------
    name         : identifier used in :func:`get_defect`
    spin         : quantum number S (0.5, 1, 1.5, 2, …)
    D0_Hz        : axial zero-field splitting D₀ (Hz). Zero for spin-1/2.
    E0_Hz        : transverse strain splitting E₀ (Hz).
    d_perp       : transverse E-field coupling d⊥ (Hz per V/m).
    d_parallel   : axial E-field coupling d∥ (Hz per V/m). Usually 0.
    gamma_Hz_T   : electron gyromagnetic ratio γₑ (Hz/T). Default 28 GHz/T.
    ms0_index    : index of the spin-0 (optically pumped) state in the
                   ordered basis {|+S⟩, |+S−1⟩, …, |−S⟩}.
                   Spin-1: 1 (|0⟩).  Spin-3/2: 1 (|+1/2⟩).  Spin-1/2: 0.
    notes        : reference / citation string.
    """

    name: str
    spin: float
    D0_Hz: float = 0.0
    E0_Hz: float = 0.0
    d_perp: float = 0.0
    d_parallel: float = 0.0
    gamma_Hz_T: float = 28e9
    ms0_index: int = 1
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"DefectType(name={self.name!r}, spin={self.spin}, "
            f"D0={self.D0_Hz / 1e9:.3f} GHz, "
            f"E0={self.E0_Hz / 1e6:.1f} MHz, "
            f"d⊥={self.d_perp:.3f} Hz/(V/m))"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Built-in presets
# ─────────────────────────────────────────────────────────────────────────────

VB_MINUS = DefectType(
    name="vb_minus",
    spin=1,
    D0_Hz=3.46e9,
    E0_Hz=50e6,
    d_perp=0.35,
    d_parallel=0.0,
    gamma_Hz_T=28e9,
    ms0_index=1,
    notes="VB⁻ spin defect in hexagonal boron nitride (hBN). "
          "Gottscholl et al., Nat. Mat. 2020.",
)

NV_MINUS = DefectType(
    name="nv_minus",
    spin=1,
    D0_Hz=2.870e9,
    E0_Hz=0.0,
    d_perp=0.17,
    d_parallel=0.0,
    gamma_Hz_T=28e9,
    ms0_index=1,
    notes="NV⁻ centre in diamond. D₀ = 2.87 GHz. "
          "d⊥ ≈ 0.17 Hz/(V/m). Dolde et al., Nat. Phys. 2011.",
)

V_SIC = DefectType(
    name="v_sic",
    spin=1,
    D0_Hz=1.28e9,
    E0_Hz=0.0,
    d_perp=0.1,
    d_parallel=0.0,
    gamma_Hz_T=28e9,
    ms0_index=1,
    notes="V2 silicon vacancy in 4H-SiC. D₀ ≈ 1.28 GHz.",
)

P1_CENTER = DefectType(
    name="p1",
    spin=0.5,
    D0_Hz=0.0,
    E0_Hz=0.0,
    d_perp=0.0,
    d_parallel=0.0,
    gamma_Hz_T=28e9,
    ms0_index=0,
    notes="P1 substitutional nitrogen in diamond (spin-1/2). "
          "Single Zeeman-split ODMR transition. No ZFS.",
)

CR_GAN = DefectType(
    name="cr_gaN",
    spin=1.5,
    D0_Hz=1.8e9,
    E0_Hz=0.0,
    d_perp=0.0,
    d_parallel=0.0,
    gamma_Hz_T=28e9,
    ms0_index=1,
    notes="Cr⁴⁺ in GaN, approximate values. Spin-3/2.",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, DefectType] = {
    "vb_minus": VB_MINUS,
    "nv_minus": NV_MINUS,
    "v_sic":    V_SIC,
    "p1":       P1_CENTER,
    "cr_gan":   CR_GAN,
    "cr_gaN":   CR_GAN,
}


def get_defect(name: Union[str, "DefectType"]) -> "DefectType":
    """
    Return a :class:`DefectType` by name, or pass through an existing instance.

    Parameters
    ----------
    name : str or DefectType
        If a string: one of ``'vb_minus'``, ``'nv_minus'``, ``'v_sic'``,
        ``'p1'``, ``'cr_gaN'``.  If already a DefectType, returned unchanged.

    Raises
    ------
    KeyError if the name is not in the registry.
    """
    if isinstance(name, DefectType):
        return name
    key = str(name).lower().replace("-", "_")
    if key not in _REGISTRY:
        available = list(dict.fromkeys(_REGISTRY.values()))  # deduplicated
        names = [dt.name for dt in available]
        raise KeyError(
            f"Unknown defect type {name!r}. Available: {names}"
        )
    return _REGISTRY[key]


def list_defects() -> None:
    """Print a formatted table of all built-in defect types."""
    seen: set[str] = set()
    rows: list[DefectType] = []
    for dt in _REGISTRY.values():
        if dt.name not in seen:
            seen.add(dt.name)
            rows.append(dt)
    header = f"{'Name':12s}  {'Spin':5s}  {'D₀ (GHz)':10s}  {'ms₀ idx':8s}  Notes"
    sep    = "─" * (len(header) + 4)
    print(header)
    print(sep)
    for dt in rows:
        print(
            f"{dt.name:12s}  {dt.spin:<5.1f}  {dt.D0_Hz / 1e9:<10.3f}  "
            f"{dt.ms0_index:<8d}  {dt.notes[:55]}"
        )
