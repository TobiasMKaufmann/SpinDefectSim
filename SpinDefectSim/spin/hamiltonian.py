"""
spin/hamiltonian.py — General ODMR spin Hamiltonian and SpinDefect class.

Supports spin-1/2, spin-1, spin-3/2, and any higher spin via the general
odmr_hamiltonian_Hz() builder.  Built-in defect types (VB⁻, NV⁻, …) live in
:mod:`spin.defects`.  Any combination of spin, ZFS, E-field coupling,
gyromagnetic ratio, and quantization axis is supported.

The electron-only Hamiltonian H/h (in Hz) for spin-S with lab-frame B and
quantization axis z′:

  H = D₀·(Sz′²  − S(S+1)/3·I)                          [ZFS axial,   S ≥ 1]
    + E₀·(Sx′²  − Sy′²)                                  [ZFS transverse, S ≥ 1]
    + d∥·Ez′·(Sz′²  − S(S+1)/3·I)                        [E axial,     S ≥ 1]
    + d⊥·[Ey′·(Sx′²−Sy′²) + Ex′·{Sx′,Sy′}]              [E transverse, S ≥ 1]
    + γₑ·(Bx′·Sx′ + By′·Sy′ + Bz′·Sz′)                  [Zeeman, all S]

When nuclear spins are included (via full_hyperfine_hamiltonian_Hz), the
Hilbert space is extended to H_e ⊗ H_n1 ⊗ H_n2 ⊗ … and the full Hamiltonian
is assembled in this tensor-product space.

Applied B and E are given in the *lab* frame and rotated into the defect's
local frame (z′ = quantization_axis) before H is assembled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.linalg import eigh

from .matrices import spin_matrices
from ..base.params import PhysicalParams, Defaults


# ─────────────────────────────────────────────────────────────────────────────
#  SpinParams — data container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SpinParams:
    """
    Parameters that fully specify a single-spin Hamiltonian.

    Attributes
    ----------
    D0, E0           : ZFS axial / transverse splitting (Hz). Ignored for S < 1.
    d_perp_Hz_per_Vpm: transverse E-field coupling d⊥ (Hz/(V/m)).
    B_T              : bias B-field in the *lab* frame, shape (3,), Tesla.
    gamma_e_Hz_per_T : gyromagnetic ratio (Hz/T). Default 28 GHz/T.
    d_parallel_Hz_per_Vpm: axial E coupling (Hz/(V/m)). Usually 0.
    B_extra_T        : additional lab-frame B-field, shape (3,).
    spin             : quantum number S (0.5, 1, 1.5, …). Default 1.
    ms0_index        : index of the |ms=0⟩-like state in {|+S⟩,…,|−S⟩}.
    quantization_axis: z′-axis direction in the lab frame, shape (3,).
                       Default [0, 0, 1] means no rotation is applied.
    """
    D0: float
    E0: float
    d_perp_Hz_per_Vpm: float
    B_T: np.ndarray                     # lab-frame field, shape (3,)
    gamma_e_Hz_per_T: float = 28e9
    d_parallel_Hz_per_Vpm: float = 0.0
    B_extra_T: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spin: float = 1.0
    ms0_index: int = 1
    quantization_axis: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 1.])
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Rotation helper
# ─────────────────────────────────────────────────────────────────────────────
def _local_frame_rotation(z_hat: np.ndarray) -> np.ndarray:
    """
    Build a 3×3 rotation matrix R such that  R @ v_lab = v_local,
    where the local z′-axis is z_hat.

    Rows of R are [x′, y′, z′] expressed in lab-frame coordinates,
    constructed via Gram-Schmidt.
    """
    z = np.asarray(z_hat, float)
    z = z / np.linalg.norm(z)
    seed = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])
    x = seed - (seed @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])   # shape (3, 3), rows = local axes in lab frame


# ─────────────────────────────────────────────────────────────────────────────
#  Core Hamiltonian builder (pre-rotated fields, any spin)
# ─────────────────────────────────────────────────────────────────────────────
def _odmr_hamiltonian_local_Hz(
    sp: SpinParams,
    B_local: np.ndarray,
    E_local: np.ndarray,
) -> np.ndarray:
    """
    Build H/h (Hz) assuming B_local and E_local are *already* in the defect's
    local frame.  Works for any spin S = sp.spin.
    """
    S = float(sp.spin)
    Sx, Sy, Sz, I = spin_matrices(S)
    Bx, By, Bz = map(float, B_local)
    Ex, Ey, Ez = map(float, E_local)

    H = np.zeros(Sz.shape, dtype=complex)

    if S >= 1.0:
        SSp1_3 = S * (S + 1) / 3.0
        H += sp.D0 * (Sz @ Sz - SSp1_3 * I)
        H += sp.E0 * (Sx @ Sx - Sy @ Sy)
        H += sp.d_parallel_Hz_per_Vpm * Ez * (Sz @ Sz - SSp1_3 * I)
        H += sp.d_perp_Hz_per_Vpm * (
            Ey * (Sx @ Sx - Sy @ Sy) + Ex * (Sx @ Sy + Sy @ Sx)
        )

    H += sp.gamma_e_Hz_per_T * (Bx * Sx + By * Sy + Bz * Sz)
    return H


# ─────────────────────────────────────────────────────────────────────────────
#  Public Hamiltonian API
# ─────────────────────────────────────────────────────────────────────────────
def odmr_hamiltonian_Hz(sp: SpinParams, E_vec_lab: np.ndarray) -> np.ndarray:
    """
    Build the ODMR spin Hamiltonian H/h (Hz) for a given lab-frame E-field.

    Supports any spin S (via sp.spin) and any quantization axis (via
    sp.quantization_axis).  Applied fields are rotated from the lab frame
    into the defect's local frame before H is built.

    Parameters
    ----------
    sp        : SpinParams
    E_vec_lab : (3,) array, E-field in the lab frame (V/m)

    Returns
    -------
    H : complex128 ndarray, shape (2S+1, 2S+1)
    """
    B_total = np.asarray(sp.B_T, float) + np.asarray(sp.B_extra_T, float)
    E_vec   = np.asarray(E_vec_lab, float)
    q       = np.asarray(sp.quantization_axis, float)

    if np.allclose(q, [0., 0., 1.]):          # fast path — no rotation
        return _odmr_hamiltonian_local_Hz(sp, B_total, E_vec)

    R = _local_frame_rotation(q / np.linalg.norm(q))
    return _odmr_hamiltonian_local_Hz(sp, R @ B_total, R @ E_vec)


# Backward-compatible alias
vb_spin_hamiltonian_Hz = odmr_hamiltonian_Hz


# ─────────────────────────────────────────────────────────────────────────────
#  Tensor-product helpers for hyperfine Hamiltonians
# ─────────────────────────────────────────────────────────────────────────────

def _kron_embed(
    ops: "dict[int, np.ndarray]",
    dims: "list[int]",
) -> np.ndarray:
    """
    Embed one or more operators into a tensor-product Hilbert space.

    Parameters
    ----------
    ops  : mapping from subspace index k → operator acting on that subspace.
           Any subspace *not* listed receives an identity of appropriate dimension.
    dims : list of subspace dimensions [d_0, d_1, …, d_{N-1}].

    Returns
    -------
    result : complex128 ndarray of shape (∏ dims, ∏ dims).

    Examples
    --------
    Embed S_z (electron, subspace 0) acting on space e ⊗ n (dims [3, 3])::

        _kron_embed({0: Sz_e}, [3, 3])          # → Sz_e ⊗ I_3

    Hyperfine term S_i · I_j for nucleus at subspace 1::

        _kron_embed({0: S_i, 1: I_j}, [3, 3])   # → S_i ⊗ I_j
    """
    parts = [
        ops.get(k, np.eye(d, dtype=complex)).astype(complex)
        for k, d in enumerate(dims)
    ]
    result = parts[0]
    for p in parts[1:]:
        result = np.kron(result, p)
    return result


def full_hyperfine_hamiltonian_Hz(
    sp: "SpinParams",
    E_vec_lab: np.ndarray,
    nuclear_spins: list,
) -> np.ndarray:
    """
    Build the complete electron + nuclear Hamiltonian H/h (Hz) including
    hyperfine coupling, nuclear Zeeman, and nuclear quadrupole terms.

    The Hilbert space is ordered as  H_e ⊗ H_n1 ⊗ H_n2 ⊗ …
    with total dimension  (2S+1) · Π_k(2I_k+1).

    If *nuclear_spins* is empty this returns the same result as
    :func:`odmr_hamiltonian_Hz` (only the electron spin, same dimension).

    Parameters
    ----------
    sp            : :class:`SpinParams` — electron spin parameters.
    E_vec_lab     : (3,) lab-frame electric field (V/m).
    nuclear_spins : list of :class:`~spin.nuclear.NuclearSpin`.

    Returns
    -------
    H : complex128 ndarray, shape (dim_total, dim_total).

    Notes
    -----
    The full Hamiltonian is::

        H / h = [H_electron ⊗ 𝟙_nuclear]
              + Σ_k [𝟙_e ⊗ … ⊗ H_nuclear_k ⊗ …]
              + Σ_k Σ_{i,j} A_k[i,j] · (S_i ⊗ I_{kj})

    where H_nuclear_k includes nuclear Zeeman (γ_n · B · I) and, for I ≥ 1,
    the electric quadrupole  P · (Iz² − I(I+1)/3 · 𝟙).

    All field components (B, E) are rotated into the defect local frame
    (z′ = sp.quantization_axis) before assembly.
    """
    if not nuclear_spins:
        return odmr_hamiltonian_Hz(sp, E_vec_lab)

    from .matrices import spin_matrices

    # ── Rotate fields into local frame ────────────────────────────────────────
    B_total = np.asarray(sp.B_T, float) + np.asarray(sp.B_extra_T, float)
    E_vec   = np.asarray(E_vec_lab, float)
    q       = np.asarray(sp.quantization_axis, float)
    if np.allclose(q, [0., 0., 1.]):
        B_local = B_total
        E_local = E_vec
    else:
        R = _local_frame_rotation(q / np.linalg.norm(q))
        B_local = R @ B_total
        E_local = R @ E_vec

    # ── Hilbert-space dimensions ──────────────────────────────────────────────
    dim_e  = int(round(2 * sp.spin + 1))
    dims_n = [int(round(2 * ns.spin + 1)) for ns in nuclear_spins]
    dims   = [dim_e] + dims_n          # [d_e, d_n1, d_n2, …]

    dim_total = 1
    for d in dims:
        dim_total *= d
    H = np.zeros((dim_total, dim_total), dtype=complex)

    # ── Electron spin Hamiltonian (subspace 0) ────────────────────────────────
    H_e = _odmr_hamiltonian_local_Hz(sp, B_local, E_local)
    H += _kron_embed({0: H_e}, dims)

    Se = spin_matrices(sp.spin)[:3]    # (Sx_e, Sy_e, Sz_e)

    # ── Nuclear contributions (one per nucleus) ───────────────────────────────
    for k, ns in enumerate(nuclear_spins):
        subspace = k + 1               # electron is subspace 0
        I_ops = spin_matrices(ns.spin)[:3]    # (Ix, Iy, Iz)
        dim_nk = dims_n[k]

        # Nuclear Zeeman:  γ_n · (Bx·Ix + By·Iy + Bz·Iz)
        H_nZ = ns.gamma_Hz_T * sum(
            float(B_local[i]) * I_ops[i] for i in range(3)
        )
        H += _kron_embed({subspace: H_nZ}, dims)

        # Nuclear electric quadrupole:  P · (Iz² − I(I+1)/3 · 𝟙)   [I ≥ 1]
        if ns.quadrupole_Hz != 0.0 and ns.spin >= 1.0:
            Iz = I_ops[2]
            I_id = np.eye(dim_nk, dtype=complex)
            H_Q = ns.quadrupole_Hz * (
                Iz @ Iz - ns.spin * (ns.spin + 1) / 3.0 * I_id
            )
            H += _kron_embed({subspace: H_Q}, dims)

        # Hyperfine  S · A · I  =  Σ_{i,j} A[i,j] · S_i ⊗ I_j
        for i in range(3):
            for j in range(3):
                A_ij = float(ns.A_tensor_Hz[i, j])
                if A_ij == 0.0:
                    continue
                H += A_ij * _kron_embed({0: Se[i], subspace: I_ops[j]}, dims)

    return H


def odmr_transitions_Hz(
    H: np.ndarray,
    electron_dim: int,
    ms0_basis_index: int,
    overlap_threshold: float = 0.1,
) -> np.ndarray:
    """
    Extract ODMR-relevant transition frequencies from a full electron+nuclear
    Hamiltonian.

    "ODMR-relevant" means the initial state has dominant electron |ms=0⟩
    character (overlap ≥ *overlap_threshold* with |ms=0⟩_e ⊗ 𝟙_n) and the
    final state does not (i.e. has |ms = ±1⟩_e character).

    Parameters
    ----------
    H                : full Hamiltonian (dim_total × dim_total), complex.
    electron_dim     : dimension of the electron spin subspace (2S+1).
    ms0_basis_index  : index of |ms=0⟩ in the electron Sz eigenbasis.
    overlap_threshold: minimum |ms=0⟩_e overlap to classify a state as
                       belonging to the |ms=0⟩ manifold.

    Returns
    -------
    freqs : float64 ndarray, ODMR transition frequencies (Hz), sorted ascending.
    """
    evals, evecs = eigh(H)
    dim_total  = H.shape[0]
    dim_nuclear = dim_total // electron_dim

    # Projector diagonal: 1 for rows that belong to |ms=0⟩_e ⊗ any nuclear state
    P0_diag = np.zeros(dim_total)
    start = ms0_basis_index * dim_nuclear
    P0_diag[start : start + dim_nuclear] = 1.0

    # Overlap of each eigenstate with the |ms=0⟩_e subspace
    overlaps_0 = np.einsum("ij,j,ij->i", evecs.conj(), P0_diag, evecs).real

    ms0_mask = overlaps_0 >= overlap_threshold

    freqs = []
    for i0 in np.where(ms0_mask)[0]:
        for j in np.where(~ms0_mask)[0]:
            df = abs(float(evals[j]) - float(evals[i0]))
            if df > 0.0:
                freqs.append(df)

    return np.sort(np.unique(np.round(freqs, decimals=0)))


def diagonalize_hamiltonian(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonalise a Hermitian Hamiltonian.

    Returns
    -------
    eigvals : real float64 array, shape (N,), ascending order
    eigvecs : complex128 array, shape (N, N), column eigenvectors
    """
    eigvals, eigvecs = eigh(H)
    return eigvals.real, eigvecs


def extract_ms0_like_transitions_Hz(
    evals_Hz: np.ndarray,
    evecs: np.ndarray,
    ms0_basis_index: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Find the eigenstate with maximum overlap with the |ms=0⟩-like basis state
    and return all transition frequencies from it to the other eigenstates.

    Works for any spin dimension dim = 2S+1.

    Parameters
    ----------
    evals_Hz        : eigenvalues (Hz), shape (dim,)
    evecs           : eigenvectors (column vectors), shape (dim, dim)
    ms0_basis_index : index of |ms=0⟩ in the ordered Sz eigenbasis.
                      Spin-1: 1.  Spin-3/2: 1 (|+1/2⟩).  Spin-1/2: 0 or 1.

    Returns
    -------
    freqs : ndarray, shape (dim−1,), sorted ascending — transition frequencies (Hz)
    i0    : index of the |ms=0⟩-like eigenstate in the energy-ordered basis
    """
    dim = len(evals_Hz)
    ket0 = np.zeros(dim, dtype=complex)
    ket0[ms0_basis_index] = 1.0
    overlaps = np.abs(evecs.conj().T @ ket0) ** 2
    i0 = int(np.argmax(overlaps))
    freqs = sorted(abs(evals_Hz[j] - evals_Hz[i0]) for j in range(dim) if j != i0)
    return np.array(freqs), i0


# ─────────────────────────────────────────────────────────────────────────────
#  SpinDefect — high-level wrapper
# ─────────────────────────────────────────────────────────────────────────────
class SpinDefect(PhysicalParams):
    """
    A single ODMR-active spin defect.

    The defect species is selected via *defect_type* (a name string or a
    :class:`~spin.defects.DefectType` instance).  Individual Hamiltonian
    parameters can be overridden with keyword arguments.

    Parameters
    ----------
    defect_type       : str or DefectType. Default ``"vb_minus"``.
                        Built-in names: ``'vb_minus'``, ``'nv_minus'``,
                        ``'v_sic'``, ``'p1'``, ``'cr_gaN'``.
    spin              : override the spin quantum number S.
    B_mT              : scalar bias field magnitude (mT), applied along the
                        lab x-axis (for in-plane fields, e.g. VB⁻/hBN).
    B_vec_mT          : 3-vector bias field in the *lab* frame (mT).
                        Overrides *B_mT* when supplied.
    quantization_axis : (3,) unit vector defining the defect's z′-axis in the
                        lab frame.  Default [0, 0, 1].
    D0_Hz, E0_Hz, d_perp, d_parallel : override the defect-type values.
    defaults          : :class:`~base.params.Defaults` instance.

    Examples
    --------
    >>> SpinDefect()                              # VB⁻, default params
    >>> SpinDefect("nv_minus", B_mT=3.0)          # NV⁻, 3 mT along x
    >>> SpinDefect("nv_minus", B_vec_mT=[0,0,3])  # NV⁻, B along z′
    >>> SpinDefect("p1", B_mT=10.0)               # spin-1/2 P1 centre
    >>> SpinDefect(D0_Hz=2.0e9, spin=1)           # fully custom spin-1
    >>> SpinDefect(quantization_axis=[1,0,0])     # VB⁻, z′ = lab x
    """

    def __init__(
        self,
        defect_type: Union[str, "DefectType", None] = None,
        *,
        spin: Optional[float] = None,
        B_mT: Optional[float] = None,
        B_vec_mT=None,
        quantization_axis=None,
        D0_Hz: Optional[float] = None,
        E0_Hz: Optional[float] = None,
        d_perp: Optional[float] = None,
        d_parallel: Optional[float] = None,
        nuclear_spins=None,
        defaults: Optional[Defaults] = None,
    ):
        super().__init__(defaults=defaults)

        from .defects import get_defect, VB_MINUS
        dt = VB_MINUS if defect_type is None else get_defect(defect_type)

        # Individual overrides take priority over defect-type defaults
        spin_val   = float(spin)    if spin       is not None else dt.spin
        D0_val     = float(D0_Hz)   if D0_Hz      is not None else dt.D0_Hz
        E0_val     = float(E0_Hz)   if E0_Hz      is not None else dt.E0_Hz
        d_perp_val = float(d_perp)  if d_perp     is not None else dt.d_perp
        d_par_val  = float(d_parallel) if d_parallel is not None else dt.d_parallel
        gamma_val  = dt.gamma_Hz_T

        # Bias field
        if B_vec_mT is not None:
            B_T = np.asarray(B_vec_mT, float) * 1e-3
        else:
            B_mT_val = self._resolve(B_mT, "B_mT")
            B_T = np.array([float(B_mT_val) * 1e-3, 0.0, 0.0])

        # Quantization axis
        if quantization_axis is not None:
            q = np.asarray(quantization_axis, float)
            q = q / np.linalg.norm(q)
        else:
            q = np.array([0., 0., 1.])

        self.spin_params = SpinParams(
            D0=D0_val,
            E0=E0_val,
            d_perp_Hz_per_Vpm=d_perp_val,
            d_parallel_Hz_per_Vpm=d_par_val,
            B_T=B_T,
            gamma_e_Hz_per_T=gamma_val,
            spin=spin_val,
            ms0_index=dt.ms0_index,
            quantization_axis=q,
        )
        self.defect_type = dt

        # Nuclear spins: explicit list overrides the defect-type default
        if nuclear_spins is not None:
            self.nuclear_spins = list(nuclear_spins)
        else:
            self.nuclear_spins = list(dt.nuclear_spins)

    # ── query methods ─────────────────────────────────────────────────────────
    def hamiltonian(self, E_vec_Vpm=(0., 0., 0.)) -> np.ndarray:
        """Return the electron-only H/h (Hz) as a (2S+1)×(2S+1) complex matrix."""
        return odmr_hamiltonian_Hz(self.spin_params, np.asarray(E_vec_Vpm, float))

    def full_hamiltonian(self, E_vec_Vpm=(0., 0., 0.)) -> np.ndarray:
        """
        Return H/h (Hz) in the full electron ⊗ nuclear Hilbert space.

        Dimension: (2S+1) · Π_k(2I_k+1).  If no nuclear spins are set
        (``self.nuclear_spins`` is empty) this is identical to
        :meth:`hamiltonian`.

        Parameters
        ----------
        E_vec_Vpm : (3,) lab-frame E-field (V/m).

        Returns
        -------
        H : complex128 ndarray.
        """
        return full_hyperfine_hamiltonian_Hz(
            self.spin_params,
            np.asarray(E_vec_Vpm, float),
            self.nuclear_spins,
        )

    def hyperfine_transitions(
        self,
        E_vec_Vpm=(0., 0., 0.),
        overlap_threshold: float = 0.1,
    ) -> np.ndarray:
        """
        ODMR transition frequencies (Hz) from the full electron+nuclear
        Hamiltonian, sorted ascending.

        These are transitions from states with dominant |ms=0⟩_e character
        to states with dominant |ms=±1⟩_e character (or ±3/2, etc.).  Each
        nuclear-spin substate produces a separate line, giving the hyperfine
        multiplet structure visible in a high-resolution ODMR spectrum.

        Parameters
        ----------
        E_vec_Vpm        : (3,) lab-frame E-field (V/m).
        overlap_threshold: minimum |ms=0⟩_e amplitude-squared to classify a
                           state as the initial manifold. Default 0.1.

        Returns
        -------
        freqs : float64 ndarray (Hz).
        """
        H = self.full_hamiltonian(E_vec_Vpm)
        sp = self.spin_params
        dim_e = int(round(2 * sp.spin + 1))
        return odmr_transitions_Hz(
            H, dim_e, sp.ms0_index, overlap_threshold=overlap_threshold
        )

    def diagonalize(
        self, E_vec_Vpm=(0., 0., 0.)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (eigenvalues_Hz, eigenvectors) for the given lab-frame E-field."""
        return diagonalize_hamiltonian(self.hamiltonian(E_vec_Vpm))

    def transition_frequencies(
        self, E_vec_Vpm=(0., 0., 0.)
    ) -> np.ndarray:
        """
        Return all ODMR transition frequencies from the |ms=0⟩-like state (Hz),
        sorted ascending.  Returns 2 values for spin-1, 1 for spin-1/2, etc.
        """
        evals, evecs = self.diagonalize(E_vec_Vpm)
        freqs, _ = extract_ms0_like_transitions_Hz(
            evals, evecs, self.spin_params.ms0_index
        )
        return freqs

    def zero_field_splitting(self) -> float:
        """Return the maximum transition splitting at zero E-field (Hz)."""
        freqs = self.transition_frequencies(np.zeros(3))
        return float(freqs[-1] - freqs[0]) if len(freqs) >= 2 else float(freqs[0])

    def to_experiment(
        self,
        E_vec_Vpm=(0., 0., 0.),
        B_extra_T=(0., 0., 0.),
        *,
        sensing: str = "both",
    ) -> "SensingExperiment":
        """
        Wrap this single defect into a :class:`~sensing.protocols.SensingExperiment`.

        Enables the same protocol methods (CW ODMR, Ramsey, Hahn-echo, SNR …)
        to be called on a single defect as on a full ensemble.

        Parameters
        ----------
        E_vec_Vpm : (3,) lab-frame E-field for the *signal* branch (V/m).
        B_extra_T : (3,) lab-frame stray B-field for the *signal* branch (T).
        sensing   : which field(s) contribute to the contrast.

            ``"E"``    — only E_vec_Vpm differs between signal and reference.
            ``"B"``    — only B_extra_T differs.
            ``"both"`` — both differ.

        Returns
        -------
        SensingExperiment with N = 1

        Examples
        --------
        E-field sensing on a single NV centre::

            defect = SpinDefect("nv_minus", B_mT=3.0)
            exp = defect.to_experiment(E_vec_Vpm=[1e4, 0, 0], sensing="E")
            tau, *_, dS_peak = exp.echo_static()

        B-field sensing (stray field from a magnetic sample)::

            exp = defect.to_experiment(B_extra_T=[0, 0, 5e-5], sensing="B")
            f_with, f_no, df = exp.cw_odmr(f_axis)

        Both fields active simultaneously::

            exp = defect.to_experiment(
                E_vec_Vpm=[1e4, 0, 0],
                B_extra_T=[0, 0, 5e-5],
                sensing="both",
            )
        """
        from ..sensing.protocols import SensingExperiment
        from .spectra import ensemble_transitions_from_Efields

        sensing = sensing.lower()
        if sensing not in ("e", "b", "both"):
            raise ValueError(f"sensing must be 'E', 'B', or 'both'; got {sensing!r}")

        use_E = sensing in ("e", "both")
        use_B = sensing in ("b", "both")

        E_w = np.asarray(E_vec_Vpm, float).reshape(1, 3) if use_E else np.zeros((1, 3))
        B_w = np.asarray(B_extra_T, float).reshape(1, 3) if use_B else np.zeros((1, 3))
        B_no = np.zeros((1, 3))

        sp = self.spin_params
        tr_with = ensemble_transitions_from_Efields(E_w, sp, B_extra_fields=B_w)
        tr_no   = ensemble_transitions_from_Efields(np.zeros((1, 3)), sp, B_extra_fields=B_no)

        exp = SensingExperiment(
            sp, sp, E_w, defaults=self.defaults,
            B_extra_fields_with=B_w,
            B_extra_fields_no=B_no,
        )
        exp._tr_with = tr_with
        exp._tr_no   = tr_no
        return exp

    def __repr__(self) -> str:
        sp  = self.spin_params
        B_mT = np.linalg.norm(sp.B_T) * 1e3
        nuc = f", {len(self.nuclear_spins)} nuclear spin(s)" if self.nuclear_spins else ""
        return (
            f"SpinDefect(type={self.defect_type.name!r}, S={sp.spin}, "
            f"D0={sp.D0 / 1e9:.3f} GHz, E0={sp.E0 / 1e6:.1f} MHz, "
            f"|B|={B_mT:.2f} mT{nuc})"
        )
