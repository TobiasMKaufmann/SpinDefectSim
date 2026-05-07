"""spin — spin Hamiltonian, lineshapes, pulsed ODMR, defect types, rate model."""
from .matrices import spin_matrices, spin_1_matrices, spin_half_matrices
from .nuclear import (
    NuclearSpin,
    axial_A_tensor,
    isotropic_A_tensor,
    GAMMA_14N, GAMMA_15N, GAMMA_11B, GAMMA_10B, GAMMA_13C, GAMMA_29Si,
)
from .rates import (
    RateParams, RateModel, compute_odmr_contrast,
    NV_RATES, VB_RATES, VSIC_RATES, P1_RATES, CRGAN_RATES,
)
from .defects import DefectType, get_defect, list_defects, VB_MINUS, NV_MINUS, V_SIC, P1_CENTER, CR_GAN
from .hamiltonian import (
    SpinParams, SpinDefect,
    odmr_hamiltonian_Hz, vb_spin_hamiltonian_Hz,
    full_hyperfine_hamiltonian_Hz, odmr_transitions_Hz,
    diagonalize_hamiltonian, extract_ms0_like_transitions_Hz,
    _local_frame_rotation,
)
from .spectra import lorentzian, PL_model, ensemble_odmr_spectrum, ensemble_transitions_from_Efields
from .echo import (
    spin_echo_effective_fwhm, echo_detected_odmr_spectrum,
    ensemble_echo_signal, lock_in_difference_echo,
    lock_in_odmr_spectrum, lock_in_difference_ramsey,
)

__all__ = [
    # matrices
    "spin_matrices", "spin_1_matrices", "spin_half_matrices",
    # nuclear
    "NuclearSpin", "axial_A_tensor", "isotropic_A_tensor",
    "GAMMA_14N", "GAMMA_15N", "GAMMA_11B", "GAMMA_10B", "GAMMA_13C", "GAMMA_29Si",
    # rate model
    "RateParams", "RateModel", "compute_odmr_contrast",
    "NV_RATES", "VB_RATES", "VSIC_RATES", "P1_RATES", "CRGAN_RATES",
    # defects
    "DefectType", "get_defect", "list_defects",
    "VB_MINUS", "NV_MINUS", "V_SIC", "P1_CENTER", "CR_GAN",
    # hamiltonian
    "SpinParams", "SpinDefect",
    "odmr_hamiltonian_Hz", "vb_spin_hamiltonian_Hz",
    "full_hyperfine_hamiltonian_Hz", "odmr_transitions_Hz",
    "diagonalize_hamiltonian", "extract_ms0_like_transitions_Hz",
    # spectra
    "lorentzian", "PL_model", "ensemble_odmr_spectrum",
    "ensemble_transitions_from_Efields",
    # echo / ramsey
    "spin_echo_effective_fwhm", "echo_detected_odmr_spectrum",
    "ensemble_echo_signal", "lock_in_difference_echo",
    "lock_in_odmr_spectrum", "lock_in_difference_ramsey",
]
