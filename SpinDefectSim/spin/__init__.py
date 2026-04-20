"""spin — spin Hamiltonian, lineshapes, pulsed ODMR, defect types."""
from .matrices import spin_matrices, spin_1_matrices, spin_half_matrices
from .defects import DefectType, get_defect, list_defects, VB_MINUS, NV_MINUS, V_SIC, P1_CENTER, CR_GAN
from .hamiltonian import (
    SpinParams, SpinDefect,
    odmr_hamiltonian_Hz, vb_spin_hamiltonian_Hz,
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
    # defects
    "DefectType", "get_defect", "list_defects",
    "VB_MINUS", "NV_MINUS", "V_SIC", "P1_CENTER", "CR_GAN",
    # hamiltonian
    "SpinParams", "SpinDefect",
    "odmr_hamiltonian_Hz", "vb_spin_hamiltonian_Hz",
    "diagonalize_hamiltonian", "extract_ms0_like_transitions_Hz",
    # spectra
    "lorentzian", "PL_model", "ensemble_odmr_spectrum",
    "ensemble_transitions_from_Efields",
    # echo / ramsey
    "spin_echo_effective_fwhm", "echo_detected_odmr_spectrum",
    "ensemble_echo_signal", "lock_in_difference_echo",
    "lock_in_odmr_spectrum", "lock_in_difference_ramsey",
]
