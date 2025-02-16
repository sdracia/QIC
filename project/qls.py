import numpy as np
from molecule import Molecule
from scipy.constants import h, k
from scipy.sparse import csr_array, sparray
from typing import Tuple, Optional, NamedTuple
from wigners import clebsch_gordan
import pandas as pd


class Polarization(NamedTuple):
    pi: float
    sp: float
    sm: float


def get_excitation_probabilities(
    molecule: Molecule,
    frequency: float,
    duration_us: float,
    rabi_rate_mhz: float,
    dephased: bool = False,
    coherence_time_us: float = 100.0,
    is_minus: bool = True,
) -> np.ndarray:
    """Returns the excitation probabilities for given frequency and other parameters

    Args:
        molecule (Molecule): The molecule to calculate the excitation probabilities for
        frequency (float): The frequency of the excitation pulse in MHz
        duration_us (float): The duration of the excitation pulse in microseconds
        rabi_rate_mhz (float): The Rabi rate in MHz
        dephased (bool): If True, the excitation is dephased
        coherence_time_us (float): The coherence time in us for rabi flopping
        is_minus (bool): If True, the excitation is for dm = -1
    Returns:
        np.ndarray: The excitation probabilities for each state
    """
    state_exc_probs = np.zeros(len(molecule.state_df))

    detunings = 2 * np.pi * (frequency - molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    omegas = rabi_rate_mhz * molecule.transition_df["coupling"].to_numpy(dtype=float)

    if dephased:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * ((1 - np.cos(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us) * np.exp(-duration_us / coherence_time_us)) / 2)
    else:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * np.sin(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us / 2) ** 2

    if is_minus:
        # state1 --> state2
        states_index = molecule.transition_df["index1"].to_numpy(dtype=int)
    else:
        # state2 --> state1
        states_index = molecule.transition_df["index2"].to_numpy(dtype=int)

    for i in range(len(molecule.transition_df)):
        state_exc_probs[states_index[i]] += transition_exc_probs[i]

    return state_exc_probs



def get_spectrum(
    molecule: Molecule,
    state_distribution: np.ndarray,
    duration_us: float,
    rabi_rate_mhz: float,
    max_frequency_mhz: float,
    scan_points: int,
    dephased: bool = True,
    coherence_time_us: float = 100.0,
    is_minus: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the spectrum for given parameters

    Args:
        molecule (Molecule): The molecule to calculate the spectrum for
        duration_us (float): The duration of the excitation pulse in microseconds
        rabi_rate_mhz (float): The Rabi rate in MHz
        max_frequency_mhz (float): The maximum frequency in MHz
        scan_points (int): The number of scan points
        dephased (bool): If True, the excitation is dephased
        is_minus (bool): If True, the excitation is for dm = -1; otherwise, dm = +1
    Returns:
        np.ndarray: The excitation probabilities for each frequency
    """
    frequencies = np.linspace(-max_frequency_mhz, max_frequency_mhz, scan_points)
    exc_probs = [
        np.dot(
            get_excitation_probabilities(molecule, frequency, duration_us, rabi_rate_mhz, dephased, coherence_time_us, is_minus),
            state_distribution,
        )
        for frequency in frequencies
    ]
    return frequencies, exc_probs




def excitation_matrix(
    molecule: Molecule,
    frequency: float,
    duration_us: float,
    rabi_rate_mhz: float,
    dephased: bool = False,
    coherence_time_us: float = 1000.0,
    is_minus: bool = True,
) -> sparray:
    """Returns the excitation probabilities for given frequency and other parameters

    Args:
        molecule (Molecule): The molecule to calculate the excitation probabilities for
        frequency (float): The frequency of the excitation pulse in MHz
        duration_us (float): The duration of the excitation pulse in microseconds
        rabi_rate_mhz (float): The Rabi rate in MHz
        dephased (bool): If True, the excitation is dephased
        coherence_time_us (float): The coherence time in us for rabi flopping
        is_minus (bool): If True, the excitation is for dm = -1
    Returns:
        np.ndarray: The excitation probabilities for each state
    """
    num_states = len(molecule.state_df)

    detunings = 2 * np.pi * (frequency - molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    omegas = rabi_rate_mhz * molecule.transition_df["coupling"].to_numpy(dtype=float)

    if dephased:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * ((1 - np.cos(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us) * np.exp(-duration_us / coherence_time_us)) / 2)
    else:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * np.sin(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us / 2) ** 2
        
    if is_minus:
        # state1 --> state2
        rows = molecule.transition_df["index2"].to_numpy(dtype=int)
        cols = molecule.transition_df["index1"].to_numpy(dtype=int)
        exc_matrix = csr_array((transition_exc_probs, (rows, cols)), shape=(num_states, num_states)) + csr_array((-transition_exc_probs, (cols, cols)), shape=(num_states, num_states))
    else:
        # state2 --> state1
        rows = molecule.transition_df["index1"].to_numpy(dtype=int)
        cols = molecule.transition_df["index2"].to_numpy(dtype=int)
        exc_matrix = csr_array((transition_exc_probs, (rows, cols)), shape=(num_states, num_states)) + csr_array((-transition_exc_probs, (cols, cols)), shape=(num_states, num_states))

    return exc_matrix


def get_thermal_distribution(molecule: Molecule, temperature: float) -> np.ndarray:
    """Returns the thermal distribution for a given temperature

    Args:
        molecule (Molecule): The molecule to calculate the thermal distribution for
        temperature (float): The temperature in Kelvin
    Returns:
        np.ndarray: The thermal distribution for each state
    """
    rotational_energy_ghz = molecule.state_df["rotation_energy_ghz"].to_numpy()



    state_distribution = np.exp(-h * rotational_energy_ghz * 1e9 / (k * temperature))


    state_distribution /= np.sum(state_distribution)

    return state_distribution




class States:
    """Class to store the state distribution of the given molecule"""

    def __init__(self, molecule: Molecule, temperature: Optional[float] = None):
        """Initializes the States object

        Args:
            molecule (Molecule): The molecule to store the state distribution for
            temperature (float): The temperature in Kelvin, if None, the state distribution is uniform
        """
        self.molecule = molecule
        self.num_states = len(molecule.state_df)
        self.j = molecule.state_df["j"].to_numpy(dtype=int)

        if temperature is not None:
            self.dist = get_thermal_distribution(molecule, temperature)
        else:
            self.dist = np.ones(len(molecule.state_df)) / len(molecule.state_df)

    def j_distribution(self) -> np.ndarray:
        """Returns the distribution of the rotational states"""
        return np.bincount(self.j, weights=self.dist)
