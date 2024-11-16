import numpy as np
import math
import debugger as db
from scipy.special import factorial

def hermite(x, n):
    """
    hermite:
        Computes the Hermite polynomial of order 'n' over the real space grid 'x'.

    Parameters
    ----------
    x : np.ndarray
        Real space grid as a 1D numpy array.
    n : int
        Order of the Hermite polynomial. Must be a non-negative integer (n >= 0).

    Returns
    -------
    herm_pol : np.ndarray
        Hermite polynomial of order 'n', evaluated over the input grid 'x'.

    Pre-condition
    -------------
    - n >= 0. If n < 0, the function triggers a debugging checkpoint and halts execution.
    """
    # Pre-condition: n >= 0
    if n < 0:
        db.checkpoint(debug=True, msg=f"The order of the Hermite polynomial is not valid (n={n}, expected n>=0)", stop=True)

    # Coefficients set to 0 except for the one of order n.
    herm_coeffs = np.zeros(n + 1)
    herm_coeffs[n] = 1

    # Actual computation of the polynomial over the space grid.
    herm_pol = np.polynomial.hermite.hermval(x, herm_coeffs)
    return herm_pol


def harmonic_en(omega=1.0, n=0):
    """
    harmonic_en:
        Computes the energy levels for a quantum harmonic oscillator.

    Parameters
    ----------
    omega : float, optional
        Angular frequency of the harmonic potential. Default is 1.0.
    n : int, optional
        Quantum number (energy level index). Default is 0 (ground state). Must be n >= 0.

    Returns
    -------
    energy : float
        The energy of the n-th level, given by E_n = ℏ * ω * (n + 1/2).

    Pre-condition
    -------------
    - n >= 0. If n < 0, the function triggers a debugging checkpoint and halts execution.
    """
    # Constants set to 1 in atomic units.
    hbar = 1.0

    # Pre-condition: n >= 0
    if n < 0:
        db.checkpoint(debug=True, msg=f"The order of the energy level is not valid (n={n}, expected n>=0)", stop=True)

    # Energy of the n-th level.
    energy = hbar * omega * (n + 1 / 2)
    return energy


def harmonic_wfc(x, omega, n):
    """
    harmonic_wfc:
        Computes the wavefunction of order 'n' for a quantum harmonic oscillator, 
        defined over the real space grid 'x'.

    Parameters
    ----------
    x : np.ndarray
        Real space grid as a 1D numpy array.
    omega : float
        Angular frequency of the harmonic oscillator. Must be positive (ω > 0).
    n : int
        Quantum number (wavefunction order). Must be n >= 0.

    Returns
    -------
    psi : np.ndarray
        Normalized wavefunction of order 'n', evaluated over the input grid 'x'.

    Pre-condition
    -------------
    - n >= 0. If n < 0, the function triggers a debugging checkpoint and halts execution.
    """
    # Constants set to 1 in atomic units.
    hbar = 1.0
    m = 1.0

    # Components of the analytical solution for stationary states.
    prefactor = 1 / np.sqrt(2**n * factorial(n, exact=False)) * ((m * omega) / (np.pi * hbar))**0.25
    x_coeff = np.sqrt(m * omega / hbar)
    exponential = np.exp(-(m * omega * x**2) / (2 * hbar))

    # Complete wavefunction.
    psi = prefactor * exponential * hermite(x_coeff * x, n)
    return psi


def analytic_eigenv(x_i, omega, k):
    """
    analytic_eigenv:
        Computes the first 'k' eigenvalues and eigenvectors for a quantum harmonic oscillator,
        defined over a real-space grid 'x_i'.

    Parameters
    ----------
    x_i : np.ndarray
        Real space grid as a 1D numpy array.
    omega : float
        Angular frequency of the harmonic oscillator.
    k : int
        Number of eigenvalues and eigenvectors to compute. Must be k >= 0.

    Returns
    -------
    eigenvalues_analy : np.ndarray
        Array of the first 'k' eigenvalues of the harmonic oscillator.
    eigenvectors_analy : np.ndarray
        Array of the first 'k' normalized eigenvectors (wavefunctions).

    Pre-condition
    -------------
    - k >= 0. If k < 0, the function triggers a debugging checkpoint and halts execution.
    """
    eigenvalues_analy = []
    eigenvectors_analy = []

    for i in range(k):
        wavefunction = harmonic_wfc(x_i, omega, i)

        # Normalize the wavefunction.
        norm = np.sqrt(np.sum(np.abs(wavefunction)**2))
        eigenvector = wavefunction / norm

        # Compute the eigenvalue.
        eigenvalue = harmonic_en(omega, i)

        eigenvalues_analy.append(eigenvalue)
        eigenvectors_analy.append(eigenvector)

    eigenvalues_analy = np.array(eigenvalues_analy)
    eigenvectors_analy = np.array(eigenvectors_analy)

    return eigenvalues_analy, eigenvectors_analy
