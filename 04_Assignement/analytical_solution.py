import numpy as np
import math
import debugger as db
from scipy.special import factorial

def hermite(x, n):
  """
  hermite:
      Hermite polinomial of order 'n', 
      defined over the real space grid 'x'.

  Parameters
  ----------
  x : np.ndarray
      Real space grid.
  n : int
      Order of the polinomial.

  Returns
  -------
  herm_pol: np.ndarray
            Hermite polinomial of order 'n'.
  """
  # Pre-condition: n>=0
  if n<0:
    db.checkpoint(debug=True, msg=f"The order of the Hermite polynomial is not valid (n={n}, expected n>=0)", stop=True)

  # Coefficients set to 0 except for the one of order n.
  herm_coeffs = np.zeros(n+1)
  herm_coeffs[n] = 1
  
  # Actual computation of the polinomial over the space grid.
  herm_pol = np.polynomial.hermite.hermval(x, herm_coeffs)
  return herm_pol

# ===========================================================================================================

def harmonic_en(omega=1.0, n=0):
  """
  harmonic_en:
      Energy levels for an harwmonic potential.

  Parameters
  ----------
  omega : float, optional
          Angular frequency of the harmonic potential. By default 1.0.
  n : int, optional
      Energy level. By default 0.

  Returns
  -------
  energy: float
          Energy of level 'n'.
      
  """
  # Constants set to 1 in atomic units.
  hbar = 1.0
  
  # Pre-condition: n>=0
  if n<0:
    db.checkpoint(debug=True, msg=f"The order of the energy level is not valid (n={n}, expected n>=0)", stop=True)
    
  # Complete wavefunction.
  energy = hbar * omega * (n + 1/2)
  return energy

# ===========================================================================================================

def harmonic_wfc(x, omega, n):
  """
  harmonic_wfc:
      Wavefunction of order 'n' for an harmonic potential, 
      defined over the real space grid 'x'.
      
      V(x) = 0.5* m * omega * x**2
      
  Parameters
  ----------
  x : np.ndarray
      Real space grid.
  omega: float, optional
         Angular frequency of the harmonic potential. By default 1.0.
  n : int, optional
      Order of the wavefunction. By default 0 (ground state).

  Returns
  -------
  psi: np.ndarray
       Wavefucntion of order 'n'.
  """
  # Constants set to 1 in atomic units.
  hbar = 1.0
  m = 1.0
  
  # Components of the analytical solution for stationary states.
  prefactor = 1/np.sqrt(2**n * factorial(n, exact=False)) * ((m * omega)/(np.pi * hbar))**(0.25)
  x_coeff = np.sqrt(m * omega / hbar)
  exponential = np.exp(- (m * omega * x**2) / (2 * hbar))
  
  # Complete wavefunction.
  psi = prefactor * exponential * hermite(x_coeff * x, n)
  return psi


def analytic_eigenv(x_i, omega, k):

    eigenvalues_analy = []
    eigenvectors_analy = []

    for i in range(k):
        wavefunction = harmonic_wfc(x_i, omega, i)

        norm = np.sqrt(np.sum(np.abs(wavefunction)**2))
        eigenvector = wavefunction / norm

        eigenvalue = harmonic_en(omega, i)

        eigenvalues_analy.append(eigenvalue)
        eigenvectors_analy.append(eigenvector)

    eigenvalues_analy = np.array(eigenvalues_analy)
    eigenvectors_analy = np.array(eigenvectors_analy)

    return eigenvalues_analy, eigenvectors_analy