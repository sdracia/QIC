import numpy as np
import debugger as db

from scipy.linalg import eigh
from scipy.special import factorial
from scipy.sparse import diags



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
  if n < 0:
    db.checkpoint(debug=True, msg=f"The order of the Hermite polynomial is not valid (n={n}, expected n>=0)", stop=True)

  # Coefficients set to 0 except for the one of order n.
  herm_coeffs = np.zeros(n + 1)
  herm_coeffs[n] = 1
  
  # Actual computation of the polinomial over the space grid.
  herm_pol = np.polynomial.hermite.hermval(x, herm_coeffs)
  return herm_pol

# ===========================================================================================================

def harmonic_wfc(x, omega, n=0):
  """
  harmonic_wfc:
    Wavefunction of order 'n' for a harmonic potential, 
    defined over the real space grid 'x'.
  
    V(x) = 0.5 * omega * x**2
        
  Parameters
  ----------
  x : np.ndarray
    Spatial grid used for discretization.
  omega : float
    Angular frequency of the harmonic oscillator.
  n : int, optional
    Order of the wavefunction. By default 0 (ground state).

  Returns
  -------
  psi: np.ndarray
    Normalized wavefunction of order 'n'.
  """
  # Grid
  dx = x[1] - x[0]
  
  # Components of the analytical solution for stationary states.
  prefactor = 1 / np.sqrt(2**n * factorial(n)) * (omega / np.pi)**0.25
  exponential = np.exp(- (omega * x**2) / 2)
  
  # Complete wavefunction, with normalization
  psi = prefactor * exponential * hermite(x * np.sqrt(omega), n)
  psi_normalized = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)
  
  return psi_normalized