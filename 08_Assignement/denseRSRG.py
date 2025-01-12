import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================================================

def pauli_matrices():
  """
  pauli_matrices:
    Builds the Puali matrices.

  Returns
  -------
  s_x, s_y, s_z: tuple of np.ndarray
    Pauli matrices for a 2x2 system.
  """
  s_x = np.array([[0, 1], [1, 0]])
  s_y = np.array([[0, -1j], [1j, 0]])
  s_z = np.array([[1, 0], [0, -1]])
  return s_x, s_y, s_z

# ===========================================================================================================

def ising_hamiltonian(N, l):
  """
  ising_hamiltonian:
    Builds the Ising model Hamiltonian.

  Parameters
  ----------
  N : int
    Number of spins.
  l : float
    Interaction strength.

  Returns
  -------
  H : np.ndarray
    Ising Hamiltonian.
  """

  dim = 2 ** N
  H_nonint = np.zeros((dim, dim))
  H_int = np.zeros((dim, dim))
  
  s_x, _, s_z = pauli_matrices()
  
  for i in range(N):
    zterm = np.kron(np.eye(2**i), np.kron(s_z, np.eye(2**(N - i - 1))))
    H_nonint += zterm
    
  for i in range(N - 1):
    xterm = np.kron(np.eye(2**i), np.kron(s_x, np.kron(s_x, np.eye(2**(N - i - 2)))))
    H_int += xterm
  
  H =  l * H_nonint + H_int 
  return H

# ===========================================================================================================



def projector(H, d_eff):
  """
  projector:
    Builds the projector used to truncate the operators
  
  Parameters
  ----------
  H : np.ndarray
    Current Hamiltonian.
  d_eff : int
    Indices to keep.

  Returns
  -------
  proj : np.ndarray
    projector
  """

  # H is diagonalized, than the eigenvalues and corresponding eigenvectors are sorted from the lowest.
  # Than the projector is built building the matrix with the first d_eff eigenvalues

  eigvals, eigvecs = np.linalg.eigh(H)
  sorted_indices = np.argsort(eigvals)  
  eigvecs_sorted = eigvecs[:, sorted_indices]  
    
  proj = eigvecs_sorted[:, :d_eff]

  return proj

# ===========================================================================================================


def initialize_A_B(N):
  """
  initialize_A_B :
    It initializes the operators A_0 and B_0 for the inizial step, which are used for the interaction hamiltonian

  Parameters
  ----------
  N : int
    Number of spins.

  Returns
  -------
  A_0, B_0 : np.ndarray
    Operators for the interaction hamiltonian
  """

  s_x, _, _ = pauli_matrices()
  
  A_0 = np.kron(np.eye(2**(N - 1)), s_x)
  B_0 = np.kron(s_x, np.eye(2**(N - 1)))
  
  return A_0, B_0

# ===========================================================================================================

def compute_H_2N(N, H, A, B):  
  """
  compute_H_2N :
    Builds the doubled Hamiltonian 
  
  Parameters
  ----------
  N : int
    Number of Spins
  H : np.ndarray
    Current Hamiltonian.
  A, B : np.ndarray
    Interaction Hamiltonian components 

  Returns
  -------
  H_2N : np.ndarray
    double Hamiltonian
  """

  # The doubled Hamiltoninan is built with the operators given as input, with the kron product

  H_2N = np.kron(H, np.eye(2**(N))) + np.kron(np.eye(2**(N)), H) + np.kron(A, B)
  return H_2N

# ===========================================================================================================


def update_operators(N, H_2N, A, B):
  """
  update_operators :
    Updates the operators truncating them using the Projector built with the first d_eff eigenvectors
  
  Parameters
  ----------
  N : int
    Number of Spins
  H_2N : np.ndarray
    Doubled Hamiltonian.
  A, B : np.ndarray
    Interaction Hamiltonian components 

  Returns
  -------
  H_eff, A_eff, B_eff, P : np.ndarray
    It returns the truncated operators after the application of the projector operator, and the projector itself
  """
  # I compute the projector operator with the first d_eff eigenvectors
  P = projector(H_2N, d_eff=2**N)
  
  P_dagger = P.conj().T
  I_N = np.eye(2**N)

  # Compute the truncated operators, with the correct dimensions
  H_eff = P_dagger @ H_2N @ P
  A_eff = P_dagger @ np.kron(I_N, A) @ P
  B_eff = P_dagger @ np.kron(B, I_N) @ P
  
  return H_eff, A_eff, B_eff, P

# ===========================================================================================================

def real_space_rg(N, l, threshold, d_eff, max_iter=100):
  """
  real_space_rg :
    Runs the RSRG algorithm for the precise value of lambda
  
  Parameters
  ----------
  N : int
    Number of Spins
  l : float
    lambda values for the non-interacting hamiltonian
  threshold : float
    threshold value for the difference of consecutive GS energy densities
  d_eff : int
    Number eigenvectors and eigevalues to keep in the truncation of the operators
  max_iter : int
    maximum number of iterations that can be done in the algorithm

  Returns
  -------
  gs_energies_dict : dictionary
    dictionary of gs energy densities at each iteration
  eigvecs[:, 0] : list
    ground state eigenvector corresponding to the last iteration (biggest dimension)
  deltas_dict : dictionary
    dictionary of deltas (i.e. differences between consecutvie gs energy densisties) at each iteration
  actual_dim: int
    Dimension of the last computed hamiltonian
  """
  prev_energy_density = 10

  # Initialization of the Hamiltonian and the interaction operators
  H = ising_hamiltonian(N, l)
  A, B = initialize_A_B(N)
  

  actual_dim = N

  # Dictionaries to store the values
  gs_energies_dict = {}
  deltas_dict = {}


  for iteration in range(1, max_iter + 1):

    # Step 1: computation of the doubled Hamiltonian
    H_2N = compute_H_2N(N, H, A, B)

    # The dimension is doubled
    actual_dim = actual_dim * 2

    
    # Step 2: Diagonalization of the doubled Hamiltonian and computation of the first d_eff energy density and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H_2N)
    sorted_indices = np.argsort(eigvals)  

    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    eigvals = eigvals[:d_eff]
    eigvecs = eigvecs[:, :d_eff]
    
    current_energy_density = eigvals[0]/actual_dim

    gs_energies_dict[actual_dim] = current_energy_density

    # Check for convergence 
    delta = abs(current_energy_density - prev_energy_density)

    deltas_dict[actual_dim] = delta

    if delta > threshold:
      # Step 3: If not converged: update of the operators
      H, A, B, P = update_operators(N, H_2N, A, B)

    else:
      break

    # Update previous energy density for next iteration
    prev_energy_density = current_energy_density

  print(f"Convergence achieved at iteration {iteration}: ε = {current_energy_density}", '\n')
  print(f"Converged reached for a system with N = {actual_dim} sites, with precision: delta = {delta}")

  return gs_energies_dict, eigvecs[:, 0], deltas_dict, actual_dim


# ===========================================================================================================


def update_hamiltonian(N, l_values, threshold, max_iter=100):
  """
  update_hamiltonian :
    Runs the RSRG algorithm for different values of lambda, storing the results into dictionaries
  
  Parameters
  ----------
  N : int
    Number of Spins
  l_values : list
    lambda values for the non-interacting hamiltonian
  threshold : float
    threshold value for the difference of consecutive GS energy densities
  max_iter : int
    maximum number of iterations that can be done in the algorithm

  Returns
  -------
  eigenvalues_dict : dictionary
    dictionary over lambda of dictionary over N of gs energy densities
  last_eigenvectors_dict : dictionary
    dictionary of ground state eigenvector corresponding to the final dimension (biggest dimension)
  """
  # Initialize dictionaries to store eigenvalues and eigenvectors
  eigenvalues_dict = {}
  last_eigenvectors_dict = {}

  # Running over lambda
  for l in l_values:      
    d_eff = 2**N    
    # Call the RSRG algorithm
    normgs_eigval_dict, last_eigvec, deltas_dim, actual_dim = real_space_rg(N, l, threshold, d_eff, max_iter)  
    
    # Store the values
    eigenvalues_dict[l] = normgs_eigval_dict
    last_eigenvectors_dict[l] = last_eigvec
    
  print("-----------------------------------------")
    
  return eigenvalues_dict, last_eigenvectors_dict


# ===========================================================================================================


def plot_dict_N_GSen(eigenvalues, type):
    """
    plot_dict_N_GSen :
      Plots the results of update_hamiltonian

    Parameters
    ----------
    eigenvalues : dictionary
      dictionary of eigenvalues
    type : string
      Either "hvline" or "plot" depending on the type of representation one wants to obtain

    Returns:
      None
    """
    N_prec = 0
    
    if type == "hlines":
        for idx, (N, value) in enumerate(eigenvalues.items()):
            # For a correct representation of the lines
            delta_N = (N - N_prec)/2
            plt.hlines(value, N - delta_N, (3*N)/2, colors=f'C{idx}', linewidth=2.5, label=f'N={N}')
            N_prec = N
    elif type == "plot":
        for idx, (N, value) in enumerate(list(eigenvalues.items())):
            plt.plot(N,value, "s--", markersize = 4, label=f'N={N}')
    else:
        print("Invalid type")


# ===========================================================================================================


def Mag(N):
  """
  Mag :
    Computes the Magnetization operator to apply in order to compute the magnetization
  
  Parameters
  ----------
  N : int
    Number of Spins

  Returns
  -------
  M_z : np.ndarray
    The magnetization operator
  """
  _, _, s_z = pauli_matrices()  

  M_z = np.zeros((2**N, 2**N), dtype=complex)

  for i in range(N):
    M_z_i = np.kron(np.eye(2**i), np.kron(s_z, np.eye(2**(N - i - 1))))
    M_z += M_z_i

  M_z /= N
  return M_z


# ===========================================================================================================


def compute_magnetization(N, l_vals, threshold, d_eff, max_iter):
    """
    compute_magnetization :
      Computes the magnetization of a GS system, using the RSRG algorithm

    Parameters
    ----------
    N : int
      Number of Spins
    l_valeus : list
      lambda values for the non-interacting hamiltonian
    threshold : float
      threshold value for the difference of consecutive GS energy densities
    d_eff : int
      Number eigenvectors and eigevalues to keep in the truncation of the operators
    max_iter : int
      maximum number of iterations that can be done in the algorithm

    Returns:
    magnetizations : list
      values of the magnetization computed for a spicific ground state system
    """

    # List to store the values
    magnetizations = []  

    for l in l_vals:
        # Varying lambda I run the RSRG algorithm
        _, last_eigvec, _, _ = real_space_rg(N, l, threshold, d_eff, max_iter)

        # I compute the number of sites
        number_sites = int(np.log2(len(last_eigvec)))

        # I build the magnetization operator
        M_z = Mag(number_sites)

        # I compute the magnetization
        magnetization = np.dot(last_eigvec.conj().T, np.dot(M_z, last_eigvec)).real
        magnetizations.append(magnetization)

    return magnetizations