import numpy as np
import matplotlib.pyplot as plt

import denseRSRG as df



def initialize_operators(m, l):
  """
  initialize_operators :
    It initializes the operators A_0 and B_0 for the inizial step, which are used for the interaction hamiltonian

  Parameters
  ----------
  m : int
    Number of spins of half the system.
  l : float
    Strength value of the trasverse field
  Returns
  -------
  A_L_0, B_L_0, A_R_0, B_R_0, H_L_0, H_R_0 : np.ndarray
    Operators at the initial step n = 0
  """
  s_x, _, s_z = df.pauli_matrices()
  
  A_L_0 = A_R_0 = np.eye(2**m)
  B_L_0 = np.kron(np.eye(2**(m - 1)), s_x)
  B_R_0 = np.kron(s_x, np.eye(2**(m - 1)))
  
  # Hamiltonians that describe the left and right system
  H_L_0 = H_R_0 = df.ising_hamiltonian(m,l) 
  
  return A_L_0, B_L_0, A_R_0, B_R_0, H_L_0, H_R_0

# ===========================================================================================================

def compute_H_LR(H_L, H_R, AL, BL, AR, BR, l):
  """
  compute_H_LR :
    Builds the Hamiltonians H_{L+1} (H_{R+1}) which describe the enlarged left (right)
    block, where one site has been added
  
  Parameters
  ----------
  l : float
    Strength value of the trasverse field
  H_L, H_R, AL, BL, AR, BR : np.ndarray
    Hamiltonian and interacting operators before the system is enlarged

  Returns
  -------
  H_L1, H_R1 : np.ndarray
    Enlarged Hamiltonians for the left and right subsystems where one site has been 
    added for both.
  """
  s_x, _, s_z = df.pauli_matrices()
  H_L1 = np.kron(H_L, np.eye(2)) + np.kron(AL, l * s_z) + np.kron(BL, s_x)
  H_R1 = np.kron(np.eye(2), H_R) + np.kron(l * s_z, AR) + np.kron(s_x, BR)
  return H_L1, H_R1

# ===========================================================================================================

def update_operators(A_L, A_R):
  """
  update_operators :
    Updates the operators A_L, A_R, B_L and B_R which are used to enlarge the Hamiltonian
    with more sites. Their dimensionality must be fixed
  
  Parameters
  ----------
  A_L, A_R : np.ndarray
    The previous operator used to enlarge the Hamiltonian in the previous step

  Returns
  -------
  A_L_new, B_L_new, A_R_new, B_R_new : np.ndarray
    The updated version of the operator used in the next step of the algorithm
    to enlarge the system with 2 sites. Their dimensionality has been fixed
  """
  s_x, _, _ = df.pauli_matrices()
  
  A_L_new = np.kron(A_L, np.identity(2))
  B_L_new = np.kron(np.eye(2), s_x)
  
  A_R_new = np.kron(np.identity(2), A_R)
  B_R_new = np.kron(s_x, np.eye(2))
  
  return A_L_new, B_L_new, A_R_new, B_R_new

# ===========================================================================================================

def compute_H_2m(H_L1, H_R1, m):
  """
  compute_H_2m :
    Computes the Hamiltonian for the system of size 2m + 2
    from the previous step
  
  Parameters
  ----------
  H_L1, H_R1 : np.ndarray
    Operator representing the enlarged left and right block by adding one site

  Returns
  -------
  H_2m : np.ndarray
    Hamiltonian for the system of size 2m+2
  """
  s_x, _, _ = df.pauli_matrices()
  I_m = np.identity(2**(m))
  I_m1 = np.identity(H_R1.shape[1])
  
  H_int = np.kron(s_x, s_x)
  H_LR = np.kron(I_m, np.kron(H_int, I_m))
  
  H_2m = np.kron(H_L1, I_m1) + np.kron(I_m1, H_R1) + H_LR
  
  return H_2m



def get_reduced_density_matrix(psi, loc_dim, n_sites, keep_indices,
    print_rho=False):
    """
    Parameters
    ----------
    psi : ndarray
        state of the Quantum Many-Body system
    loc_dim : int
        local dimension of each single site of the QMB system
    n_sites : int
        total number of sites in the QMB system
    keep_indices (list of ints):
        Indices of the lattice sites to keep.
    print_rho : bool, optional
        If True, it prints the obtained reduced density matrix], by default False

    Returns
    -------
    ndarray
        Reduced density matrix
    """
    if not isinstance(psi, np.ndarray):
        raise TypeError(f'density_mat should be an ndarray, not a {type(psi)}')

    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f'loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}')

    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f'n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}')

    # Ensure psi is reshaped into a tensor with one leg per lattice site
    psi_tensor = psi.reshape(*[loc_dim for _ in range(int(n_sites))])
    # Determine the environmental indices
    all_indices = list(range(n_sites))
    env_indices = [i for i in all_indices if i not in keep_indices]
    new_order = keep_indices + env_indices
    # Rearrange the tensor to group subsystem and environment indices
    psi_tensor = np.transpose(psi_tensor, axes=new_order)
    # print(f"Reordered psi_tensor shape: {psi_tensor.shape}")
    # Determine the dimensions of the subsystem and environment for the bipartition
    subsystem_dim = np.prod([loc_dim for i in keep_indices])
    env_dim = np.prod([loc_dim for i in env_indices])
    # Reshape the reordered tensor to separate subsystem from environment
    psi_partitioned = psi_tensor.reshape((subsystem_dim, env_dim))
    # Compute the reduced density matrix by tracing out the env-indices
    RDM = np.tensordot(psi_partitioned, np.conjugate(psi_partitioned), axes=([1], [1]))
    # Reshape rho to ensure it is a square matrix corresponding to the subsystem
    RDM = RDM.reshape((subsystem_dim, subsystem_dim))

    # PRINT RHO
    if print_rho:
        print('----------------------------------------------------')
        print(f'DENSITY MATRIX TRACING SITES ({str(env_indices)})')
        print('----------------------------------------------------')
        print(RDM)

    return RDM

# ===========================================================================================================

def projector(rho_L, k):
  """
  projector:
    Builds the projector used to truncate the operators
  
  Parameters
  ----------
  rho_L : np.ndarray
    Reduced density matrix of the left or right system of the current Hamiltonian H_2m,
    computed on the ground state
  k : int
    Indices to keep.

  Returns
  -------
  proj : np.ndarray
    projector
  """
  
  if k > rho_L.shape[0]:
    raise ValueError(f"'k' must be <= the dimension of rho_L, got k={k} and dim={rho_L.shape[0]}")

  eigvals, eigvecs = np.linalg.eigh(rho_L)
  sorted_indices = np.argsort(eigvals)[::-1]  ##TO CHECK

  eigvals = eigvals[sorted_indices]
  eigvecs = eigvecs[:, sorted_indices]

  eigvals = eigvals[:k]
  eigvecs = eigvecs[:, :k]

  proj = eigvecs
  return proj

# ===========================================================================================================

def truncate_operators(P_L, P_R, A_L, B_L, A_R, B_R, H_L, H_R):
  """
  truncate_operators :
    Updates the operators truncating them using the Projector built with the first k eigenvectors
  
  Parameters
  ----------
  P_L, P_R : np.ndarray
    Projector built for the left and right subsystems
  A_L, B_L, A_R, B_R, H_L, H_R : np.ndarray
    Current operators representing the left and right systems

  Returns
  -------
  A_L_trunc, B_L_trunc, A_R_trunc, B_R_trunc, H_L_trunc, H_R_trunc : np.ndarray
    It returns the truncated operators after the application of the projector operator, and the projector itself,
    describing both the left amd right systems
  """
  P_L_dagger = P_L.conj().T
  P_R_dagger = P_R.conj().T
  
  A_L_trunc = P_L_dagger @ A_L @ P_L
  B_L_trunc = P_L_dagger @ B_L @ P_L
  A_R_trunc = P_R_dagger @ A_R @ P_R
  B_R_trunc = P_R_dagger @ B_R @ P_R
  
  H_L_trunc = P_L_dagger @ H_L @ P_L
  H_R_trunc = P_R_dagger @ H_R @ P_R

  return A_L_trunc, B_L_trunc, A_R_trunc, B_R_trunc, H_L_trunc, H_R_trunc

# ===========================================================================================================

def dmrg(l, m_max, threshold=1e-6, max_iter=100):
  """
  Infinite DMRG function for a 1D quantum system.

  Parameters
  ----------
  l : float
      Coupling parameter (e.g., transverse field strength).
  m_max : int
      Maximum number of states to retain during truncation.
  threshold : float, optional
      Convergence criterion for the energy difference between iterations.
  max_iter : int, optional
      Maximum number of iterations allowed.

  Returns
  -------
  gs_energies_dict : dictionary
    dictionary of gs energy densities at each iteration
  psi_ground : list
    ground state eigenvector corresponding to the last iteration (biggest dimension)
  deltas_dict : dictionary
    dictionary of deltas (i.e. differences between consecutive gs energy densisties) at each iteration
  actual_dim: int
    Dimension of the last computed hamiltonian
  """
  prev_energy_density = np.inf

  # Initialize operators and Hamiltonians
  m = 1  # Initial single-site system
  A_L, B_L, A_R, B_R, H_L, H_R = initialize_operators(m, l)


  actual_dim = 2*m

  # Dictionaries to store the values
  gs_energies_dict = {}
  deltas_dict = {}
  
  for iteration in range(max_iter):

    # Step 1: Enlarge Hamiltonians by adding one site: L->L+1 ; R->R+1
    H_L1, H_R1 = compute_H_LR(H_L, H_R, A_L, B_L, A_R, B_R, l)
    
    # Step 2: Combine into full system Hamiltonian: 2m->2m+2
    H_2m = compute_H_2m(H_L1, H_R1, m)
    actual_dim = actual_dim + 2   # The dimension is updated by consequence

    # Step 3: Diagonalization of the enlarged Hamiltonian and computation of the first energy density and eigenvector
    E, psi = np.linalg.eigh(H_2m)
    sorted_indices = np.argsort(E)  

    E = E[sorted_indices]
    psi = psi[:, sorted_indices]

    E_ground = E[0]
    psi_ground = psi[:, 0]

    # Step 4: Compute reduced density matrix
    N = int(np.log2(H_2m.shape[0]))
    D = 2                                       # Local Hilbert space dimension

    # Left system
    keep_indices_left = list(range(0, N // 2))  # Keep left block sites
    rho_L = get_reduced_density_matrix(psi_ground, D, N, keep_indices_left, print_rho=False)
    
    # Right system
    keep_indices_right = list(range(N // 2, N))  # Keep left block sites
    rho_R = get_reduced_density_matrix(psi_ground, D, N, keep_indices_right, print_rho=False)


    # Step 5: Construct the projectors of left and right system
    k = min(2 ** m_max, rho_L.shape[0] - 1)  # Ensure k does not exceed the dimension
    P_L= projector(rho_L, k) 
    P_R = projector(rho_R, k)    
    
    # Step 6: Update operators and Truncate operators and Hamiltonians
    A_L, B_L, A_R, B_R = update_operators(A_L, A_R)
    A_L, B_L, A_R, B_R, H_L, H_R = truncate_operators(P_L, P_R, A_L, B_L, A_R, B_R, H_L1, H_R1)

    # Step 7: Check convergence
    current_energy_density = E_ground / actual_dim

    delta = abs(current_energy_density - prev_energy_density)

    deltas_dict[actual_dim] = delta
    gs_energies_dict[actual_dim] = current_energy_density

    if delta < threshold:
      print(f"Converged after {iteration + 1} iterations.")
      break

    # Update for the next iteration
    prev_energy_density = current_energy_density
      
    # if iteration % 10 == 0:
      # print(f"Starting iteration {iteration} ...")
    
  print(f"Reached N = {actual_dim} with precision: delta = {delta}")
  return gs_energies_dict, psi_ground, deltas_dict, actual_dim


# ===========================================================================================================


def update_hamiltonian(m_max, l_values, convergence_threshold, max_iterations):
  """
  update_hamiltonian :
    Runs the RSRG algorithm for different values of lambda, storing the results into dictionaries
  
  Parameters
  ----------
  m_max : int
    # Maximum number of states to retain during truncation
  l_values : list
    lambda values for the non-interacting hamiltonian
  convergence_threshold : float
    threshold value for the difference of consecutive GS energy densities
  max_iterations : int
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
    # Call the Infinite-DMRG algorithm
    normgs_eigval_dict, last_eigvec, _, _ = dmrg(l, m_max, convergence_threshold, max_iterations)

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
    
    if type == "hlines":
        for idx, (N, value) in enumerate(eigenvalues.items()):
            # For a correct representation of the lines
            plt.hlines(value, N - 1, N+1, colors=f'C{idx}', linewidth=2.5, label=f'N={N}')
    elif type == "plot":
        for idx, (N, value) in enumerate(list(eigenvalues.items())):
            plt.plot(N,value, "s--", markersize = 4, label=f'N={N}')
    else:
        print("Invalid type")