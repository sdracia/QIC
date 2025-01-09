import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

import functions as fu


def pauli_matrices():
  """
  pauli_matrices:
    Builds the Pauli matrices as sparse matrices.

  Returns
  -------
  s_x, s_y, s_z: tuple of sp.csr_matrix
    Pauli matrices for a 2x2 system in sparse format.
  """
  s_x = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
  s_y = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
  s_z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)
  return s_x, s_y, s_z


# ===========================================================================================================

def initialize_A_B(m):
  s_x, _, _ = pauli_matrices()
  
  A_0 = sp.identity(2**(m - 1), format='csr')
  B_0 = sp.kron(s_x, sp.identity(2**(m - 1), format='csr'))
  
  return A_0, B_0

# ===========================================================================================================

def compute_H_LR(H_L, H_R, A, B, l):
  s_x, _, s_z = pauli_matrices()
  H_L1 = sp.kron(H_L, sp.identity(2, format='csr')) + sp.kron(A, l * s_z) + sp.kron(B, s_x)
  H_R1 = sp.kron(sp.identity(2, format='csr'), H_R) + sp.kron(l * s_z, A) + sp.kron(s_x, B)
  return H_L1, H_R1

# ===========================================================================================================

def update_operators(m):
  s_x, _, _ = pauli_matrices()
  A_eff = sp.identity(2**m, format='csr')
  B_eff = sp.kron(sp.identity(2**(m-1), format='csr'), s_x)
  
  return A_eff, B_eff

# ===========================================================================================================

def compute_H_2m(H_L1, H_R1):
  s_x, _, _ = pauli_matrices()
  H_LR = sp.kron(s_x, s_x)
  
  H_2m = sp.kron(H_L1, sp.identity(H_R1.shape[0], format='csr')) + sp.kron(sp.identity(H_L1.shape[0], format='csr'), H_R1) + H_LR
  
  return H_2m

# ===========================================================================================================
  
def rdm(psi, N, D, keep_indices):
  """
  rdm :
    Computes the reduced density matrix of a quantum state by tracing out the 
    degrees of freedom of the environment.

  Parameters
  ----------
  psi : np.ndarray
    Wavefunction of the quantum many-body system, represented as a complex vector of 
    size D^N.
  N : int
    Number of subsystems.
  D : int
    Dimension of each subsystem.
  keep_indices : list of int
    Indices of the sites to retain in the subsystem (all other sites are traced out).

  Returns
  -------
  rdm : np.ndarray
    Reduced density matrix for the subsystem specified by keep_indices, which is a 
    square matrix of size (D^len(keep_indices), D^len(keep_indices)).
  """
  # Check correct values for 'keep_indices'
  if not all(0 <= idx < N for idx in keep_indices):
    raise ValueError(f"'keep_indices' must be valid indices within range(n_sites), got {keep_indices}")
    
  # Compute subsystem and environment dimensions
  n_keep = len(keep_indices)
  subsystem_dim = D ** n_keep
  env_dim = D ** (N - n_keep)

  # Reshape the wavefunction into a sparse tensor (use csr_matrix for efficient sparse storage)
  psi_tensor = sp.csr_matrix(psi.reshape([D] * N))

  # Reorder the axes to group subsystem (first) and environment (second)
  all_indices = list(range(N))
  env_indices = [i for i in all_indices if i not in keep_indices]  # complement of keep_indices
  reordered_tensor = np.transpose(psi_tensor, axes=keep_indices + env_indices)

  # Partition into subsystem and environment (reshape back)
  psi_partitioned = reordered_tensor.reshape((subsystem_dim, env_dim))

  # Compute the reduced density matrix (use sparse matrix multiplication)
  rdm = psi_partitioned.dot(psi_partitioned.conj().T)

  return rdm

# ===========================================================================================================

def projector(rho_L, k):
  _, eigvecs = sp.linalg.eigsh(rho_L, k=k, which='LA')  # Compute the largest `k` eigenvalues
    
  proj = sp.csr_matrix(eigvecs)

  return proj

# ===========================================================================================================

def update_hamiltonian(N, l_values, threshold, d_eff, max_iter=100):
  # Initialize dictionaries to store eigenvalues and eigenvectors
  eigenvalues_dict = {}
  eigenvectors_dict = {}
  
  for l in l_values:
    print(f"Analysis with l={l}")
    
    prev_energy_density = 10
    H = fu.ising_hamiltonian(N, l)
    A, B = initialize_A_B(N)

    # Build Hamiltonian of size 2N and projector
    H_2N = fu.compute_H_2N(N, H, A, B)

    for iteration in range(1, max_iter + 1):
      # Compute the current energy density and eigenvectors
      eigvals, eigvecs = sp.linalg.eigsh(H_2N, k=d_eff, which='SA')
      current_energy_density = eigvals[0]
      
      # Save eigenvalues and eigenvectors in dictionaries with the key (2**iteration, l)
      eigenvalues_dict[(2**iteration, l)] = eigvals
      eigenvectors_dict[(2**iteration, l)] = eigvecs  # Save all eigenvectors, or choose a subset if needed

      # Check for convergence 
      delta = abs(current_energy_density - prev_energy_density)

      if delta > threshold:
        H, A, B, P = update_operators(N, H_2N, A, B)
      else:
        print(f"Convergence achieved at iteration {iteration}: ε = {current_energy_density}")
        break

      # Update previous energy density for next iteration
      prev_energy_density = current_energy_density

      # Compute new H_2N and P
      H_2N = fu.compute_H_2N(N, H, A, B)
    
    print("-----------------------------------------")
    
  return eigenvalues_dict, eigenvectors_dict


# ===========================================================================================================