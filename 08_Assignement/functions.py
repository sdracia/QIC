import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csr_matrix, kron, eye, diags

import meanfield as mf


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

def ising_hamiltonian(N, l):
  """
  ising_hamiltonian:
    Builds the Ising model Hamiltonian using sparse matrices.

  Parameters
  ----------
  N : int
    Number of spins.
  l : float
    Interaction strength.

  Returns
  -------
  H : sp.csr_matrix
    Sparse Ising Hamiltonian.
  """
  dim = 2 ** N
  H_nonint = sp.csr_matrix((dim, dim), dtype=complex)
  H_int = sp.csr_matrix((dim, dim), dtype=complex)
  
  s_x, _, s_z = pauli_matrices()
  
  for i in range(N):
    zterm = sp.kron(sp.identity(2**i, format='csr'), sp.kron(s_z, sp.identity(2**(N - i - 1), format='csr')))
    H_nonint += zterm
    
  for i in range(N - 1):
    xterm = sp.kron(sp.identity(2**i, format='csr'), sp.kron(s_x, sp.kron(s_x, sp.identity(2**(N - i - 2), format='csr'))))
    H_int += xterm
  
  H = H_int + l * H_nonint
  return H

# ===========================================================================================================

def projector(H, d_eff):
  _, eigvecs = sp.linalg.eigsh(H, k=d_eff, which='SA')  # Compute the smallest `k` eigenvalues
    
  proj = sp.csr_matrix(eigvecs)

  return proj

# ===========================================================================================================

def initialize_A_B(N):
  s_x, _, _ = pauli_matrices()
  
  A_0 = sp.kron(sp.identity(2**(N - 1), format='csr'), s_x)
  B_0 = sp.kron(s_x, sp.identity(2**(N - 1), format='csr'))
  
  return A_0, B_0

# ===========================================================================================================

def compute_H_2N(N, H, A, B):  
  H_2N = sp.kron(H, sp.identity(2**(N), format='csr')) + sp.kron(sp.identity(2**(N), format='csr'), H) + sp.kron(A, B)
  return H_2N

# ===========================================================================================================

def update_operators(N, H_2N, A, B):
  P = projector(H_2N, d_eff=2**N)
  # print("projector dim", P.shape)
  
  P_dagger = P.conj().T
  I_N = sp.identity(2**N, format='csr')

  # Compute H_Nnn, Ann, Bnn, Pnn
  H_eff = P_dagger @ H_2N @ P
  A_eff = P_dagger @ sp.kron(I_N, A) @ P
  B_eff = P_dagger @ sp.kron(B, I_N) @ P
  
  return H_eff, A_eff, B_eff, P

# ===========================================================================================================
  
def real_space_rg(N, l, threshold, d_eff, max_iter=100):
  prev_energy_density = 10
  H = ising_hamiltonian(N, l)
  A, B = initialize_A_B(N)
  
  actual_dim = N


  gs_energies_dict = {}
  deltas_dict = {}


  for iteration in range(1, max_iter + 1):
    # print("size current H = ", H.shape)

    H_2N = compute_H_2N(N, H, A, B)
    # print("size double H_2N = ", H_2N.shape)

    actual_dim = actual_dim * 2

    
    # Compute the current energy density and eigenvectors
    eigvals, eigvecs = sp.linalg.eigsh(H_2N, k=d_eff, which='SA')
    current_energy_density = eigvals[0]/actual_dim


    gs_energies_dict[actual_dim] = current_energy_density

    # Check for convergence 
    delta = abs(current_energy_density - prev_energy_density)

    deltas_dict[actual_dim] = delta

    if delta > threshold:
      H, A, B, P = update_operators(N, H_2N, A, B)
      # print("size new H = ", H.shape)

    else:
      
      break

    # Update previous energy density for next iteration
    prev_energy_density = current_energy_density

  print(f"Convergence achieved at iteration {iteration}: ε = {current_energy_density}", '\n')
  print(f"Converged reached for a system with N = {actual_dim} sites, i.e. H.shape = ({2**actual_dim}x{2**actual_dim}), with precision: delta = {delta}")

  return gs_energies_dict, eigvecs[:, 0], deltas_dict, actual_dim


# ===========================================================================================================

def update_hamiltonian(N_values, l_values, threshold, max_iter=100):
  # Initialize dictionaries to store eigenvalues and eigenvectors
  eigenvalues_dict = {}
  eigenvectors_dict = {}
  
  for N in N_values:
    print(f"Analysis with N={N}...")

    for l in l_values:      
      d_eff = 2**N    
      eigvals, eigvecs = real_space_rg(N, l, threshold, d_eff, max_iter)  
      
      eigenvalues_dict[(2*N, l)] = eigvals
      eigenvectors_dict[(2*N, l)] = eigvecs
    
    print("-----------------------------------------")
    
  return eigenvalues_dict, eigenvectors_dict


def plot_dict_N_GSen(eigenvalues, type):

    N_prec = 0
    
    if type == "hlines":
        for idx, (N, value) in enumerate(eigenvalues.items()):
            # Aggiunge una linea orizzontale per ogni valore
            delta_N = (N - N_prec)/2
            plt.hlines(value, N - delta_N, (3*N)/2, colors=f'C{idx}', linewidth=2.5, label=f'N={N}')
            N_prec = N
    elif type == "plot":
        for idx, (N, value) in enumerate(list(eigenvalues.items())[1:]):
            plt.plot(N,value, "s--", markersize = 4, label=f'N={N}')
    else:
        print("Invalid type")










def plot_eigenvalues(N_values, l_values, eigenvalues):
  # Loop over the values of N (many plots)

  eigenvalues_mf, _ = mf.diagonalize_ising(N_values, l_values)

  for N in N_values:
    plt.figure(figsize=(8, 5))
      
    # Loop over the first k levels
    energies = []
    for l in l_values:
      energies.append(eigenvalues[(N, l)] / N)

    energies_mf = [(eigenvalues_mf[(N, l)][0])/N for l in l_values]

    plt.plot(l_values, energies, label=f'RSRG')
    plt.plot(l_values, energies_mf, label=f'mean field')

    
    plt.axvline(x = -1, linestyle="--", color = "red", label="Critical point")
        
    # Plot formatting
    plt.xlabel('Interaction strength (λ)')
    plt.ylabel('Energy')
    plt.title(f'Ground state energy vs λ (N={N})')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()



def magnetization(N, l_vals, eigenvectors):
  """
  magnetization:
    Computes the magnetization of the ground state vector for an N-spin system.

  Parameters
  ----------
  ground_state : np.ndarray
    Ground state vector of the system.
  N : int
    Number of spins in the system.

  Returns
  -------
  M : float
    Expectation value of the normalized total magnetization operator.
  """
  _, _, s_z = pauli_matrices()  # Retrieve sparse Pauli matrices
  
  M_z = sp.csr_matrix((2**N, 2**N), dtype=complex)
  
  for i in range(N):
    M_z_i = sp.kron(sp.identity(2**i, format='csr'), sp.kron(s_z, sp.identity(2**(N - i - 1), format='csr')))
    M_z += M_z_i
    
  M_z /= N
  
  # M = (ground_state.conj().T @ (M_z @ ground_state)).real


  magnetizations = []

  for l in l_vals:

    ground_state = eigenvectors[(N, l)]

    # Compute magnetization
    magnetization = ground_state.conj().T @ (M_z @ ground_state)
    magnetizations.append(magnetization.real)

  return magnetizations

  # return M

# ===========================================================================================================

def plot_magnetization(N_values, l_values, eigenvectors):
  """
  plot_magnetization :
    Plot the magnetization as a function of l for different N.
  
  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  eigenvecttors : np.ndarray
    Precomputed eigenvectors for every (N, l).
  
  Returns
  -------
  None
  """  

  plt.figure(figsize=(8, 6))

  ensemble_magnetiz_total = []

  for N in N_values: 
      magnetizations = magnetization(N, l_values, eigenvectors)
      ensemble_magnetiz_total.append(magnetizations)



  for i, mag in enumerate(ensemble_magnetiz_total): 
      plt.plot(l_values, mag, "s--", markersize = 1, label=f'N = {i+2}')

  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  plt.xscale('log')  # Scala logaritmica sull'asse x
  plt.xlabel('Lambda (λ)')
  plt.ylabel('Magnetization ⟨M_z⟩')
  plt.title('Magnetization vs λ for different N')
  plt.legend()
  plt.grid(True)
  plt.show()