import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================================================
# ISING MODEL
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
  
  H = H_int + l * H_nonint
  return H

# ===========================================================================================================
# EIGENVALUES
# ===========================================================================================================

def diagonalize_ising(N_values, l_values):
  """
  diagonalize_ising :
    Diagonalize the Ising Hamiltonian for different values of N and l.

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.

  Returns
  -------
  eigenvalues, eigenvectors : tuple of np.ndarray
    Eigenvalues and eigenstates of the Ising Hamiltonian for different
    values of N and l.
  """
  eigenvalues = {}
  eigenvectors = {}
  
  for N in N_values:
    print(f"Diagonalizing Ising Hamiltonian with N={N} ...")
    
    for l in l_values:
      eigval, eigvec = np.linalg.eigh(ising_hamiltonian(N, l))
      eigenvalues[(N, l)] = eigval
      eigenvectors[(N, l)] = eigvec
  
  return eigenvalues, eigenvectors

# ===========================================================================================================

def plot_eigenvalues(N_values, l_values, k):
  """
  Plot the first k energy levels as a function of l for different N.
  
  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  k : int
    Number of lowest energy levels to plot.
  
  Returns
  ----------
  None
  """
  # Get eigenvalues
  eigenvalues, _ = diagonalize_ising(N_values, l_values)
  
  # Loop over the values of N (many plots)
  for N in N_values:
    plt.figure(figsize=(8, 5))
      
    # Loop over the first k levels
    for level in range(k):
      # Extract the first k energies given fixed values for N and l
      energies = [eigenvalues[(N, l)][level] for l in l_values]
      plt.plot(l_values, energies, label=f'Level {level + 1}')
        
    # Plot formatting
    plt.xlabel('Interaction strength (λ)')
    plt.ylabel('Energy')
    plt.title(f'First {k} energy levels vs λ (N={N})')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()