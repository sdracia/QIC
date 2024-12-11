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
  
  H =  l * H_nonint + H_int 
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


def magnetization_z(N):
    s_x, _, s_z = pauli_matrices()
    
    M_z = np.zeros((2**N, 2**N))
    for i in range(N):
        m_term = np.kron(np.eye(2**i), np.kron(s_z, np.eye(2**(N - i - 1))))
        M_z += m_term
    
    M_z = M_z / N

    return M_z


def compute_magnetization(N, l_vals):
    M_z = magnetization_z(N)

    magnetizations = []  
    ground_states = []

    for l in l_vals:
        eigval, eigvec = np.linalg.eigh(ising_hamiltonian(N, l))

        ground_state = eigvec[:, 0]
        magnetization = np.dot(ground_state.conj().T, np.dot(M_z, ground_state))

        ground_states.append(ground_state)
        magnetizations.append(magnetization)

    return magnetizations



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

  if not isinstance(k, int) or k <= 0:
    raise ValueError("k must be a positive integer.")
  if not all(k <= 2**N for N in N_values):
    raise ValueError(f"k must not exceed 2^N for any N in N_values.")


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



def plot_normalized_eigenvalues(N_values, l_values, k):
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

  if not isinstance(k, int) or k <= 0:
    raise ValueError("k must be a positive integer.")
  if not all(k <= 2**N for N in N_values):
    raise ValueError(f"k must not exceed 2^N for any N in N_values.")


  # Get eigenvalues
  eigenvalues, _ = diagonalize_ising(N_values, l_values)
  
  # Loop over the values of N (many plots)
  for N in N_values:
    plt.figure(figsize=(8, 5))
      
    # Loop over the first k levels
    for level in range(k):
      # Extract the first k energies given fixed values for N and l
      energies = [(eigenvalues[(N, l)][level])/N for l in l_values]
      plt.plot(l_values, energies, label=f'Level {level + 1}')
        
    # Plot formatting
    plt.xlabel('Interaction strength (λ)')
    plt.ylabel('Energy')
    plt.title(f'First {k} normalized energy levels vs λ (N={N})')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def plot_fixed_lambda(N_values, l, k):
    """
    Plot the first k energy levels as a function of N for a fixed \u03bb.

    Parameters
    ----------
    N_values : list of int
        Values of N, number of spins in the system.
    l : float
        Fixed interaction strength.
    k : int
        Number of lowest energy levels to plot.

    Returns
    ----------
    None
    """
    # if not isinstance(k, int) or k <= 0:
    #     raise ValueError("k must be a positive integer.")
    # if not isinstance(l, (int, float)) or not np.isfinite(l):
    #     raise ValueError("\u03bb must be a finite number.")
    # if not all(isinstance(N, int) and N > 0 for N in N_values):
    #     raise ValueError("All values of N must be positive integers.")

    # Dictionary to store eigenvalues
    eigenvalues = {}

    # Compute eigenvalues for each N
    for N in N_values:
        print(f"Computing eigenvalues for N={N}, \u03bb={l}...")
        H = ising_hamiltonian(N, l)
        eigval, _ = np.linalg.eigh(H)
        eigenvalues[N] = eigval[:k]  # Store only the first k eigenvalues

    # for N in N_values:
    #   eigval, _ = diagonalize_ising(N_values, l)
    #   eigenvalues[N] = eigval[:k]

    # Plot the first k eigenvalues as a function of N
    plt.figure(figsize=(8, 5))
    for level in range(k):
        for idx, N in enumerate(N_values):
            energy = eigenvalues[N][level]
            plt.hlines(energy, N - 0.5, N + 0.5, colors=f'C{level}', linewidth=2.5, label=f'Level {level + 1}' if idx == 0 else None)

    # Plot formatting
    plt.xticks(N_values)  # Set x-axis ticks to be exactly the N values
    plt.xlabel('Number of Spins (N)')
    plt.ylabel('Energy')
    plt.title(f'First {k} Energy Levels vs N (λ={l})')
    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.show()
