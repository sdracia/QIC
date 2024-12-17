import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, kron, eye, diags
from scipy.sparse.linalg import eigsh

# ===========================================================================================================
# ISING MODEL
# ===========================================================================================================

def pauli_matrices():
    """
    pauli_matrices:
      Builds the Pauli matrices as sparse matrices.

    Returns
    -------
    s_x, s_y, s_z: tuple of scipy.sparse.csr_matrix
      Pauli matrices for a 2x2 system.
    """
    s_x = csr_matrix([[0, 1], [1, 0]])
    s_y = csr_matrix([[0, -1j], [1j, 0]])
    s_z = csr_matrix([[1, 0], [0, -1]])
    return s_x, s_y, s_z


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
    H : scipy.sparse.csr_matrix
      Ising Hamiltonian.
    """

    dim = 2 ** N
    H_nonint = csr_matrix((dim, dim), dtype=np.complex128)
    H_int = csr_matrix((dim, dim), dtype=np.complex128)

    s_x, _, s_z = pauli_matrices()

    # Non-interacting term
    for i in range(N):
        zterm = kron(eye(2**i), kron(s_z, eye(2**(N - i - 1))), format="csr")
        H_nonint += zterm

    # Interaction term
    for i in range(N - 1):
        xterm = kron(eye(2**i), kron(s_x, kron(s_x, eye(2**(N - i - 2))), format="csr"), format="csr")
        H_int += xterm

    H = l * H_nonint + H_int
    return H


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
    eigenvalues, eigenvectors : tuple of dict
      Eigenvalues and eigenstates of the Ising Hamiltonian for different
      values of N and l.
    """

    eigenvalues = {}
    eigenvectors = {}

    for N in N_values:
        print(f"Diagonalizing Ising Hamiltonian with N={N} ...")

        for l in l_values:
            H = ising_hamiltonian(N, l)

            # Use sparse eigenvalue solver
            eigval, eigvec = eigsh(H, k=1, which='SA')  # Smallest algebraic eigenvalue
            eigenvalues[(N, l)] = eigval
            eigenvectors[(N, l)] = eigvec

    return eigenvalues, eigenvectors



# ===========================================================================================================
# MAGNETIZATION
# ===========================================================================================================


def magnetization_z(N):
    """
    magnetization_z:
      Constructs the z-component of the magnetization operator.

    Parameters
    ----------
    N : int
      Number of spins.

    Returns
    -------
    M_z : scipy.sparse.csr_matrix
      Magnetization operator in the z-direction.
    """
    s_x, _, s_z = pauli_matrices()

    dim = 2 ** N
    M_z = csr_matrix((dim, dim), dtype=np.complex128)

    for i in range(N):
        m_term = kron(eye(2**i), kron(s_z, eye(2**(N - i - 1))), format="csr")
        M_z += m_term

    M_z = M_z / N
    return M_z


def compute_magnetization(N, l_vals):
    """
    compute_magnetization:
      Computes the magnetization for the ground state of the Ising model.

    Parameters
    ----------
    N : int
      Number of spins.
    l_vals : list of float
      Values of the interaction strength l.

    Returns
    -------
    magnetizations : list of float
      Magnetization values for the ground state at different l.
    """
    M_z = magnetization_z(N)

    magnetizations = []

    for l in l_vals:
        H = ising_hamiltonian(N, l)

        # Compute ground state
        eigval, eigvec = eigsh(H, k=1, which='SA')
        ground_state = eigvec[:, 0]

        # Compute magnetization
        magnetization = ground_state.conj().T @ (M_z @ ground_state)
        magnetizations.append(magnetization.real)

    return magnetizations



# ===========================================================================================================
# DELTA E
# ===========================================================================================================


def compute_DeltaE(N, l_vals):

    DeltasE = []

    for l in l_vals:
        H = ising_hamiltonian(N, l)

        # Compute ground state
        eigval, eigvec = eigsh(H, k=3, which='SA')

        first_eigval = eigval[0]
        second_eigval = eigval[2]

        delta = np.abs(first_eigval/N - second_eigval/N)

        DeltasE.append(delta)

    return DeltasE



# ===========================================================================================================
# ENTROPY
# ===========================================================================================================



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


def entropy(N_values, l, loc_dim):

    entropies = []

    for i,N in enumerate(N_values):
        N = int(N)
        # Costruzione dell'Hamiltoniano
        H = ising_hamiltonian(N, l)
        # H = sp.ising_hamiltonian(N, l_critical[i])


        # Calcolo del ground state
        eigval, eigvec = eigsh(H, k=1, which='SA')
        ground_state = eigvec[:, 0]

        rho = np.outer(ground_state, np.conj(ground_state))


        keep_indices = list(range(N // 2))  # Mantengo i primi N/2 spin
        rhoA = get_reduced_density_matrix(ground_state, loc_dim, N, keep_indices)


        # SVD per ottenere i valori singolari
        U, Lambda, Vh = np.linalg.svd(rhoA, full_matrices=False)

        # Calcolo dell'entropia di entanglement
        entropy = -np.sum(Lambda**2 * np.log(Lambda**2))
        entropies.append(entropy)



    return entropies



# ===========================================================================================================
# PLOTTING AND VISUALIZATION
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

    eigenvalues, _ = diagonalize_ising(N_values, l_values)

    for N in N_values:
        plt.figure(figsize=(8, 5))

        for level in range(k):
            energies = [eigenvalues[(N, l)][level] for l in l_values]
            plt.plot(l_values, energies, label=f'Level {level + 1}')

        plt.xlabel('Interaction strength (λ)')
        plt.ylabel('Energy')
        plt.title(f'First {k} energy levels vs λ (N={N})')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()


def plot_normalized_eigenvalues(N_values, l_values, k):
    """
    Plot the first k energy levels as a function of l for different N, normalized by N.

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

    eigenvalues, _ = diagonalize_ising(N_values, l_values)

    for N in N_values:
        plt.figure(figsize=(8, 5))

        for level in range(k):
            energies = [eigenvalues[(N, l)][level] / N for l in l_values]
            plt.plot(l_values, energies, label=f'Level {level + 1}')

        plt.xlabel('Interaction strength (λ)')
        plt.ylabel('Normalized Energy')
        plt.title(f'First {k} normalized energy levels vs λ (N={N})')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()


def plot_fixed_lambda(N_values, l, k):
    """
    Plot the first k energy levels as a function of N for a fixed λ.

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
    eigenvalues = {}

    for N in N_values:
        print(f"Computing eigenvalues for N={N}, λ={l}...")
        H = ising_hamiltonian(N, l)
        eigval, _ = eigsh(H, k=k, which='SA')
        eigenvalues[N] = eigval

    plt.figure(figsize=(8, 5))
    for level in range(k):
        for idx, N in enumerate(N_values):
            energy = eigenvalues[N][level]
            plt.hlines(energy, N - 0.5, N + 0.5, colors=f'C{level}', linewidth=2.5, label=f'Level {level + 1}' if idx == 0 else None)

    plt.xticks(N_values)
    plt.xlabel('Number of Spins (N)')
    plt.ylabel('Energy')
    plt.title(f'First {k} Energy Levels vs N (λ={l})')
    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.show()
