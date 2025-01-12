from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


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


def projector(H, d_eff):
  eigvals, eigvecs = np.linalg.eigh(H)
  sorted_indices = np.argsort(eigvals)  # Ordina per autovalori crescenti
  eigvecs_sorted = eigvecs[:, sorted_indices]  # Riordina gli autovettori
    
  # Prendi i primi `d_eff` autovettori
  proj = eigvecs_sorted[:, :d_eff]

  return proj


def initialize_A_B(N):
  s_x, _, _ = pauli_matrices()
  
  A_0 = np.kron(np.eye(2**(N - 1)), s_x)
  B_0 = np.kron(s_x, np.eye(2**(N - 1)))
  
  return A_0, B_0

# ===========================================================================================================

def compute_H_2N(N, H, A, B):  
  H_2N = np.kron(H, np.eye(2**(N))) + np.kron(np.eye(2**(N)), H) + np.kron(A, B)
  return H_2N


def update_operators(N, H_2N, A, B):
  P = projector(H_2N, d_eff=2**N)
  # print("projector dim", P.shape)
  
  P_dagger = P.conj().T
  I_N = np.eye(2**N)

  # Compute H_Nnn, Ann, Bnn, Pnn
  H_eff = P_dagger @ H_2N @ P
  A_eff = P_dagger @ np.kron(I_N, A) @ P
  B_eff = P_dagger @ np.kron(B, I_N) @ P
  
  return H_eff, A_eff, B_eff, P


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
      H, A, B, P = update_operators(N, H_2N, A, B)
      # print("size new H = ", H.shape)

    else:
      
      break

    # Update previous energy density for next iteration
    prev_energy_density = current_energy_density

  print(f"Convergence achieved at iteration {iteration}: ε = {current_energy_density}", '\n')
  print(f"Converged reached for a system with N = {actual_dim} sites, with precision: delta = {delta}")

  return gs_energies_dict, eigvecs[:, 0], deltas_dict, actual_dim


def update_hamiltonian(N, l_values, threshold, max_iter=100):
  # Initialize dictionaries to store eigenvalues and eigenvectors
  eigenvalues_dict = {}
  last_eigenvectors_dict = {}
  
  # for N in N_values:
  #   print(f"Analysis with N={N}...")

  for l in l_values:      
    d_eff = 2**N    
    normgs_eigval_dict, last_eigvec, deltas_dim, actual_dim = real_space_rg(N, l, threshold, d_eff, max_iter)  
      
    eigenvalues_dict[l] = normgs_eigval_dict
    last_eigenvectors_dict[l] = last_eigvec
    
  print("-----------------------------------------")
    
  return eigenvalues_dict, last_eigenvectors_dict


def plot_dict_N_GSen(eigenvalues, type):

    N_prec = 0
    
    if type == "hlines":
        for idx, (N, value) in enumerate(eigenvalues.items()):
            # Aggiunge una linea orizzontale per ogni valore
            delta_N = (N - N_prec)/2
            plt.hlines(value, N - delta_N, (3*N)/2, colors=f'C{idx}', linewidth=2.5, label=f'N={N}')
            N_prec = N
    elif type == "plot":
        for idx, (N, value) in enumerate(list(eigenvalues.items())):
            plt.plot(N,value, "s--", markersize = 4, label=f'N={N}')
    else:
        print("Invalid type")


def Mag(N):
  _, _, s_z = pauli_matrices()  

  M_z = np.zeros((2**N, 2**N), dtype=complex)

  for i in range(N):
    M_z_i = np.kron(np.eye(2**i), np.kron(s_z, np.eye(2**(N - i - 1))))
    M_z += M_z_i

  M_z /= N
  return M_z


def compute_magnetization(N, l_vals, threshold, d_eff, max_iter):

    magnetizations = []  

    for l in l_vals:
        normgs_eigval_dict, last_eigvec, deltas_dim, actual_dim = real_space_rg(N, l, threshold, d_eff, max_iter)
        # print(len(last_eigvec))
        # print(actual_dim)
        a = int(np.log2(len(last_eigvec)))

        M_z = Mag(a)
        # magnetization = (last_eigvec.conj().T @ (M_z @ last_eigvec)).real
        # magnetization = (last_eigvec.conj().transpose().dot(M_z.dot(last_eigvec))).real
        magnetization = np.dot(last_eigvec.conj().T, np.dot(M_z, last_eigvec)).real
        magnetizations.append(magnetization)

    return magnetizations