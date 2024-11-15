import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import analytical_solution as anso

def kinetic_gen(size, deltax, order = 2):
    factor = 1/(2*(deltax**2))

    if (order == 2):
        main_diag = 2 * np.ones(size)
        off_diag = -1 * np.ones(size - 1)

        K = factor * (np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))
    elif (order == 4):
        main_diag = (5 / 2) * np.ones(size)  # Main diagonal
        first_off_diag = (-4 / 3) * np.ones(size - 1)  # First off-diagonals
        second_off_diag = (1 / 12) * np.ones(size - 2)  # Second off-diagonals

        K = factor * (
            np.diag(main_diag) + np.diag(first_off_diag, k=1) + np.diag(first_off_diag, k=-1) +  
            np.diag(second_off_diag, k=2) + np.diag(second_off_diag, k=-2)   
        )

    # elif (order == 6):

    else:
        print("Invalid order value. It must be either 2, 4 or 6.")
    return K

def potential_gen(size, x_i, omega):
    factor = (omega**2)/2
    main_diag = (x_i**2) * np.ones(size)

    V = factor * np.diag(main_diag)

    return V

def hamiltonian_gen(size, deltax, x_i, omega, order = 2):

    K = kinetic_gen(size, deltax, order)
    V = potential_gen(size, x_i, omega)

    A = K + V

    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)  # Get indices of sorted eigenvalues
    eigenvalues = eigenvalues[sorted_indices]  # Sort eigenvalues
    eigenvectors = eigenvectors[:, sorted_indices]  # Sort eigenvectors to match eigenvalues

    # Select the first k eigenvalues and corresponding eigenvectors
    norm = np.sqrt(np.sum(np.abs(eigenvectors)**2, axis=0))
    for i in range(len(eigenvalues)):
        eigenvectors[:, i] = eigenvectors[:, i] / norm

    center_index = size // 2  # assuming symmetric grid centered around x = 0

    for i in range(len(eigenvectors)):
        if i % 2 == 0:  # Even states
            # Ensure the wavefunction is positive at the center
            if ((i//2)%2==0 and eigenvectors[:, i][center_index] < 0) or ((i//2)%2!=0 and eigenvectors[:, i][center_index] > 0):
                eigenvectors[:, i] *= -1
        else:  # Odd states
            # Find the first peak after the center and make it positive
            first_peak_index = center_index + np.argmax(np.abs(eigenvectors[i][center_index:]))
            if eigenvectors[:, i][first_peak_index] < 0:
                eigenvectors[:, i] *= -1
    
    return A, eigenvalues, eigenvectors.T

def plot(number_to_print, eigenvalues, eigenvectors, x_i, L, omega):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    colors = plt.cm.tab10.colors[:number_to_print]
    
    for i in range(number_to_print):
      ax1.plot(x_i, eigenvectors[i], label=f"{i}-th eigenvector $\psi(x)$", color=colors[i], linewidth=1.5, linestyle="--")

    ax1.set_xlabel("Position $x$")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Wavefunctions of Quantum Harmonic Oscillator")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax1.set_xlim(-L,L)

    # Plot the potential and energy levels in the second subplot
    ax2.plot(x_i, 0.5 * omega**2 * x_i**2, label="Harmonic potential $V(x)$", color="red", linestyle="-", linewidth=1.5)
    for i in range(number_to_print):
      ax2.axhline(eigenvalues[i], label=f"{i}-th eigenvelue", color=colors[i], linestyle="-.", linewidth=1.5, alpha=0.8)

    ax2.set_xlabel("Position $x$")
    ax2.set_ylabel("Energy")
    ax2.set_title("Energy Levels and Harmonic Potential")
    ax2.set_ylim(-0.5, 8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax2.set_xlim(-L,L)

    fig.tight_layout()
    plt.show()

#############################
### CORRECTNESS FUNCTIONS ###
#############################

def correctness(k, eigenvalues, eigenvectors, eigenvalues_analy, eigenvectors_analy):
    # Eigenvalue errors
    eigval_errors = np.abs(eigenvalues - eigenvalues_analy)
    relative_eigval_errors = eigval_errors / np.abs(eigenvalues_analy)

    # Eigenvector errors
    eigvec_dot = []
    for i in range(k):
        # Normalize both eigenvectors
        vec_approx = eigenvectors[i] / np.linalg.norm(eigenvectors[i])
        vec_analytical = eigenvectors_analy[i] / np.linalg.norm(eigenvectors_analy[i])

        # Calculate the cosine similarity
        dot = np.dot(vec_approx, vec_analytical)
        # theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical errors

        # # Store the angle in degrees for better interpretability
        # eigvec_angles.append(np.degrees(theta))
        eigvec_dot.append(1 - np.abs(dot))

    # Output results
    # print("Eigenvalue Errors:", eigval_errors)
    # print("Relative Eigenvalue Errors:", relative_eigval_errors)
    # print("Eigenvector Angles (in degrees):", eigvec_dot)

    return relative_eigval_errors, eigvec_dot

def plot_correctness(k, rel_eigval_err, eigvec_dot):
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Eigenvalue Errors
    axes[0].bar(range(1, k + 1), rel_eigval_err, color='teal', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel("Eigenvalue Index", fontsize=12)
    axes[0].set_ylabel("Relative Error", fontsize=12)
    axes[0].set_title("Relative Error in Eigenvalues", fontsize=14)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Eigenvector Angles
    axes[1].bar(range(1, k + 1), eigvec_dot, color='orange', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel("Eigenvector Index", fontsize=12)
    axes[1].set_ylabel("1 - dot product", fontsize=12)
    axes[1].set_title("Dot Product Between Approximate and Analytical Eigenvectors", fontsize=14)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Display the plots
    plt.tight_layout()
    plt.show()

def plot_both_correctness(k, eigval2, eigval4, eigvec2, eigvec4):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Eigenvalue Errors
    axes[0].bar(range(1, k + 1), eigval2, color='teal', alpha=0.7, edgecolor='black', label="Order 2")
    axes[0].bar(range(1, k + 1), eigval4, color='red', alpha=0.7, edgecolor='black', label="Order 4")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Eigenvalue Index", fontsize=12)
    axes[0].set_ylabel("Relative Error", fontsize=12)
    axes[0].set_title("Relative Error in Eigenvalues", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Eigenvector Angles
    axes[1].bar(range(1, k + 1), eigvec2, color='teal', alpha=0.7, edgecolor='black', label="Order 2")
    axes[1].bar(range(1, k + 1), eigvec4, color='red', alpha=0.7, edgecolor='black', label="Order 4")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Eigenvalue Index", fontsize=12)
    axes[1].set_ylabel("Relative Error", fontsize=12)
    axes[1].set_title("Relative Error in Eigenvalues", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Display the plots
    plt.tight_layout()
    plt.show()

#############################
### STABILITY FUNCTIONS ###
#############################

def stability(num_runs, order, k, N, deltax, x_i, omega):
    eigenvalues_runs = []
    eigenvectors_runs = []

    # Repeat eigenvalue calculation and store results
    for _ in range(num_runs):
        _, eigenvalues, eigenvectors = hamiltonian_gen(N, deltax, x_i, omega, order)  # Adjust parameters as needed
        eigenvalues_runs.append(eigenvalues[:k])          # Store the first k eigenvalues
        eigenvectors_runs.append(eigenvectors[:k])     # Store the first k eigenvectors

    # Convert lists to numpy arrays for easier manipulation
    eigenvalues_runs = np.array(eigenvalues_runs)
    eigenvectors_runs = np.array(eigenvectors_runs)

    # Calculate mean and standard deviation of eigenvalues across runs
    eigenvalues_mean = np.mean(eigenvalues_runs, axis=0)
    eigenvalues_std = np.std(eigenvalues_runs, axis=0)

    dot_matrix = np.zeros((k, num_runs - 1))

    for i in range(k):
        for j in range(1, num_runs):
            dot_product = np.dot(eigenvectors_runs[j, i, :], eigenvectors_runs[j - 1, i, :])
            dot_matrix[i, j - 1] = np.abs(1 - np.abs(dot_product))

    eigvec_dot_mean = np.mean(dot_matrix, axis=1)

    return eigenvalues_std, eigvec_dot_mean, dot_matrix


#########################################
### ACCURATE DISCRETIZATION FUNCTIONS ###
#########################################

def discretization_size(N_min, N_max, step, k, omega, L, order):

    sizes = list(range(N_min, N_max, step))
    num_sizes = len(sizes)

    # Inizializza le matrici per salvare gli errori per le heatmaps
    eigval_errors_matrix = np.zeros((num_sizes, k))
    eigvec_dots_matrix = np.zeros((num_sizes, k))

    # Loop per calcolare gli errori
    for idx, size in enumerate(sizes):
        deltax = (2*L)/size
        x_i = np.array([(((2*L)/size)*i - L) for i in range(size)])

        _, eigenvalues, eigenvectors = hamiltonian_gen(size, deltax, x_i, omega, order)

        # Limitare a k autovalori/autovettori
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:k]

        eigenvalues_analy, eigenvectors_analy = anso.analytic_eigenv(x_i, omega, k)

        # Calcolo dell'errore
        eigval_err, eigvec_dot = correctness(k, eigenvalues, eigenvectors, eigenvalues_analy, eigenvectors_analy)

        # Inserisci i risultati nelle matrici
        eigval_errors_matrix[idx, :] = eigval_err
        eigvec_dots_matrix[idx, :] = eigvec_dot
    
    return eigval_errors_matrix, eigvec_dots_matrix, sizes


def omega_variation(omega_min, omega_max, omega_step, k, N, L, order):

    omega_sizes = list(np.round(np.arange(omega_min, omega_max, omega_step), decimals=2))
    num_sizes = len(omega_sizes)
    
    deltax = (2*L)/N
    x_i = np.array([(((2*L)/N)*i - L) for i in range(N)])

    # Inizializza le matrici per salvare gli errori per le heatmaps
    eigval_errors_matrix = np.zeros((num_sizes, k))
    eigvec_dots_matrix = np.zeros((num_sizes, k))

    # Loop per calcolare gli errori
    for idx, omega in enumerate(omega_sizes):

        A, eigenvalues, eigenvectors = hamiltonian_gen(N, deltax, x_i, omega, order)

        # Limitare a k autovalori/autovettori
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:k]

        eigenvalues_analy, eigenvectors_analy = anso.analytic_eigenv(x_i, omega, k)

        # Calcolo dell'errore
        eigval_err, eigvec_dot = correctness(k, eigenvalues, eigenvectors, eigenvalues_analy, eigenvectors_analy)

        # Inserisci i risultati nelle matrici
        eigval_errors_matrix[idx, :] = eigval_err
        eigvec_dots_matrix[idx, :] = eigvec_dot

    return eigval_errors_matrix, eigvec_dots_matrix, omega_sizes


def scaling_heatmap(eigval_errors_matrix, eigvec_dots_matrix, sizes, k):
    # Visualizzazione delle heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Heatmap per gli errori sugli autovalori
    sns.heatmap(np.log10(eigval_errors_matrix), ax=axes[0], annot=False, cmap="YlGnBu", 
                xticklabels=np.arange(1, k+1), yticklabels=sizes)
    axes[0].set_title("Log10 of Errors on Eigenvalues", fontsize=16)
    axes[0].set_xlabel("Eigenvalue Index", fontsize=16)
    axes[0].set_ylabel("Matrix Size", fontsize=16)

    # Heatmap per il prodotto scalare degli autovettori
    sns.heatmap(np.log10(eigvec_dots_matrix), ax=axes[1], annot=False, cmap="YlOrRd", 
                xticklabels=np.arange(1, k+1), yticklabels=sizes)
    axes[1].set_title(" Log of Error $|1 - |dot||$ on Eigenvectors", fontsize=16)
    axes[1].set_xlabel("Eigenvector Index", fontsize=16)
    axes[1].set_ylabel("Matrix Size", fontsize=16)

    plt.tight_layout()
    plt.show()