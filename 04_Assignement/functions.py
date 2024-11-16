import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import analytical_solution as anso

def kinetic_gen(size, deltax, order = 2):
    """
    Generates the kinetic energy matrix for a discretized system with a specified accuracy order.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    deltax : float
        The spacing of the grid points.
    order : int, optional
        The order of accuracy for the finite difference method. Supported values are:
        - 2: Second-order accuracy.
        - 4: Fourth-order accuracy.
        Default is 2.

    Returns
    -------
    K : numpy.ndarray
        The generated kinetic energy matrix.

    Notes
    -----
    - For `order=2`, the matrix is constructed using the second-order central difference method.
    - For `order=4`, the matrix uses the fourth-order central difference method.
    - If an unsupported order is provided, an error message is printed, and the function returns nothing.
    """
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
    """
    Generates the potential energy matrix for a quantum harmonic oscillator.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    x_i : numpy.ndarray
        The positions in the discretized grid.
    omega : float
        The angular frequency of the harmonic oscillator.

    Returns
    -------
    V : numpy.ndarray
        The generated potential energy matrix.

    Notes
    -----
    - The potential is calculated as V(x) = (1/2) * omega^2 * x^2.
    - The matrix is diagonal, as the potential is position-dependent.
    """
    factor = (omega**2)/2
    main_diag = (x_i**2) * np.ones(size)

    V = factor * np.diag(main_diag)

    return V

def hamiltonian_gen(size, deltax, x_i, omega, order = 2):
    """
    Generates the Hamiltonian matrix and computes its eigenvalues and eigenvectors.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    deltax : float
        The spacing of the grid points.
    x_i : numpy.ndarray
        The positions in the discretized grid.
    omega : float
        The angular frequency of the harmonic oscillator.
    order : int, optional
        The order of accuracy for the finite difference method. Supported values are 2 or 4.
        Default is 2.

    Returns
    -------
    A : numpy.ndarray
        The Hamiltonian matrix (K + V).
    eigenvalues : numpy.ndarray
        The eigenvalues of the Hamiltonian, sorted in ascending order.
    eigenvectors : numpy.ndarray
        The eigenvectors of the Hamiltonian, normalized and aligned.

    Notes
    -----
    - The Hamiltonian is computed as H = K + V.
    - The eigenvectors are normalized and adjusted for consistent sign convention.
    """

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
    """
    Plots the eigenfunctions and energy levels of the quantum harmonic oscillator.

    Parameters
    ----------
    number_to_print : int
        Number of eigenfunctions and energy levels to display.
    eigenvalues : numpy.ndarray
        The eigenvalues of the Hamiltonian.
    eigenvectors : numpy.ndarray
        The eigenvectors of the Hamiltonian.
    x_i : numpy.ndarray
        The positions in the discretized grid.
    L : float
        The limit of the grid (half-width).
    omega : float
        The angular frequency of the harmonic oscillator.

    Returns
    -------
    None
        Displays the plots.
    """
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
    """
    Computes the errors between approximate and analytical eigenvalues and eigenvectors.

    Parameters
    ----------
    k : int
        Number of eigenvalues and eigenvectors to consider.
    eigenvalues : np.ndarray
        Approximate eigenvalues.
    eigenvectors : np.ndarray
        Approximate eigenvectors.
    eigenvalues_analy : np.ndarray
        Analytical eigenvalues for comparison.
    eigenvectors_analy : np.ndarray
        Analytical eigenvectors for comparison.

    Returns
    -------
    tuple
        A tuple containing:
        - relative_eigval_errors (np.ndarray): Relative errors of the eigenvalues.
        - eigvec_dot (list): Dot product differences between approximate and analytical eigenvectors.

    Notes
    -----
    - The eigenvectors are normalized before calculating dot products.
    - Dot products indicate similarity, with values closer to zero implying higher similarity.
    """
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
    """
    Plots the relative errors in eigenvalues and dot product differences for eigenvectors.

    Parameters
    ----------
    k : int
        Number of eigenvalues and eigenvectors to display.
    rel_eigval_err : np.ndarray
        Relative errors of the eigenvalues.
    eigvec_dot : list
        Dot product differences between approximate and analytical eigenvectors.

    Returns
    -------
    None
        Displays two bar plots for eigenvalue and eigenvector errors.
    """
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
    """
    Plots relative errors in eigenvalues and dot product differences for eigenvectors 
    comparing second-order and fourth-order approximations.

    Parameters
    ----------
    k : int
        Number of eigenvalues and eigenvectors to display.
    eigval2 : np.ndarray
        Relative errors in eigenvalues for the second-order approximation.
    eigval4 : np.ndarray
        Relative errors in eigenvalues for the fourth-order approximation.
    eigvec2 : list
        Dot product differences for eigenvectors in the second-order approximation.
    eigvec4 : list
        Dot product differences for eigenvectors in the fourth-order approximation.

    Returns
    -------
    None
        Displays a side-by-side comparison of the correctness plots for the two orders.
    """
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
    axes[1].set_xlabel("Eigenvector Index", fontsize=12)
    axes[1].set_ylabel("1 - dot product", fontsize=12)
    axes[1].set_title("Dot Product Between Approximate and Analytical Eigenvectors", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Display the plots
    plt.tight_layout()
    plt.show()

#############################
### STABILITY FUNCTIONS ###
#############################

def stability(num_runs, order, k, N, deltax, x_i, omega):
    """
    Analyzes the stability of eigenvalues and eigenvectors over multiple runs.

    Parameters
    ----------
    num_runs : int
        Number of repetitions for stability analysis.
    order : int
        Order of discretization (2 or 4).
    k : int
        Number of eigenvalues and eigenvectors to consider.
    N : int
        Size of the Hamiltonian matrix.
    deltax : float
        Discretization step size.
    x_i : np.ndarray
        Grid points.
    omega : float
        Frequency parameter for the potential.

    Returns
    -------
    tuple
        - eigenvalues_std (np.ndarray): Standard deviations of eigenvalues across runs.
        - eigvec_dot_mean (np.ndarray): Mean dot product differences for eigenvectors.
        - dot_matrix (np.ndarray): Dot product differences for all eigenvectors across runs.

    Notes
    -----
    - Eigenvalues and eigenvectors are calculated `num_runs` times for consistency analysis.
    - Dot product differences between eigenvectors indicate variability across runs.
    """
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
    """
    Analyzes the effect of discretization size on the accuracy of eigenvalues and eigenvectors.

    Parameters
    ----------
    N_min : int
        Minimum number of grid points.
    N_max : int
        Maximum number of grid points.
    step : int
        Step size for the number of grid points.
    k : int
        Number of eigenvalues and eigenvectors to consider.
    omega : float
        Frequency parameter for the potential.
    L : float
        Half-length of the spatial domain.
    order : int
        Order of discretization (2 or 4).

    Returns
    -------
    tuple
        A tuple containing:
        - eigval_errors_matrix (np.ndarray): Relative errors of eigenvalues for each N.
        - eigvec_dots_matrix (np.ndarray): Dot product differences for eigenvectors for each N.
        - sizes (list): List of discretization sizes analyzed.
    """
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
    """
    Examines how the eigenvalue and eigenvector errors vary with the frequency parameter, omega.

    Parameters
    ----------
    omega_min : float
        Minimum value of omega.
    omega_max : float
        Maximum value of omega.
    omega_step : float
        Step size for omega variation.
    k : int
        Number of eigenvalues and eigenvectors to consider.
    N : int
        Size of the Hamiltonian matrix.
    L : float
        Half-length of the spatial domain.
    order : int
        Order of discretization (2 or 4).

    Returns
    -------
    tuple
        - eigval_errors_matrix (np.ndarray): Relative errors of eigenvalues for each omega.
        - eigvec_dots_matrix (np.ndarray): Dot product differences for eigenvectors for each omega.
        - omegas (list): List of omega values analyzed.
    """

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


def scaling_heatmap(eigval_errors_matrix, eigvec_dots_matrix, sizes, k, type):
    """
    Generates heatmaps to visualize eigenvalue and eigenvector errors across discretization sizes.

    Parameters
    ----------
    eigval_errors_matrix : np.ndarray
        Relative errors of eigenvalues for different discretization sizes.
    eigvec_dots_matrix : np.ndarray
        Dot product differences of eigenvectors for different discretization sizes.
    sizes : list
        Discretization sizes corresponding to the rows of the matrices.
    k : int
        Number of eigenvalues and eigenvectors analyzed.
    type : string
        Value of type of heatmap generated, which can be either "size" or "omega",
        depending on the kind of scaling we are performing

    Returns
    -------
    None
        Displays heatmaps for eigenvalue and eigenvector errors.
    """
    # Visualizzazione delle heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Heatmap per gli errori sugli autovalori
    sns.heatmap(np.log10(eigval_errors_matrix), ax=axes[0], annot=False, cmap="YlGnBu", 
                xticklabels=np.arange(1, k+1), yticklabels=sizes)
    axes[0].set_title("Log10 of Errors on Eigenvalues", fontsize=16)
    axes[0].set_xlabel("Eigenvalue Index", fontsize=16)
    if (type == "size"):
        axes[0].set_ylabel("Matrix Size", fontsize=16)
    elif (type == "omega"):
        axes[0].set_ylabel("Omega value", fontsize=16)
    else:
        print("Invalid type")



    # Heatmap per il prodotto scalare degli autovettori
    sns.heatmap(np.log10(eigvec_dots_matrix), ax=axes[1], annot=False, cmap="YlOrRd", 
                xticklabels=np.arange(1, k+1), yticklabels=sizes)
    axes[1].set_title(" Log of Error $|1 - |dot||$ on Eigenvectors", fontsize=16)
    axes[1].set_xlabel("Eigenvector Index", fontsize=16)
    if (type == "size"):
        axes[0].set_ylabel("Matrix Size", fontsize=16)
    elif (type == "omega"):
        axes[0].set_ylabel("Omega value", fontsize=16)
    else:
        print("Invalid type")

    plt.tight_layout()
    plt.show()