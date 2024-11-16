import scipy.sparse as sp
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spl
import seaborn as sns

import functions as fu


def sparse_matrix_one_eig(L, omega, sizes, order):
    """
    sparse_matrix_one_eig:
        Compares the computational time to find the lowest eigenvalue 
        of matrices using sparse and dense methods.

    Parameters
    ----------
    L : float
        Half-length of the domain over which the matrix is defined.
    omega : float
        Angular frequency of the potential.
    sizes : list[int]
        List of matrix sizes to evaluate.
    order : int
        Order of the finite difference scheme used to compute the kinetic energy matrix.

    Returns
    -------
    matrix : scipy.sparse.csc_matrix
        Sparse representation of the last matrix generated.
    timings : np.ndarray
        Array of shape (len(sizes), 2) containing computational times:
        - timings[:, 0]: Times for sparse computations.
        - timings[:, 1]: Times for dense computations.
    """
    num_sizes = len(sizes)  # Number of different matrix sizes to evaluate
    timings = np.zeros((num_sizes, 2))  # Array to store timings for sparse and dense methods

    for idx, size in enumerate(sizes):
        deltax = (2 * L) / size  # Grid spacing
        x_i = np.array([(((2 * L) / size) * i - L) for i in range(size)])  # Real space grid

        # Generate kinetic and potential energy matrices
        K = fu.kinetic_gen(size, deltax, order)
        V = fu.potential_gen(size, x_i, omega)

        A = K + V  # Total Hamiltonian matrix

        # Convert the Hamiltonian matrix to sparse format
        matrix = sp.csc_matrix(A, shape=(size, size))

        # Compute the lowest eigenvalue using the sparse solver
        tic = time.time()
        eigvl, eigvc = spl.eigsh(matrix, k=1, which="SA")  # Smallest algebraic eigenvalue
        timings[idx, 0] = time.time() - tic

        # Compute the lowest eigenvalue using the dense solver
        tic = time.time()
        eigvl, eigvc = np.linalg.eigh(matrix.todense())  # Dense solver for all eigenvalues
        timings[idx, 1] = time.time() - tic

    # Plotting the computational time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sizes, timings[:, 0], "o--", label="Sparse", color="forestgreen")  # Sparse timings
    ax.plot(sizes, timings[:, 1], "s--", label="Dense", color="firebrick")  # Dense timings
    ax.legend(fontsize=16)
    ax.set_xscale("log")  # Use a logarithmic scale for matrix size
    ax.set_xlabel("Size of the matrix", fontsize=14)
    ax.set_ylabel("Computational time [s]", fontsize=14)
    ax.set_title("Time to find the lowest eigenvalue", fontsize=16)

    plt.grid(True, linestyle="--", alpha=0.7)  # Add grid for better visualization
    plt.show()

    return matrix, timings


def sparse_matrix_heatmap(L, omega, sizes, order, num_eig):
    """
    sparse_matrix_heatmap:
        Creates a heatmap comparing the computational time difference between 
        sparse and dense methods for eigenvalue computations.

    Parameters
    ----------
    L : float
        Half-length of the domain over which the matrix is defined.
    omega : float
        Angular frequency of the potential.
    sizes : list[int]
        List of matrix sizes to evaluate.
    order : int
        Order of the finite difference scheme used to compute the kinetic energy matrix.
    num_eig : int
        Number of eigenvalues to compute using the sparse solver.

    Returns
    -------
    matrix : scipy.sparse.csc_matrix
        Sparse representation of the last matrix generated.
    timings : np.ndarray
        Array of shape (num_eig, len(sizes)) containing the time differences 
        (dense - sparse) for eigenvalue computations.
    """
    num_sizes = len(sizes)  # Number of different matrix sizes to evaluate
    timings = np.zeros((num_eig, num_sizes))  # Array to store time differences for each eigenvalue

    for idx_1, size in enumerate(sizes):
        deltax = (2 * L) / size  # Grid spacing
        x_i = np.array([(((2 * L) / size) * i - L) for i in range(size)])  # Real space grid

        # Generate kinetic and potential energy matrices
        K = fu.kinetic_gen(size, deltax, order)
        V = fu.potential_gen(size, x_i, omega)

        A = K + V  # Total Hamiltonian matrix

        # Convert the Hamiltonian matrix to sparse format
        matrix = sp.csc_matrix(A, shape=(size, size))

        # Time the dense solver for all eigenvalues
        tic = time.time()
        eigvl, eigvc = np.linalg.eigh(matrix.todense())  # Dense solver
        dense_timing = time.time() - tic

        # Compute time differences for each number of eigenvalues
        list_values = np.arange(1, num_eig + 1)  # Eigenvalues to compute
        for idx_2, n in enumerate(list_values):
            tic = time.time()
            eigvl, eigvc = spl.eigsh(matrix, k=n, which="SA")  # Sparse solver
            sparse_timing = time.time() - tic
            diff_timing = dense_timing - sparse_timing  # Time difference
            timings[idx_2, idx_1] = diff_timing

    # Heatmap visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        timings, annot=False, cmap="YlGnBu",
        xticklabels=sizes, yticklabels=np.arange(1, num_eig + 1)
    )
    plt.title("Time Difference (Dense - Sparse) for Eigenvalue Calculation", fontsize=16)
    plt.xlabel("Matrix Size", fontsize=16)
    plt.ylabel("Number of Eigenvalues", fontsize=16)

    plt.tight_layout()
    plt.show()

    return matrix, timings
