import scipy.sparse as sp
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spl
import seaborn as sns

import functions as fu

def sparse_matrix_one_eig(L, omega, sizes, order):

    num_sizes = len(sizes)

    timings = np.zeros((num_sizes, 2))

    for idx, size in enumerate(sizes):
        deltax = (2*L)/size
        x_i = np.array([(((2*L)/size)*i - L) for i in range(size)])

        K = fu.kinetic_gen(size, deltax, order)
        V = fu.potential_gen(size, x_i, omega)

        A = K + V

        #SPARSE MATRIX
        matrix = sp.csc_matrix(A, shape=(size, size))

        tic = time.time()
        eigvl, eigvc = spl.eigsh(matrix, k=1, which="SA")
        timings[idx, 0] = time.time() - tic

        tic = time.time()
        eigvl, eigvc = np.linalg.eigh(matrix.todense())
        timings[idx, 1] = time.time() - tic

        
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sizes, timings[:, 0], "o--", label="Sparse", color="forestgreen")
    ax.plot(sizes, timings[:, 1], "s--", label="Dense", color="firebrick")
    ax.legend(fontsize = 16)
    ax.set_xscale("log")

    ax.set_xlabel("Size of the matrix", fontsize=14)
    ax.set_ylabel("Computational time [s]", fontsize=14)
    ax.set_title("Time to find the lowest eigenvalue", fontsize=16)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return matrix, timings



def sparse_matrix_heatmap(L, omega, sizes, order, num_eig):

    num_sizes = len(sizes)

    timings = np.zeros((num_eig, num_sizes))

    for idx_1, size in enumerate(sizes):
        deltax = (2*L)/size
        x_i = np.array([(((2*L)/size)*i - L) for i in range(size)])

        K = fu.kinetic_gen(size, deltax, order)
        V = fu.potential_gen(size, x_i, omega)

        A = K + V

        #SPARSE MATRIX
        matrix = sp.csc_matrix(A, shape=(size, size))

        tic = time.time()
        eigvl, eigvc = np.linalg.eigh(matrix.todense())
        dense_timing = time.time() - tic

        list_values = np.arange(1, num_eig+1)

        for idx_2, n in enumerate(list_values):
            tic = time.time()
            eigvl, eigvc = spl.eigsh(matrix, k=n, which="SA")
            sparse_timing = time.time() - tic
            diff_timing = dense_timing - sparse_timing
            timings[idx_2, idx_1] = diff_timing


        
    # Visualizzazione delle heatmaps
    plt.figure(figsize=(12,8))

    # Heatmap per gli errori sugli autovalori
    sns.heatmap(timings,  annot=False, cmap="YlGnBu", 
                xticklabels=sizes, yticklabels=np.arange(1, num_eig+1))
    plt.title("Time Difference (Dense - Sparse) for Eigenvalue Calculation", fontsize=16)
    plt.xlabel("Matrix Size", fontsize=16)
    plt.ylabel("Number of Eigenvalues", fontsize=16)

    plt.tight_layout()
    plt.show()

    return matrix, timings