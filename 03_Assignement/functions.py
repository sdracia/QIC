# generation of random hermitian matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import os
from scipy.optimize import curve_fit
import seaborn as sns
from time import time
import scipy.sparse.linalg as spl
import networkx as nx

from netgraph import ArcDiagram

def generate_random_hermitian_matrix(N, seed, flag):
    """Generates a random Hermitian matrix of size NxN with complex entries.

    Args:
        N (int): Dimension of the matrix.
        seed (int): Random seed for reproducibility.
        flag (int): Determines distribution type:
                    - 0: Standard normal distribution.
                    - 1: Uniform distribution in range [-1, 1].
                    - 2: Data from QRNG.

    Raises:
        ValueError: If bytes_list is insufficient for flag 2.
        ValueError: If flag value is invalid.

    Returns:
        np.ndarray: Hermitian matrix of complex values.
    """

    np.random.seed(seed)
    
    if flag == 0:
        # Generate using standard normal distribution
        real_part = np.random.randn(N, N)
        imag_part = np.random.randn(N, N)
    elif flag == 1:
        # Generate values between -1 and 1
        real_part = np.random.uniform(-1, 1, (N, N))
        imag_part = np.random.uniform(-1, 1, (N, N))
    elif flag == 2:
        # START IMPORTING DATA
        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_0.txt')
        df0 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_1.txt')
        df1 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_2.txt')
        df2 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        df0 = df0.diff()
        df1 = df1.diff()
        df2 = df2.diff()

        df_stat = pd.concat([df0, df1, df2])
        df_stat = df_stat[df_stat.Time_Tag > 3900]

        data = np.array(df_stat.Time_Tag)
        #FINISH IMPORTING DATA

        bytes_list = qrng_list(data)

        # Generate values from bytes_list and normalize to [-1, 1]
        if len(bytes_list) < N * N:
            raise ValueError("Not enough bytes in bytes_list to fill the matrix.")
        
        # Randomly sample N*N values from bytes_list
        sampled_bytes = np.random.choice(bytes_list, size=(N, N), replace=False)
        # Normalize to range [-1, 1]
        real_part = (sampled_bytes / 127.5) - 1  # Scale from [0, 255] to [-1, 1]
        imag_part = (sampled_bytes / 127.5) - 1  # Using the same method for imaginary part
    else:
        raise ValueError("Invalid flag value. Use 0 for normal distribution or 1 for [-1, 1].")
    
    # Combine real and imaginary parts to form a complex matrix
    A = real_part + 1j * imag_part
    
    # Make the matrix Hermitian
    A = np.tril(A) + np.tril(A, -1).T.conj()  # Lower triangle + conjugate transpose of upper
    
    return A

#generate random real diagonal matrix
def generate_real_diag_matrix(N, seed, flag):
    """Generates a random real diagonal matrix of size NxN.

    Args:
        N (int): Dimension of the matrix.
        seed (int): Random seed for reproducibility.
        flag (int): Determines distribution type:
                    - 0: Standard normal distribution.
                    - 1: Uniform distribution in range [-1, 1].
                    - 2: Data from QRNG.

    Raises:
        ValueError: If bytes_list is insufficient for flag 2.
        ValueError: If flag value is invalid.

    Returns:
        tuple: (np.ndarray, np.ndarray) Diagonal matrix and its diagonal entries.
    """

    np.random.seed(seed)
    
    if flag == 0:
        # Generate using standard normal distribution
        diagonal_entries = np.random.normal(loc=0, scale=1, size=N)
    elif flag == 1:
        # Generate values between -1 and 1
        diagonal_entries = np.random.uniform(low=-1, high=1, size=N)
    elif flag == 2:
        # START IMPORTING DATA
        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_0.txt')
        df0 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_1.txt')
        df1 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_2.txt')
        df2 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        df0 = df0.diff()
        df1 = df1.diff()
        df2 = df2.diff()

        df_stat = pd.concat([df0, df1, df2])
        df_stat = df_stat[df_stat.Time_Tag > 3900]

        data = np.array(df_stat.Time_Tag)
        #FINISH IMPORTING DATA

        bytes_list = qrng_list(data)

        # Generate values from bytes_list and normalize to [-1, 1]
        if len(bytes_list) < N:
            raise ValueError("Not enough bytes in bytes_list to fill the matrix.")
        
        # Randomly sample N*N values from bytes_list
        sampled_bytes = np.random.choice(bytes_list, size=N, replace=False)
        # Normalize to range [-1, 1]
        diagonal_entries = (sampled_bytes / 127.5) - 1  # Scale from [0, 255] to [-1, 1]
    else:
        raise ValueError("Invalid flag value. Use 0 for normal distribution or 1 for [-1, 1].")
    
    A = np.diag(diagonal_entries)
    
    return A, diagonal_entries


def generate_sparse_random(N, seed, density, flag, sparse=True):
    """Generates a sparse Hermitian matrix with given density.

    Args:
        N (int): Dimension of the matrix.
        seed (int): Random seed for reproducibility.
        density (float): Proportion of non-zero elements.
        flag (int): Distribution type:
                    - 0: Standard normal distribution.
                    - 1: Uniform distribution in range [-1, 1].
                    - 2: Data from QRNG.
        sparse (bool, optional): If True, returns sparse matrix; otherwise, dense. Defaults to True.

    Raises:
        ValueError: If bytes_list is insufficient for flag 2.
        ValueError: If flag is invalid.

    Returns:
        tuple: (scipy.sparse.csc_matrix, np.ndarray) Hermitian matrix and eigenvalues.
    """


    np.random.seed(seed)

    # Determine the number of non-zero elements based on density
    num_nonzero_elems = int(np.ceil(N**2 * density / 2))

    if flag == 0:
        # Standard normal distribution for real and imaginary parts
        real_part = np.random.randn(num_nonzero_elems)
        imag_part = np.random.randn(num_nonzero_elems)
    elif flag == 1:
        # Uniform distribution between -1 and 1
        real_part = np.random.uniform(-1, 1, num_nonzero_elems)
        imag_part = np.random.uniform(-1, 1, num_nonzero_elems)
    elif flag == 2:
        # START IMPORTING DATA
        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_0.txt')
        df0 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_1.txt')
        df1 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        path_spin =os.path.abspath('/home/sdruci/Q-OpticsLaser/lab1/Static_wheel/Part_2.txt')
        df2 = pd.read_csv(path_spin, sep=';', header=None, names=['Time_Tag', 'Channel'], skiprows=5)   

        df0 = df0.diff()
        df1 = df1.diff()
        df2 = df2.diff()

        df_stat = pd.concat([df0, df1, df2])
        df_stat = df_stat[df_stat.Time_Tag > 3900]

        data = np.array(df_stat.Time_Tag)
        #FINISH IMPORTING DATA

        bytes_list = qrng_list(data)

        # Generate values from bytes_list and normalize to [-1, 1]
        if len(bytes_list) < num_nonzero_elems * num_nonzero_elems:
            raise ValueError("Not enough bytes in bytes_list to fill the matrix.")
        
        # Randomly sample N*N values from bytes_list
        sampled_bytes = np.random.choice(bytes_list, size=(num_nonzero_elems, num_nonzero_elems), replace=False)
        # Normalize to range [-1, 1]
        real_part = (sampled_bytes / 127.5) - 1  # Scale from [0, 255] to [-1, 1]
        imag_part = (sampled_bytes / 127.5) - 1  # Using the same method for imaginary part
    else:
        raise ValueError("Invalid flag value. Use 0 for normal distribution or 1 for [-1, 1].")
    
    # Generate indices for non-zero elements
    elem_idxs = np.random.randint(0, N, size=(num_nonzero_elems, 2))
    elems = real_part + 1j * imag_part

    # Create sparse Hermitian matrix
    A = sp.csc_matrix((elems, (elem_idxs[:, 0], elem_idxs[:, 1])), shape=(N, N))
    A = A + A.conj().T

    # Calculate eigenvalues
    if sparse:
        eigenvalues, _ = sp.linalg.eigsh(A, k=min(N-2, A.shape[0] - 2))  # Ensure k < dimension of A
    else:
        dense_matrix = A.todense()  # Convert to dense matrix if required
        eigenvalues, _ = np.linalg.eigh(dense_matrix)
    
    return A, eigenvalues

def get_sparsity(matrix):
    """Calculates and prints sparsity percentage of the matrix.

    Args:
        matrix (scipy.sparse.csc_matrix): Sparse matrix to calculate sparsity.
    """

    num_nonzero = matrix.nnz
    total_elements = matrix.shape[0] * matrix.shape[1]
    # Calculate sparsity
    sparsity = 1 - (num_nonzero / total_elements)
    print(f"SPARSITY: {sparsity*100:.4f}%")


def compute_normalized_spacings(N, seed, flag, type, density=0.1):
    """Generates matrix and computes normalized spacings between eigenvalues.

    Args:
        N (int): Dimension of the matrix.
        seed (int): Random seed for reproducibility.
        flag (int): Distribution type for matrix generation.
        type (str): Matrix type - "hermitian", "diagonal", or "sparse".
        density (float, optional): Density of non-zero elements for sparse matrices. Defaults to 0.1.

    Returns:
        tuple: (np.ndarray, np.ndarray, np.ndarray) Generated matrix, eigenvalues, and normalized spacings.
    """

    # Generate Hermitian matrix and compute eigenvalues
    if (type == "hermitian"): 
        A = generate_random_hermitian_matrix(N, seed, flag)
        eigenvalues = np.linalg.eigh(A)[0]  # Sorted real eigenvalues
    elif (type == "diagonal"):
        A, eigenvalues = generate_real_diag_matrix(N, seed, flag)
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
    elif (type == "sparse"):
        A, eigenvalues = generate_sparse_random(N, seed, density, flag)
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
    else:
        print("Invalid type for matrix generation")

    # Compute spacings and normalize by mean spacing
    spacings = np.diff(eigenvalues)
    avg_spacing = np.mean(spacings)
    normalized_spacings = spacings / avg_spacing
    return A, eigenvalues, normalized_spacings

# calculate P(s) distribution, depending on type == "hermitian" or type == "diagonal"
def calculate_Ps_distribution(N, num_matrices, flag, type):
    """Calculates the distribution of normalized spacings P(s) across multiple matrices.

    Args:
        N (int): Dimension of the matrices.
        num_matrices (int): Number of matrices for averaging.
        flag (int): Distribution type.
        type (str): Type of matrix - "hermitian" or "diagonal".

    Returns:
        np.ndarray: Array of normalized spacings.
    """

    all_normalized_spacings = []
    
    # Accumulate normalized spacings from multiple random matrices
    for i in range(num_matrices):
        seed = i  # Use a different seed for each matrix
        A, eigenvalues, normalized_spacings = compute_normalized_spacings(N, seed, flag, type)

        all_normalized_spacings.extend(normalized_spacings)
    
    # Convert to numpy array
    all_normalized_spacings = np.array(all_normalized_spacings)
    
    
    return all_normalized_spacings



def plot_distr_and_fit(N_bins, spacings, num_matrices, N, flag, fitted_P_s):
    """Plots histogram of normalized spacings and fitted function.

    Args:
        N_bins (int): Number of bins for histogram.
        spacings (np.ndarray): Array of normalized spacings.
        num_matrices (int): Number of matrices used.
        N (int): Matrix size.
        flag (int): Type of distribution for labeling plot.
        fitted_P_s (np.ndarray): Fitted values for the distribution function.

    Raises:
        ValueError: If flag is invalid for plot label.
    """
    spacings_range = (np.min(spacings), np.max(spacings))
    s_vals = np.linspace(spacings_range[0], spacings_range[1], 1000)
    
    if flag == 0:
        msg = "# Generate using standard normal distribution"
    elif flag == 1:
        msg = "# Generate values between -1 and 1"
    elif flag == 2:
        msg = "# Generate values with quantum number generator"
    else:
        raise ValueError("Invalid flag value. Use 0 for normal distribution, or 1 for [-1, 1].")
    
    plt.figure(figsize=(10, 6))
    plt.hist(spacings, bins=N_bins, range=spacings_range, color='blue', edgecolor='blue', alpha=0.7, label='Normalized histogram', density=True)
    plt.plot(s_vals, fitted_P_s, 'r-', label='Fitted function', linewidth=1)

    # Set the title with larger font size
    plt.title(f'Distribution of Normalized Spacings $P(s)$\nwith {num_matrices} Matrices of Size {N}\n{msg}', 
              fontsize=18)

    # Increase the font size of axis titles
    plt.xlabel('Normalized Spacing $s$', fontsize=16)
    plt.ylabel('$P(s)$', fontsize=16)

    # Increase the font size of tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Increase the font size of the legend
    plt.legend(fontsize=14, loc='upper right')

    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()




def target_function(s, a, b, alpha, beta):
    """Target fitting function for P(s) = a * s^alpha * exp(b * s^beta).

    Args:
        s (np.ndarray): Array of spacings.
        a (float): Scale parameter.
        b (float): Exponential parameter.
        alpha (float): Power parameter.
        beta (float): Exponential power parameter.

    Returns:
        np.ndarray: Evaluated function values.
    """

    return a * (s ** alpha) * np.exp(b * ( s ** beta))

def fitting(N_bins, spacings, flag, type):
    """Fits distribution P(s) to normalized spacing data.

    Args:
        N_bins (int): Number of bins for histogram.
        spacings (np.ndarray): Array of normalized spacings.
        flag (int): Type of distribution for output labeling.
        type (str): Matrix type - "hermitian", "diagonal", or "sparse".

    Raises:
        ValueError: If flag is invalid for labeling.

    Returns:
        tuple: Fitted parameters a, b, alpha, beta, chi-square value, and fitted P(s) values.
    """

    counts, bin_edges = np.histogram(spacings, bins=N_bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    P_s = counts / (np.sum(counts) * np.diff(bin_edges)[0])  # Normalize counts
    
    # Initial guesses for parameters a, b, alpha, and beta
    if (type == "hermitian"):
        initial_guess = [1, 1, 1, 1]
    elif (type == "diagonal"):
        initial_guess = [6, 0.5, -3, 0.5]
    elif (type == "sparse"):
        initial_guess = [15, -3, 2.5, 1.3]
    else:
        print("Invalid type format")
    
    # Perform curve fitting
    params, _ = curve_fit(target_function, bin_centers, P_s, p0=initial_guess)
    a, b, alpha, beta = params
    
    # Print fitted parameters

    if flag == 0:
        msg = "# Generate using standard normal distribution"
    elif flag == 1:
        msg = "# Generate values between -1 and 1"
    elif flag == 2:
        msg = "# Generate values with quantum number generator"
    else:
        raise ValueError("Invalid flag value. Use 0 for normal distribution, or 1 for [-1, 1].")

    print(msg)
    print(f"Fitted parameters:\n a = {a:.4f}, b = {b:.4f}, alpha = {alpha:.4f}, beta = {beta:.4f}")
    
    # Generate fitted values for plotting
    estimated = target_function(bin_centers, a, b, alpha, beta)

    # Calculate chi-square
    # Approximate uncertainties as sqrt of counts (Poisson uncertainty)
    uncertainties = np.sqrt(counts)
    # Avoid division by zero in case of zero counts
    uncertainties[uncertainties == 0] = 1  # Assign a minimal uncertainty if count is zero
    
    chi_square = np.sum(((P_s - estimated) ** 2) / (uncertainties ** 2))

    print(f"Chi-square: {chi_square}")
    print('\n')

    spacings_range = (np.min(spacings), np.max(spacings))
    s_vals = np.linspace(spacings_range[0], spacings_range[1], 1000)
    fitted_P_s = target_function(s_vals, a, b, alpha, beta)
    
    return a, b, alpha, beta, chi_square, fitted_P_s


def bits_to_byte(bits):
    """Converts a list of bits to a byte.

    Args:
        bits (list): List of 8 bits.

    Returns:
        int: Byte representation of the bits.
    """

    byte = 0
    for bit in bits:
      byte = (byte << 1) | bit
    return byte
  
def qrng_list(data):
    """Generates a list of random bytes based on QRNG data.

    Args:
        data (np.ndarray): Array of time-tag data.

    Returns:
        list: List of generated bytes.
    """

    bits_list = []
    bytes_list = []
    i = 1
    while i < len(data):
      bits_list.append(int(data[i] > data[i-1]))
      i += 2

    bits_list = np.array(bits_list)
  
    for i in range(0, len(bits_list), 8):
      byte_chunk = bits_list[i:i+8]
      if len(byte_chunk) == 8:
        bytes_list.append(bits_to_byte(byte_chunk))

    return bytes_list


def results(N, num_matrices, N_bins, flag, type):
    """Computes best fit parameters for P(s) distribution and plots result.

    Args:
        N (int): Dimension of the matrices.
        num_matrices (int): Number of matrices for averaging.
        N_bins (int): Number of bins for histogram.
        flag (list): List of distribution flags.
        type (str): Matrix type - "hermitian", "diagonal", or "sparse".

    Returns:
        tuple: Best flag, chi-square value, and parameters a, b, alpha, beta.
    """

    # Initialize variables to keep track of the best chi-square value and corresponding flag
    best_chi_square = float('inf')
    best_flag = None
    best_params = None
    best_fit_P_s = None
    best_spacings = None

    # Loop through each flag and fit the distribution
    for i in flag:
        spacings = calculate_Ps_distribution(N, num_matrices, i, type)

        print(f"Fitting for flag {i}...")
        a, b, alpha, beta, chi_square, fitted_P_s = fitting(N_bins, spacings, i, type)

        # Check if this chi-square is the best one so far
        if chi_square < best_chi_square:
            best_chi_square = chi_square
            best_flag = i
            best_params = (a, b, alpha, beta)
            best_fit_P_s = fitted_P_s
            best_spacings = spacings

    # After finding the best flag, print the result
    print(f"Best flag: {best_flag}")
    print(f"Best chi-square value: {best_chi_square}")
    print(f"Best parameters: a = {best_params[0]:.4f}, b = {best_params[1]:.4f}, alpha = {best_params[2]:.4f}, beta = {best_params[3]:.4f}")

    # Plot the graph for the best flag
    plot_distr_and_fit(N_bins, best_spacings, num_matrices, N, best_flag, best_fit_P_s)

    return best_flag, best_chi_square, best_params[0], best_params[1], best_params[2], best_params[3]