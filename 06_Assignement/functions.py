import numpy as np
import itertools
import time


def initialize_coefficients(N, D, seed, type, init_coeff=None, random_init=False):

    np.random.seed(seed)

    if (type == "separable"):
        if random_init:
            coefficients = []
            for _ in range(N):
                # Generate random complex coefficients and normalize them
                c = np.random.random(D) + 1j * np.random.random(D)  # Random complex numbers
                c /= np.linalg.norm(c)  # Normalize
                coefficients.append(c)
        else: 
            coefficients = init_coeff

    elif (type == "general"):
        total_dim = D**N  # Dimension of the composite Hilbert space
    
        if random_init:
            # Generate random complex coefficients and normalize them
            coefficients = np.random.random(total_dim) + 1j * np.random.random(total_dim)
            coefficients /= np.linalg.norm(coefficients)  # Normalize
        else: 
            coefficients = init_coeff

    else:
        print("Invalid type")


    # Validate coefficients
    try:
        is_valid = validate_coefficients(coefficients, N, D, type)
        print("Coefficients are valid:", is_valid)
    except ValueError as e:
        print("Validation error:", e)
    
    return coefficients


def validate_coefficients(coefficients, N, D, type):
    """
    Validates the coefficients for a separable quantum state.
    
    Parameters:
        coefficients (list of np.array): List of coefficient arrays for each subsystem.
        N (int): Number of subsystems.
        D (int): Dimension of the Hilbert space for each subsystem.
        
    Raises:
        ValueError: If the coefficients fail any of the validation checks.
        
    Returns:
        bool: True if all checks pass.
    """
    if (type == "separable"):
        # Check number of subsystems
        if len(coefficients) != N:
            raise ValueError(f"Expected coefficients for {N} subsystems, but got {len(coefficients)}.")

        # Check dimensionality and normalization for each subsystem
        for idx, c in enumerate(coefficients):
            if len(c) != D:
                raise ValueError(f"Subsystem {idx+1} coefficients must have dimension {D}, but got {len(c)}.")
            if not np.isclose(np.sum(np.abs(c)**2), 1):
                raise ValueError(
                    f"Subsystem {idx+1} coefficients are not normalized. "
                    f"Expected sum(|c_j|^2) = 1, but got {np.sum(np.abs(c)**2)}."
                )
            
    elif (type == "general"):
        total_dim = D**N  # Total number of basis states in the composite Hilbert space
    
        # Check dimensionality of the coefficients
        if len(coefficients) != total_dim:
            raise ValueError(f"Expected {total_dim} coefficients, but got {len(coefficients)}.")

        # Check normalization of the coefficients
        if not np.isclose(np.sum(np.abs(coefficients)**2), 1):
            raise ValueError(
                f"Coefficients are not normalized. "
                f"Expected sum(|c_j|^2) = 1, but got {np.sum(np.abs(coefficients)**2)}."
            )
    
    else:
        print("Invalid type")

    
    # All checks passed
    return True


def create_state(N, D, coefficients, type):
    """
    Creates a generic separable state for a system composed of N subsystems.
    
    Parameters:
        N (int): Number of subsystems.
        D (int): Dimension of the Hilbert space for each subsystem.
        coefficients (list of np.array): List of NumPy arrays containing the coefficients 
                                         c_j for each subsystem.
        
    Returns:
        np.array: State of the total system in the composite Hilbert space.
    """

    if (type == "separable"):
        # Initialize with the state of the first subsystem
        basis_vectors = [np.eye(D)[:, j] for j in range(D)]  # Basis vectors of the D-dimensional Hilbert space

        # Construct the weighted sum for the first subsystem
        state = sum(coefficients[0][j] * basis_vectors[j] for j in range(D))

        # Compute the tensor product with the remaining subsystems
        for i in range(1, N):
            sub_state = sum(coefficients[i][j] * basis_vectors[j] for j in range(D))
            state = np.kron(state, sub_state)

    elif (type == "general"):
        total_dim = D**N  # Dimension of the composite Hilbert space
    
        # Generate all multi-indices (j1, j2, ..., jN) for the composite system
        indices = list(itertools.product(range(D), repeat=N))

        # Build the composite state
        basis_vectors = [np.eye(D)[:, j] for j in range(D)]  # Basis vectors for each subsystem
        state = np.zeros(total_dim, dtype=complex)

        for idx, multi_index in enumerate(indices):
            tensor_product = basis_vectors[multi_index[0]]
            for j in multi_index[1:]:
                tensor_product = np.kron(tensor_product, basis_vectors[j])
            state += coefficients[idx] * tensor_product

    else:
        print("Invalid type")
    
    return state



def comput_time(N_max, D_max, seed, type):

    N_min = 1
    N_step = 1

    D_min = 1
    D_step = 1

    N_sizes = list(range(N_min, N_max, N_step))
    D_sizes = list(range(D_min, D_max, D_step))

    cpu_times_matrix = np.zeros((len(N_sizes), len(D_sizes)))

    for idx_1, N_i in enumerate(N_sizes):

        for idx_2, D_i in enumerate(D_sizes):
            # print(N_i, " ", D_i)
            start_time = time.time()
            
            coefficients = initialize_coefficients(N_i, D_i, seed, type, random_init=True)  # Use random initialization
            separable_state = create_state(N_i, D_i, coefficients, type)

            end_time = time.time()

            # Store the elapsed time in the matrix
            elapsed_time = end_time - start_time
            cpu_times_matrix[idx_1, idx_2] = elapsed_time
    

    return N_sizes, D_sizes, cpu_times_matrix



# def initialize_general_coefficients(N, D, seed, init_coeff, random_init=False):

#     np.random.seed(seed)

#     total_dim = D**N  # Dimension of the composite Hilbert space
    
#     if random_init:
#         # Generate random complex coefficients and normalize them
#         coefficients = np.random.random(total_dim) + 1j * np.random.random(total_dim)
#         coefficients /= np.linalg.norm(coefficients)  # Normalize
#     else: 
#         coefficients = init_coeff

#     # Validate coefficients
#     try:
#         is_valid = validate_general_coefficients(coefficients, N, D)
#         print("Coefficients are valid:", is_valid)
#     except ValueError as e:
#         print("Validation error:", e)
    
#     return coefficients


# def validate_general_coefficients(coefficients, N, D):
#     """
#     Validates the coefficients for a general non-separable quantum state.

#     Parameters:
#         coefficients (np.array): Array of coefficients for the composite state.
#         N (int): Number of subsystems.
#         D (int): Dimension of the Hilbert space for each subsystem.
    
#     Raises:
#         ValueError: If the coefficients fail any of the validation checks.
    
#     Returns:
#         bool: True if all checks pass.
#     """
#     total_dim = D**N  # Total number of basis states in the composite Hilbert space
    
#     # Check dimensionality of the coefficients
#     if len(coefficients) != total_dim:
#         raise ValueError(f"Expected {total_dim} coefficients, but got {len(coefficients)}.")
    
#     # Check normalization of the coefficients
#     if not np.isclose(np.sum(np.abs(coefficients)**2), 1):
#         raise ValueError(
#             f"Coefficients are not normalized. "
#             f"Expected sum(|c_j|^2) = 1, but got {np.sum(np.abs(coefficients)**2)}."
#         )
    
#     # All checks passed
#     return True



# def create_general_state(N, D, coefficients):
#     """
#     Constructs a general non-separable quantum state for a system of N subsystems.
    
#     Parameters:
#         N (int): Number of subsystems.
#         D (int): Dimension of the Hilbert space for each subsystem.
#         coefficients (np.array, optional): Array of complex coefficients for each basis state. 
#                                            Default is None.
#         random_init (bool, optional): If True, initializes random normalized coefficients. 
#                                       Ignored if `coefficients` is provided. Default is False.
    
#     Returns:
#         np.array: The state vector for the composite Hilbert space of dimension D^N.
    
#     Raises:
#         ValueError: If the coefficients are not normalized or their size is incorrect.
#     """
#     total_dim = D**N  # Dimension of the composite Hilbert space
    
#     # Generate all multi-indices (j1, j2, ..., jN) for the composite system
#     indices = list(itertools.product(range(D), repeat=N))
    
#     # Build the composite state
#     basis_vectors = [np.eye(D)[:, j] for j in range(D)]  # Basis vectors for each subsystem
#     state = np.zeros(total_dim, dtype=complex)
    
#     for idx, multi_index in enumerate(indices):
#         tensor_product = basis_vectors[multi_index[0]]
#         for j in multi_index[1:]:
#             tensor_product = np.kron(tensor_product, basis_vectors[j])
#         state += coefficients[idx] * tensor_product
    
#     return state
