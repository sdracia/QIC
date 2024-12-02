import numpy as np
import itertools
import time
import sys


def initialize_coefficients(N, D, seed, type, init_coeff=None, random_init=False):
    """
    Initializes the coefficients for a quantum state, either separable or general.

    Parameters
    ----------
    N : int
        Number of subsystems in the quantum system.
    D : int
        Dimension of the Hilbert space for each subsystem.
    seed : int
        Seed for the random number generator to ensure reproducibility.
    type : str
        Type of quantum state. Must be either "separable" or "general".
    init_coeff : list or np.array, optional
        Predefined coefficients. If provided, these are used instead of random initialization.
        For "separable", it is a list of NumPy arrays (one per subsystem).
        For "general", it is a single NumPy array for the entire state.
        Default is None.
    random_init : bool, optional
        If True, coefficients are randomly initialized. If False, `init_coeff` is used.
        Default is False.

    Returns
    -------
    list or np.array
        Coefficients for the quantum state.
        - For "separable", returns a list of NumPy arrays, each containing the coefficients 
          for a single subsystem.
        - For "general", returns a single NumPy array containing coefficients for the 
          composite state.

    Raises
    ------
    ValueError
        If the validation of the coefficients fails.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Case 1: Separable state
    if type == "separable":
        if random_init:
            coefficients = []
            for _ in range(N):
                # Generate random complex coefficients for a subsystem
                c = np.random.random(D) + 1j * np.random.random(D)  # Complex random numbers
                c /= np.linalg.norm(c)  # Normalize the coefficients
                coefficients.append(c)  # Append to the list
        else:
            coefficients = init_coeff  # Use predefined coefficients if provided

    # Case 2: General state
    elif type == "general":
        total_dim = D**N  # Total dimension of the composite Hilbert space

        if random_init:
            # Generate random complex coefficients for the entire composite system
            coefficients = np.random.random(total_dim) + 1j * np.random.random(total_dim)
            coefficients /= np.linalg.norm(coefficients)  # Normalize the coefficients
        else:
            coefficients = init_coeff  # Use predefined coefficients if provided

    else:
        # Handle invalid type with an error message
        raise ValueError(f"Invalid state type '{type}'. Must be either 'separable' or 'general'.")

    # Validate the initialized coefficients
    try:
        is_valid = validate_coefficients(coefficients, N, D, type)
        # Uncomment the next line for debugging
        # print("Coefficients are valid:", is_valid)
    except ValueError as e:
        print("Validation error:", e)
        raise  # Re-raise the exception for debugging purposes

    # Return the validated coefficients
    return coefficients



def validate_coefficients(coefficients, N, D, type):
    """
    Validates the coefficients of a quantum state to ensure they meet the expected
    physical and dimensional requirements based on the state type (separable or general).

    Parameters
    ----------
    coefficients : list or np.array
        For a separable state, this is a list of NumPy arrays, each containing the 
        coefficients for a single subsystem. For a general state, it is a single 
        NumPy array containing the coefficients for the entire composite system.
    N : int
        Number of subsystems (e.g., lattice sites or qudits) in the system.
    D : int
        Local dimension of the Hilbert space for each subsystem (e.g., 2 for qubits, 3 for qutrits).
    type : str
        Type of quantum state. Must be either "separable" or "general".
    
    Raises
    ------
    ValueError
        If the coefficients fail any of the validation checks, such as mismatched dimensions 
        or lack of normalization.
    
    Returns
    -------
    bool
        Returns True if all validation checks pass, indicating that the coefficients 
        are correctly initialized and physically meaningful.
    """
    if type == "separable":
        # Ensure the number of coefficient arrays matches the number of subsystems
        if len(coefficients) != N:
            raise ValueError(
                f"Expected coefficients for {N} subsystems, but got {len(coefficients)}."
            )

        # Check each subsystem's coefficients for correct dimensionality and normalization
        for idx, c in enumerate(coefficients):
            # Verify that the coefficients for this subsystem have the correct dimension
            if len(c) != D:
                raise ValueError(
                    f"Subsystem {idx + 1} coefficients must have dimension {D}, but got {len(c)}."
                )
            # Ensure the coefficients are normalized (sum of squared magnitudes equals 1)
            if not np.isclose(np.sum(np.abs(c) ** 2), 1):
                raise ValueError(
                    f"Subsystem {idx + 1} coefficients are not normalized. "
                    f"Expected sum(|c_j|^2) = 1, but got {np.sum(np.abs(c) ** 2)}."
                )

    elif type == "general":
        total_dim = D ** N  # Total dimension of the composite Hilbert space
        
        # Verify that the total number of coefficients matches the Hilbert space size
        if len(coefficients) != total_dim:
            raise ValueError(
                f"Expected {total_dim} coefficients for the composite state, but got {len(coefficients)}."
            )

        # Ensure the coefficients are normalized
        if not np.isclose(np.sum(np.abs(coefficients) ** 2), 1):
            raise ValueError(
                f"Coefficients are not normalized. "
                f"Expected sum(|c_j|^2) = 1, but got {np.sum(np.abs(coefficients) ** 2)}."
            )

    else:
        # Handle invalid state type with an error message
        raise ValueError(f"Invalid state type '{type}'. Must be either 'separable' or 'general'.")

    # If all checks pass, return True
    return True



def create_state(N, D, coefficients, type):
    """
    Constructs the quantum state for a system composed of N subsystems, 
    either as a separable state or as a general entangled state.

    Parameters
    ----------
    N : int
        Number of subsystems (e.g., qudits or lattice sites).
    D : int
        Dimension of the Hilbert space for each subsystem.
    coefficients : list or np.array
        - For "separable": A list of NumPy arrays, each containing the coefficients 
          for a single subsystem.
        - For "general": A single NumPy array containing coefficients for the entire 
          composite system.
    type : str
        Type of quantum state. Must be either "separable" or "general".

    Returns
    -------
    np.array
        The full quantum state as a NumPy array in the composite Hilbert space 
        of dimension \( D^N \).

    Raises
    ------
    ValueError
        If an invalid `type` is specified.
    """
    # Case 1: Separable state
    if type == "separable":
        # Generate basis vectors for the D-dimensional Hilbert space
        basis_vectors = [np.eye(D)[:, j] for j in range(D)]

        # Construct the state for the first subsystem
        state = sum(coefficients[0][j] * basis_vectors[j] for j in range(D))

        # Iterate over the remaining subsystems and compute the tensor product
        for i in range(1, N):
            # Weighted sum for the current subsystem
            sub_state = sum(coefficients[i][j] * basis_vectors[j] for j in range(D))
            # Tensor product with the state constructed so far
            state = np.kron(state, sub_state)

    # Case 2: General state
    elif type == "general":
        total_dim = D**N  # Total dimension of the composite Hilbert space

        # Generate all possible multi-indices for the composite system
        indices = list(itertools.product(range(D), repeat=N))

        # Initialize the composite state as a zero vector
        state = np.zeros(total_dim, dtype=complex)

        # Generate basis vectors for each subsystem
        basis_vectors = [np.eye(D)[:, j] for j in range(D)]

        # Construct the composite state by summing weighted tensor products
        for idx, multi_index in enumerate(indices):
            tensor_product = basis_vectors[multi_index[0]]
            for j in multi_index[1:]:
                tensor_product = np.kron(tensor_product, basis_vectors[j])
            # Add the weighted tensor product to the state
            state += coefficients[idx] * tensor_product

    else:
        # Handle invalid type with an error message
        raise ValueError(f"Invalid state type '{type}'. Must be either 'separable' or 'general'.")

    return state



def comput_time(N_max, D_max, seed, type):
    """
    Computes the execution time and memory usage for generating quantum states 
    across varying subsystem sizes (N) and local dimensions (D).

    Parameters
    ----------
    N_max : int
        Maximum number of subsystems (inclusive range: 1 to N_max).
    D_max : int
        Maximum local dimension of each subsystem (inclusive range: 1 to D_max).
    seed : int
        Seed for the random number generator to ensure reproducibility.
    type : str
        Type of quantum state. Must be either "separable" or "general".

    Returns
    -------
    tuple
        N_sizes : list[int]
            List of subsystem sizes (N) considered.
        D_sizes : list[int]
            List of local dimensions (D) considered.
        cpu_times_matrix : np.ndarray
            2D array where each element [i, j] contains the time (in seconds) 
            required to generate the quantum state for N_sizes[i] and D_sizes[j].
        bytes_matrix : np.ndarray
            2D array where each element [i, j] contains the memory usage 
            (in bytes) of the quantum state for N_sizes[i] and D_sizes[j].
    """
    # Define the range and step size for subsystem sizes (N) and local dimensions (D)
    N_min = 1
    N_step = 1
    D_min = 1
    D_step = 1

    # Generate lists of subsystem sizes and local dimensions
    N_sizes = list(range(N_min, N_max, N_step))
    D_sizes = list(range(D_min, D_max, D_step))

    # Initialize matrices to store execution times and memory usage
    cpu_times_matrix = np.zeros((len(N_sizes), len(D_sizes)))
    bytes_matrix = np.zeros((len(N_sizes), len(D_sizes)))

    # Loop over each combination of subsystem size (N) and local dimension (D)
    for idx_1, N_i in enumerate(N_sizes):
        for idx_2, D_i in enumerate(D_sizes):
            # Start timing the state generation process
            start_time = time.time()

            # Generate random coefficients for the state
            coefficients = initialize_coefficients(N_i, D_i, seed, type, random_init=True)

            # Create the full quantum state using the initialized coefficients
            state = create_state(N_i, D_i, coefficients, type)

            # End timing after the state has been created
            end_time = time.time()

            # Calculate the memory usage of the generated state in bytes
            N_bytes = state.nbytes

            # Calculate and store the elapsed time for state generation
            elapsed_time = end_time - start_time
            cpu_times_matrix[idx_1, idx_2] = elapsed_time

            # Store the memory usage in bytes
            bytes_matrix[idx_1, idx_2] = N_bytes

    # Return the results: subsystem sizes, local dimensions, and the collected data
    return N_sizes, D_sizes, cpu_times_matrix, bytes_matrix
