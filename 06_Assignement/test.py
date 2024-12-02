import numpy as np

import rhos as rh
import functions as fu


def assert_equal(actual, expected, tol=1e-10):
    """
    Compares two matrices or vectors for equality within a specified tolerance.

    Parameters
    ----------
    actual : np.array
        The actual result obtained from a function.
    expected : np.array
        The expected result to compare against.
    tol : float, optional
        The numerical tolerance for comparison. Default is 1e-10.

    Raises
    ------
    AssertionError
        If the matrices or vectors differ by more than the specified tolerance.
    """
    if not np.allclose(actual, expected, atol=tol):
        raise AssertionError(f"Test failed!\nActual:\n{actual}\nExpected:\n{expected}")
    print("Test passed.")



def test_separable_state_simple():
    """
    Validates the construction of a simple separable state |ψ⟩ = |0⟩|0⟩.

    Checks include:
    - Correct wavefunction generation.
    - Accuracy of the full density matrix.
    - Correctness of reduced density matrices for subsystems.
    """
    print("### TEST: Separable State Simple ###")
    N = 2  # Number of subsystems
    D = 2  # Dimension of each subsystem
    seed = 12345
    type = "separable"
    
    init_coeff = [
        np.array([1, 0]),  # Subsystem 1 (|0>)
        np.array([1, 0])   # Subsystem 2 (|0>)
    ]

    # Generate wavefunction
    coefficients = fu.initialize_coefficients(N, D, seed, type, init_coeff, random_init=False)
    psi = fu.create_state(N, D, coefficients, type)
    
    # Expected result
    expected_psi = np.array([1, 0, 0, 0])  # |00> in the computational basis

    assert_equal(psi, expected_psi)

    # Generate density matrix
    rho = rh.rho(psi)
    expected_rho = rh.rho(expected_psi)
    
    assert_equal(rho, expected_rho)

    # Reduced density matrices
    rdm_left = rh.get_reduced_density_matrix(psi, D, N, [0])
    rdm_right = rh.get_reduced_density_matrix(psi, D, N, [1])
    expected_rdm = np.array([[1, 0], [0, 0]])  # For both left and right subsystems
    assert_equal(rdm_left, expected_rdm)
    assert_equal(rdm_right, expected_rdm)


def test_separable_state_combination():
    """
    Validates a separable state with a superposition: |ψ⟩ = |0⟩ ⊗ (|0⟩ + |1⟩)/sqrt(2).

    Checks include:
    - Proper handling of superposition in wavefunction generation.
    - Accuracy of the full density matrix.
    - Correctness of reduced density matrices for subsystems.
    """
    print("### TEST: Separable State Combination ###")
    N = 2
    D = 2
    seed = 12345
    type = "separable"

    init_coeff = [
        np.array([1, 0]),  # Subsystem 1 (|0>)
        np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Subsystem 2 (|0> + |1>) / sqrt(2)
    ]

    # Generate wavefunction
    coefficients = fu.initialize_coefficients(N, D, seed, type, init_coeff, random_init=False)
    psi = fu.create_state(N, D, coefficients, type)
    
    # Expected result
    expected_psi = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])  # |0>|(0+1)/sqrt(2)>
    assert_equal(psi, expected_psi)

    # Generate density matrix
    rho = rh.rho(psi)
    expected_rho = np.outer(expected_psi, np.conj(expected_psi))
    assert_equal(rho, expected_rho)

    # Reduced density matrices
    rdm_left = rh.get_reduced_density_matrix(psi, D, N, [0])
    rdm_right = rh.get_reduced_density_matrix(psi, D, N, [1])
    expected_rdm_left = np.array([[1, 0], [0, 0]])
    expected_rdm_right = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert_equal(rdm_left, expected_rdm_left)
    assert_equal(rdm_right, expected_rdm_right)


def test_general_state_bell():
    """
    Validates the entangled Bell state |ψ⟩ = (|00⟩ + |11⟩)/sqrt(2).

    Checks include:
    - Correct wavefunction generation for an entangled state.
    - Accuracy of the full density matrix.
    - Validation of reduced density matrices reflecting maximally mixed states.
    """
    print("### TEST: General State Bell ###")
    N = 2
    D = 2
    seed = 12345
    type = "general"

    init_coeff = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]  # Bell state (|00> + |11>) / sqrt(2)

    # Generate wavefunction
    coefficients = fu.initialize_coefficients(N, D, seed, type, init_coeff, random_init=False)
    psi = fu.create_state(N, D, coefficients, type)
    
    # Expected result
    expected_psi = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # |00> + |11>
    assert_equal(psi, expected_psi)

    # Generate density matrix
    rho = rh.rho(psi)
    expected_rho = np.outer(expected_psi, np.conj(expected_psi))
    assert_equal(rho, expected_rho)

    # Reduced density matrices
    rdm_left = rh.get_reduced_density_matrix(psi, D, N, [0])
    rdm_right = rh.get_reduced_density_matrix(psi, D, N, [1])
    expected_rdm = np.array([[0.5, 0], [0, 0.5]])  # Same for both left and right
    assert_equal(rdm_left, expected_rdm)
    assert_equal(rdm_right, expected_rdm)


def test_general_state_single_nonzero():
    """
    Validates a general state with a single non-zero component: |ψ⟩ = |11⟩.

    Checks include:
    - Proper wavefunction generation for the sparse state.
    - Accuracy of the full density matrix.
    - Correctness of reduced density matrices for subsystems.
    """
    print("### TEST: General State Single Non-Zero Component ###")
    N = 2
    D = 2
    init_coeff = [0, 0, 0, 1]  # Only |11> is non-zero
    type="general"
    seed = 12345

    # Generate wavefunction
    coefficients = fu.initialize_coefficients(N, D, seed, type, init_coeff, random_init=False)
    psi = fu.create_state(N, D, coefficients, type)
    
    # Expected result
    expected_psi = np.array([0, 0, 0, 1])  # |11>
    assert_equal(psi, expected_psi)

    # Generate density matrix
    rho = rh.rho(psi)
    expected_rho = np.outer(expected_psi, np.conj(expected_psi))
    assert_equal(rho, expected_rho)

    # Reduced density matrices
    rdm_left = rh.get_reduced_density_matrix(psi, D, N, [0])
    rdm_right = rh.get_reduced_density_matrix(psi, D, N, [1])
    expected_rdm = np.array([[0, 0], [0, 1]])  # For both left and right subsystems
    assert_equal(rdm_left, expected_rdm)
    assert_equal(rdm_right, expected_rdm)


def test_separable_state_equal_superposition():
    """
    Validates a separable state with equal superposition: |ψ⟩ = (|0⟩ + |1⟩) ⊗ (|0⟩ + |1⟩)/2.

    Checks include:
    - Proper wavefunction generation with balanced probabilities.
    - Accuracy of the full density matrix.
    - Correctness of reduced density matrices reflecting the equal superposition.
    """
    print("### TEST: Separable State Equal Superposition ###")
    N = 2
    D = 2
    init_coeff = [
        np.array([1/np.sqrt(2), 1/np.sqrt(2)]),  # Subsystem 1 (|0> + |1>) / sqrt(2)
        np.array([1/np.sqrt(2), 1/np.sqrt(2)])   # Subsystem 2 (|0> + |1>) / sqrt(2)
    ]
    seed = 12345
    type = "separable"

    # Generate wavefunction
    coefficients = fu.initialize_coefficients(N, D, seed, type, init_coeff, random_init=False)
    psi = fu.create_state(N, D, coefficients, type)
    
    # Expected result
    expected_psi = np.array([0.5, 0.5, 0.5, 0.5])  # |psi>
    assert_equal(psi, expected_psi)

    # Generate density matrix
    rho = rh.rho(psi)
    expected_rho = np.outer(expected_psi, np.conj(expected_psi))
    assert_equal(rho, expected_rho)

    # Reduced density matrices
    rdm_left = rh.get_reduced_density_matrix(psi, D, N, [0])
    rdm_right = rh.get_reduced_density_matrix(psi, D, N, [1])
    expected_rdm = np.array([[0.5, 0.5], [0.5, 0.5]])  # Same for both left and right
    assert_equal(rdm_left, expected_rdm)
    assert_equal(rdm_right, expected_rdm)
