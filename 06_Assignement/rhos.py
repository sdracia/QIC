import numpy as np


def rho(state):
    return np.outer(state, np.conj(state))


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
    print(f"Reordered psi_tensor shape: {psi_tensor.shape}")
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