import analytical_solution as anso
import numpy as np
from numpy.linalg import norm

class Param:
    """
    Container for holding all simulation parameters

    Parameters
    ----------
    xmax : float
        The real space is between [-xmax, xmax]
    num_x : int
        Number of intervals in [-xmax, xmax]
    dt : float
        Time discretization
    timesteps : int
        Number of timestep
    im_time : bool, optional
        If True, use imaginary time evolution.
        Default to False.
    """
    def __init__(self,
                 xmax: float,
                 num_x: int,
                 dt: float,
                 timesteps: int,
                 im_time: bool = False) -> None:

        self.xmax = xmax
        self.num_x = num_x
        self.dt = dt
        self.timesteps = timesteps
        self.im_time = im_time

        self.T = dt * timesteps
        self.dx = 2 * xmax / num_x      # it's like 2L/N
        # Real time grid
        self.x = np.arange(-xmax + xmax / num_x, xmax, self.dx)
        # Definition of the momentum
        self.dk = np.pi / xmax          # k = pi / L
        # Momentum grid
        self.k = np.concatenate((np.arange(0, num_x / 2), np.arange(-num_x / 2, 0))) * self.dk


class Operators:
    """Container for holding operators and wavefunction coefficients."""
    def __init__(self, res: int) -> None:

        self.V = np.empty(res, dtype=complex)
        self.R = np.empty(res, dtype=complex)
        self.K = np.empty(res, dtype=complex)
        self.wfc = np.empty(res, dtype=complex)


def init_and_splitop(par, omega, init_wfc):

    opr = Operators(len(par.x))

    res = np.zeros((2, par.timesteps, par.num_x))
    potential_evolution = np.zeros((1, par.timesteps, par.num_x))

    # Initial Wavefunction initialization
    opr.wfc = init_wfc

    density = np.abs(opr.wfc) ** 2
    density_momentum = np.abs(np.fft.fft(opr.wfc))**2


    # Renormalizing for imaginary time
    if par.im_time:
        # Compute the renormalization factor
        renorm_factor = np.sum( density * par.dx )
        # Renormalize the wavefunction opr.wfc
        opr.wfc /= np.sqrt(renorm_factor)
    

    # Save to res
    res[0, 0, :] = np.real(density)  # Real space
    res[1, 0, :] = np.real(density_momentum)  # Momentum space


    opr.V = 0.5 * omega**2 * (par.x)**2
    potential_evolution[0, 0, :] = opr.V 


    coeff = 1 if par.im_time else 1j
    for i in range(1, par.timesteps):
        # Time-dependent q_0
        q_0 = (i * par.dt) / par.T
        opr.V = 0.5 * omega**2 * (par.x - q_0)**2

        potential_evolution[0, i, :] = opr.V        # I append the time evolution of the potential
        
        opr.R = np.exp(-0.5 * opr.V * par.dt * coeff)
        opr.K = np.exp(-0.5 * (par.k ** 2) * par.dt * coeff)

        # Half-step in real space
        opr.wfc *= opr.R

        # FFT to momentum space
        opr.wfc = np.fft.fft(opr.wfc)

        # Full step in momentum space
        opr.wfc *= opr.K

        # iFFT back
        opr.wfc = np.fft.ifft(opr.wfc)

        # Final half-step in real space
        opr.wfc *= opr.R

        # Density for plotting and potential
        density = np.abs(opr.wfc) ** 2
        density_momentum = np.abs(np.fft.fft(opr.wfc))**2

        # Renormalizing for imaginary time
        if par.im_time:
            # Compute the renormalization factor
            renorm_factor = np.sum( density * par.dx )
            # Renormalize the wavefunction opr.wfc
            opr.wfc /= np.sqrt(renorm_factor)

        # Save to res
        res[0, i, :] = np.real(density)  # Real space
        res[1, i, :] = np.real(density_momentum)  # Momentum space


    return res, opr, potential_evolution


def calculate_energy(par: Param, opr: Operators) -> float:
    """Calculate the energy <Psi|H|Psi>."""
    # Creating real, momentum, and conjugate wavefunctions.
    wfc_r = opr.wfc
    wfc_k = np.fft.fft(wfc_r)
    wfc_c = np.conj(wfc_r)

    # Finding the momentum and real-space energy terms
    energy_k = 0.5 * wfc_c * np.fft.ifft((par.k ** 2) * wfc_k)
    energy_r = wfc_c * opr.V * wfc_r

    # Integrating over all space
    energy_final = sum(energy_k + energy_r).real

    return energy_final * par.dx