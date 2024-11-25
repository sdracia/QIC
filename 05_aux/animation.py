import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os
from scipy.optimize import curve_fit


def plot_average_position(par, avg_position, n, filename, im_time):
  """
  plot_average_position :
    Plot the average position of the particle over time.

  Parameters
  ----------
  par : Param
    Parameters of the simulation.
  avg_position : np.ndarray
    Array of average position values.

  Returns
  -------
  None
  """
  output_dir = f"n={n}"

  os.makedirs(output_dir, exist_ok=True)
    
  full_path = os.path.join(output_dir, filename)

  def oscillatory_fit(t, m):

    return m * t


  # Compute time array (100 elements)
  time = np.linspace(0, par.tsim, len(avg_position))

  popt, _ = curve_fit(oscillatory_fit, time, avg_position, p0=[1.0])
  fitted_curve = oscillatory_fit(time, *popt)
  
  # Plot
  plt.figure(figsize=(8, 6))
  plt.plot(time, avg_position, color="blue", label="Average position $\langle x(t) \\rangle$")
  if (im_time == False):
    plt.plot(time, fitted_curve, label="Fit", color="red", linestyle="--")
  
  plt.xlabel("Time (t)", fontsize = 16)
  plt.ylabel("Position (x)", fontsize = 16)
  plt.title("Average position of the particle over time", fontsize = 16)
#   plt.xlim(4.01,4.03)
#   plt.ylim(0.297, 0.3025)
  plt.legend()

  plt.grid(True, linestyle="--", alpha=0.7)

  plt.savefig(full_path)

  # Mostra il grafico (opzionale, dopo aver salvato)
  plt.show()

  # Chiudi la figura per evitare sovrapposizioni
  plt.close()

  return popt, fitted_curve
  
# ===========================================================================================================

def gif_animation(par, density, potential, avg_position, n, filename='qho_time_evolution.gif'):
  """
  gif_animation:
    Creates an animated GIF showing the wave function, potential, and average position
    at each timestep.

  Parameters
  ----------
  par : Param
    Parameters of the simulation (contains spatial grid, etc.)
  density : numpy.ndarray
    Array of the wave function density at each timestep.
  potential : numpy.ndarray
    Array of the potential at each timestep.
  avg_position : numpy.ndarray
    Array of the average position of the particle at each timestep.
  filename : str, optional
    Name of the output GIF file. Default is 'qho_time_evolution.gif'.
  """


  
  # Set up the figure and axis for the animation
  output_dir = f"n={n}"
    
  # Crea la cartella se non esiste
  os.makedirs(output_dir, exist_ok=True)
    
  # Prepara il percorso completo del file
  full_path = os.path.join(output_dir, filename)


  fig, ax = plt.subplots()
  ax.set_xlim(par.x_min, par.x_max)
  ax.set_ylim(0, 1)
  ax.set_xlabel("Position (x)")
  ax.set_ylabel("Probability density |ψ(x)|^2")

  # Lines for the wave function, potential, and average position
  line_wfc, = ax.plot([], [], lw=1.5, label="|ψ(x)|^2", color='blue')
  line_pot, = ax.plot([], [], lw=1.5, label="V(x)", color='gray', linestyle='--')
  line_avg_pos, = ax.plot([], [], lw=1, label="Average position", color='red', linestyle='-.')

  ax.legend(loc="upper left")

  # Initialization function for the animation
  def init_line():
    """
    init_line:
      Initialization function for the animation.
    """
    line_wfc.set_data([], [])
    line_pot.set_data([], [])
    line_avg_pos.set_data([], [])
    return line_wfc, line_pot, line_avg_pos

  def animate(i):
    """
    animate:
      Animation function to update the plot at each frame.
    """
    # Extract density, potential and average position for frame i
    y_wfc = density[i, :par.num_x]
    y_pot = potential[i, :]
    avg_pos = avg_position[i]

    # Set data for the wave function and potential
    line_wfc.set_data(par.x, y_wfc)
    line_pot.set_data(par.x, y_pot)
    
    # Update the average position line
    line_avg_pos.set_data([avg_pos, avg_pos], [0, 1.5])  # Vertical line from y=0 to y=1.5

    return line_wfc, line_pot, line_avg_pos

  # Create the animation and save as a GIF
  anim = animation.FuncAnimation(fig, animate, init_func=init_line, frames=100, interval=10, blit=True)
  writer = animation.PillowWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
  anim.save(full_path, writer=writer)

  # Close the figure to prevent it from displaying in the notebook
  plt.close(fig)
    
  # Final print statement
  print(f"Animation saved as '{filename}'")


def analyze_spectrum(time, diff, filename):
    """
    Performs the analysis and generates a plot with two subplots:
    1. The original average position of the particle over time with a fitted curve.
    2. The frequency spectrum of the difference between the average position and the fitted curve.
    
    Parameters
    ----------
    time : np.ndarray
        The array of time values.
    avg_position : np.ndarray
        The average position of the particle at each time step.
    fitted_curve : np.ndarray
        The fitted curve to subtract from the average position to calculate the difference.
    
    Returns
    -------
    None
    """
        
    output_dir = f"Fourier"

    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, filename)

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(time, diff, label="average position", color="blue")
    axs[0].set_xlabel("Time (t)", fontsize=16)
    axs[0].set_ylabel("Position (x)", fontsize=16)
    axs[0].set_title("Average position of the particle with fit", fontsize=16)
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.7)
    
    # Fourier transform analysis for the second subplot
    # Remove DC component (mean) from the difference signal
    diff_no_dc = diff - np.mean(diff)
    
    # Compute the Fourier transform
    fft_values = np.fft.fft(diff_no_dc)
    
    # Calculate the corresponding frequencies
    freq = np.fft.fftfreq(len(time), time[1] - time[0])  # Constant time step
    
    # Calculate the magnitude of the Fourier transform
    fft_magnitude = np.abs(fft_values)
    
    # Plot the frequency spectrum in the second subplot
    axs[1].plot(freq[:len(freq)//2], fft_magnitude[:len(freq)//2], color="blue", label="Frequency spectrum")
    axs[1].set_xlabel("Frequency (Hz)", fontsize=14)
    axs[1].set_ylabel("Magnitude", fontsize=14)
    axs[1].set_title("Frequency spectrum of the difference from the fit", fontsize=16)
    # axs[1].set_xlim(0, 1)  # Limit the frequency axis for better visualization
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.7)
    
    # Show the plot with both subplots
    plt.tight_layout()
    
    plt.savefig(full_path)
    plt.show()

    plt.close()

    return freq, fft_magnitude
