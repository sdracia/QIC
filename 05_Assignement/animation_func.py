import matplotlib.pyplot as plt
from matplotlib import animation

import functions as fu

def plot_average_position(time, avg_position):
  """
  plot_average_position :
    Plot the average position of the particle over time.

  Parameters
  ----------
  time : np.ndarray
    Array of time values.
  avg_position : np.ndarray
    Array of average position values.

  Returns
  -------
  None
  """
  plt.figure(figsize=(8, 6))
  plt.plot(time, avg_position, label="Average position $\langle x(t) \\rangle$")
  plt.xlabel("Time (t)")
  plt.ylabel("Position (x)")
  plt.legend()
  plt.title("Average position of the particle over time")

  plt.grid(True)
  plt.show()


def gif_animation(par, density, potential, avg_position, filename='real_space_with_avg_position.gif'):
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
      Name of the output GIF file. Default is 'real_space_with_avg_position.gif'.
    """
    # Set up the figure and axis for the animation
    fig, ax = plt.subplots()
    ax.set_xlim(-par.xmax, par.xmax)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Probability Density |ψ(x)|^2")

    # Lines for the wave function, potential, and average position
    line_wfc, = ax.plot([], [], lw=2, label="Wave Function |ψ(x)|^2", color='blue')
    line_pot, = ax.plot([], [], lw=2, label="Potential V(x)", color='gray', linestyle='--')
    line_avg_pos, = ax.plot([], [], lw=2, label="Average Position", color='red', linestyle='-.')

    ax.legend()

    # Initialization function for the animation
    def init_line():
        line_wfc.set_data([], [])
        line_pot.set_data([], [])
        line_avg_pos.set_data([], [])
        return line_wfc, line_pot, line_avg_pos

    # Animation function to update the plot at each frame
    def animate(i):
        x = par.x
        y_wfc = density[i, :par.num_x]  # Extract density for frame i
        y_pot = potential[i, :]  # Extract potential from the stored potential array
        avg_pos = avg_position[i]  # Extract the average position for the current timestep

        # Set data for the wave function and potential
        line_wfc.set_data(x, y_wfc)
        line_pot.set_data(x, y_pot)

        # Update the average position line
        line_avg_pos.set_data([avg_pos, avg_pos], [0, 1.5])  # Vertical line from y=0 to y=1.5

        return line_wfc, line_pot, line_avg_pos

    # Create the animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init_line, frames=100, interval=20, blit=True
    )

    # Save the animation as a GIF
    writer = animation.PillowWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(filename, writer=writer)
    
    # Close the figure to prevent it from displaying in the notebook
    plt.close(fig)
    
    # Final print statement
    print(f"Animation saved as '{filename}'")