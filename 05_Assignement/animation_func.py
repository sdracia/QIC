import matplotlib.pyplot as plt
from matplotlib import animation
import os

import functions as fu

def plot_average_position(time, avg_position, n, filename):
  """
  plot_average_position :f
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

  output_dir = f"n={n}"
    
   # Crea la cartella se non esiste
  os.makedirs(output_dir, exist_ok=True)
    
  # Prepara il percorso completo del file
  full_path = os.path.join(output_dir, filename)

  plt.figure(figsize=(8, 6))
  plt.plot(time, avg_position, label="Average position $\langle x(t) \\rangle$")
  plt.xlabel("Time (t)", fontsize = 16)
  plt.ylabel("Position (x)", fontsize = 16)
  plt.legend()
  plt.title("Average position of the particle over time", fontsize = 16)

  plt.grid(True, linestyle="--", alpha=0.7)

  plt.savefig(full_path)

  # Mostra il grafico (opzionale, dopo aver salvato)
  plt.show()

  # Chiudi la figura per evitare sovrapposizioni
  plt.close()


def gif_animation(n, par, density, potential, avg_position, filename='real_space_with_avg_position.gif'):
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


    output_dir = f"n={n}"
    
    # Crea la cartella se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepara il percorso completo del file
    full_path = os.path.join(output_dir, filename)


    # Set up the figure and axis for the animation
    fig, ax = plt.subplots()
    ax.set_xlim(-par.xmax, par.xmax)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Position (x)", fontsize = 16)
    ax.set_ylabel("Probability Density |ψ(x)|^2", fontsize = 16)

    # Lines for the wave function, potential, and average position
    line_wfc, = ax.plot([], [], lw=2, label="Wave Function |ψ(x)|^2", color='blue')
    line_pot, = ax.plot([], [], lw=2, label="Potential V(x)", color='gray', linestyle='--')
    line_avg_pos, = ax.plot([], [], lw=2, label="Average Position", color='red', linestyle='-.')

    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)

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
    anim.save(full_path, writer=writer)
    
    # Close the figure to prevent it from displaying in the notebook
    plt.close(fig)
    
    # Final print statement
    print(f"Animation saved as '{filename}'")



def plot_time_evolution(n, T, par, density, potential, avg_position, timesteps, filename='time_evolution.png'):
    """
    plot_time_evolution_with_gradient:
      Creates a single plot showing the time evolution of the wave function, potential, 
      and average position, with a color gradient representing time progression.

    Parameters
    ----------
    par : Param
      Parameters of the simulation (contains spatial grid, etc.).
    density : numpy.ndarray
      Array of the wave function density at each timestep.
    potential : numpy.ndarray
      Array of the potential at each timestep.
    avg_position : numpy.ndarray
      Array of the average position of the particle at each timestep.
    timesteps : list of int
      List of timesteps to include in the plot.
    filename : str, optional
      Name of the output image file. Default is 'time_evolution_gradient.png'.
    """

    output_dir = f"n={n}"
    
    # Crea la cartella se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepara il percorso completo del file
    full_path = os.path.join(output_dir, filename)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-par.xmax, par.xmax)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Position (x)", fontsize=16)
    ax.set_ylabel("Probability Density |ψ(x)|^2", fontsize=16)
    ax.set_title("Time Evolution of Wave Function with Color Gradient", fontsize=18)
    
    # Define a colormap and normalize it across timesteps
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(timesteps), vmax=max(timesteps))

    # Plot data for each timestep with gradient colors
    for t in timesteps:
        x = par.x
        y_wfc = density[t, :par.num_x]  # Extract density for timestep t
        avg_pos = avg_position[t]  # Extract average position for timestep t
        
        # Determine color based on timestep
        color = cmap(norm(t))
        
        # Plot wave function with color gradient
        ax.plot(x, y_wfc, color=color, lw=1.5, label=f"t={t}" if t == timesteps[0] else None)
        
        # Plot average position as a vertical dashed line
        ax.axvline(avg_pos, linestyle='--', color=color, alpha=0.7)

    # Plot the potential (assuming it's time-independent or representative)
    ax.plot(x, potential[0, :], label="Potential V(x)", color='black', linestyle='-', lw=2, alpha=0.7)

    # Add a colorbar to indicate time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label("Time Progression", fontsize=14)

    # Add legend and grid
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Save the figure to a file
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close(fig)
    
    # Final print statement
    print(f"Time evolution plot with gradient saved as '{filename}'")
