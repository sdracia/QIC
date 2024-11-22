import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import functions as fu

def plot_git(params, res, pot_evol):

    # Set up the figure and axis for the animation
    step_interval = max(params.timesteps // 100, 1)  # Ensure at least 200 frames, adjust as needed
    selected_frames = range(0, params.timesteps, step_interval)
    print(selected_frames)

    fig, ax = plt.subplots()
    ax.set_xlim(-params.xmax, params.xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Probability Density |ψ(x)|^2")

    # Prepare the line for the wavefunction and potential
    line, = ax.plot([], [], lw=2, label="Wave Function |ψ(x)|^2", color='blue')
    potential_line, = ax.plot([], [], lw=1, label="Potential V(x)", color='gray', linestyle='--')
    ax.legend(loc="upper right")

    # Initialization function for the animation
    def init_line():
        line.set_data([], [])
        potential_line.set_data([], [])
        return line, potential_line

    # Animation function that updates the plot at each frame
    def animate(i):
        x = params.x
        y_wavefunction = res[0, i, :]  # Probability density for the wavefunction
        y_potential = pot_evol[0, i, :]  # Potential at timestep i

        line.set_data(x, y_wavefunction)
        potential_line.set_data(x, y_potential)
        return line, potential_line

    # Create the animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init_line, frames=selected_frames, interval=20, blit=True, repeat=True
    )

    # Save the animation as a GIF
    writer = animation.PillowWriter(fps=1000, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('real_space.gif', writer=writer)

    plt.show()