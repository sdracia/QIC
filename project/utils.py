import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from molecule import Molecule

from molecule import CaOH, CaH, mu_N, gI


def plot_transitions(molecule: Molecule, text: bool = True):
    
    # I analyze the following transitions at different Js, and in different regimes of B:
    # - Signature transition: it's the target transition in the molecule
    # - Penultimate transition: it's the transition at m+1 to the right of the signature transition
    # - Sub-manifold splitting: it's the splitting of the manifold between the states at $\xi = -$ and $\xi = +$ 
    

    signature_transitions = np.abs(np.array([molecule.transition_df.loc[molecule.transition_df["j"]==j].iloc[0]["energy_diff"] for j in range(1,molecule.j_max+1)]))
    penultimate_transitions = np.abs(np.array([molecule.transition_df.loc[molecule.transition_df["j"]==j].iloc[1]["energy_diff"] for j in range(1,molecule.j_max+1)]))
    sub_manifold_splitting = []

    for i,j in enumerate(range(1, molecule.j_max + 1)):

        gj = molecule.gj_list[i] if molecule.gj_list != [] else molecule.gj
        cij = molecule.cij_list[i] if molecule.gj_list != [] else molecule.cij_khz

        x = 1 / 2 * np.sqrt(cij**2 * ((j + 1 / 2) ** 2) + (- molecule.cb_khz * (gj - gI)) ** 2)

        sub_manifold_splitting.append(2*x)

    sub_manifold_splitting = np.array(sub_manifold_splitting)

    J = np.arange(1, molecule.j_max + 1)

    bar_width = 0.2 

    plt.figure(figsize=(12, 6))
    plt.bar(J - bar_width, signature_transitions, width=bar_width, color='red', label='Signature transition')
    plt.bar(J, penultimate_transitions, width=bar_width, color='blue', label='Penultimate transition')
    plt.bar(J + bar_width, sub_manifold_splitting, width=bar_width, color='green', label='Sub-manifold splitting')
    plt.xlabel('J')
    plt.ylabel('Frequency (kHz)')
    plt.title(f'$B = {molecule.b_field_gauss}$')
    plt.xticks(J)
    plt.legend()

    # text written up to the bars
    if text:
        for i in range(len(J)):
            plt.text(J[i] - bar_width, signature_transitions[i] + 1, f'{signature_transitions[i]:.2f}', 
                     ha='center', va='bottom', color='red', rotation=90)
            plt.text(J[i], penultimate_transitions[i] + 1, f'{penultimate_transitions[i]:.2f}', 
                     ha='center', va='bottom', color='blue', rotation=90)
            plt.text(J[i] + bar_width, sub_manifold_splitting[i] + 1, f'{sub_manifold_splitting[i]:.2f}', 
                     ha='center', va='bottom', color='green', rotation=90)

    plt.tight_layout()
    plt.show()



def plot_state_dist(molecule, j):

    states_in_j = molecule.state_df.loc[molecule.state_df["j"] == j]
    transitions_in_j = molecule.transition_df[molecule.transition_df["j"] == j]
    m = states_in_j["m"].to_numpy()
    energies = states_in_j["zeeman_energy_khz"].to_numpy()

    state_dist = states_in_j["state_dist"].to_numpy()
    # print(state_dist)
    state_dist = state_dist / np.sum(state_dist)
    # print(state_dist)



    # spin_up = states_in_j["spin_up"].to_numpy()
    # spin_down = states_in_j["spin_down"].to_numpy()
    colors = state_dist

    fig, ax = plt.subplots(figsize=(12, 8))
    for mi, ei, ci in zip(m, energies, colors):
        ax.hlines(ei, mi - 0.3, mi + 0.3, colors=plt.cm.plasma(ci), linewidth=3)

    ax.set_xlabel("m")
    ax.set_ylabel("Zeeman energy (kHz)")
    ax.set_title(f"Zeeman energies of all states in j={j}, B={molecule.b_field_gauss} G")

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap="plasma"), ax=ax)
    cbar.set_label("spin")

    ax.set_xlim(-j-1, j+1)
    ax.set_xticks([i+0.5 for i in range(-j - 1, j + 1)])


    # plot the difference between neibouring states on arrows conecting them
    for transition in transitions_in_j.itertuples():
        m1 = transition.m1
        xi1 = transition.xi1
        energy1 = molecule.state_df.loc[(molecule.state_df["j"] == j) & (molecule.state_df["m"] == m1) & (molecule.state_df["xi"] == xi1)].iloc[0].zeeman_energy_khz
        m2 = transition.m2
        xi2 = transition.xi2
        energy2 = molecule.state_df.loc[(molecule.state_df["j"] == j) & (molecule.state_df["m"] == m2) & (molecule.state_df["xi"] == xi2)].iloc[0].zeeman_energy_khz
        energy_diff = transition.energy_diff
        coupling = transition.coupling
        ax.annotate(
            "", 
            xy=(float(m1)-1.0, float(energy1) + float(energy_diff)), 
            xytext=(float(m1), float(energy1)),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1)
        )
        # add the energy difference as text on the arrow
        ax.text((3*m1 + m2) / 4.0 -0.5, (3*energy1 + energy2) / 4.0, f"{energy_diff:.2f} kHz", fontsize=8, color="gray")
        # add the coupling strength as text on the arrow
        ax.text((3*m1 + m2) / 4.0 -0.5, (3*energy1 + energy2) / 4.0 - 0.9, f"{coupling:.2f}", fontsize=8, color="red")

    plt.show()
    plt.close()



import matplotlib as mpl
from numpy import pi
plt.rcParams['font.family'] = 'DejaVu Sans'

def heatmap_state_pop(dataframe_molecule, j_max, normalize = True):

        
    vh_cmap="hsv"
    cmap_shift=0
    
    dataframe = dataframe_molecule[["j", "m", "xi", "state_dist"]].copy()

    ## Renormalize
    if normalize:
        for j in range(j_max + 1):
            states_in_j = dataframe.loc[dataframe["j"] == j]

            state_dist = states_in_j["state_dist"].to_numpy()
            state_dist = state_dist / np.sum(state_dist)

            dataframe.loc[dataframe["j"] == j, "state_dist"] = state_dist




    df_grouped = dataframe.groupby(['j', 'm']).agg({'xi': 'first', 'state_dist': list}).reset_index()
    
    for index, row in df_grouped.iterrows():
        if len(row['state_dist']) == 1:  # Se la lista contiene un solo valore
            if row['xi'] == False:
                df_grouped.at[index, 'state_dist'] = [row['state_dist'][0], np.nan]  # Aggiungi NaN come secondo elemento
            elif row['xi'] == True:
                df_grouped.at[index, 'state_dist'] = [np.nan, row['state_dist'][0]]  # Aggiungi NaN come primo elemento
    
    
    df = df_grouped[["j", "m"]]
    state = df_grouped["state_dist"].tolist()
    
    
    sq_array = np.zeros((2 * (j_max+1), j_max + 1), dtype=object)    # *2
    sq_array[:] = np.nan
    
    list_index = []
    
    for index, row in df.iterrows():
        j = row['j']
        m = row['m']
        list_index.append([int(j_max + m + 0.5), int(j)])
    
    
    for data_idx, idx_tuple in enumerate(list_index):
        # print(idx_tuple[0], idx_tuple[1])
        # print(state[data_idx])
        sq_array[idx_tuple[0], idx_tuple[1]] = state[data_idx]
    
    
    for i in range(sq_array.shape[0]):
        for j in range(sq_array.shape[1]):
            if isinstance(sq_array[i, j], list):
                sq_array[i, j] = np.array(sq_array[i, j])  # Converte la lista in un array numpy
                sq_array[i, j] = np.where(np.abs(sq_array[i, j]) < 1e-10, 0, sq_array[i, j])  # Applica l'azzeramento
    
    matrix = sq_array
    
    
    cmap = mpl.colormaps.get_cmap(vh_cmap)
    norm = mpl.colors.Normalize(
        vmin=-pi + np.finfo(float).eps + cmap_shift * 2 * pi,
        vmax=pi + cmap_shift * 2 * pi,
    )
    
    plt.figure(figsize=(16, 6)) 
    # ax_facecolor = '#D3D3D3'
    # # ax = ax if ax is not None else plt.gca()
    ax = plt.gca()
    # ax.patch.set_facecolor(ax_facecolor)
    # ax.set_aspect("equal", "box")
    
    vh_amp = False
    max_weight = 1
    ax_facecolor='#D3D3D3'
    ax_bkgdcolor="white"
    
    
    for (x, y), w in np.ndenumerate(matrix):
        # print(x,y,w)
        var = isinstance(w, float)
        # print(var)
    
        #single values: alwasy nan
        if var:
            # print("Ok")
            if not np.isnan(w):
                face_color = cmap(norm(np.angle(w)))
                edge_color = None
    
                if not vh_amp:
                    # if vh_amp is False, then hinton plot has rectangles area ~ norm squared of amplitude
                    w_plot = abs(w) ** 2
                else:
                    # else hinton plot has rectangles ~ norm of amplitude
                    w_plot = abs(w)
                size = np.sqrt(w_plot / max_weight)
            else:
                size = 1.0
                face_color = ax_bkgdcolor
                edge_color = ax_bkgdcolor
    
            # print("one", x - size / 2 - j_max, "and two", y - size / 2)
            rect = plt.Rectangle(
                [x-0.5 - size / 2 - j_max, y - size / 2],
                size,
                size,
                facecolor=face_color,
                edgecolor=edge_color,
            )
            ax.add_patch(rect)
        else:
            w_false = w[0]
            w_true = w[1]
    
            if not np.isnan(w_false):
                face_color = "blue"
                edge_color = "blue"
    
                if not vh_amp:
                    # if vh_amp is False, then hinton plot has rectangles area ~ norm squared of amplitude
                    w_plot = abs(w_false) ** 2
                else:
                    # else hinton plot has rectangles ~ norm of amplitude
                    w_plot = abs(w_false)
                size = np.sqrt(w_plot / max_weight)
            else:
                size = 1.0
                face_color = ax_bkgdcolor
                edge_color = ax_bkgdcolor
    
            size_x = size
            size_y = size*0.5
            rect = plt.Rectangle(
                [x-0.5 - size_x / 2 - j_max, y + 0.25 - size_y / 2],
                size_x,
                size_y,
                facecolor=face_color,
                edgecolor=edge_color,
            )
            ax.add_patch(rect)
    
            if not np.isnan(w_true):
                face_color = "blue"
                edge_color = "blue"
    
                if not vh_amp:
                    # if vh_amp is False, then hinton plot has rectangles area ~ norm squared of amplitude
                    w_plot = abs(w_true) ** 2
                else:
                    # else hinton plot has rectangles ~ norm of amplitude
                    w_plot = abs(w_true)
                size = np.sqrt(w_plot / max_weight)
            else:
                size = 1.0
                face_color = ax_bkgdcolor
                edge_color = ax_bkgdcolor
    
            size_x = size
            size_y = size*0.5
            rect = plt.Rectangle(
                [x-0.5 - size_x / 2 - j_max, y - 0.25 - size_y / 2],
                size_x,
                size_y,
                facecolor=face_color,
                edgecolor=edge_color,
            )
            ax.add_patch(rect)
            
    ax_facecolor = '#D3D3D3'
    # ax = ax if ax is not None else plt.gca()
    # ax = plt.gca()
    ax.patch.set_facecolor(ax_facecolor)
    ax.set_aspect("equal", "box")
    
    label_color="k"
    grid_color="w"
    grid_bool = True
    ax_labels_bool=True
    ax_color='k'
    
    ax.set_ylim([-0.5, matrix.shape[1] - 0.5])
    ax.set_xlim([-j_max-1, j_max+1])
    ax.set_yticks(np.arange(0, j_max + 1))
    ax.set_yticks(np.arange(0, j_max + 2) - 0.5, minor=True)
    # ax.set_xticks(np.arange(-j_max, j_max + 1))
    
    ax.grid(grid_bool, which="minor", color=grid_color)
    ax.tick_params(which="minor", bottom=False, left=False)
    if ax_labels_bool:
        ax.set_xlabel("$m$")
        ax.set_ylabel("$J$")
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    ax.tick_params(axis="x", colors=ax_color)
    ax.tick_params(axis="y", colors=ax_color)
    ax.spines["left"].set_color(ax_color)
    ax.spines["bottom"].set_color(ax_color)
    ax.spines["right"].set_color(ax_color)
    ax.spines["top"].set_color(ax_color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-j_max - 0.5, j_max + 1.5, 1))

    return matrix