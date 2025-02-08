import numpy as np
import matplotlib.pyplot as plt

from molecule import Molecule

from molecule import CaOH, CaH, mu_N, gI




def plot_transitions(molecule: Molecule, text: bool = True):
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

    bar_width = 0.2  # Larghezza delle barre
    plt.figure(figsize=(12, 6))

    # Plottiamo le barre
    plt.bar(J - bar_width, signature_transitions, width=bar_width, color='red', label='Signature transition')
    plt.bar(J, penultimate_transitions, width=bar_width, color='blue', label='Penultimate transition')
    plt.bar(J + bar_width, sub_manifold_splitting, width=bar_width, color='green', label='Sub-manifold splitting')

    # Aggiunta di etichette e titolo
    plt.xlabel('J')
    plt.ylabel('Frequency (kHz)')
    plt.title('$B = 3 \, G$')
    plt.xticks(J)
    # plt.ylim(0, 100)  # Range delle frequenze
    plt.legend()

    if text:
        for i in range(len(J)):
            plt.text(J[i] - bar_width, signature_transitions[i] + 1, f'{signature_transitions[i]:.2f}', 
                     ha='center', va='bottom', color='red', rotation=90)
            plt.text(J[i], penultimate_transitions[i] + 1, f'{penultimate_transitions[i]:.2f}', 
                     ha='center', va='bottom', color='blue', rotation=90)
            plt.text(J[i] + bar_width, sub_manifold_splitting[i] + 1, f'{sub_manifold_splitting[i]:.2f}', 
                     ha='center', va='bottom', color='green', rotation=90)

    # Mostra il grafico
    plt.tight_layout()
    plt.show()
