import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from typing import TypeAlias
from pathlib import Path
from wigners import clebsch_gordan
from scipy.constants import h, physical_constants
# import qls

matplotlib.use("TkAgg")

# physical constants
mu_N = physical_constants["nuclear magneton"][0]
gI = physical_constants["proton g factor"][0]


class CaH:
    name: str = "CaH"
    """name of the molecule"""
    
    gj: float = -1.36
    """g factor for J"""
    
    cij_khz: float = 8.52   # kHz
    """coupling strength between proton spin and molecule rotation, in kHz"""

    br_ghz: float = 142.5017779
    """rotational constant, in GHz"""
    
    omega_0_thz: float = 750.0      # THz
    """frequency of the vibrational transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 285.5        # THz
    """frequency of the Raman beam, in THz"""

    coupling_coefficient: float = 1.
    """coupling coefficient"""

    # CONSTRUCTOR
    def __init__(self, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None) -> None:
        self.b_field_gauss = b_field_gauss
        """magnetic field in Gauss"""

        self.j_max = j_max
        """maximum j value to consider"""

        self.gj_list = gj_list if gj_list is not None else []
        self.cij_list = cij_list if cij_list is not None else []
        """If the lists exist, they are taken as gj and cij for different j's, otherwise the single value is considered"""

        if self.gj_list and self.cij_list:
            if len(self.gj_list) != (j_max +1) or len(self.cij_list) != (j_max +1):
                raise ValueError("Wrong input dimensions for j_max, cij_list, gj_list")

        self.cb_khz = mu_N * b_field_gauss * 1e-4 / h / 1e3
        """zeeman coefficient mu_N * B / h"""

        self.state_df: pd.DataFrame = pd.DataFrame()
        self.state_df_columns = ["j", "m", "xi", "spin_up", "spin_down", "zeeman_energy_khz", "rotation_energy_ghz"]
        """
        j: int, the j value of the state
        m: int, the m value of the state
        xi: Xi, the xi value of the state, boolean, False for Xi.minus and True for Xi.plus
        spin_up: float, the component for state with nuclear spin aligned to rotation
        spin_down: float, the component for state with nuclear spin anti-aligned to rotation
        zeeman_energy_khz: float, the Zeeman energy in kHz
        rotation_energy_ghz: float, the rotational energy in GHz
        """

        self.transition_df: pd.DataFrame = pd.DataFrame()
        self.transition_df_columns = ["j", "m1", "xi1", "m2", "xi2", "index1", "index2", "energy_diff", "coupling"]
        """
        j: int, the j value of the state
        m1: int, the m value of the initial state
        xi1: Xi, the xi value of the initial state, boolean, False for Xi.minus and True for Xi.plus
        m2: int, the m value of the final state
        xi2: Xi, the xi value of the final state, boolean, False for Xi.minus and True for Xi.plus
        energy_diff: float, the energy difference between the final and initial state in kHz
        coupling: float, the coupling strength between the initial and final state
        """

    @classmethod
    def read_molecule_data(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
        """
        Load the molecule data from the file. If the file does not exist, returns an error.
        """
        new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
        states_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_states.csv")
        transitions_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_transitions.csv")


        if states_file.exists() and transitions_file.exists():
            new_instance.state_df = pd.read_csv(states_file)
            new_instance.transition_df = pd.read_csv(transitions_file)
        else:
            print("The molecule data do not exist.'\n' Create the new molecule with class method create_molecule_data")
        return new_instance

    @classmethod
    def create_molecule_data(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
        """
        Calculate the molecule data given the input parameters.
        """
        new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
        new_instance.init_states()
        new_instance.init_transition_dataframe()
        new_instance.save_data()
        return new_instance

    @classmethod
    def m_csi_minus(cls, j: int) -> np.array:
        """
        Computes the values of m, for a given J, with csi = -

        j = 4
        m_csi_minus(j) --> [-4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5]
        """
        return np.arange(-j - 0.5, j, 1)

    @classmethod
    def m_csi_plus(cls, j: int) -> np.array:
        """
        Computes the values of m, for a given J, with csi = +

        j = 4
        m_csi_minus(j) --> [-3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5]
        """
        return np.arange(-j + 0.5, j + 1, 1)

    def init_states(self):
        """
        Initialize the states in a given j value.
        Calculate the Zeeman energy and the state vector for each state.
        The results are stored in the state_df as a Pandas dataframe.
        States are listed in the order of J values, starting from the lowest J to J_max.
        For each J, first Xi.minus and then Xi.plus states.
        The states of each Xi manifold are listed in the order of m values, starting from the lowest m to the highest m.
        """
        # In order to minimize the overheads and improve the efficiency for the creation of the dataframe i compute it here

        state_list = []

        for i,j in enumerate(range(self.j_max + 1)):

            gj = self.gj_list[i] if self.gj_list != [] else self.gj
            cij = self.cij_list[i] if self.gj_list != [] else self.cij_khz
            """If the lists exist, they are taken as gj and cij for different j's, otherwise the single value is considered"""

            rotation_energy_ghz = self.br_ghz * j * (j + 1)
            
            zeeman_edge_minus = (gj * j + gI / 2) * self.cb_khz - cij * j / 2
            zeeman_edge_plus = -(gj * j + gI / 2) * self.cb_khz - cij * j / 2
            """j-independent energies for the edge states"""

            xi = False  # calculate xi = - states
            for m in self.m_csi_minus(j):
                x = 1 / 2 * sqrt(cij**2 * ((j + 1 / 2) ** 2 - m**2) + (cij * m - self.cb_khz * (gj - gI)) ** 2)
                y = -self.cb_khz * (gj - gI) / 2  + m * cij / 2
                if m == -j - 0.5:   # Extreme state. Edge left state
                    spin_up = 0.0
                    spin_down = 1.0
                    zeeman_energy_khz = zeeman_edge_minus

                    state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

                else:               # Other non extreme states
                    spin_up = sqrt((x - y) / (2 * x))
                    spin_down = -sqrt((x + y) / (2 * x))
                    zeeman_energy_khz = cij / 4 - self.cb_khz * gj * m + x

                    state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

            xi = True  # calculate xi = + states
            for m in self.m_csi_plus(j):
                x = 1 / 2 * sqrt(cij**2 * ((j + 1 / 2) ** 2 - m**2) + (cij * m - self.cb_khz * (gj - gI)) ** 2)
                y = -self.cb_khz / 2 * (gj - gI) + m * cij / 2

                if m == j + 0.5:        # Extreme state. Right edge state
                    spin_up = 1.0
                    spin_down = 0.0
                    zeeman_energy_khz = zeeman_edge_plus

                    state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

                else:                   # Non-extreme states
                    spin_up = sqrt((x + y) / (2 * x))
                    spin_down = sqrt((x - y) / (2 * x))
                    zeeman_energy_khz = cij / 4 - self.cb_khz * gj * m - x

                    state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

        """state dataframe"""
        self.state_df = pd.DataFrame(state_list, columns=self.state_df_columns)

    def init_transition_dataframe(self):
        """
        Initialize the transition dataframe.
        """
        transition_list = []

        for j in range(self.j_max + 1):
            """I take the molecule states at given j, and calculate molecule transitions for those, in a proper order"""
            states_in_j = self.state_df.loc[self.state_df["j"] == j]
            states_index = states_in_j.index.to_numpy()
            states_array = states_in_j.to_numpy()
            m_len = 2 * j + 1
            

            # index1 <--> index2. They are just the two states involved in the transition, not necessarily the initial and final.
            # initial and final states are defined by is_minus variable in qls.py
            for index, state1 in enumerate(states_array):
                "1 state"
                index1 = states_index[index]
                m1, xi1, zeeman_energy_khz1 = state1[1], state1[2], state1[5]

                if index == 0:  
                    """no transition from the Xi.minus left edge state"""
                    continue

                if index == 1 or index == m_len:
                    """the 2 states right next to the Xi.minus left edge state. They only have 1 transition with the left edge state"""
                    index2 = states_index[0]
                    state2 = states_array[0]
                    m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]

                    energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                    coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                    transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
                    continue
                
                """For all the other states I have two transitions: one on the same value of csi, one on the opposite value of csi"""
                """Same value of csi"""
                index2 = states_index[index - 1]
                state2 = states_array[index - 1]
                m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

                """Opposite value of csi (depending on which csi has the initial state)"""
                if xi1:  # Xi.plus
                    index2 = states_index[index - m_len]
                    state2 = states_array[index - m_len]
                    m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                    energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                    coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                    transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
                else:  # Xi.minus
                    index2 = states_index[index + m_len - 2]
                    state2 = states_array[index + m_len - 2]
                    m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                    energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                    coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                    transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

        """Transition Dataframe is built following this specific order of calculations"""
        self.transition_df = pd.DataFrame(transition_list, columns=self.transition_df_columns)

    def get_raman_coupling(self, index1, index2, qa, qb):
        """
        Calculates the coupling between two Zeeman levels, |m1,xi1> & |m2,xi2>
        with two Raman beam with polarization qa and qb.
        Detuning and the electronic transition term is not considered.
        """


        def j_coupling(j, j_exc, mj1, mj2, qa, qb) -> float:
            """
            Function which computes the coupling coefficients based on the selection rules on j and m.
            Input parameters: 
            - j: the value of j of the initial and final state of the transition (both states belong to the same manifold)
            - j_exc: excited value of j due to the Raman beam. It can be j+1 or j-1
            - mj1: projection of rotational angular momentum for initial level
            - mj2: projection of rotational angular momentum for final level
            - qa: polarization of the first beam
            - qb: polarization of the second beam
            """
            if j < 0 or j_exc < 0:
                return 0
            if j < abs(mj1) or j < abs(mj2) or j_exc < abs(mj1 + qa):
                return 0
            return (
                np.sqrt((2 * j_exc + 1) / (2 * j + 1))      # """First transition: initial --> excited """
                * clebsch_gordan(1, 0, j_exc, 0, j, 0)
                * clebsch_gordan(1, -qa, j_exc, mj1 + qa, j, mj1)   

                * np.sqrt((2 * j + 1) / (2 * j_exc + 1))    # """Second transition: excited --> final"""  
                * clebsch_gordan(1, 0, j, 0, j_exc, 0)
                * clebsch_gordan(1, -qb, j, mj2, j_exc, mj1 + qa)
            )


        state1 = self.state_df.loc[index1]
        state2 = self.state_df.loc[index2]
        j = state1.j                  
        m1 = state1.m               
        m2 = state2.m
        m1_up = int(m1 - 0.5)       # è mj = m - 1/2, quindi è l'm per lo stato spin_up, ossia quello con mI = 1/2
        m1_down = int(m1 + 0.5)     # è mj = m + 1/2, quindi è l'm per lo stato spin_down, ossia quello con mI = -1/2
        m2_up = int(m2 - 0.5)       # stessa cosa per il secondo stato 
        m2_down = int(m2 + 0.5)

        # counter-rotating terms S-
        """
        Counter-rotating terms S- : I consider all combinations of spin_up and spin_down between first and second states.
        Then it is weighted according to the formula in Chou et al. 
        Both for J+1 and J-1
        Combinations of spin_up and spin_down are neglected, since I is conserved in the transition
        """
        coupling_minus = (
            1.0
            / (self.omega_thz - self.omega_0_thz)
            * (
                state1.spin_down * state2.spin_down * j_coupling(j, j + 1, m1_down, m2_down, qa, qb)
                # + state1.spin_down * state2.spin_up * j_coupling(j, j + 1, m1_down, m2_up, qa, qb)
                # + state1.spin_up * state2.spin_down * j_coupling(j, j + 1, m1_up, m2_down, qa, qb)
                + state1.spin_up * state2.spin_up * j_coupling(j, j + 1, m1_up, m2_up, qa, qb)

                + state1.spin_down * state2.spin_down * j_coupling(j, j - 1, m1_down, m2_down, qa, qb)
                # + state1.spin_down * state2.spin_up * j_coupling(j, j - 1, m1_down, m2_up, qa, qb)
                # + state1.spin_up * state2.spin_down * j_coupling(j, j - 1, m1_up, m2_down, qa, qb)
                + state1.spin_up * state2.spin_up * j_coupling(j, j - 1, m1_up, m2_up, qa, qb)
            )
        )

        """
        co-rotating terms S+ : I consider all combinations of spin_up and spin_down between first and second states.
        The path is inversed: initial polarization is qa, second polarization is qb
        Then it is weighted according to the formula in Chou et al. 
        Both for J+1 and J-1
        Combinations of spin_up and spin_down are neglected, since I is conserved in the transition
        """
        coupling_plus = (
            1.0
            / (self.omega_0_thz + self.omega_thz)
            * (
                state1.spin_down * state2.spin_down * j_coupling(j, j + 1, m1_down, m2_down, qb, qa)
                # + state1.spin_down * state2.spin_up * j_coupling(j, j + 1, m1_down, m2_up, qb, qa)
                # + state1.spin_up * state2.spin_down * j_coupling(j, j + 1, m1_up, m2_down, qb, qa)
                + state1.spin_up * state2.spin_up * j_coupling(j, j + 1, m1_up, m2_up, qb, qa)

                + state1.spin_down * state2.spin_down * j_coupling(j, j - 1, m1_down, m2_down, qb, qa)
                # + state1.spin_down * state2.spin_up * j_coupling(j, j - 1, m1_down, m2_up, qb, qa)
                # + state1.spin_up * state2.spin_down * j_coupling(j, j - 1, m1_up, m2_down, qb, qa)
                + state1.spin_up * state2.spin_up * j_coupling(j, j - 1, m1_up, m2_up, qb, qa)
            )
        )


        """The result is a weighted mean of the co-rotating and counter-rotating terms"""
        return (coupling_minus + coupling_plus) / (1.0 / (self.omega_thz - self.omega_0_thz) + 1.0 / (self.omega_0_thz + self.omega_thz))

    def plot_zeeman_levels(self, j: int):
        """
        Plot the Zeeman energies of all states in a given j value.
        """
        states_in_j = self.state_df.loc[self.state_df["j"] == j]
        transitions_in_j = self.transition_df[self.transition_df["j"] == j]
        m = states_in_j["m"].to_numpy()
        energies = states_in_j["zeeman_energy_khz"].to_numpy()
        spin_up = states_in_j["spin_up"].to_numpy()
        spin_down = states_in_j["spin_down"].to_numpy()
        colors = spin_up**2 - spin_down**2

        fig, ax = plt.subplots(figsize=(12, 8))
        for mi, ei, ci in zip(m, energies, colors):
            ax.hlines(ei, mi - 0.3, mi + 0.3, colors=plt.cm.plasma(ci), linewidth=3)

        ax.set_xlabel("m")
        ax.set_ylabel("Zeeman energy (kHz)")
        ax.set_title(f"Zeeman energies of all states in j={j}, B={self.b_field_gauss} G")

        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="plasma"), ax=ax)
        cbar.set_label("spin")

        ax.set_xlim(-j-1, j+1)
        ax.set_xticks([i+0.5 for i in range(-j - 1, j + 1)])


        # plot the difference between neibouring states on arrows conecting them
        for transition in transitions_in_j.itertuples():
            m1 = transition.m1
            xi1 = transition.xi1
            energy1 = self.state_df.loc[(self.state_df["j"] == j) & (self.state_df["m"] == m1) & (self.state_df["xi"] == xi1)].iloc[0].zeeman_energy_khz
            m2 = transition.m2
            xi2 = transition.xi2
            energy2 = self.state_df.loc[(self.state_df["j"] == j) & (self.state_df["m"] == m2) & (self.state_df["xi"] == xi2)].iloc[0].zeeman_energy_khz
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


    def fom(self, b_start: float, b_stop: float, j_start: float, j_stop: float):

        gj = self.gj
        cij = self.cij_khz

        num_points = np.abs(b_stop-b_start)*100
        B_values = np.linspace(b_start, b_stop, num_points) 
        J_values = np.linspace(j_start, j_stop, num_points) 

        B, J = np.meshgrid(B_values, J_values)

        cb = mu_N * B * 1e-4 / h / 1e3

        x = 1 / 2 * np.sqrt(cij**2 * ((J + 1 / 2) ** 2) + (- cb * (gj - gI)) ** 2)
        y = - cb * (gj - gI) / 2 

        h0 = 1 - np.abs(y/x)

        F = 1 - h0

        plt.figure(figsize=(8, 6))
        c = plt.contourf(J, B, F, 200, cmap='viridis', vmin=0, vmax=1)  
        plt.colorbar(c, label='F')
        plt.xlabel('J')
        plt.ylabel('B')
        plt.title('Heatmap of F')
        plt.grid(True)
        plt.show()
        


    def save_data(self):
        self.state_df.to_csv(f"molecule_data/{self.name}_B[{self.b_field_gauss:.2f}]_Jmax[{self.j_max}]_states.csv", index=False)
        self.transition_df.to_csv(f"molecule_data/{self.name}_B[{self.b_field_gauss:.2f}]_Jmax[{self.j_max}]_transitions.csv", index=False)


class CaOH(CaH):
    name: str = "CaOH"
    """name of the molecule"""
    gj: float = -0.036
    """g factor for J"""
    cij_khz: float = 1.49
    """coupling strength between proton spin and molecule rotation, in kHz"""
    br_ghz: float = 10.96    
    """rotational constant, in GHz"""
    omega_0_thz: float = 118.49
    # omega_0_thz: float = 1100.0
    """frequency of the electronic transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 280.1869
    """frequency of the Raman beam, in THz"""
    coupling_coefficient: float = 1.
    """coupling coefficient, for now to be able to compare with the NIST value"""

    def __init__(self, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None) -> None:
        super().__init__(b_field_gauss, j_max, gj_list, cij_list)


Molecule: TypeAlias = CaH | CaOH 
