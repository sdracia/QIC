# I suppose that the molecule i give in input is not optically pumped. I do it inside the class, if it's possible
import numpy as np
from molecule import CaOH, CaH, mu_N, gI
import qls


class BayesianStateEstimation:

    def __init__(self, model = None, temperature = 300, b_field_gauss = 3.6, j_max = 15):
        self.b_field_gauss = b_field_gauss
        self.j_max = j_max

        if model is None:
            model = CaOH.create_molecule_data(b_field_gauss=self.b_field_gauss, j_max=self.j_max)
        
        self.model = model

        # - If model = None, this if condition will be verified (since no "state_dist" is present)
        # - I pass a model to the class: if qls.States have been previously called, and/or optical pumping already applied, 
        # this if condition will not be verified. If qls.States has not been called to the model before, the if condition will be  
        # verified
        if "state_dist" not in self.model.state_df.columns:
            states1 = qls.States(molecule = self.model, temperature = temperature)

        self.temperature = temperature
        self.init_prior()

        self.history_list = []
        # If I want the dataframe
        # self.df = pd.DataFrame(self.history_list)     


    # prior is the column in the dataframe in the molecule
    def init_prior(self):
        self.prior = self.model.state_df["state_dist"]


    # optical pumping: updates the prior
    def optical_pumping(self, pump_frequency_mhz, num_pumps, pump_duration_us, pump_rabi_rate_mhz, pump_dephased, coherence_time_us, is_minus, noise_params = None, seed = None):
        # BE CAREFUL: if pumping is applied also in the prior updating, be careful that the prior changes and so
        # "state_dist" can be changed accordingly with the updated prior.

        # this function changes the state_dist column of the molecule dataframe
        qls.apply_pumping(self.model, pump_frequency_mhz, num_pumps, pump_duration_us, pump_rabi_rate_mhz, pump_dephased, coherence_time_us, is_minus, noise_params, seed)
        #for this reason, I update the prior
        self.prior = self.model.state_df["state_dist"]
        

    def measurement_setting(self, rabi_rate_mhz, dephased, coherence_time_us, is_minus, noise_params = None, seed = None):
        # the laser field fixes the rabi_rate: it's the same for the measurements that drive transitions and optical pumping.
        # [dephased, coh_time, is_minus] are considered the same for all measurements. can be done differently
        df_trans = self.model.transition_df

        measurements = [[df_trans.loc[df_trans["j"]==j].iloc[0]["energy_diff"] * 1e-3, 
                         np.pi/(rabi_rate_mhz*df_trans.loc[df_trans["j"]==j].iloc[0]["coupling"]), 
                         dephased, 
                         coherence_time_us,
                         is_minus
                         ] for j in range(1, self.j_max+1)]
        
        self.Probs_exc_list = []

        for frequency, duration, deph, coh_time, is_min in measurements:
            state_exc_probs = qls.get_excitation_probabilities(self.model, frequency, duration, rabi_rate_mhz, deph, coh_time, is_min, noise_params, seed)
            self.Probs_exc_list.append(state_exc_probs)

        self.measurements = measurements


    def update_distibution(self, num_updates, apply_pumping = False, save_data = True):
        for _ in range (num_updates):

            # if apply_pumping:
            #   IF PUMPING IS DONE AT EACH PRIOR UPDATE (NAMELY IF IT IS PART OF THE DESIGN)
            #   BE CAREFUL: once the prior is updated, see if the pumping should be applied considering 
            #   as "state_dist" the current prior or not. if yes, then in apply_pumping write 
            #   at the very beginning self.model.state_df["state_dist"] = self.prior

            self.meas_idx = self.get_next_setting()
            data = self.prior

            prob_exc = self.Probs_exc_list[self.meas_idx]

            lh0 = prob_exc
            lh1 = 1 - prob_exc

            self.p0 = np.sum(data*lh0)

            self.outcome = self.outcome_simulation(self.p0) 

            if self.outcome == 0:
                lihelihood = lh0
            else:
                lihelihood = lh1
            
            posterior = self.prior * lihelihood
            posterior = posterior / np.sum(posterior)


            self.likelihood = lihelihood
            self.posterior = posterior

            if save_data:
                self.history_list.append({
                    "meas_idx": self.meas_idx,
                    "measurement": self.measurements[self.meas_idx],
                    "p0": self.p0,
                    "outcome": self.outcome,
                    "prior": self.prior.tolist(),
                    "likelihood": self.likelihood.tolist(),
                    "posterior": self.posterior.tolist()
                })

            # prior updating:
            self.prior = posterior



    def get_next_setting(self):
        # TO CHANGE: for the moment I choose it by myself; it's fixed. Here is where the design should be applied
        idx = 3
        return idx
    

    def outcome_simulation(self, p0):
        # TO CHANGE: for the moment I throw a random number between 0 and 1, and choose accordingly to p0
        # needs to be changes since im extracting the outcome based on the prior knowledge.
        # this_rand = np.random.rand()
        # if this_rand <= p0:
        #     outcome = 0
        # else:
        #     outcome = 1


        this_rand = np.random.rand()
        if this_rand <= 0.5:
            outcome = 0
        else:
            outcome = 1
        return outcome       


        