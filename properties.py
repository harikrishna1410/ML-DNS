import torch
from params import SimulationParameters
class FluidProperties:
    def __init__(self, params: SimulationParameters):
        self.sim_params = params
        # molecular weight in kg/mol
        self.MW = params.MW
        
        # Ratio of specific heats (gamma) for air
        self.gamma = params.gamma
        
        # Universal gas constant in J/(mol·K)
        self.R_universal = 8.314462618
        ##
        self.R_univ_nd = self.R_universal\
            /(self.sim_params.P_ref*self.sim_params.l_ref**3\
              /self.sim_params.T_ref)
        #
        self.R = self.R_univ_nd / self.MW

        self.Cv = self.R/(self.gamma - 1)
