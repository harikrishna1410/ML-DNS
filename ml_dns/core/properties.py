import torch
from .params import SimulationParameters
class FluidProperties:
    def __init__(self, params: SimulationParameters):
        self.sim_params = params
        # molecular weight in kg/mol
        self.MW = params.MW
        
        # Ratio of specific heats (gamma) for air
        self.gamma = params.gamma

        ##prandtl number
        self.Pr = params.Pr
        
        # Universal gas constant in J/(mol·K)
        self.R_universal = 8.314462618
        ##
        self.R_univ_nd = self.R_universal\
            /(self.sim_params.P_ref*self.sim_params.l_ref**3\
              /self.sim_params.T_ref)
        #
        self.R = self.R_univ_nd / self.MW

        self.Cv = self.R/(self.gamma - 1)
        self.Cp = self.gamma*self.Cv
    ##expects a dimensional temperature
    def calculate_viscosity(self, T):
        # Sutherland's law constants for air
        mu_ref = 1.716e-5  # Reference viscosity in kg/(m·s)
        T_ref = 273.15  # Reference temperature in K
        S = 110.4  # Sutherland temperature in K

        # Calculate viscosity using Sutherland's law
        mu = mu_ref * (T / T_ref)**(3/2) * (T_ref + S) / (T + S)

        # Non-dimensionalize the viscosity
        mu_nd = mu / (self.sim_params.rho_ref * self.sim_params.a_ref * self.sim_params.l_ref)

        return mu_nd

    ##expects a dimensional temperature
    def calculate_thermal_conductivity(self, T):

        if self.Pr is None:
            # Sutherland's law constants for thermal conductivity of air
            k_ref = 0.0241  # Reference thermal conductivity in W/(m·K)
            T_ref = 273.15  # Reference temperature in K
            S = 194.0  # Sutherland temperature for thermal conductivity in K

            # Calculate thermal conductivity using Sutherland's law
            k = k_ref * (T / T_ref)**(3/2) * (T_ref + S) / (T + S)

            # Non-dimensionalize the thermal conductivity
            # Using characteristic values from sim_params
            #(J/(kg*K))*(kg/m**3)*(m/s)*(m)
            k_nd = k / (self.sim_params.rho_ref * self.sim_params.cp_ref * self.sim_params.a_ref * self.sim_params.l_ref)
        else:
            mu_nd = self.calculate_viscosity(T)
            k_nd = mu_nd*self.Cp/self.Pr

        return k_nd
    
    ##expects a dimensional temperature
    def calculate_species_diffusivity(self, T):
        return torch.zeros_like(T)