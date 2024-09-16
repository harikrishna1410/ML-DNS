import torch
from params import SimulationParameters
class FluidProperties:
    def __init__(self, params: SimulationParameters):
        # Air molecular weight in kg/mol
        self.MW_air = 0.02897
        
        # Ratio of specific heats (gamma) for air
        self.gamma = params.gamma
        
        # Universal gas constant in J/(mol·K)
        self.R_universal = 8.314462618

    def compute_pressure(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute pressure using the ideal gas equation.
        
        Args:
        rho (torch.Tensor): Density in kg/m^3
        T (torch.Tensor): Temperature in K
        
        Returns:
        torch.Tensor: Pressure in Pa
        """
        R_specific = self.R_universal / self.MW_air  # Specific gas constant for air
        return rho * R_specific * T
