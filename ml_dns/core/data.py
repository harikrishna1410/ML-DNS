import torch
from abc import ABC,abstractmethod
from .params import SimulationParameters
from .properties import FluidProperties
from .grid import Grid


class BaseSimulationState(ABC):
    def __init__(self, params: SimulationParameters, props: FluidProperties):
        self.nvars = params.nvars
        self.sim_params = params
        self.Fluid_props = props

        ##primitive maps
        self.ptoi = {}
        self.itop = {}
        ##soln maps
        self.stoi = {}
        self.itos = {}
        self.refs = {}

        self.__primitives = None
        self.__soln = None
        self.__time = 0.0 
        #TODO
        self.__particles = None
    
    @abstractmethod
    def get_time(self):
        pass
    
    @abstractmethod
    def set_time(self,time):
        pass
    
    @abstractmethod
    def update_time(self,dt):
        pass

    @abstractmethod
    def get_solution(self):
        pass
    
    @abstractmethod
    def set_solution(self,soln):
        pass
    
    @abstractmethod
    def get_solution_var(self,varname:str):
        pass
    
    ##this method updates the solution in place
    @abstractmethod
    def update_solution(self,rhs_dt):
        pass
    
    @abstractmethod
    def get_primitives(self):
        pass
    
    @abstractmethod
    def get_primitive_var(self,varname):
        pass
    
    @abstractmethod
    def set_primitives(self,values):
        pass

    @abstractmethod
    def set_primitive_var(self,values,varname):
        pass

    @abstractmethod
    def compute_primitives_from_soln(self):
        pass
    
    @abstractmethod
    def compute_soln_from_primitives(self):
        pass

class CompressibleFlowState(BaseSimulationState):
    def __init__(self, params:SimulationParameters, props:FluidProperties):
        super().__init__(params,props)
        # Create tensors for primitives and solution
        #rho,u,v,w,T,Ys,....,E (total energy),p (pressure)
        self.__primitives = torch.zeros((self.nvars+2,) + tuple(params.nl))
        ##soln: rho,rho*u,rho*v,rho*w,rho*E,rho*Ys
        self.__soln = torch.zeros((self.nvars,) + tuple(params.nl))
        self.__time = 0.0

        # Initialize primitive variable mappings
        self.ptoi['rho'] = 0
        for i in range(self.sim_params.ndim):
            self.ptoi[f'u{i}'] = i+1
        self.ptoi['T'] = self.sim_params.ndim+1
        for i in range(self.sim_params.num_species):
            self.ptoi[f'Y{i}'] = self.sim_params.ndim+2+i
        self.ptoi['E'] = -2
        self.ptoi['P'] = -1

        # Initialize solution variable mappings
        self.stoi['rho'] = 0
        for i in range(self.sim_params.ndim):
            self.stoi[f'rho_u{i}'] = i+1
        self.stoi['rho_E'] = self.sim_params.ndim+1
        for i in range(self.sim_params.num_species):
            self.stoi[f'rho_Y{i}'] = self.sim_params.ndim+2+i

        # Create reverse mappings
        self.itop = {v: k for k, v in self.ptoi.items()}
        self.itos = {v: k for k, v in self.stoi.items()}

        ##init the refs for primitives
        self.refs["rho"] = self.sim_params.rho_ref
        for i in range(self.sim_params.ndim):
            self.refs[f"u{i}"] = self.sim_params.a_ref
        self.refs["T"] = self.sim_params.T_ref
        self.refs["E"] = self.sim_params.a_ref**2
        self.refs["P"] = self.sim_params.P_ref

    def get_time(self):
        return self.__time
    
    def set_time(self,time):
        self.__time = time
    
    def update_time(self,dt):
        self.__time += dt

    def get_solution(self):
        return self.__soln

    def set_solution(self,soln):
        self.__soln = soln
    
    def get_solution_var(self, varname: str):
        if varname in self.stoi:
            return self.__soln[self.stoi[varname]]
        else:
            if varname == "rho_u":
                u_s = self.stoi["rho_u0"]
                return self.__soln[u_s:u_s+self.sim_params.ndim]
            elif varname == "rho_Y":
                if self.sim_params.num_species == 0:
                    return None
                y_s = self.stoi["rho_Y0"]
                return self.__soln[y_s:y_s+self.sim_params.num_species]
            
        raise ValueError(f"Unknown solution variable {varname}")
    
    def update_solution(self,rhs_dt):
        self.__soln += rhs_dt
    
    def get_primitives(self):
        return self.__primitives
    
    def get_primitive_var(self, varname):
        if varname in self.ptoi:
            return self.__primitives[self.ptoi[varname]]
        else:
            if varname == "u":
                u_s = self.ptoi["u0"]
                return self.__primitives[u_s:u_s+self.sim_params.ndim]
            elif varname == "Y":
                if self.sim_params.num_species == 0:
                    return None
                y_s = self.ptoi["Y0"]
                return self.__primitives[y_s:y_s+self.sim_params.num_species]
        raise ValueError(f"Unknown primitive variable {varname}")
    
    def set_primitives(self,values):
        self.__primitives = values
    
    def set_primitive_var(self, value, varname):
        if varname in self.ptoi:
            self.__primitives[self.ptoi[varname]] = value
        elif varname == "u":
            u_s = self.ptoi["u0"]
            self.__primitives[u_s:u_s+self.sim_params.ndim] = value
        elif varname == "Y":
                u_s = self.ptoi["Y0"]
                self.__primitives[u_s:u_s+self.sim_params.num_species] = value
        else:
            raise ValueError(f"Unknown primitive variable {varname}")
    
    def compute_primitives_from_soln(self):
        # Compute density (rho)
        self.__primitives[0] = self.__soln[0]
        
        # Compute velocities (u, v, w)
        for i in range(self.sim_params.ndim):
            self.__primitives[i+1] = self.__soln[i+1] / self.__soln[0]
        
        # Compute mass fractions (Ys)
        for i in range(self.sim_params.num_species):
            self.__primitives[self.sim_params.ndim+2+i] = self.__soln[self.sim_params.ndim+2+i] / self.__soln[0]
        
        self.__primitives[-2] = self.__soln[self.sim_params.ndim+1]/self.__soln[0]
        # Compute temperature (T)
        self.__primitives[self.sim_params.ndim+1] = (self.__primitives[-2] \
                                            - 0.5 * torch.sum(self.__primitives[1:self.sim_params.ndim+1]**2, dim=0))\
                                            / (self.Fluid_props.Cv)
        self.__primitives[-1] = self.compute_pressure(self.__primitives[0], self.__primitives[self.sim_params.ndim+1])

    def compute_soln_from_primitives(self):
        # Compute density (rho)
        self.__soln[0] = self.__primitives[0]
        
        # Compute momentum (rho * u, rho * v, rho * w)
        for i in range(self.sim_params.ndim):
            self.__soln[i+1] = self.__primitives[0] * self.__primitives[i+1]
        
        # Compute total energy (rho * E)
        kinetic_energy = 0.5 * torch.sum(self.__primitives[1:self.sim_params.ndim+1]**2, dim=0)
        internal_energy = self.__primitives[self.sim_params.ndim+1] * self.Fluid_props.Cv
        
        self.__soln[self.sim_params.ndim+1] = self.__primitives[0] * (kinetic_energy + internal_energy)
        
        # Compute species densities (rho * Ys)
        for i in range(self.nvars - self.sim_params.ndim - 2):
            self.__soln[self.sim_params.ndim+2+i] = self.__primitives[0] * self.__primitives[self.sim_params.ndim+2+i]
        
    def compute_pressure(self, rho: torch.Tensor, T: torch.Tensor):
        return rho * self.Fluid_props.R * T

    def compute_density_from_pressure(self, P: torch.Tensor, T: torch.Tensor):
        return P / (self.Fluid_props.R * T)
    
    def compute_speed_of_sound(self):
        return torch.sqrt(self.Fluid_props.gamma * self.Fluid_props.R * self.__primitives[self.ptoi['T']])
        
    def min_max_primitives(self):
        """
        Compute the minimum and maximum values for each variable in the state.
        
        Returns:
        dict: A dictionary containing the min and max values for each variable.
        """
        result = {}
        
        # Compute min and max for primitives using ptoi dictionary
        for var_name, idx in self.ptoi.items():
            ref_val = self.refs.get(var_name, 1.0)
            min_val = torch.amin(self.__primitives[idx]).item() * ref_val
            max_val = torch.amax(self.__primitives[idx]).item() * ref_val
            result[f'{var_name}_min'] = min_val
            result[f'{var_name}_max'] = max_val
        
        return result


class SimulationData:
    def __init__(self, params : SimulationParameters,state: BaseSimulationState, grid: Grid):
        self.state = state
        self.sim_params = params
        self.grid = grid


