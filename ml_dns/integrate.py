import torch
import torch.nn as nn
from .data import SimulationState
from .grid import Grid
from .properties import FluidProperties

class Integrator:

    def __init__(self, dt, method="rk4", use_nn=False, neural_integrator=None):
        self.dt = dt
        self.neural_integrator = neural_integrator
        self.use_nn = use_nn
        if method == "rk4":
            self.integrator = self.rk4
        elif method == "euler":
            self.integrator = self.euler
        else:
            raise ValueError(f"Unsupported integration method: {method}")

    def integrate(self, state: SimulationState, rhs):
        return self.integrator(state, rhs)

    def set_dt(self, dt):
        self.dt = dt

    def set_neural_integrator(self, neural_integrator):
        self.neural_integrator = neural_integrator
    
    def get_use_nn(self):
        return self.use_nn

    def set_use_nn(self, value: bool):
        self.use_nn = value

    def toggle_use_nn(self):
        self.use_nn = not self.use_nn

    def euler(self, state: SimulationState, rhs):
        if self.use_nn and self.neural_integrator:
            return self.neural_integrator(state)
        state.soln += self.dt * rhs(state)

        state.time += self.dt
        
        return state

    def rk4(self, state: SimulationState, rhs):
        if self.use_nn and self.neural_integrator:
            return self.neural_integrator(state)
        
        k1 = rhs(state)
        
        state.soln += 0.5 * self.dt * k1
        state.compute_primitives_from_soln()
        k2 = rhs(state)
        
        state.soln -= 0.5 * self.dt * k1  # Revert to original state
        state.compute_primitives_from_soln()
        state.soln += 0.5 * self.dt * k2
        state.compute_primitives_from_soln()
        k3 = rhs(state)
        
        state.soln -= 0.5 * self.dt * k2  # Revert to original state
        state.compute_primitives_from_soln()
        state.soln += self.dt * k3
        state.compute_primitives_from_soln()
        k4 = rhs(state)
        
        state.soln -= self.dt * k3  # Revert to original state
        state.compute_primitives_from_soln()
        state.soln += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Update the simulation time
        state.time += self.dt
        return state


##function fo comute timestep based on cfl
def compute_timestep(cfl, 
                     grid: Grid, 
                     state: SimulationState, 
                     fluid_props: FluidProperties):
    # Compute the maximum wave speed
    c = torch.sqrt(fluid_props.gamma * state.P / state.rho)
    max_speed = torch.max(torch.linalg.norm(state.u,dim=0) + c)

    # Compute the minimum grid spacing
    min_dx = min([dx.min() for dx in grid.dx()])

    # Compute the timestep
    dt = cfl * min_dx / max_speed

    return dt
    