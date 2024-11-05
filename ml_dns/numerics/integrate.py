import torch
import torch.nn as nn
from ..core import BaseSimulationState, CompressibleFlowState, Grid

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

    def integrate(self, state: BaseSimulationState, rhs):
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

    def euler(self, state: BaseSimulationState, rhs):
        if self.use_nn and self.neural_integrator:
            return self.neural_integrator(state)
        state.update_solution(self.dt * rhs(state))
        state.update_time(self.dt)
        return state

    def rk4(self, state: BaseSimulationState, rhs):
        if self.use_nn and self.neural_integrator:
            return self.neural_integrator(state)
        
        # Store initial solution
        initial_soln = state.get_solution().clone()
        
        k1 = rhs(state)
        
        state.update_solution(0.5 * self.dt * k1)
        state.compute_primitives_from_soln()
        k2 = rhs(state)
        
        # Reset to initial solution
        state.set_solution(initial_soln.clone())
        state.compute_primitives_from_soln()
        state.update_solution(0.5 * self.dt * k2)
        state.compute_primitives_from_soln()
        k3 = rhs(state)
        
        # Reset to initial solution
        state.set_solution(initial_soln.clone())
        state.compute_primitives_from_soln()
        state.update_solution(self.dt * k3)
        state.compute_primitives_from_soln()
        k4 = rhs(state)
        
        # Reset to initial solution and apply final update
        state.set_solution(initial_soln.clone())
        state.compute_primitives_from_soln()
        state.update_solution((self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

        # Update the simulation time
        state.update_time(self.dt)
        return state