import torch
import numpy as np
import torch.nn.functional as F
from mpi4py import MPI
from properties import FluidProperties
from derivatives import Derivatives
from grid import Grid
from params import SimulationParameters
from data import SimulationData, SimulationState
import json
from rhs import RHS
from advection import Advection
from integrate import Integrator
from force import Force

class NavierStokesSolver:
    def __init__(self,json_file):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize simulation parameters
        with open(json_file, 'r') as f:
            params = json.load(f)
        topo = params.get('topology', [1, 1, 1])
        if self.size != topo[0] * topo[1] * topo[2]:
            raise ValueError(f"MPI size {self.size} does not match topology {topo}")
        
        self.my_pidx = tuple(np.unravel_index(self.rank, topo))
        self.params = SimulationParameters(json_file, topo, self.my_pidx)
        
        # Initialize grid object
        self.grid = Grid(self.params, self.my_pidx)
        # Initialize fluid properties
        self.fluid_props = FluidProperties()
        # Initialize simulation state object
        self.state = SimulationState(self.params, self.fluid_props)
        # Initialize simulation data object
        self.data = SimulationData(self.params, self.state)
        # Initialize derivatives object with grid
        self.derivatives = Derivatives(self.grid)

        # Initialize Integrator
        self.integrator = Integrator(self.params.dt,
                                     use_nn=False,
                                     neural_integrator=None)

        # Initialize Advection object
        self.advection = Advection(
            params=self.params,
            derivatives=self.derivatives,
            integrator=self.integrator,
            nn_model=None,
            use_nn=False,
            method='compressible'
        )

        # Initialize Force object
        self.force = Force(
            params=self.params,
            derivatives=self.derivatives,
            integrator=self.integrator,
            nn_model=None,
            use_nn=False,
            use_buoyancy=self.params.use_buoyancy
        )

        # Initialize RHS object with advection and force
        self.rhs = RHS(advection=self.advection, force=self.force)


    def solve(self):
        """
        Main solver loop.
        """
        for step in range(self.params.max_iterations):
            self.state = self.rhs.integrate(self.state)
            self.state.compute_primitives_from_soln()



fname="simuation_params.json"
solver = NavierStokesSolver(fname)
solver.solve()
