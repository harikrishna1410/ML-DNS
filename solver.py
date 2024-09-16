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
from haloexchange import HaloExchange
from init import Initializer

class NavierStokesSolver:
    def __init__(self,json_file):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize simulation parameters
        with open(json_file, 'r') as f:
            params = json.load(f)
        topo = [params.get(n, 1) for n in ['npx', 'npy', 'npz']]
        if self.size != topo[0] * topo[1] * topo[2]:
            raise ValueError(f"MPI size {self.size} does not match topology {topo}")
        
        self.my_pidx = tuple(torch.unravel_index(torch.tensor(self.rank), topo))
        self.params = SimulationParameters(json_file, topo, self.my_pidx)
        
        # Initialize grid object
        self.grid = Grid(self.params)
        # Initialize fluid properties
        self.fluid_props = FluidProperties(self.params)
        # Initialize simulation state object
        self.state = SimulationState(self.params, self.fluid_props)
        # Initialize simulation data object
        self.data = SimulationData(self.params, self.state)
        ##
        self.halo_exchange = HaloExchange(self.params, self.data, self.comm)
        # Initialize derivatives object with grid
        self.derivatives = Derivatives(self.grid, self.halo_exchange, self.data)

        # Initialize Integrator
        self.integrator = Integrator(self.params.dt,
                                     use_nn=False,
                                     neural_integrator=None)
        
        # Initialize Advection object
        self.advection = Advection(
            params=self.params,
            derivatives=self.derivatives,
            integrator=self.integrator,
            use_nn=self.params.use_nn,
            nn_model=None,
            method=self.params.advection_method
        )

        # Initialize Force object
        self.force = Force(
            params=self.params,
            derivatives=self.derivatives,
            integrator=self.integrator,
            use_nn=self.params.use_nn,
            nn_model=None,
            use_buoyancy=self.params.use_buoyancy
        )

        # Initialize RHS object with advection and force
        self.rhs = RHS(advection=self.advection, force=self.force)

        self.initializer = Initializer(self.params, self.state, self.grid)
        self.initializer.initialize()


    def solve(self):
        """
        Main solver loop.
        """
        for step in range(self.params.num_steps):
            self.state = self.rhs.integrate(self.state)
            self.state.compute_primitives_from_soln()



fname="inputs/input.json"
solver = NavierStokesSolver(fname)
solver.solve()
