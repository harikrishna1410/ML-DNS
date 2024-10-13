import torch
import numpy as np
import torch.nn.functional as F
from mpi4py import MPI
from .properties import FluidProperties
from .derivatives import Derivatives
from .grid import Grid
from .params import SimulationParameters
from .data import SimulationData, SimulationState
import json
from .rhs import RHS
from .advection import Advection
from .diffusion import Diffusion
from .integrate import Integrator, compute_timestep
from .force import Force
from .haloexchange import HaloExchange
from .init import Initializer
from .DNS_io import IO

class NavierStokesSolver:
    def __init__(self, json_file):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize simulation parameters
        with open(json_file, 'r') as f:
            params = json.load(f)
        topo = [params.get(n) for n in ['npx', 'npy', 'npz']]
        if self.size != topo[0] * topo[1] * topo[2]:
            raise ValueError(f"MPI size {self.size} does not match topology {topo}")
        
        self.my_pidx = tuple(np.unravel_index(self.rank, topo, order='F'))
        self.params = SimulationParameters(json_file, topo, self.my_pidx)
        
        # Initialize grid object
        self.grid = Grid(self.params)
        # Initialize fluid properties
        self.fluid_props = FluidProperties(self.params)
        # Initialize simulation state object
        self.state = SimulationState(self.params, self.fluid_props)
        # Initialize simulation data object
        self.data = SimulationData(self.params, self.state,self.grid)
        
        self.halo_exchange = HaloExchange(self.params, self.data, self.comm)
        # Initialize derivatives object with grid
        self.derivatives = Derivatives(self.grid, self.halo_exchange, self.data)

        self.initializer = Initializer(self.params, self.state, self.grid,self.fluid_props)
        self.initializer.initialize()
        dt = compute_timestep(self.params.cfl,self.grid,self.state,self.fluid_props)
        # Initialize Integrator
        self.integrator = Integrator(dt,
                                     method=self.params.integrator,
                                     use_nn=self.params.integrator_use_nn,
                                     neural_integrator=self.params.integrator_nn_model)
        
        if self.params.use_advection:
            # Initialize Advection object
            self.advection = Advection(
                params=self.params,
                derivatives=self.derivatives,
                integrator=self.integrator,
                use_nn=self.params.advection_use_nn,
                nn_model=None,  
                method=self.params.advection_method
            )
        else:
            self.advection = None

        if self.params.use_force:
            # Initialize Force object
            self.force = Force(
                params=self.params,
                derivatives=self.derivatives,
                integrator=self.integrator,
                use_nn=self.params.force_use_nn,
                nn_model=None,  
                use_buoyancy=self.params.use_buoyancy
            )
        else:
            self.force = None
        
        if self.params.use_diffusion:
            self.diffusion = Diffusion(
                params=self.params,
                derivatives=self.derivatives,
                integrator=self.integrator,
                fluid_props=self.fluid_props,
                use_nn=False,
                nn_model=None,  
            )
        else:
            self.diffusion = None

        # Initialize RHS object with advection and force
        self.rhs = RHS(split_integrate=self.params.rhs_split_integrate
                       ,integrator=self.integrator
                       ,advection=self.advection, 
                       force=self.force,
                       diffusion=self.diffusion)


        self.io = IO(self.params, self.data)
        self.io.write()
        if self.rank == 0:
            self.io.write_grid()

    def solve(self):
        """
        Main solver loop.
        """
        for step in range(self.params.num_steps):
            if(self.rank == 0):
                if(step % 10 == 0):
                    print(f"Step {step}")
                    print(self.state.min_max())
            self.state = self.rhs.integrate(self.state)
            self.state.compute_primitives_from_soln()
            
            # Write output at specified intervals
            if step % self.params.output_frequency == 0:
                print(f"Writing output at step {step}")
                self.io.write()

        # Write final state
        self.io.write()
