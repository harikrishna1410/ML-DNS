import torch
import numpy as np
import torch.nn.functional as F
from mpi4py import MPI
import json
from .core import *
from .io import *
from .ml import *
from .numerics import *
from .physics import *
from .init import *

SUPPORTED_FLUID_TYPES = {
    "compressible_newtonian": CompressibleFlowState,
    # "Incompressible_Newtonian": IncompressibleFlowState,  # Future fluid types
    # "Non_Newtonian": NonNewtonianFlowState,
}

class Solver:
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
        if self.params.fluidtype not in SUPPORTED_FLUID_TYPES:
            raise ValueError(f"Unsupported fluid type: {self.params.fluidtype}. "
                            f"Supported types are: {list(SUPPORTED_FLUID_TYPES.keys())}")

        FlowStateClass = SUPPORTED_FLUID_TYPES[self.params.fluidtype]
        self.state = FlowStateClass(self.params, self.fluid_props)
        # Initialize simulation data object
        self.data = SimulationData(self.params, self.state,self.grid)
        
        self.halo_exchange = HaloExchange(self.params, self.comm)
        # Initialize derivatives object with grid
        self.derivatives = Derivatives(self.grid, self.halo_exchange)

        self.io = IO(self.params, self.data)
        self.initializer = Initializer(self.params, self.state, self.grid,self.fluid_props,self.io)
        self.initializer.initialize()
        self.io.write()
        if self.rank == 0:
            self.io.write_grid()
        if self.params.fluidtype == "compressible_newtonian":
            dt = self.compute_timestep(self.params.cfl,self.grid, self.state)
        else:
            dt = self.params.dt
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
                use_nn=self.params.advection_use_nn,
                nn_model=None,  
            )
        else:
            self.advection = None

        if self.params.use_force:
            # Initialize Force object
            self.force = Force(
                params=self.params,
                derivatives=self.derivatives,
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
                fluid_props=self.fluid_props,
                use_nn=False,
                nn_model=None,  
            )
        else:
            self.diffusion = None
        
        self.reaction = None

        # Initialize RHS object with advection and force
        self.rhs = RHS(advection=self.advection, 
                       force=self.force,
                       diffusion=self.diffusion,
                       reaction=self.reaction)

    def solve(self):
        """
        Main solver loop.
        """
        for step in range(self.params.num_steps):
            if(self.rank == 0):
                if(step % 10 == 0):
                    print(f"Step {step} dt {self.integrator.dt*self.params.time_ref}")
                    print(self.state.min_max_primitives())
            self.state = self.integrator.integrate(self.state,self.rhs)
            self.state.compute_primitives_from_soln()
            
            # Write output at specified intervals
            if step % self.params.output_frequency == 0:
                print(f"Writing output at step {step}")
                self.io.write()

        # Write final state
        self.io.write()

    
        ##function fo comute timestep based on cfl
    def compute_timestep(self,cfl, 
                         grid: Grid, 
                         state: CompressibleFlowState):
        # Compute the maximum wave speed
        c = state.compute_speed_of_sound()
        max_speed = torch.max(torch.linalg.norm(state.get_primitive_var("u"),dim=0) + c)

        # Compute the minimum grid spacing
        min_dx = min([dx.min() for dx in grid.dx()])

        # Compute the timestep
        dt = cfl * min_dx / max_speed

        return dt
