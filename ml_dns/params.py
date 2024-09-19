import json
import numpy as np

class SimulationParameters:
    def __init__(self, json_file, topo, my_idx):
        with open(json_file, 'r') as f:
            params = json.load(f)
        
        self.case = params.get("case", "pressure_pulse")
        self.restart = params.get("restart", False)
        self.restart_file = params.get("restart_file", "restart.h5")
        self.case_params = params.get("case_params", {})

        # number of dimensions
        self.ndim = params.get("ndim", 3)
        # number of grid points
        self.ng = (params.get("nxg", 1), params.get("nyg", 1), params.get("nzg", 1))
        # number of processors
        self.np = (params.get("npx", 1), params.get("npy", 1), params.get("npz", 1))
        # Number of time steps
        self.num_steps = params.get("num_steps", 1000)
        # my index
        self.my_idx = my_idx

        # Domain parameters 
        self.domain_extents = params.get("domain_extents", 
                                         {"xs": 0.0, "xe": 1.0, 
                                          "ys": 0.0, "ye": 1.0, 
                                          "zs": 0.0, "ze": 1.0})
        self.grid_stretching_params = params.get("grid_stretching_params", {})
        
        # Boundary conditions
        boundary_conditions = params.get("boundary_conditions", {})
        self.periodic_bc = boundary_conditions.get("periodic_bc", [True, True, True])
        
        # Governing equations parameters
        governing_equations = params.get("governing_equations", {})
        rhs = governing_equations.get("rhs",{})
        self.rhs_split_integrate = rhs.get("split_integrate",False)

        advection = governing_equations.get("advection", {})
        self.use_advection = advection.get("use_advection", False)
        self.advection_method = advection.get("method", "compressible")
        self.advection_use_nn = advection.get("use_nn", False)

        diffusion = governing_equations.get("diffusion", {})
        self.use_diffusion = diffusion.get("use_diffusion", False)
        self.diffusion_method = diffusion.get("method", "fickian")
        self.diffusion_use_nn = diffusion.get("use_nn", False)

        reaction = governing_equations.get("reaction", {})
        self.use_reaction = reaction.get("use_reaction", False)
        self.num_species = reaction.get("num_species", 0)
        if(self.num_species > 0):
            raise ValueError("number of species > 0")
        self.reaction_mechanism = reaction.get("mechanism", "")
        self.reaction_use_nn = reaction.get("use_nn", False)

        force = governing_equations.get("force", {})
        self.use_force = force.get("use_force", False)
        self.use_buoyancy = force.get("use_buoyancy", False)
        self.force_use_nn = force.get("use_nn", False)

        # Fluid properties
        fluid_properties = params.get("fluid_properties", {})
        self.gamma = fluid_properties.get("gamma", 1.4)  # Specific heat ratio
        self.MW = fluid_properties.get("MW", 0.02897)  # defaults to Air MW kg/mol

        # Numerical methods
        numerical_methods = params.get("numerical_methods", {})
        self.diff_order = numerical_methods.get("diff_order", 8)
        self.stencil = numerical_methods.get("stencil",None)
        self.integrator = numerical_methods.get("integrator", "euler")
        self.dt = numerical_methods.get("dt", 0.001)
        self.cfl = numerical_methods.get("cfl", 0.5)
        self.integrator_use_nn = numerical_methods.get("use_nn", False)
        self.integrator_nn_model = numerical_methods.get("nn_model", None)

        # Output parameters
        output = params.get("output", {})
        self.output_format = output.get("format", "hdf5")
        self.output_dir = output.get("output_dir","./data/")
        self.output_frequency = output.get("frequency", 100)

        # Compute local grid points
        ##always add the remainder points to np<dim>-1 rank
        self.nl = tuple([
            self.ng[i] // self.np[i] + 
            (self.ng[i] % self.np[i] if my_idx[i] == self.np[i] - 1 else 0)
            for i in range(3)
        ])


        #refernce values
        refs = params.get("reference")
        self.P_ref = refs.get("P_ref") ##Pa
        self.l_ref = refs.get("l_ref") ##m
        self.a_ref = refs.get("a_ref") ##m/s
        self.T_ref = refs.get("T_ref") ##K
        ##get others from the above
        self.time_ref = self.l_ref/self.a_ref
        self.rho_ref = self.P_ref*self.time_ref**2/self.l_ref**2
        self.P_atm = 101325.0 ##Pa
        self.nvars = self.ndim + 2 + self.num_species
        self.my_rank = np.ravel_multi_index(my_idx, topo)
        ##non-dimensionalise inputs
        for ext in self.domain_extents:
            self.domain_extents[ext] /= self.l_ref
        self.MW /= (self.P_ref*self.l_ref*self.time_ref**2)
        self.dt /= self.time_ref

        print("Reference values:")
        print(f"P_ref: {self.P_ref} Pa")
        print(f"l_ref: {self.l_ref} m")
        print(f"a_ref: {self.a_ref} m/s")
        print(f"T_ref: {self.T_ref} K")
        print(f"time_ref: {self.time_ref} s")
        print(f"rho_ref: {self.rho_ref} kg/m^3")
        print(f"P_atm: {self.P_atm} Pa")


    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)

