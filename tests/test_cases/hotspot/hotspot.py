import sys
import os
import torch

# Add the parent directory of ml_dns to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',"..")))

from ml_dns import Solver
import matplotlib.pyplot as plt

"""
    this case tests the advection term in 1D.
    when all pressure gradient terms are set to zero,
    Euler equations just become scalar transport equations.
"""
##
#
if __name__ == "__main__":
    import copy

    fname = "./inputs/input.json"
    solver = Solver(fname)
    
    # Store initial state
    initial_T = solver.state.get_primitive_var("T").clone()
    initial_P = solver.state.get_primitive_var("P").clone()
    initial_rho = solver.state.get_primitive_var("rho").clone()
    
    u = torch.amax(solver.state.get_primitive_var("u")).item()
    init_peak_sim = solver.grid.xg(0)[torch.argmax(solver.state.get_primitive_var("T"))]
    solver.solve()
    final_peak_sim = solver.grid.xg(0)[torch.argmax(solver.state.get_primitive_var("T"))]
    
    # Get final simulated state
    final_T_simulated = solver.state.get_primitive_var("T")
    final_P_simulated = solver.state.get_primitive_var("P")
    final_rho_simulated = solver.state.get_primitive_var("rho")
    
    ##compute the final state
    xs = solver.params.domain_extents["xs"]
    xe = solver.params.domain_extents["xe"]
    init_center = (xs + (xe-xs)*solver.params.case_params["center"][0])
    final_center = init_center + (solver.params.num_steps*solver.params.dt*u)
    r = torch.abs(solver.grid.xg(0) - final_center)/(xe-xs)
    final_T_computed =  (1.0 + solver.params.case_params.get('amplitude') \
                       * torch.exp(-1000 * r**2.0))*\
                        solver.params.case_params.get('T_ambient')/solver.params.T_ref
    
    print("max error in T:",torch.amax(torch.abs(final_T_simulated.squeeze()-final_T_computed.squeeze()))\
          /torch.amax(torch.abs(final_T_computed)))
    
    # Create subplots for T, P, and rho
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot Temperature
    ax1.plot(solver.grid.xg(0), initial_T.squeeze(), label='Initial', linestyle=':')
    ax1.plot(solver.grid.xg(0), final_T_computed.squeeze(), label='Computed')
    ax1.plot(solver.grid.xg(0), final_T_simulated.squeeze(), label='Simulated', linestyle='--')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Temperature Evolution')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Pressure
    ax2.plot(solver.grid.xg(0), initial_P.squeeze(), label='Initial', linestyle=':')
    ax2.plot(solver.grid.xg(0), final_P_simulated.squeeze(), label='Simulated', linestyle='--')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Pressure')
    ax2.set_title('Pressure Evolution')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Density
    ax3.plot(solver.grid.xg(0), initial_rho.squeeze(), label='Initial', linestyle=':')
    ax3.plot(solver.grid.xg(0), final_rho_simulated.squeeze(), label='Simulated', linestyle='--')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Density')
    ax3.set_title('Density Evolution')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig("flow_variables.png", dpi=300)