import sys
import os
import torch

# Add the parent directory of ml_dns to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_dns import NavierStokesSolver
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
    solver = NavierStokesSolver(fname)
    initial_T = solver.state.T.clone()
    u = torch.amax(solver.state.u).item()
    init_peak_sim = solver.grid.xg(0)[torch.argmax(solver.state.T)]
    solver.solve()
    final_peak_sim = solver.grid.xg(0)[torch.argmax(solver.state.T)]
    final_T_simulated = solver.state.T
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
    
    plt.figure(figsize=(10, 6))
    # Plot initial state
    plt.plot(solver.grid.xg(0), initial_T.squeeze(), label='Initial T', linestyle=':')
    plt.plot(solver.grid.xg(0), final_T_computed.squeeze(), label='Computed T')
    plt.plot(solver.grid.xg(0), final_T_simulated.squeeze(), label='Simulated T', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Computed vs Simulated Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig("T.png",dpi=300)