import sys
import os
import torch

# Add the parent directory of ml_dns to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_dns import NavierStokesSolver
import matplotlib.pyplot as plt
import numpy as np

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
    gamma = solver.fluid_props.gamma
    init_state = copy.deepcopy(solver.state)
    solver.solve()
    final_state = copy.deepcopy(solver.state)
    # Compute initial mass and energy
    initial_mass = torch.sum(init_state.rho).item()
    initial_mom = torch.sum(init_state.rho_u).item()
    initial_total_energy = torch.sum(init_state.rho_E).item()
    
    print(f"Initial mass: {initial_mass}")
    print(f"Initial mom: {initial_mom}")
    print(f"Initial total energy: {initial_total_energy}")
    
    # Compute final mass and energy
    final_mass = torch.sum(final_state.rho).item()
    final_mom = torch.sum(final_state.rho_u).item()
    final_total_energy = torch.sum(final_state.rho_E).item()
    
    print(f"Final mass: {final_mass}")
    print(f"Final mom: {final_mom}")
    print(f"Final total energy: {final_total_energy}")
        
    # Extract the middle index in the y-direction
    ny_mid = solver.grid.ng[1] // 2

    # Extract initial and final states for T, u, v, rho, and P at y = ny/2
    initial_T = init_state.T[:, ny_mid, 0]
    final_T = final_state.T[:, ny_mid, 0]
    initial_u = init_state.u[0, :, ny_mid, 0]
    final_u = final_state.u[0, :, ny_mid, 0]
    initial_v = init_state.u[1, :, ny_mid, 0]
    final_v = final_state.u[1, :, ny_mid, 0]
    initial_rho = init_state.rho[:, ny_mid, 0]
    final_rho = final_state.rho[:, ny_mid, 0]
    initial_P = init_state.P[:, ny_mid, 0]
    final_P = final_state.P[:, ny_mid, 0]
    # Extract x-coordinates
    x_coords = solver.grid.xg(0)*solver.params.l_ref

    # Plot T at t=0 and final t
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, initial_T, label='Initial T', linestyle=':')
    plt.plot(x_coords, final_T, label='Final T', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Temperature at y = ny/2')
    plt.legend()
    plt.grid(True)
    plt.savefig("Temperature_comparison.png", dpi=300)

    # Plot u at t=0 and final t
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, initial_u, label='Initial u', linestyle=':')
    plt.plot(x_coords, final_u, label='Final u', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('u-velocity')
    plt.title('u-velocity at y = ny/2')
    plt.legend()
    plt.grid(True)
    plt.savefig("u_velocity_comparison.png", dpi=300)

    # Plot v at t=0 and final t
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, initial_v, label='Initial v', linestyle=':')
    plt.plot(x_coords, final_v, label='Final v', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('v-velocity')
    plt.title('v-velocity at y = ny/2')
    plt.legend()
    plt.grid(True)
    plt.savefig("v_velocity_comparison.png", dpi=300)

    # Plot rho at t=0 and final t
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, initial_rho, label='Initial rho', linestyle=':')
    plt.plot(x_coords, final_rho, label='Final rho', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Density at y = ny/2')
    plt.legend()
    plt.grid(True)
    plt.savefig("rho_comparison.png", dpi=300)

    # Plot P at t=0 and final t
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, initial_P, label='Initial P', linestyle=':')
    plt.plot(x_coords, final_P, label='Final P', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Pressure')
    plt.title('Pressure at y = ny/2')
    plt.legend()
    plt.grid(True)
    plt.savefig("pressure_comparison.png", dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, initial_P/initial_rho**gamma, label=r'$Initial P/\rho^{\gamma}$', linestyle=':')
    plt.plot(x_coords, final_P/final_rho**gamma, label=r'$Final P/\rho^{\gamma}$', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Pressure')
    plt.title('Pressure at y = ny/2')
    plt.legend()
    plt.grid(True)
    plt.savefig("pressure_rho-gamma.png", dpi=300)