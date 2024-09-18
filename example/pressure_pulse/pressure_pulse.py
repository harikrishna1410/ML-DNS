import sys
from ml_dns import NavierStokesSolver

if __name__ == "__main__":
    fname = "./inputs/input.json"
    solver = NavierStokesSolver(fname)
    solver.solve()
