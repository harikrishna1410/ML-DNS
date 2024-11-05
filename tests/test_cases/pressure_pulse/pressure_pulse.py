import sys
import os

# Add the parent directory of ml_dns to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_dns import NavierStokesSolver

if __name__ == "__main__":
    fname = "./inputs/input.json"
    solver = NavierStokesSolver(fname)
    solver.solve()