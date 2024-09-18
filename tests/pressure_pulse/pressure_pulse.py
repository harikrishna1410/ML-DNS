from ml_dns import NavierStokesSolver

if __name__ == "__main__":
    print("hello1")
    fname = "./inputs/input.json"
    solver = NavierStokesSolver(fname)
    solver.solve()