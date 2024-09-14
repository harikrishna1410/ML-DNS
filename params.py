import json
import numpy as np

class SimulationParameters:
    def __init__(self, json_file, topo, my_idx):
        with open(json_file, 'r') as f:
            params = json.load(f)        
        self.ng = (params.get("nxg"),params.get("nyg"),params.get("nzg"))
        self.ndim = params.ndim
        self.np = (params.get("npx"),params.get("npy"),params.get("npz"))
        self.grid_stretching_params = [
            {
                'start': params.get('xs', 0),
                'end': params.get('xe', 1)
            },
            {
                'start': params.get('ys', 0),
                'end': params.get('ye', 1)
            },
            {
                'start': params.get('zs', 0),
                'end': params.get('ze', 1)
            }
        ]  # Modified grid stretching parameters to use xs, xe, ys, ye, zs, ze from json_file

