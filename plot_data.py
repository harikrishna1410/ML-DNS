import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

def plot_data(file_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Get the list of variables
        variables = list(f.keys())
        # Plot each variable
        for var in variables:
            if(var == 'time' or var == 'dt' or var == 'nl'):
                continue
            data = f[var][:]
            
            # If the data is 3D, plot a 2D slice in the middle of each dimension
            if data.ndim == 3:
                for dim in range(3):
                    slice_index = data.shape[dim] // 2
                    if dim == 0:
                        slice_data = data[slice_index, :, :]
                    elif dim == 1:
                        slice_data = data[:, slice_index, :]
                    else:
                        slice_data = data[:, :, slice_index]
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(slice_data, cmap='viridis')
                    plt.colorbar(label=var)
                    plt.title(f'{var} - Slice in dimension {dim}')
                    plt.savefig(os.path.join(output_dir, f'{var}_dim{dim}.png'))
                    plt.close()
            
            # If the data is 2D, plot it directly
            elif data.ndim == 2:
                plt.figure(figsize=(10, 8))
                plt.imshow(data, cmap='viridis')
                plt.colorbar(label=var)
                plt.title(var)
                plt.savefig(os.path.join(output_dir, f'{var}.png'))
                plt.close()
            
            # If the data is 1D, plot it as a line
            elif data.ndim == 1:
                plt.figure(figsize=(10, 6))
                plt.plot(data)
                plt.title(var)
                plt.savefig(os.path.join(output_dir, f'{var}.png'))
                plt.close()

if __name__ == "__main__":
    file_path = "./data/time_0.0000000e+00/0.h5"  # Replace with your actual file path
    output_dir = "output_plots"  # Replace with your desired output directory
    plot_data(file_path, output_dir)
