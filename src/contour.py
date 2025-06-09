import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from adios2 import Adios, Stream
from mpi4py import MPI
import os

# TODO: change to ./RESULTS
def parse_arguments():
    parser = argparse.ArgumentParser(description='Making contour plots')

    parser.add_argument('input_file', 
                        type=str,
                        help='Path to the input ADIOS2 BP5 file (REQUIRED)')

    parser.add_argument('--xml', 
                        '-x', 
                        type=str, 
                        default=None,
                        help='Path to ADIOS2 XML configuration file (optional)')

    parser.add_argument('--vars', 
                        '-v', 
                        type=str, 
                        required=True,
                        help='Variables to plot, separated by commas (REQUIRED)')

    parser.add_argument('--mode', 
                        '-m', 
                        type=str, 
                        choices=['2d', '3d'], 
                        default='2d',
                        help='2D or 3D mode, default: 2d (optional)')

    parser.add_argument('--max_steps', 
                        '-n', 
                        type=int, 
                        help='Maximum number of timesteps to process (REQUIRED)')
    
    parser.add_argument('--slice', 
                        '-s', 
                        type=int, 
                        default=16,
                        help='Slice index for 3D mode default: 16 (optional)')
 

    return parser.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f"Running with {size} MPI processes")
    
    
    args = parse_arguments()
    
    max_steps = args.max_steps
    input_file = args.input_file
    adios2_xml = args.xml
    vars = args.vars.split(',')
    slice = args.slice 
    
    mode = args.mode
    if rank == 0:
        print(f"Input file: {input_file}")
        print(f"ADIOS2 XML file: {adios2_xml}")
        print(f"Variables to plot: {vars}")
        print(f"Mode: {mode}")
    
    if max_steps <= 0:
        print("Error: max_steps must be a non-negative integer.")
        sys.exit(1)
        
    if adios2_xml is None:
        adios_obj = Adios(comm)
    else:
        adios_obj = Adios(adios2_xml, comm)
    Rio = adios_obj.declare_io("ReadIO")
    print(f"Opening input file: {input_file}")
    
    with Stream(Rio, input_file, "r") as s:
        for steps in s:
            step = s.current_step()
            print(f"Processing step {s.current_step()}")
            
            data = {}
            for var in vars:
                if var in s.available_variables():
                    data[var] = s.read(var)
                else:
                    print(f"Variable {var} not found in the input file.")
            
            if mode == '2d':
                plt.figure()
                for var, values in data.items():
                    output_dir = "../RESULTS"
                    os.makedirs(output_dir, exist_ok=True)
                    plt.contourf(np.squeeze(values), cmap="inferno", levels=50)
                    plt.title(var + f" at step {step}")
                    plt.colorbar()
                    plt.savefig(os.path.join(output_dir, f"{var}_step_{step}.png"))
                    plt.close()
                    
            elif mode == '3d':
                for var, values in data.items():
                    output_dir = "../RESULTS"
                    os.makedirs(output_dir, exist_ok=True)
                    dims = values.shape
                    axes = [0, 1, 2]
                    
                    x_dim, y_dim, z_dim = axes

                    x = np.arange(dims[x_dim])
                    y = np.arange(dims[y_dim])
                    X, Y = np.meshgrid(x, y)
                    
                    z_index = slice
                    if z_dim == 0:
                        values_2d = values[z_index, :, :]
                    elif z_dim == 1:
                        values_2d = values[:, z_index, :]
                    else:  
                        values_2d = values[:, :, z_index]


                    if values_2d.shape != X.shape:
                        values_2d = values_2d.T

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    surf = ax.plot_surface(X, Y, values_2d, cmap='inferno', linewidth=0, antialiased=False)

                    levels = np.linspace(np.min(values_2d), np.max(values_2d), 10)
                    ax.contour(X, Y, values_2d, levels=levels, cmap='inferno', linewidths=2)

                    ax.set_title(f"{var} slice at dim {mode} index {z_index}")
                    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                    plt.savefig(os.path.join(output_dir, f"{var}_3d_slice_{mode}_idx{z_index}.png"))
                    plt.close()

            
            
            if s.current_step() >= max_steps - 1:
                print(f"Reached max_steps = {max_steps}")
                break
    print("Images saved to ../RESULTS")

if __name__ == "__main__":
    main()
