import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from mpl_toolkits.mplot3d import axes3d
from adios2 import Adios, Stream, bindings
import mpi4py as MPI

def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Making contour plots')
    
    parser.add_argument('input_file', 
                       type=str, 
                       help='Path to the input ADIOS2 BP5 file (REQUIRED)')
    parser.add_argument('--xml', '-x',
                       type=str,
                       default=None,
                       help='Path to ADIOS2 XML configuration file (optional)')
    
    parser.add_argument('vars', 
                        '-v', 
                        type=str,
                        help='Variables to plot, separated by commas (REQUIRED)')
    
    parser.add_argument('--mode',
                        '-m', 
                        type=str, 
                        choices=['2d', '3d'], 
                        default='2d', 
                        help='2D or 3D mode default: 2d (optional)') 
    
    return parser.parse_args()
    




def main():
    print("hello world")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    args = parse_arguments()
    
    max_steps = args.max_steps
    input_file = args.input_file
    adios2_xml = args.xml
    vars = args.vars.split(',')
    
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
            #status = s.begin_step()
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
                    plt.contourf(values)
                    plt.title(var + f" at step {step}")
                    plt.colorbar()
                    plt.savefig(f"{var}_step_{step}.png")
                    
            elif mode == '3d':
                sys.exit("3D plotting is not implemented yet.")
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # for var, values in data.items():
                #     X, Y = np.meshgrid(np.arange(values.shape[0]), np.arange(values.shape[1]))
                #     ax.plot_surface(X, Y, values, cmap='viridis')
                #     ax.set_title(var + f" at step {step}")
                #plt.savefig(f"{var}_step_{step}.png")
            
            
            if s.current_step() >= max_steps - 1:
                print(f"Reached max_steps = {max_steps}")
                break
    

if __name__ == "__main__":
    main()
