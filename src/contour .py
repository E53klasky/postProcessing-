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
    
    if max_steps < 0:
        print("Error: max_steps must be a non-negative integer.")
        sys.exit(1)
    
    

if __name__ == "__main__":
    main()
