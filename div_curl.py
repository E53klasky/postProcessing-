from mpi4py import MPI
import numpy as np 
from adios2 import stream 


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Rank {rank} of {size} started.")
    
    
    
if __name__ == "__main__":
    main()
