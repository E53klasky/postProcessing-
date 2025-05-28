from mpi4py import MPI
import numpy as np 
from adios2 import Adios, Stream
import os
import sys

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Rank {rank} of {size} started.")
    
    
    if(len(sys.argv) < 2):
        print("Usage: python div_curl.py <PATH/TO/ADIOS2/DIR> <PATH/TO/ADIOS2/xml> ")
        return
    adios2_dir = sys.argv[1]
    adios2_xml = sys.argv[2]
    print(f"ADIOS2 directory: {adios2_dir}")
    print(f"ADIOS2 XML file: {adios2_xml}")
    
    adios_obj = Adios(adios2_xml)
    Rio = adios_obj.declare_io("readerIO")
    Wio = adios_obj.declare_io("WriteIO")
    
    
    
    
if __name__ == "__main__":
    main()
