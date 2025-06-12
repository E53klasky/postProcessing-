import adios2
from frechetdist import frdist
import os
import sys
import argparse
import numpy as np 


"""

# Example: Two curves as lists of 2D points
curve1 = [(0, 0), (1, 1), (2, 2)]
curve2 = [(0, 0), (2, 1), (2, 2)]

# Compute discrete Fréchet distance
distance = frdist(curve1, curve2)

print("Discrete Fréchet Distance:", distance)


"""
# add time steps for later
def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculating the error of streamlines given the segments')
    
    parser.add_argument("--file1", type=str, required=True, help="First Adios file with streamline segments")
    parser.add_argument("--file2", type=str, required=True, help="Second Adios file with streamline segments")

    parser.add_argument("--max_steps", type=int, required=True, help="Maximum number of steps to process")
    parser.add_argument('--xml', 
                        '-x', type=str, 
                        default=None, 
                        help='ADIOS2 XML config file default: None (optional)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    xml = args.xml
    file1 = args.file1
    file2 = args.file2
    max_step = args.max_steps
    if xml is not None:
        adios = adios2.Adios(xml)
    else:
        adios = adios2.Adios()
    
    Rio1 = adios.declare_io("reader1")
    Rio2 = adios.declare_io("reader2")
    
    with adios2.Stream(Rio1, file1, 'r' ) as f1,  adios2.Stream(Rio2, file2, 'r') as f2:
        for _ in f1:
            for _ in f2:
                step1 = f1.current_step()
                step2 = f2.current_step()
                
                print(f"processing step {step1}")
                segments_f1 = f1.read('segments')
                segments_f1_pairs = np.array(segments_f1).reshape(-1, 2)
                
                segments_f2 = f2.read('segments')
                segments_f2_pairs = np.array(segments_f2).reshape(-1,2)
                
                distance = frdist(segments_f1_pairs, segments_f2_pairs)
                print("Discrete Fréchet Distance:", distance)
                
                if step1 >= max_step -1 or step2 >= max_setep -1:
                    print(f"Reached max_steps = {max_step}")
                    break
                
                

if __name__ == "__main__":
    main()