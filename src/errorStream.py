import adios2
from frechetdist import frdist
import os
import sys
import argparse
import numpy as np 
import math 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def RK_visualization(segment_compressed, segment_uncompressed, step=None):
    errors = np.linalg.norm(segment_compressed - segment_uncompressed, axis=1)

    points = segment_compressed.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(errors.min(), errors.max()))
    lc.set_array(errors[:-1])  
    lc.set_linewidth(3)

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_aspect('equal')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Compressed Streamline Colored by Error")
    ax1.grid(False)

    cbar = plt.colorbar(lc, ax=ax1)
    cbar.set_label("Error Magnitude")

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    streamline_filename = "highlighted_compressed_streamline.png"
    if step is not None:
        streamline_filename = f"highlighted_compressed_streamline_step_{step:04d}.png"

    fig1.savefig(os.path.join(output_dir, streamline_filename), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='b')
    plt.yscale("log")
    plt.title("Distance Error Plot")
    plt.xlabel("Point Index")
    plt.ylabel("Error Magnitude")
    plt.grid(True, which="both")
    plt.tight_layout()

    errorplot_filename = "distance_error_plot.png"
    if step is not None:
        errorplot_filename = f"distance_error_plot_step_{step:04d}.png"

    plt.savefig(os.path.join(output_dir, errorplot_filename), dpi=300, bbox_inches='tight')
    plt.close(fig2)

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
                statusf1 = f1.begin_step()
                statusf2 = f2.begin_step()
                step1 = f1.current_step()
                step2 = f2.current_step()
                
                print(f"processing step {step1}")
                segments_f1 = f1.read('segments')
                segments_f1_pairs = np.array(segments_f1).reshape(-1, 2)
                
                segments_f2 = f2.read('segments')
                segments_f2_pairs = np.array(segments_f2).reshape(-1,2)
                
                distance = frdist(segments_f1_pairs, segments_f2_pairs)
                print("Discrete Fréchet Distance:", distance)
                RK_visualization(segments_f1_pairs, segments_f2_pairs)
                if  not statusf1 or not statusf2 or step1 >= max_step -1 or step2 >= max_step -1:
                    print(f"Reached max_steps = {max_step}")
                    break
                
                

if __name__ == "__main__":
    main()