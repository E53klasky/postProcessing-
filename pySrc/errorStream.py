import adios2
from frechetdist import frdist
import os
import sys
import argparse
import numpy as np 
import math 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from rich.traceback import install

# make my own code on it and incldue for rk steps code 
# change names of lower and higher res
def RK_visualization(segment_compressed, segment_uncompressed, distance, step=None):
    install()
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
    if step is not None:
        ax1.set_title(f"Lower Resolution Streamline Colored by Error (Step {step:04d})")
    else:
        ax1.set_title("Lower Resolution Streamline Colored by Error")
    ax1.grid(False)

    cbar = plt.colorbar(lc, ax=ax1)
    cbar.set_label("Error Magnitude")

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    streamline_filename = "highlighted_Lower_Resolution_streamline.png"
    if step is not None:
        streamline_filename = f"highlighted_Lower_Resolution_streamline_step_{step:04d}.png"

    fig1.savefig(os.path.join(output_dir, streamline_filename), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='b')
    plt.yscale("log")
    plt.title(f"Distance Error Plot {distance}")
    plt.xlabel("Point Index")
    plt.ylabel("Error Magnitude")
    plt.grid(True, which="both")
    plt.tight_layout()

    errorplot_filename = "distance_error_plot.png"
    if step is not None:
        errorplot_filename = f"distance_error_plot_step_{step:04d}.png"

    plt.savefig(os.path.join(output_dir, errorplot_filename), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.plot(segment_compressed[:, 0], segment_compressed[:, 1], linestyle='-', color='red', label='Lower Res')
    ax3.plot(segment_uncompressed[:, 0], segment_uncompressed[:, 1], linestyle='--', color='green', label='Higher res')
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    if step is not None:
        ax3.set_title(f"Lower Resolution vs Higher Resolution Streamlines (Step {step:04d}) error: {distance}")
    else:
        ax3.set_title(f"Lower Resolution vs Higher Resolution Streamlines error: {distance}")
    ax3.legend()
    ax3.grid(True)

    streamline_comparison_filename = "streamline_comparison.png"
    if step is not None:
        streamline_comparison_filename = f"streamline_comparison_step_{step:04d}.png"

    fig3.savefig(os.path.join(output_dir, streamline_comparison_filename), dpi=300, bbox_inches='tight')
    plt.close(fig3)


def parse_arguments():
    install()
    parser = argparse.ArgumentParser(description='Calculating the error of streamlines given the segments')

    parser.add_argument("--file1", type=str, required=True, help="First Adios file (low resolution/compressed)")
    parser.add_argument("--file2", type=str, required=True, help="Second Adios file (high resolution)")
    parser.add_argument("--max_steps", type=int, required=True, help="Maximum number of steps to process")
    parser.add_argument('--xml', '-x', type=str, default=None, help='ADIOS2 XML config file (optional)')

    parser.add_argument("--var_x", type=str, required=True, help="Variable name for x coordinates")
    parser.add_argument("--var_y", type=str, required=True, help="Variable name for y coordinates")
    parser.add_argument("--var_offset", type=str, required=True, help="Variable name for offsets")

    return parser.parse_args()

def main():
    install()
    args = parse_arguments()

    if args.xml:
        adios = adios2.Adios(args.xml)
    else:
        adios = adios2.Adios()

    io1 = adios.declare_io("reader1")
    io2 = adios.declare_io("reader2")

    with adios2.Stream(io1, args.file1, 'r') as f1, adios2.Stream(io2, args.file2, 'r') as f2:
        step = 0

        while step < args.max_steps:
            status1 = f1.begin_step()
            status2 = f2.begin_step()

            if not status1 or not status2:
                print(f"End of stream reached at step {step}")
                break

            # Read x, y from both files
            coords_x_1 = np.array(f1.read(args.var_x))
            coords_y_1 = np.array(f1.read(args.var_y))

            coords_x_2 = np.array(f2.read(args.var_x))
            coords_y_2 = np.array(f2.read(args.var_y))

            # Combine into coordinate pairs
            segments_f1 = np.column_stack((coords_x_1, coords_y_1))
            segments_f2 = np.column_stack((coords_x_2, coords_y_2))

            distance = frdist(segments_f1, segments_f2)
            print(f"Step {step} - Discrete Fréchet Distance: {distance}")
            RK_visualization(segments_f1,segments_f2,distance, step)
            f1.end_step()
            f2.end_step()
            step += 1

        print(f"Finished processing {step} steps")

if __name__ == "__main__":
    main()