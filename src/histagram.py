import numpy as np
import matplotlib.pyplot as plt
import argparse
import adios2
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate histogram from ADIOS2 BP file variable.")
    parser.add_argument("input_file", type=str, help="Path to input .bp file")
    parser.add_argument("variable", type=str, help="Variable name to create histogram for")
    parser.add_argument("num_bins", type=int, help="Number of histogram bins")
    parser.add_argument("max_steps", type=int, help="Maximum number of time steps to process")
    parser.add_argument("--xml", type=str, default=None, help="Optional ADIOS2 XML configuration")
    return parser.parse_args()

def main():
    args = parse_arguments()

    results_dir = os.path.abspath(os.path.join("..", "RESULTS"))
    os.makedirs(results_dir, exist_ok=True)

    if args.xml:
        adios = adios2.Adios(args.xml)
    else:
        adios = adios2.Adios()

    io = adios.declare_io("HistogramIO")

    with adios2.Stream(io, args.input_file, 'r') as stream:
        for _ in stream:
            step = stream.current_step()
            print(f"Reading step {step}")
            
            data = stream.read(args.variable)
            flat_data = data.flatten()
            
            min_val = flat_data.min()
            max_val = flat_data.max()
            print(f"Variable '{args.variable}' min: {min_val}, max: {max_val}")
          
            counts, bin_edges = np.histogram(flat_data, bins=args.num_bins, range=(min_val, max_val))

            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
     
            plt.figure()
            plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), edgecolor='black', align='center')
            plt.xlabel(f"{args.variable} values")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of '{args.variable}' (step {step})")
            plt.tight_layout()
            plt.savefig(f"../RESULTS/{args.variable}_step_{step}_histogram.png")
            plt.close()

            if step == args.max_steps - 1:
                print("Done")
                print(f"Images saved to ../RESULTS")
                break

if __name__ == "__main__":
    main()
