import numpy as np
import matplotlib.pyplot as plt
import argparse
import adios2
import os
import sys
from mpi4py import MPI

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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    var = args.variable
    os.makedirs(results_dir, exist_ok=True)

    if args.xml:
        adios = adios2.Adios(args.xml, comm)
    else:
        adios = adios2.Adios()

    io = adios.declare_io("HistogramIO")

    with adios2.Stream(io, args.input_file, 'r', comm) as stream:
        for _ in stream:
            step = stream.current_step()
            status = stream.begin_step()
            if rank == 0:
                print(f"Reading step {step}")

            var_in = io.inquire_variable(var)
            shape = var_in.shape()  
            Y, Z = shape[1], shape[2]

            
            base = Z // size
            rem = Z % size
            local_z = base + 1 if rank < rem else base
            local_start = rank * base + min(rank, rem)

            count = [1, Y, local_z]
            start = [0, 0, local_start]
            var_in.set_selection((start, count))

            local_data = stream.read(var)[0]
            local_data = local_data.flatten()

            local_min = local_data.min()
            local_max = local_data.max()

            global_min = comm.allreduce(local_min, op=MPI.MIN)
            global_max = comm.allreduce(local_max, op=MPI.MAX)

            local_hist, bin_edges = np.histogram(local_data, bins=args.num_bins, range=(global_min, global_max))

            global_hist = np.empty_like(local_hist)
            comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
           
            if global_min == global_max:
                global_max += 1e-6

            if rank == 0:
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                plt.figure()
                plt.bar(bin_centers, global_hist, width=(bin_edges[1] - bin_edges[0]),
                        edgecolor='black', align='center')
                plt.xlabel(f"{var} values")
                plt.ylabel("Frequency")
                plt.title(f"Histogram of '{var}' (step {step})")
                plt.tight_layout()
                plt.savefig(f"../RESULTS/{var}_step_{step}_histogram.png")
                plt.close()

            if not status or step == args.max_steps - 1:
                if rank == 0:
                    print("Done")
                    print(f"Images saved to ../RESULTS")
                break
if __name__ == "__main__":
    main()
