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
            if rank == 0:
                print(f"Reading step {step}")

            var_in = io.inquire_variable(var)
            shape = var_in.shape()  # (1, Y, Z)
            Y, Z = shape[1], shape[2]

            # Split work along Z
            base = Z // size
            rem = Z % size
            local_z = base + 1 if rank < rem else base
            local_start = rank * base + min(rank, rem)

            count = [1, Y, local_z]
            start = [0, 0, local_start]
            var_in.set_selection((start, count))

            local_data = stream.read(var)[0]  # shape: (Y, local_z)
            sendbuf = local_data.flatten()

            recvcounts = comm.gather(sendbuf.size, root=0)

            if rank == 0:
                total_size = sum(recvcounts)
                recvbuf = np.empty(total_size, dtype=sendbuf.dtype)
            else:
                recvbuf = None

            comm.Gatherv(sendbuf, (recvbuf, recvcounts), root=0)

            if rank == 0:
                # Reconstruct full data
                full_data = np.empty((Y, Z), dtype=sendbuf.dtype)

                offset = 0
                for i in range(size):
                    z_len = base + 1 if i < rem else base
                    z_start = i * base + min(i, rem)
                    chunk = recvbuf[offset:offset + Y * z_len].reshape((Y, z_len))
                    full_data[:, z_start:z_start + z_len] = chunk
                    offset += Y * z_len

                flat_data = full_data.flatten()

                min_val = flat_data.min()
                max_val = flat_data.max()
                print(f"Variable '{var}' min: {min_val}, max: {max_val}")

                counts, bin_edges = np.histogram(flat_data, bins=args.num_bins, range=(min_val, max_val))
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                plt.figure()
                plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), edgecolor='black', align='center')
                plt.xlabel(f"{var} values")
                plt.ylabel("Frequency")
                plt.title(f"Histogram of '{var}' (step {step})")
                plt.tight_layout()
                plt.savefig(f"../RESULTS/{var}_step_{step}_histogram.png")
                plt.close()

            if step == args.max_steps - 1:
                if rank == 0:
                    print("Done")
                    print(f"Images saved to ../RESULTS")
                break
if __name__ == "__main__":
    main()
