import numpy as np
import matplotlib.pyplot as plt
import argparse
import adios2
import os
from rich.traceback import install
import ReaderClass
import WrighterClass
from mpi4py import MPI


# Fix it to wright out properly
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate histogram from ADIOS2 BP file variable."
    )
    parser.add_argument(
        "--file1",
        type=str,
        required=True,
        help="First Adios file with streamline segments (lower resolution/compressed)",
    )
    parser.add_argument(
        "--IO_Name1",
        type=str,
        default="reader1",
        help="IO Name for the first Adios file (default: reader1)",
    )
    parser.add_argument(
        "--var",
        "-v",
        type=str,
        required=True,
        help="Variables name to create hisagrams",
    )
    parser.add_argument("--num_bins", type=int, help="Number of histogram bins")

    parser.add_argument(
        "--xml", type=str, default=None, help="Optional ADIOS2 XML configuration"
    )
    parser.add_argument(
        "--Declare_Write_IO", help="IO name for writing output", required=True
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="bins.bp",
        help="Output BP file for the result",
    )

    return parser.parse_args()


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    args = parse_arguments()
    if rank == 0:
        results_dir = os.path.abspath(os.path.join("..", "RESULTS"))
        os.makedirs(results_dir, exist_ok=True)

    var = args.var

    r = ReaderClass.Reader(args.IO_Name1, args.file1, args.xml, comm=comm)
    if rank == 0:
        w = WrighterClass.Writer(args.Declare_Write_IO, args.output_file, args.xml)
    while True:
        status = r.begin_step()

        if status != adios2.bindings.StepStatus.OK:
            break

        current_step = r.current_step()
        print(f"Reading step: {int(current_step)}")

        r.set_read_vars([var])
        local_data = r.read_step(var)

        local_data = local_data.flatten()

        local_min = local_data.min()
        local_max = local_data.max()

        global_min = comm.allreduce(local_min, op=MPI.MIN)
        global_max = comm.allreduce(local_max, op=MPI.MAX)

        local_hist, bin_edges = np.histogram(
            local_data, bins=args.num_bins, range=(global_min, global_max)
        )

        global_hist = np.empty_like(local_hist)
        comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)

        if global_min == global_max:
            global_max += 1e-6

        if rank == 0:
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            plt.figure()
            plt.bar(
                bin_centers,
                global_hist,
                width=(bin_edges[1] - bin_edges[0]),
                edgecolor="black",
                align="center",
            )
            plt.xlabel(f"{var} values")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of '{var}' (step {current_step})")
            plt.tight_layout()
            plt.savefig(f"../RESULTS/{var}_step_{current_step}_histogram.png")
            print(f"output saved to ../RESULTS/{var}_step_{current_step}_histogram.png")
            plt.close()
            w.begin_step()
            w.write("bins", global_hist)
            w.end_step()

        r.end_step()

    r.close()
    if rank == 0:
        w.close()

    print(f"Histogram finsished successfully")


if __name__ == "__main__":
    install()
    main()
