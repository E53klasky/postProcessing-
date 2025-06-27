import numpy as np
import matplotlib.pyplot as plt
import argparse
import adios2
import os
from rich.traceback import install
import ReaderClass
import WrighterClass

# will not do till i get classes working in parallel
# rewrite it to use the classes and work in parallel
def parse_arguments():
    install()
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
        "--writeIO",
        "-wio",
        type=str,
        required=True,
        help="IO Name for the output Adios file",
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
    parser.add_argument("--output", "-o", default="bins.bp", help="name of outputfile")
    return parser.parse_args()


def main():
    args = parse_arguments()
    results_dir = os.path.abspath(os.path.join("..", "RESULTS"))
    os.makedirs(results_dir, exist_ok=True)
    var = args.var

    r = ReaderClass.Reader(args.IO_Name1, args.file1, args.xml)
    w = WrighterClass.Writer(args.writeIO, args.output, args.xml)
    
    while True:
        status = r.begin_step()
        step_count = r.current_step
        if status != adios2.bindings.StepStatus.OK:
            break
        w.begin_step()
        r.set_read_vars([var])
        data = r.read_step(var)

        global_min = np.min(data)
        global_max = np.max(data)
        if global_min == global_max:
            global_max += 1e-6

        hist, bin_edges = np.histogram(
            data, bins=args.num_bins, range=(global_min, global_max)
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        plt.figure()
        plt.bar(
            bin_centers,
            hist,
            width=(bin_edges[1] - bin_edges[0]),
            edgecolor="black",
            align="center",
        )
        w.write("bins", hist)

        plt.xlabel(f"{var} values")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of '{var}' (step {step_count})")
        plt.tight_layout()
        plt.savefig(f"../RESULTS/{var}_step_{step_count}_histogram.png")
        plt.close()
        
        r.end_step()
        w.end_step()
    
    r.close()
    w.close()
    print(f"Hisatram outputted successfully amd data is outputed ./{args.output}")

if __name__ == "__main__":
    install()
    main()
   