import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from adios2 import bindings
import os
from rich.traceback import install
from ReaderClass import Reader


# make cleaner/DONE
def parse_arguments():
    install()
    parser = argparse.ArgumentParser(description="Making contour plots")

    parser.add_argument(
        "--bpfile1", "-f1", help="Lower-resolution input BP file", required=True
    )

    parser.add_argument(
        "--Declare_Read_Io1",
        "-d1",
        help="IO name for lower-resolution input",
        required=True,
    )

    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        help="Path to ADIOS2 XML configuration file (optional)",
    )

    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Variables to plot, separated by commas (REQUIRED)",
    )

    return parser.parse_args()


def main():
    install()

    args = parse_arguments()

    bpfile = args.bpfile1
    declare_read_io = args.Declare_Read_Io1
    adios2_xml = args.xml
    vars = args.vars.split(",")

    print(f"Input file: {bpfile}")
    print(f"ADIOS2 XML file: {adios2_xml}")
    print(f"Variables to plot: {vars}")

    r = Reader(declare_read_io, bpfile, adios2_xml)
    while True:
        status = r.begin_step()
        r.set_read_vars(vars)

        if status != bindings.StepStatus.OK:
            print("No more steps to read or an error occurred.")
            break

        for var in vars:
            data = r.read_step(var)
            if data is None:
                print(f"Variable '{var}' not found in the stream.")
                continue

            plt.figure()
            output_dir = "../RESULTS"
            os.makedirs(output_dir, exist_ok=True)
            plt.contourf(np.squeeze(data), cmap="inferno", levels=50)
            plt.title(f"{var} at step {r.current_step}")
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, f"{var}_step_{r.current_step}.png"))
            plt.close()
            print(
                f"Saved contour plot for {var} at step {r.current_step} to {output_dir} as {var}_step_{r.current_step}.png"
            )
        r.end_step()
    r.close()
    print("Contour.py plots saved to finshed successfully!")


if __name__ == "__main__":
    install()
    main()
