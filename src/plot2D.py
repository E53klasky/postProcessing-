import numpy as np
import matplotlib.pyplot as plt
import argparse
import adios2
import os
import ReaderClass
from rich.traceback import install


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate 2D plots from ADIOS2 BP file variable."
    )
    parser.add_argument(
        "--bpfile1", "-bp1", type=str, required=True, help="Path to input .bp file"
    )
    parser.add_argument(
        "--read_io",
        "-r",
        type=str,
        required=True,
        default="ReadIO",
        help="IO name for reading the BP file",
    )
    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Variables name to create 2d plots separated by commas",
    )
    parser.add_argument(
        "--xml", type=str, default=None, help="Optional ADIOS2 XML configuration"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    r = ReaderClass.Reader(args.read_io, args.bpfile1, xml=args.xml)
    vars = args.vars.split(",")

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)
    while True:
        status = r.begin_step()

        if status != adios2.bindings.StepStatus.OK:
            break
        step_count = r.current_step()
        print(f"Reading step: {int(step_count)}")

        r.set_read_vars(vars)

        for var in vars:
            data = r.read_step(var)
            if data is not None:

                if len(data.shape) == 3 and data.shape[0] == 1:
                    data = data[0, :, :]
                    plt.imshow(data, cmap="inferno")
                    plot_filename = f"{var}_step_{step_count}.png"
                    plt.title(f"{var} at step {step_count}")
                    plt.colorbar()
                    plt.savefig(
                        os.path.join(output_dir, f"{var}_step_{step_count}.png")
                    )
                    plt.close()
                    print(f"Plot saved as {plot_filename} to {output_dir}")
                elif len(data.shape) == 2:
                    plt.imshow(data, cmap="inferno")
                    plot_filename = f"{var}_step_{step_count}.png"
                    plt.title(f"{var} at step {step_count}")
                    plt.colorbar()
                    plt.savefig(
                        os.path.join(output_dir, f"{var}_step_{step_count}.png")
                    )
                    plt.close()
                    print(f"Plot saved as {plot_filename} to {output_dir}")

            else:
                print(f"Variable '{var}' not found in the stream.")
        step_count += 1
        r.end_step()

    print("Plot2D completed successfully")


if __name__ == "__main__":
    install()
    main()
