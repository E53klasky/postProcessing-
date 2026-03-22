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
        "--input",
        "-in",
        type=str,
        required=True,
        help="Adios input file",
    )
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        required=False,
        help="IO Name for the first Adios file (default: reader1)",
    )
    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Variables name to create 2d plots separated by commas",
    )
    parser.add_argument(
        "--xml", "-x", type=str, default=None, help="Optional ADIOS2 XML configuration"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    r = ReaderClass.Reader(args.readIO, args.input, xml=args.xml)
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

            if data is None:
                print(f"Variable '{var}' not found in the stream.")
                continue

            # ---- Handle dimensions ----
            if len(data.shape) == 3:
                if data.shape[0] == 1:
                    data = data[0, :, :]
                else:
                    z_mid = data.shape[0] // 2
                    data = data[z_mid, :, :]
                    print(f"Using z-mid slice: {z_mid}")

            elif len(data.shape) != 2:
                print(f"Skipping unsupported shape {data.shape}")
                continue

            # ---- Plot ----
            plt.imshow(
                data,
                cmap="inferno",
            )

            plt.title(f"{var} at step {step_count}")
            plt.colorbar()

            plot_filename = f"{var}_step_{step_count}.png"
            plt.savefig(os.path.join(output_dir, plot_filename))
            plt.close()

            print(f"Plot saved as {plot_filename} to {output_dir}")

        r.end_step()

    print("Plot2D completed successfully")


if __name__ == "__main__":
    install()
    main()
