import argparse
import numpy as np
from adios2 import Adios, Stream, bindings


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute truncation error between low-res and high-res ADIOS2 files."
    )
    parser.add_argument("bpfile1", help="First input BP file (low res)")
    parser.add_argument(
        "--var1", required=True, help="Variable name from the first file"
    )
    parser.add_argument("bpfile2", help="Second input BP file (high res, ground truth)")
    parser.add_argument(
        "--var2", required=True, help="Variable name from the second file"
    )
    parser.add_argument(
        "--output_file",
        default="truncation_error.bp",
        help="Output BP file for the result",
    )
    parser.add_argument(
        "--xml", default=None, help="Optional ADIOS2 XML configuration file"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of time steps to process",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Tolerance level - errors <= tolerance will be set to 0",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.xml is not None:
        adios = Adios(args.xml)
    else:
        adios = Adios()

    io1 = adios.declare_io("ReadIO1")
    io2 = adios.declare_io("ReadIO2")
    io_out = adios.declare_io("OutputIO")

    low_res_file = args.bpfile1
    high_res_file = args.bpfile2

    print(f"Low-res file:  {low_res_file}")
    print(f"High-res file: {high_res_file}")

    with Stream(io1, low_res_file, "r") as f1, Stream(
        io2, high_res_file, "r"
    ) as f2, Stream(io_out, args.output_file, "w") as fout:

        step = 0

        while True:
            print(f"\n--- Step {step} ---")

            status1 = f1.begin_step()
            if status1 != bindings.StepStatus.OK:
                print("End of stream in low-res file.")
                break

            var_low = f1.inquire_variable(args.var1)
            data_low = f1.read(var_low)
            f1.end_step()

            status2 = f2.begin_step()
            if status2 != bindings.StepStatus.OK:
                print("End of stream in high-res file.")
                break

            var_high = f2.inquire_variable(args.var2)
            data_high = f2.read(var_high)
            f2.end_step()

            # Remove singleton dimension
            data_low_2d = data_low.squeeze()
            data_high_2d = data_high.squeeze()

            ny_low, nx_low = data_low_2d.shape
            ny_high, nx_high = data_high_2d.shape

            # Calculate skip factor from grid ratio
            # skip = (N_high - 1) / (N_low - 1)
            skip_y = (ny_high - 1) / (ny_low - 1)
            skip_x = (nx_high - 1) / (nx_low - 1)

            print(f"Low-res shape: {ny_low} x {nx_low}")
            print(f"High-res shape: {ny_high} x {nx_high}")
            print(f"Skip factors: y={skip_y:.6f}, x={skip_x:.6f}")

            truncation_error = np.zeros_like(data_low_2d)

            for i in range(ny_low):
                for j in range(nx_low):
                    gt_i = int(i * skip_y)
                    gt_j = int(j * skip_x)
                    # note you can take the np.abs of this
                    truncation_error[i, j] = (
                        data_high_2d[gt_i, gt_j] - data_low_2d[i, j]
                    )


            max_idx = np.unravel_index(np.argmax(np.abs(truncation_error)), truncation_error.shape)
            print(f"  Max error location: row={max_idx[0]}, col={max_idx[1]} (out of {ny_low}x{nx_low})")
            if args.tolerance is not None:
                truncation_error[np.abs(truncation_error) <= args.tolerance] = 0.0

            fout.begin_step()
            output_shape = (ny_low, nx_low)
            fout.write(
                f"{args.var1}_truncation_error",
                truncation_error,
                output_shape,
                [0, 0],
                output_shape,
            )
            fout.end_step()

            step += 1

            if args.max_steps is not None and step >= args.max_steps:
                break

    print(f"\nCompleted. Output written to: {args.output_file}")


if __name__ == "__main__":
    main()
