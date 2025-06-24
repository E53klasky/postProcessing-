import numpy as np
import argparse
from adios2 import Adios, Stream, bindings
from rich.traceback import install
from ReaderClass import Reader
from WrighterClass import Writer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Subtract variables from two ADIOS2 files and write the difference."
    )

    parser.add_argument(
        "--bpfile1", help="Lower-resolution input BP file", required=True
    )
    parser.add_argument(
        "--bpfile2", help="Higher-resolution input BP file", required=True
    )

    parser.add_argument(
        "--Declare_Read_Io1", help="IO name for lower-resolution input", required=True
    )
    parser.add_argument(
        "--Declare_Read_Io2", help="IO name for higher-resolution input", required=True
    )

    parser.add_argument(
        "--Declare_Write_IO", help="IO name for writing output", required=True
    )

    parser.add_argument("--var", help="Variable name to subtract", required=True)

    parser.add_argument(
        "--output_file", default="subtract.bp", help="Output BP file for the result"
    )

    parser.add_argument(
        "--xml", default=None, help="Optional ADIOS2 XML configuration (default: None)"
    )

    parser.add_argument(
        "--tolerance",
        default=None,
        type=float,
        help="Tolerance level: differences <= tolerance will be set to 0",
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of points to skip in the high-resolution file",
    )

    return parser.parse_args()


def main():
    install()
    args = parse_arguments()

    bpfile1 = args.bpfile1
    bpfile2 = args.bpfile2

    Declare_Io1 = args.Declare_Read_Io1
    Declare_Io2 = args.Declare_Read_Io2

    write_IO = args.Declare_Write_IO

    skip_factor = args.skip

    GT = args.bpfile2
    E = args.bpfile1
    output_File = args.output_file
    print(f"Opening input streams: {E} and {GT}")

    r_low = Reader(Declare_Io1, bpfile1)
    r_high = Reader(Declare_Io2, bpfile2)
    w = Writer(write_IO, output_File)

    var = args.var
    print(f"Variable to subtract: {var}")
    var_not_defined = True
    while True:
        status_low = r_low.begin_step()
        status_high = r_high.begin_step()
        w.begin_step()

        r_low.set_read_vars([var])
        r_high.set_read_vars([var])

        if (
            bindings.StepStatus.OK != status_low
            or bindings.StepStatus.OK != status_high
        ):
            break
        low_res = r_low.read_step(var)
        ground_truth = r_high.read_step(var)

        diff = np.zeros_like(low_res)
        for i in range(low_res.shape[1]):
            for j in range(low_res.shape[2]):
                gt_i = int(i * skip_factor)
                gt_j = int(j * skip_factor)
                gt_value = ground_truth[0, gt_i, gt_j]
                e_value = low_res[0, i, j]
                # can make it abs if needed
                diff[0, i, j] = (gt_value - e_value)

        if args.tolerance is not None:
            diff[diff <= float(args.tolerance)] = 0.0
        if var_not_defined:
            w.set_write_vars(diff, var)

        var_not_defined = False
        w.write(var, diff)

        w.end_step()
        r_low.end_step()
        r_high.end_step()

    r_low.close()
    r_high.close()
    w.close()

    print("\nSubtraction completed and written to", args.output_file)


if __name__ == "__main__":
    install()
    main()
