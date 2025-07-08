import numpy as np
import argparse
from adios2 import bindings
from rich.traceback import install
from ReaderClass import Reader
from WrighterClass import Writer
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator

# TODO: why am I stil getting the worng values
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
        "--output_file",
        "-o",
        default="subtract.bp",
        help="Output BP file for the result",
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
    return parser.parse_args()


def upscale_array(low_res, high_res_shape):
    old_shape = low_res.shape
    old_axes = [np.linspace(0, 1, s) for s in old_shape]
    interpolator = RegularGridInterpolator(old_axes, low_res.astype(np.float64))

    new_axes = [np.linspace(0, 1, s) for s in high_res_shape]
    mesh = np.meshgrid(*new_axes, indexing="ij")
    points = np.stack([axis.ravel() for axis in mesh], axis=-1)

    high_res = interpolator(points).reshape(high_res_shape)
    return high_res


def subtraction(low_res, ground_truth, tolerance):
    low_res = low_res.astype(np.float64)
    ground_truth = ground_truth.astype(np.float64)

    diff = np.abs(ground_truth - low_res, dtype=np.float64)

    if tolerance is not None:
        diff[diff <= tolerance] = 0
    return diff


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = parse_arguments()

    r_low = Reader(args.Declare_Read_Io1, args.bpfile1, comm=comm)
    r_high = Reader(args.Declare_Read_Io2, args.bpfile2, comm=comm)
    w = Writer(args.Declare_Write_IO, args.output_file, comm=comm)

    var = args.var
    tolerance = args.tolerance

    while True:
        status_low = r_low.begin_step()
        status_high = r_high.begin_step()
        if (
            bindings.StepStatus.OK != status_low
            or bindings.StepStatus.OK != status_high
        ):
            break

        current_step = r_low.current_step()
        print(f"Rank {rank}: Reading step {int(current_step)}")

        w.begin_step()
        r_low.set_read_vars([var])
        r_high.set_read_vars([var])

        low_res = r_low.read_step(var)
        ground_truth = r_high.read_step(var)

        if low_res.shape != ground_truth.shape:
            low_res = upscale_array(low_res, ground_truth.shape)

        diff = subtraction(low_res, ground_truth, tolerance)
        print(f"max diff: {np.max(diff)}")
        w.write(f"diff_{var}", diff)
        w.end_step()

        r_low.end_step()
        r_high.end_step()

    r_low.close()
    r_high.close()
    w.close()

    if rank == 0:
        print(f"\nSubtraction completed and written to {args.output_file}")


if __name__ == "__main__":
    install()
    main()
