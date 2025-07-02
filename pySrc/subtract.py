import numpy as np
import argparse
from adios2 import bindings
from rich.traceback import install
from ReaderClass import Reader
from WrighterClass import Writer
from mpi4py import MPI

# clean up 
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
        "--output_file","-o", default="subtract.bp", help="Output BP file for the result"
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
        default=1,
        help="Number of points to skip in the high-resolution file",
    )

    return parser.parse_args()


def subtraction_2D(low_res, ground_truth, skip_factor, tolerance, comm):
    diff = np.zeros_like(low_res)
    for i in range(low_res.shape[0]):
        for j in range(low_res.shape[1]):
            gt_i = int(i * skip_factor)
            gt_j = int(j * skip_factor)

            if gt_i >= ground_truth.shape[0] or gt_j >= ground_truth.shape[1]:
                continue  

            gt_value = ground_truth[gt_i, gt_j]
            e_value = low_res[i, j]
            diff[i, j] = abs(gt_value - e_value)

    if tolerance is not None:
        diff[diff <= float(tolerance)] = 0.0

    if comm is not None:
        comm.Barrier()

    return diff



def subtraction_3D(low_res, ground_truth, skip_factor, tolerance, comm):
    diff = np.zeros_like(low_res)
    for i in range(low_res.shape[0]):
        for j in range(low_res.shape[1]):
            for k in range(low_res.shape[2]):
                gt_i = int(i * skip_factor)
                gt_j = int(j * skip_factor)
                gt_k = int(k * skip_factor)

                if (
                    gt_i >= ground_truth.shape[0] or
                    gt_j >= ground_truth.shape[1] or
                    gt_k >= ground_truth.shape[2]
                ):
                    continue  

                gt_value = ground_truth[gt_i, gt_j, gt_k]
                e_value = low_res[i, j, k]
                diff[i, j, k] = abs(gt_value - e_value)

    if tolerance is not None:
        diff[diff <= float(tolerance)] = 0.0

    if comm is not None:
        comm.Barrier()

    return diff



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
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

    r_low = Reader(Declare_Io1, bpfile1, comm=comm)
    r_high = Reader(Declare_Io2, bpfile2, comm=comm)
    w = Writer(write_IO, output_File, comm=comm)

    var = args.var
    print(f"Variable to subtract: {var}")

    while True:
        status_low = r_low.begin_step()
        status_high = r_high.begin_step()

        if (
            bindings.StepStatus.OK != status_low
            or bindings.StepStatus.OK != status_high
        ):
            break
        w.begin_step()
        r_low.set_read_vars([var])
        r_high.set_read_vars([var])

        if r_low.vars_Out.get(var) is None or r_high.vars_Out.get(var) is None:
            print("Variables not found in the low resolution stream.")
            break

        low_res = r_low.read_step(var)
        ground_truth = r_high.read_step(var)
        if len(low_res.shape) == 3 and low_res.shape[0] == 1:
            low_res = np.squeeze(low_res, axis=0)
            ground_truth = np.squeeze(ground_truth, axis=0)
            diff = subtraction_2D(low_res, ground_truth, skip_factor, args.tolerance, comm)
        else:
            diff = subtraction_3D(low_res, ground_truth, skip_factor, args.tolerance, comm)

        w.write(f"diff_{var}", diff)

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
