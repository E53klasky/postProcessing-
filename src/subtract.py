import numpy as np
import argparse
from adios2 import bindings
from rich.traceback import install
from ReaderClass import Reader
from WrighterClass import Writer
from mpi4py import MPI

# this does not work  
# idk what to do
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Subtract variables from two ADIOS2 files and write the difference."
    )
    parser.add_argument(
        "--bpfile1", required=True, help="Lower-resolution input BP file"
    )
    parser.add_argument(
        "--bpfile2", required=True, help="Higher-resolution input BP file"
    )
    parser.add_argument(
        "--Declare_Read_Io1", required=True, help="IO name for lower-resolution input"
    )
    parser.add_argument(
        "--Declare_Read_Io2", required=True, help="IO name for higher-resolution input"
    )
    parser.add_argument(
        "--Declare_Write_IO", required=True, help="IO name for writing output"
    )
    parser.add_argument("--var", required=True, help="Variable name to subtract")
    parser.add_argument(
        "--output_file", "-o", default="subtract.bp", help="Output BP file"
    )
    parser.add_argument("--xml", default=None, help="Optional ADIOS2 XML config")
    parser.add_argument(
        "--tolerance", type=float, default=None, help="Tolerance for subtraction"
    )
    return parser.parse_args()

# code only fials with 3 cores fails with 3 
# 2d 
def subtraction_2D(low_res, ground_truth, tolerance, comm):
    skip_i = ground_truth.shape[0] / low_res.shape[0]
    skip_j = ground_truth.shape[1] / low_res.shape[1]
    skip_i_int = int(np.ceil(skip_i))
    skip_j_int = int(np.ceil(skip_j))

    data1_upsampled = np.repeat(np.repeat(low_res, skip_i_int, axis=0), skip_j_int, axis=1)

    data1_upsampled = data1_upsampled[:ground_truth.shape[0], :ground_truth.shape[1]]


    diff = np.abs(ground_truth - data1_upsampled)

    if tolerance is not None:
        diff = np.where(diff <= tolerance, 0.0, diff)
    return diff


def subtraction_3D(low_res, ground_truth, tolerance=None, comm=None):
    skip_i = ground_truth.shape[0] / low_res.shape[0]
    skip_j = ground_truth.shape[1] / low_res.shape[1]
    skip_k = ground_truth.shape[2] / low_res.shape[2]

    skip_i_int = int(np.ceil(skip_i))
    skip_j_int = int(np.ceil(skip_j))
    skip_k_int = int(np.ceil(skip_k))

    upsampled = np.repeat(
        np.repeat(
            np.repeat(low_res, skip_i_int, axis=0),
            skip_j_int, axis=1),
        skip_k_int, axis=2
    )

    upsampled = upsampled[:ground_truth.shape[0], :ground_truth.shape[1], :ground_truth.shape[2]]

    diff = np.abs(ground_truth - upsampled)

    if tolerance is not None:
        diff = np.where(diff <= tolerance, 0.0, diff)

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

 
        if len(low_res.shape) == 3 and low_res.shape[0] == 1:
            low_res = low_res[0, :, :]
            ground_truth = ground_truth[0,:,:]
            diff = subtraction_2D(low_res, ground_truth, tolerance, comm)
            w.write(f"diff_{var}", diff)
            w.end_step()
        elif len(low_res.shape) == 2:
            diff = subtraction_2D(low_res, ground_truth, tolerance, comm)
            w.write(f"diff_{var}", diff)
            w.end_step()
        else:
            diff = subtraction_3D(low_res, ground_truth, tolerance, comm)
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
