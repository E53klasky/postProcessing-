import adios2
import numpy as np
import argparse
from rich.traceback import install
import ReaderClass
import WrighterClass
from mpi4py import MPI


# NOTE only works in parallel for same sizes
def RMSE2D(GT, E, step, var_NAME, skip_factor=2):
    error = np.zeros_like(E)
    diff_sq = 0
    count = 0

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            gt_i = int(i * skip_factor)
            gt_j = int(j * skip_factor)

            if gt_i < GT.shape[0] and gt_j < GT.shape[1]:
                gt_value = GT[gt_i, gt_j]
                e_value = E[i, j]
                error[i, j] = gt_value - e_value
                diff_sq += (error[i, j]) ** 2
                count += 1

    if count == 0:
        print("Warning: No valid data points found for RMSE calculation")
        return 0.0

    rmse = np.sqrt(diff_sq / count)
    print("=" * 60)
    print(f"Step {step}: The RMSE for {var_NAME} is: {rmse}")
    print("=" * 60)
    return rmse


def RMSE3D(GT, E, step, var_NAME, skip_factor=2):
    error = np.zeros_like(E)
    diff_sq = 0
    count = 0

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            for k in range(E.shape[2]):
                gt_i = int(i * skip_factor)
                gt_j = int(j * skip_factor)
                gt_k = int(k * skip_factor)

                if gt_i < GT.shape[0] and gt_j < GT.shape[1] and gt_k < GT.shape[2]:
                    gt_value = GT[gt_i, gt_j, gt_k]
                    e_value = E[i, j, k]
                    error[i, j, k] = gt_value - e_value
                    diff_sq += (error[i, j, k]) ** 2
                    count += 1

    if count == 0:
        print("Warning: No valid data points found for RMSE calculation")
        return 0.0

    rmse = np.sqrt(diff_sq / count)
    print("=" * 60)
    print(f"Step {step}: The RMSE for {var_NAME} is: {rmse}")
    print("=" * 60)
    return rmse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute RMSE from ADIOS2 files")
    parser.add_argument(
        "--lowres", required=True, help="Path to the lower resolution ADIOS2 file"
    )
    parser.add_argument(
        "--highres",
        required=True,
        help="Path to the ground truth (high resolution) ADIOS2 file",
    )
    parser.add_argument(
        "--lowres_io", default="readerIOLow", help="IO name for low resolution reader"
    )
    parser.add_argument(
        "--highres_io",
        default="readerIOHigh",
        help="IO name for high resolution reader",
    )
    parser.add_argument(
        "--writer_io",
        default="writerIO",
        help="IO name for the writer to output RMSE results",
    )
    parser.add_argument(
        "--output_file",
        default="rmse_results.bp",
        help="Output file to write RMSE error data",
    )
    parser.add_argument(
        "--xml", default=None, help="Optional ADIOS2 XML configuration file"
    )
    parser.add_argument(
        "--var", required=True, help="Variable name to read from the files"
    )
    parser.add_argument(
        "--skip_factor",
        type=int,
        required=True,
        help="The skip factor for the higher resolution",
    )
    return parser.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = parse_arguments()

    print(f"Opening input streams: {args.lowres} and {args.highres}")
    print(f"Variable to analyze: {args.var}")
    print(f"Skip factor: {args.skip_factor}")

    r_low = ReaderClass.Reader(args.lowres_io, args.lowres, xml=args.xml, comm=comm)
    r_high = ReaderClass.Reader(args.highres_io, args.highres, xml=args.xml, comm=comm)
    w = WrighterClass.Writer(args.writer_io, args.output_file, xml=args.xml, comm=comm)

    var = args.var
    rmse_values = []

    try:
        while True:
            status_low = r_low.begin_step()
            status_high = r_high.begin_step()

            if (
                status_low != adios2.bindings.StepStatus.OK
                or status_high != adios2.bindings.StepStatus.OK
            ):
                print("End of data stream reached")
                break

            step_count = r_low.current_step()
            print(f"Reading step: {int(step_count)}")
            w.begin_step()

            r_low.set_read_vars([var])
            r_high.set_read_vars([var])

            if r_low.vars_Out.get(var) is None or r_high.vars_Out.get(var) is None:
                print(f"Variable '{var}' not found in one or both streams")
                w.end_step()
                r_low.end_step()
                r_high.end_step()
                break

            data_low = r_low.read_step(var)
            data_high = r_high.read_step(var)

            if data_low is not None and data_high is not None:
                print(f"Low res data shape: {data_low.shape}")
                print(f"High res data shape: {data_high.shape}")

                if len(data_low.shape) == 2 and len(data_high.shape) == 2:
                    rmse_value = RMSE2D(
                        data_high, data_low, step_count, var, args.skip_factor
                    )
                elif (
                    len(data_low.shape) == 3
                    and data_low.shape[0] == 1
                    and len(data_high.shape) == 3
                    and data_high.shape[0] == 1
                ):
                    data_low = data_low[0, :, :]
                    data_high = data_high[0, :, :]
                    rmse_value = RMSE2D(
                        data_high, data_low, step_count, var, args.skip_factor
                    )
                else:
                    rmse_value = RMSE3D(
                        data_high, data_low, step_count, var, args.skip_factor
                    )

                rmse_values.append(rmse_value)

                w.write("RMSE", np.array([rmse_value], dtype=np.float64))

            else:
                print(f"Variable '{var}' data is None in one of the streams")

            w.end_step()
            r_low.end_step()
            r_high.end_step()

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()

    finally:
        try:
            w.close()
            r_low.close()
            r_high.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    if rmse_values:
        print("\n" + "=" * 60)
        print("RMSE SUMMARY")
        print("=" * 60)
        print(f"Average RMSE: {np.mean(rmse_values):.6f}")
        print(f"Minimum RMSE: {np.min(rmse_values):.6f}")
        print(f"Maximum RMSE: {np.max(rmse_values):.6f}")
        print("=" * 60)
        print("RMSE computation completed successfully.")
        print(f"RMSE finished successfully and saved to ./{args.output_file}")
    else:
        print("No RMSE values calculated")


if __name__ == "__main__":
    install()
    main()
