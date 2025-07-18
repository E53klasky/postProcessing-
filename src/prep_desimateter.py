import numpy as np
import argparse
from adios2 import bindings
from rich.traceback import install
from ReaderClass import Reader
from WrighterClass import Writer

# from mpi4py import MPI not for now
from scipy.interpolate import RegularGridInterpolator
import math


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Resample data to nearest power of two dimensions for decimation preprocessing."
    )
    parser.add_argument(
        "--input_file", "-in", help="Input BP file to resample", required=True
    )
    parser.add_argument(
        "--input_io",
        help="IO name for input file",
        required=False,
        default="reaio",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="resampled.bp",
        required=False,
        help="Output BP file for resampled data",
    )
    parser.add_argument(
        "--output_io",
        help="IO name for output file",
        required=False,
        default="wio",
    )
    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Variable names to resample (order matters), separated by commas, e.g., temperature,pressure,velocity (REQUIRED)",
    )
    parser.add_argument(
        "--xml", default=None, help="Optional ADIOS2 XML configuration (default: None)"
    )
    parser.add_argument(
        "--method",
        choices=["nearest", "upsample", "downsample"],
        default="nearest",
        help="Resampling method: 'nearest' (default), 'upsample', or 'downsample'",
    )

    return parser.parse_args()


def find_nearest_power_of_two_forced(n, method):
    if n <= 0:
        return 1

    is_power_of_2 = (n & (n - 1)) == 0

    log_n = math.log2(n)

    if method == "upsample":
        if is_power_of_2:
            power = int(log_n) + 1
        else:
            power = math.ceil(log_n)
    elif method == "downsample":
        if is_power_of_2:
            power = max(0, int(log_n) - 1)
        else:
            power = math.floor(log_n)

    return 2**power


def find_nearest_power_of_two(n, method="nearest"):
    if n <= 0:
        return 1

    log_n = math.log2(n)

    if method == "nearest":
        power = round(log_n)
    elif method == "upsample":
        power = math.ceil(log_n)
    elif method == "downsample":
        power = math.floor(log_n)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = 2**power

    return result


def resample_to_power_of_two(data, target_shape):
    if data.shape == target_shape:
        return data

    original_shape = data.shape
    original_axes = [np.linspace(0, 1, s) for s in original_shape]

    interpolator = RegularGridInterpolator(
        original_axes,
        data.astype(np.float64),
        method="cubic",
        bounds_error=False,
        fill_value=0,
    )

    target_axes = [np.linspace(0, 1, s) for s in target_shape]
    mesh = np.meshgrid(*target_axes, indexing="ij")
    points = np.stack([axis.ravel() for axis in mesh], axis=-1)

    resampled = interpolator(points).reshape(target_shape)

    return resampled.astype(data.dtype)


def calculate_target_shape(original_shape, method="nearest"):

    target_shape = []
    for dim in original_shape:
        if method == "upsample":
            target_dim = find_nearest_power_of_two_forced(dim, "upsample")
        elif method == "downsample":
            target_dim = find_nearest_power_of_two_forced(dim, "downsample")
        else:
            target_dim = find_nearest_power_of_two(dim, "nearest")
        target_shape.append(target_dim)

    return tuple(target_shape)


def main():
    # need sends and reives for mpi not for now
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    args = parse_arguments()

    reader = Reader(args.input_io, args.input_file, args.xml)
    writer = Writer(args.output_io, args.output_file, args.xml)

    variables = [var.strip() for var in args.vars.split(",")]
    step_count = 0

    print(f"Processing variables: {variables}")
    print(f"Number of variables: {len(variables)}")

    try:
        while True:
            status = reader.begin_step()

            if status != bindings.StepStatus.OK:
                break

            current_step = reader.current_step()

            print(f"Processing step: {int(current_step)}")

            print(f"  Setting read variables: {variables}")
            reader.set_read_vars(variables)

            writer.begin_step()

            for var in variables:
                print(f"  Processing variable: '{var}'")

                try:
                    data = reader.read_step(var)

                    if data is None:
                        print(
                            f"    Warning: Variable '{var}' returned None, skipping..."
                        )
                        continue

                except Exception as e:

                    print(f"    Error reading variable '{var}': {e}")
                    print(f"    Skipping variable '{var}'")
                    continue

                if len(data.shape) == 3 and data.shape[0] == 1:
                    data = data[0, :, :]

                    print(f"    Squeezed 3D data to 2D: {data.shape}")

                original_shape = data.shape

                target_shape = calculate_target_shape(original_shape, args.method)

                print(f"    Original shape: {original_shape}")
                print(f"    Target shape: {target_shape}")

                if original_shape != target_shape:

                    print(f"    Resampling from {original_shape} to {target_shape}")

                    resampled_data = resample_to_power_of_two(data, target_shape)

                    original_size = np.prod(original_shape)
                    target_size = np.prod(target_shape)
                    ratio = target_size / original_size

                    print(f"    Resampling ratio: {ratio:.4f}")
                    print(
                        f"    Data range: [{np.min(resampled_data):.6f}, {np.max(resampled_data):.6f}]"
                    )
                else:
                    resampled_data = data
                    print(
                        "    No resampling needed - data already has power-of-two dimensions"
                    )

                writer.write(var, resampled_data)

            writer.end_step()
            reader.end_step()

            step_count += 1

    except Exception as e:
        print(f"Error during processing: {e}")
        raise

    finally:
        reader.close()
        writer.close()

    print(f"\nResampling completed successfully!")
    print(f"Processed {step_count} steps")
    print(f"Variables processed: {variables}")
    print(f"Output written to: {args.output_file}")
    print(f"Resampling method: {args.method}")
    print(f"Data is now ready for decimation processing")


if __name__ == "__main__":
    install()
    main()
