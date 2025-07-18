import numpy as np
from WrighterClass import Writer
from ReaderClass import Reader
from rich.traceback import install
import adios2
import argparse
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv


def calculate_errors(original, reconstructed):
    if original.shape != reconstructed.shape:
        reconstructed = upscale_with_regular_grid_interpolator(
            reconstructed, original.shape, method="cubic"
        )

    diff = np.abs(original - reconstructed)
    l1_error = np.sum(diff)
    l2_error = np.sqrt(np.sum(diff**2))
    linf_error = np.max(diff)

    return l1_error, l2_error, linf_error


def build_progressive_array_2d(data, min_size=8):
    for i, dim_size in enumerate(data.shape):
        assert (
            dim_size & (dim_size - 1)
        ) == 0, f"Dimension {i} size {dim_size} must be power of two run prep_desimater"

    ny, nx = data.shape

    sizes = []
    curr_shape = (ny, nx)

    while all(s >= min_size for s in curr_shape):
        sizes.append(curr_shape)
        curr_shape = tuple(s // 2 for s in curr_shape)

    sizes = sizes[::-1]

    output_chunks = []

    for i, (sy, sx) in enumerate(sizes):
        step_y = ny // sy
        step_x = nx // sx

        if i == 0:
            down = data[::step_y, ::step_x]
            output_chunks.append(down.flatten())
        else:
            full = data[::step_y, ::step_x]
            new_points = []

            new_points.append(full[1::2, ::2].flatten())
            new_points.append(full[::2, 1::2].flatten())
            new_points.append(full[1::2, 1::2].flatten())

            output_chunks.append(np.concatenate(new_points))

    return np.concatenate(output_chunks), sizes


def build_progressive_array_3d(data, min_size=8):
    for i, dim_size in enumerate(data.shape):
        assert (
            dim_size & (dim_size - 1)
        ) == 0, f"Dimension {i} size {dim_size} must be power of two run prep_desimater"

    nz, ny, nx = data.shape

    sizes = []
    curr_shape = (nz, ny, nx)

    while all(s >= min_size for s in curr_shape):
        sizes.append(curr_shape)
        curr_shape = tuple(s // 2 for s in curr_shape)

    sizes = sizes[::-1]

    output_chunks = []

    for i, (sz, sy, sx) in enumerate(sizes):
        step_z = nz // sz
        step_y = ny // sy
        step_x = nx // sx

        if i == 0:
            down = data[::step_z, ::step_y, ::step_x]
            output_chunks.append(down.flatten())
        else:
            full = data[::step_z, ::step_y, ::step_x]
            new_points = []

            new_points.append(full[1::2, ::2, ::2].flatten())
            new_points.append(full[::2, 1::2, ::2].flatten())
            new_points.append(full[::2, ::2, 1::2].flatten())
            new_points.append(full[1::2, 1::2, ::2].flatten())
            new_points.append(full[1::2, ::2, 1::2].flatten())
            new_points.append(full[::2, 1::2, 1::2].flatten())
            new_points.append(full[1::2, 1::2, 1::2].flatten())

            output_chunks.append(np.concatenate(new_points))

    return np.concatenate(output_chunks), sizes


def extract_level_2d(progressive_array, target_shape, sizes):
    index = sizes.index(target_shape)

    chunks = []
    for i, shape in enumerate(sizes):
        if i == 0:
            chunks.append(np.prod(shape))
        else:
            prev_shape = sizes[i - 1]
            added = np.prod(shape) - np.prod(prev_shape)
            chunks.append(added)

    start = sum(chunks[:index])
    end = start + chunks[index]
    flat = progressive_array[start:end]

    if index == 0:
        return flat.reshape(target_shape)

    prev_shape = sizes[index - 1]
    prev_data = extract_level_2d(progressive_array, prev_shape, sizes)

    out = np.zeros(target_shape)
    out[::2, ::2] = prev_data

    ty, tx = target_shape
    half_y, half_x = ty // 2, tx // 2
    i = 0

    quadrant_size_y_even = half_y * half_x
    quadrant_size_x_even = half_y * half_x
    quadrant_size_both_odd = half_y * half_x

    out[1::2, ::2] = flat[i : i + quadrant_size_y_even].reshape((half_y, half_x))
    i += quadrant_size_y_even
    out[::2, 1::2] = flat[i : i + quadrant_size_x_even].reshape((half_y, half_x))
    i += quadrant_size_x_even
    out[1::2, 1::2] = flat[i : i + quadrant_size_both_odd].reshape((half_y, half_x))

    return out


def extract_level_3d(progressive_array, target_shape, sizes):
    index = sizes.index(target_shape)

    chunks = []
    for i, shape in enumerate(sizes):
        if i == 0:
            chunks.append(np.prod(shape))
        else:
            prev_shape = sizes[i - 1]
            added = np.prod(shape) - np.prod(prev_shape)
            chunks.append(added)

    start = sum(chunks[:index])
    end = start + chunks[index]
    flat = progressive_array[start:end]

    if index == 0:
        return flat.reshape(target_shape)

    prev_shape = sizes[index - 1]
    prev_data = extract_level_3d(progressive_array, prev_shape, sizes)

    out = np.zeros(target_shape)
    out[::2, ::2, ::2] = prev_data

    tz, ty, tx = target_shape
    half_z, half_y, half_x = tz // 2, ty // 2, tx // 2
    i = 0

    octant_size = half_z * half_y * half_x

    out[1::2, ::2, ::2] = flat[i : i + octant_size].reshape((half_z, half_y, half_x))
    i += octant_size
    out[::2, 1::2, ::2] = flat[i : i + octant_size].reshape((half_z, half_y, half_x))
    i += octant_size
    out[::2, ::2, 1::2] = flat[i : i + octant_size].reshape((half_z, half_y, half_x))
    i += octant_size
    out[1::2, 1::2, ::2] = flat[i : i + octant_size].reshape((half_z, half_y, half_x))
    i += octant_size
    out[1::2, ::2, 1::2] = flat[i : i + octant_size].reshape((half_z, half_y, half_x))
    i += octant_size
    out[::2, 1::2, 1::2] = flat[i : i + octant_size].reshape((half_z, half_y, half_x))
    i += octant_size
    out[1::2, 1::2, 1::2] = flat[i : i + octant_size].reshape((half_z, half_y, half_x))

    return out


def upscale_with_regular_grid_interpolator(data, target_shape, method="cubic"):
    dims = len(data.shape)

    if dims == 2:
        ny, nx = data.shape
        ty, tx = target_shape

        y_old = np.linspace(0, 1, ny)
        x_old = np.linspace(0, 1, nx)
        # send and recive for parallel
        interpolator = RegularGridInterpolator((y_old, x_old), data, method=method)
        y_new = np.linspace(0, 1, ty)
        x_new = np.linspace(0, 1, tx)

        Y_new, X_new = np.meshgrid(y_new, x_new, indexing="ij")
        points = np.stack([Y_new.ravel(), X_new.ravel()], axis=1)

        result = interpolator(points).reshape(target_shape)

    elif dims == 3:
        nz, ny, nx = data.shape
        tz, ty, tx = target_shape

        z_old = np.linspace(0, 1, nz)
        y_old = np.linspace(0, 1, ny)
        x_old = np.linspace(0, 1, nx)
        # send and recive  for parallel
        interpolator = RegularGridInterpolator(
            (z_old, y_old, x_old), data, method=method
        )

        z_new = np.linspace(0, 1, tz)
        y_new = np.linspace(0, 1, ty)
        x_new = np.linspace(0, 1, tx)

        Z_new, Y_new, X_new = np.meshgrid(z_new, y_new, x_new, indexing="ij")
        points = np.stack([Z_new.ravel(), Y_new.ravel(), X_new.ravel()], axis=1)

        result = interpolator(points).reshape(target_shape)

    else:
        raise ValueError(f"Unsupported dimensionality: {dims}")

    return result


def find_best_resolution(progressive_array, sizes, full_data, error_bound):
    dims = len(full_data.shape)
    best_resolution = sizes[-1]

    for size_or_shape in sizes:
        if dims == 2:
            reconstructed = extract_level_2d(progressive_array, size_or_shape, sizes)
        elif dims == 3:
            reconstructed = extract_level_3d(progressive_array, size_or_shape, sizes)
        else:
            raise ValueError(f"Unsupported dimensionality: {dims}")

        if reconstructed.shape != full_data.shape:
            reconstructed = upscale_with_regular_grid_interpolator(
                reconstructed, full_data.shape, method="cubic"
            )

        l1_error, l2_error, linf_error = calculate_errors(full_data, reconstructed)

        print(
            f"Shape {size_or_shape}: Linf error = {linf_error:.6f}, L1 error = {l1_error:.6f}, L2 error = {l2_error:.6f}"
        )

        if linf_error <= error_bound:
            best_resolution = size_or_shape
            break

    return best_resolution


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Decimates a variable to be smaller by some error bound."
    )
    parser.add_argument(
        "--input", "-in", type=str, required=True, help="First Adios file with data"
    )
    parser.add_argument(
        "--IO_Name1",
        type=str,
        default="reader1",
        help="IO Name for the first Adios file",
    )
    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Variable to decimate seperated by a comma",
    )
    parser.add_argument(
        "--error_bound", "-eb", type=float, required=False, default=0, help="Error tolerance level this is the linf error"
    )
    parser.add_argument(
        "--xml", type=str, default=None, help="Optional ADIOS2 XML configuration"
    )
    parser.add_argument(
        "--Declare_Write_IO",
        help="IO name for writing output",
        required=False,
        default="wio",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="Decimated.bp",
        help="Output BP file for the result",
    )
    parser.add_argument(
        "--min_size", type=int, default=8, help="Minimum resolution level"
    )
    
    parser.add_argument("--level", "-l", type=int, default=None, help="level 0  is the coorest it is done by the lowest level increase by powers of 2", required=False)

    return parser.parse_args()


def main():
    args = parse_arguments()

    r = Reader(args.IO_Name1, args.input, args.xml)
    wrighter = Writer(args.Declare_Write_IO, args.output_file, args.xml)

    while True:
        status = r.begin_step()

        if status != adios2.bindings.StepStatus.OK:
            break
        wrighter.begin_step()
        current_step = r.current_step()
        print(f"Reading step: {int(current_step)}")

        var_names = [v.strip() for v in args.vars.split(",")]
        for var in var_names:
            print(f"  Processing variable: '{var}'")

            r.set_read_vars(var_names)
            data = r.read_step(var)

            if len(data.shape) == 3 and data.shape[0] == 1:
                data = data[0, :, :]

            print(f"Input data shape: {data.shape}")
            dims = len(data.shape)
            
            if args.level is not None:
                max_levels = [int(np.log2(s)) for s in data.shape]
                max_level = min(max_levels) - int(np.log2(args.min_size))

                if args.level < 0 or args.level > max_level:
                    
                    raise ValueError(
                        f"Invalid level {args.level}. Allowed range: 0 to {max_level} "
                        f"based on data shape {data.shape}."
                    )

            
            if dims == 2:
                for i, dim_size in enumerate(data.shape):
                    assert (
                        dim_size & (dim_size - 1)
                    ) == 0, f"Dimension {i} size {dim_size} must be power of two run prep_desimater"
                progressive, sizes = build_progressive_array_2d(
                    data, min_size=args.min_size
                )
            elif dims == 3:
                for i, dim_size in enumerate(data.shape):
                    assert (
                        dim_size & (dim_size - 1)
                    ) == 0, f"Dimension {i} size {dim_size} must be power of two run prep_desimater"
                progressive, sizes = build_progressive_array_3d(
                    data, min_size=args.min_size
                )
            else:
                raise ValueError(f"Unsupported dimensionality: {dims}")
            

            if args.level is not None:
                best_size_or_shape =  sizes[args.level]
            else:
                best_size_or_shape = find_best_resolution(
                    progressive, sizes, data, args.error_bound
                )

            if dims == 2:
                level_data = extract_level_2d(progressive, best_size_or_shape, sizes)
            else:
                level_data = extract_level_3d(progressive, best_size_or_shape, sizes)

            l1_error, l2_error, linf_error = calculate_errors(data, level_data)

            print(f"Final resolution: {best_size_or_shape}")
            print(
                f"Final errors - L1: {l1_error:.6f}, L2: {l2_error:.6f}, Linf: {linf_error:.6f}"
            )

            wrighter.write(var, level_data)
            wrighter.write("l1_error", np.array([l1_error]))
            wrighter.write("l2_error", np.array([l2_error]))
            wrighter.write("linf_error", np.array([linf_error]))
            wrighter.write("error_bound", np.array([args.error_bound]))

            if dims == 2:
                ny, nx = level_data.shape
                grid = pv.ImageData()
                grid.dimensions = (nx, ny, 1)
                grid.origin = (0.0, 0.0, 0.0)
                grid.spacing = (1.0 / (nx - 1), 1.0 / (ny - 1), 1.0)
                grid.point_data["values"] = level_data.flatten(order="F")
                grid.save("output.vtk")
            else:
                nz, ny, nx = level_data.shape
                grid = pv.ImageData()
                grid.dimensions = (nx, ny, nz)
                grid.origin = (0.0, 0.0, 0.0)
                grid.spacing = (1.0 / (nx - 1), 1.0 / (ny - 1), 1.0 / (nz - 1))
                grid.point_data["values"] = level_data.flatten(order="F")
                grid.save("output.vtk")
        r.end_step()
        wrighter.end_step()

    r.close()
    wrighter.close()
    print(f"Decimation completed. Output written to {args.output_file}")


if __name__ == "__main__":
    install()
    main()
