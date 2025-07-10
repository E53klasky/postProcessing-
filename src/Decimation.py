import numpy as np
from WrighterClass import Writer
from ReaderClass import Reader
from rich.traceback import install
import adios2
import argparse
from scipy.ndimage import zoom

# this really does not work  
# TODO: fix error wright out, gernalize to mxn not 2^m x2^n,3d, parallel

def calculate_errors(original, reconstructed):
    # this does not work   
# --------------------------------------------------------------------
    skip_factor = original.shape[0] / reconstructed.shape[0]
    diff = np.zeros_like(reconstructed, dtype=np.float64)
    for i in range(reconstructed.shape[0]):
        for j in range(reconstructed.shape[1]):
            gt_i = int(i * skip_factor)
            gt_j = int(j * skip_factor)
            if gt_i < original.shape[0] and gt_j < original.shape[1]:
                gt_value = original[gt_i, gt_j]
                e_value = reconstructed[i, j]
                diff[i, j] = np.abs(gt_value - e_value)
            else:
                diff[i, j] = np.nan
# -----------------------------------------------------------------
    l1_error = np.sum(diff)
    l2_error = np.sqrt(np.sum(diff**2))
    linf_error = np.nanmax(diff)
    return linf_error, l1_error, l2_error, linf_error


def build_progressive_array_general(data, min_size=8):
    H, W = data.shape
    h_sizes = []
    w_sizes = []

    h, w = H, W
    while h >= min_size and w >= min_size:
        h_sizes.append(h)
        w_sizes.append(w)
        h //= 2
        w //= 2

    h_sizes.reverse()
    w_sizes.reverse()

    output_chunks = []
    for i, (h_size, w_size) in enumerate(zip(h_sizes, w_sizes)):
        step_h = H // h_size
        step_w = W // w_size
        sampled = data[::step_h, ::step_w]

        if i == 0:
            output_chunks.append(sampled.flatten())
        else:
            new_points = []
            new_points.append(sampled[1::2, ::2].flatten())
            new_points.append(sampled[::2, 1::2].flatten())
            new_points.append(sampled[1::2, 1::2].flatten())
            output_chunks.append(np.concatenate(new_points))
    return np.concatenate(output_chunks), list(zip(h_sizes, w_sizes))


def find_best_resolution(progressive_array, shape_seq, full_data, error_bound):
    for shape in shape_seq:
        reconstructed = extract_level_general(progressive_array, shape, shape_seq)

        if reconstructed.shape != full_data.shape:

            zoom_factors = (
                full_data.shape[0] / reconstructed.shape[0],
                full_data.shape[1] / reconstructed.shape[1],
            )
            resized = zoom(reconstructed, zoom_factors, order=1)
        else:
            resized = reconstructed

        error, l1_error, l2_error, linf_error = calculate_errors(full_data, resized)
        print(f"Shape {shape}: error = {error:.6f}")
        print(f"The L1 error = {l1_error}")
        print(f"the L2 error = {l2_error} ")
        print(f"the L inf error = {linf_error} ")

        if error <= error_bound:
            return shape

    return shape_seq[-1]


def extract_level_general(progressive_array, target_shape, shape_sequence, min_size=8):
    target_h, target_w = target_shape
    sizes = shape_sequence
    index = sizes.index((target_h, target_w))

    chunks = []
    for i, (h_size, w_size) in enumerate(sizes):
        if i == 0:
            chunks.append(h_size * w_size)
        else:
            prev = sizes[i - 1]
            added = h_size * w_size - prev[0] * prev[1]
            chunks.append(added)

    start = sum(chunks[:index])
    end = start + chunks[index]
    flat = progressive_array[start:end]

    if index == 0:
        return flat.reshape(target_h, target_w)

    prev_h, prev_w = sizes[index - 1]
    prev_data = extract_level_general(
        progressive_array, (prev_h, prev_w), shape_sequence, min_size
    )

    out = np.zeros((target_h, target_w))
    out[::2, ::2] = prev_data

    i = 0
    h_half = target_h // 2
    w_half = target_w // 2

    out[1::2, ::2] = flat[i : i + h_half * w_half].reshape((h_half, w_half))
    i += h_half * w_half
    out[::2, 1::2] = flat[i : i + h_half * w_half].reshape((h_half, w_half))
    i += h_half * w_half
    out[1::2, 1::2] = flat[i : i + h_half * w_half].reshape((h_half, w_half))
    return out


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Decimates a variable to be smaller by some error bound."
    )
    parser.add_argument(
        "--file1",
        type=str,
        required=True,
        help="First Adios file with data",
    )
    parser.add_argument(
        "--IO_Name1",
        type=str,
        default="reader1",
        help="IO Name for the first Adios file (default: reader1)",
    )
    parser.add_argument(
        "--var",
        "-v",
        type=str,
        required=True,
        help="Variable to decimate",
    )
    parser.add_argument(
        "--error_bound", "-eb", type=float, required=True, help="Error tolerance level"
    )
    parser.add_argument(
        "--xml", type=str, default=None, help="Optional ADIOS2 XML configuration"
    )
    parser.add_argument(
        "--Declare_Write_IO", help="IO name for writing output", required=True
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="Decimated.bp",
        help="Output BP file for the result",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    var = args.var
    error_bound = args.error_bound

    r = Reader(args.IO_Name1, args.file1, args.xml)
    wrighter = Writer(args.Declare_Write_IO, args.output_file, args.xml)

    while True:
        status = r.begin_step()

        if status != adios2.bindings.StepStatus.OK:
            break

        current_step = r.current_step()
        print(f"Reading step: {int(current_step)}")

        r.set_read_vars([var])
        data = r.read_step(var)
        # add logic here to make it 2d  also may die bc of the -1
        if len(data.shape) == 3 and data.shape[0] == 1:
            data = data[0, :, :]
        h, w = data.shape
        trimmed_h = h if h % 2 == 0 else h - 1
        trimmed_w = w if w % 2 == 0 else w - 1
        data = data[:trimmed_h, :trimmed_w]

        progressive, shape_seq = build_progressive_array_general(data, min_size=8)

        error_bound = args.error_bound
        best_shape = find_best_resolution(progressive, shape_seq, data, error_bound)

        level_data = extract_level_general(
            progressive,
            best_shape,
            shape_sequence=shape_seq,
            min_size=8,
        )

        diff, l1, l2, linf = calculate_errors(data, level_data)

        wrighter.begin_step()
        wrighter.write(var, level_data)
        # this does not work -------------------------------------------------
        wrighter.write("l1_error", np.array([l1]))
        wrighter.write("l2_error", np.array([l2]))
        wrighter.write("linf_error", np.array([linf]))
        wrighter.write("error_bound", np.array([error_bound]))
        # this ----------------------------------------------------------------
        wrighter.end_step()

        r.end_step()

    r.close()
    wrighter.close()
    print(f"Decimation completed. Output written to {args.output_file}")


if __name__ == "__main__":
    install()
    main()
