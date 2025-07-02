from adios2 import bindings
from frechetdist import frdist
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rich.traceback import install
from ReaderClass import Reader


# make this also work for 3d
def extract_streamlines_from_segments(x_coords, y_coords, offsets):
    if len(offsets) <= 1:
        print(
            f"Warning: Only {len(offsets)} offset(s) found. Treating all data as one streamline."
        )
        if len(x_coords) > 0:
            streamlines = [np.column_stack((x_coords, y_coords))]
        else:
            streamlines = []
        return streamlines
    # rethink
    # streamlines = []
    # for i in range(len(offsets) - 1):
    #     start = offsets[i]
    #     end = offsets[i + 1]
    #     streamline = np.column_stack((x_coords[start:end], y_coords[start:end]))
    #     streamlines.append(streamline)

    return streamlines


def plot_pointwise_errors(segments_compressed, segments_uncompressed, step=None):

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    if len(segments_compressed) == 0 or len(segments_uncompressed) == 0:
        print("Warning: No streamlines to calculate point-wise errors")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    all_errors = []
    for i, (seg_comp, seg_uncomp) in enumerate(
        zip(segments_compressed, segments_uncompressed)
    ):
        if len(seg_comp) == 0 or len(seg_uncomp) == 0:
            continue

        errors = []
        for point in seg_comp:
            distances_to_points = np.linalg.norm(seg_uncomp - point, axis=1)
            min_error = np.min(distances_to_points)
            errors.append(min_error)

        all_errors.extend(errors)

    if all_errors:
        ax.plot(
            range(len(all_errors)),
            all_errors,
            marker="o",
            markersize=3,
            linestyle="-",
            color="blue",
            alpha=0.7,
        )
        ax.set_yscale("log")
        title = (
            f"Point-wise Error Distribution (Step {step:04d})"
            if step is not None
            else "Point-wise Error Distribution"
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("RK steps", fontsize=12)
        ax.set_ylabel("Error Magnitude", fontsize=12)
        ax.grid(True, which="both")
    else:
        ax.text(
            0.5,
            0.5,
            "No error data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
        )
        ax.set_title("Point-wise Error (No Data)", fontsize=14)

    filename = (
        f"pointwise_errors_step_{step:04d}.png"
        if step is not None
        else "pointwise_errors.png"
    )
    filepath = os.path.join(output_dir, filename)
    print(f"Saving point-wise errors plot to: {filepath}")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return errors


def RK_visualization(segments_compressed, segments_uncompressed, distance, step=None):
    errors = plot_pointwise_errors(
        segments_compressed, segments_uncompressed, step=step
    )

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    compressed_points = (
        np.vstack(segments_compressed)
        if len(segments_compressed) > 0
        else np.empty((0, 2))
    )
    uncompressed_points = (
        np.vstack(segments_uncompressed)
        if len(segments_uncompressed) > 0
        else np.empty((0, 2))
    )

    fig_streamlines, ax_streamlines = plt.subplots(figsize=(10, 8))
    if compressed_points.size > 0:
        ax_streamlines.plot(
            compressed_points[:, 0],
            compressed_points[:, 1],
            linestyle="-",
            color="red",
            label="Lower Res (Compressed)",
        )
    if uncompressed_points.size > 0:
        ax_streamlines.plot(
            uncompressed_points[:, 0],
            uncompressed_points[:, 1],
            linestyle="--",
            color="green",
            label="Higher Res (Uncompressed)",
        )
    ax_streamlines.set_xlabel("X", fontsize=12)
    ax_streamlines.set_ylabel("Y", fontsize=12)
    title_str = (
        f"Streamlines Comparison (Step {step:04d})"
        if step is not None
        else "Streamlines Comparison"
    )
    ax_streamlines.set_title(title_str, fontsize=14)
    ax_streamlines.legend(fontsize=12)
    ax_streamlines.grid(True)
    plt.tight_layout()

    streamlines_fname = (
        f"streamlines_comparison_step_{step:04d}.png"
        if step is not None
        else "streamlines_comparison.png"
    )
    streamlines_path = os.path.join(output_dir, streamlines_fname)
    print(f"Saving streamlines plot to: {streamlines_path}")
    fig_streamlines.savefig(streamlines_path, dpi=300, bbox_inches="tight")
    plt.close(fig_streamlines)

    fig_combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    if errors:
        ax1.plot(
            range(len(errors)),
            errors,
            marker="o",
            markersize=3,
            linestyle="-",
            color="blue",
        )
        ax1.set_yscale("log")
    ax1.set_title("Point-wise RK Error", fontsize=14)
    ax1.set_xlabel("RK steps", fontsize=12)
    ax1.set_ylabel("Error Magnitude", fontsize=12)
    ax1.grid(True, which="both")

    if compressed_points.size > 0:
        ax2.plot(
            compressed_points[:, 0],
            compressed_points[:, 1],
            linestyle="-",
            color="red",
            label="Lower Res (Compressed)",
        )
    if uncompressed_points.size > 0:
        ax2.plot(
            uncompressed_points[:, 0],
            uncompressed_points[:, 1],
            linestyle="--",
            color="green",
            label="Higher Res (Uncompressed)",
        )
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("Y", fontsize=12)
    ax2.set_title("Streamlines Comparison", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True)

    plt.tight_layout()

    combined_fname = (
        f"combined_plot_step_{step:04d}.png"
        if step is not None
        else "combined_plot.png"
    )
    combined_path = os.path.join(output_dir, combined_fname)
    print(f"Saving combined plot to: {combined_path}")
    fig_combined.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig_combined)

    return errors


def parse_arguments():
    install()
    parser = argparse.ArgumentParser(
        description="Calculating the error of streamlines given the segments"
    )

    parser.add_argument(
        "--file1",
        type=str,
        required=True,
        help="First Adios file with streamline segments (lower resolution/compressed)",
    )
    parser.add_argument(
        "--file2",
        type=str,
        required=True,
        help="Second Adios file with streamline segments (higher resoltuion)",
    )
    parser.add_argument(
        "--IO_Name1",
        type=str,
        default="reader1",
        help="IO Name for the first Adios file (default: reader1)",
    )
    parser.add_argument(
        "--IO_Name2",
        type=str,
        default="reader2",
        help="IO Name for the first Adios file (default: reader1)",
    )

    parser.add_argument(
        "--xml", "-x", type=str, default=None, help="ADIOS2 XML config file (optional)"
    )

    parser.add_argument(
        "--var_x", type=str, required=True, help="Variable name for x coordinates"
    )
    parser.add_argument(
        "--var_y", type=str, required=True, help="Variable name for y coordinates"
    )
    # rember to find out how to use the offsets
    parser.add_argument(
        "--var_offset", type=str, required=True, help="Variable name for offsets"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    r_low = Reader(args.IO_Name1, args.file1, xml=args.xml)
    r_high = Reader(args.IO_Name2, args.file2, xml=args.xml)

    while True:
        status_low = r_low.begin_step()
        status_high = r_high.begin_step()

        if (
            bindings.StepStatus.OK != status_low
            or bindings.StepStatus.OK != status_high
        ):
            break
        current_step = r_low.current_step()
        print(f"Reading step: {int(current_step)}")

        r_high.set_read_vars([args.var_x, args.var_y, args.var_offset])
        r_low.set_read_vars([args.var_x, args.var_y, args.var_offset])

        if (
            r_low.vars_Out.get(args.var_x) is None
            or r_low.vars_Out.get(args.var_y) is None
            or r_low.vars_Out.get(args.var_offset) is None
        ):
            print("Variables not found in the low resolution stream.")
            break

        segment_compressed_x = r_low.read_step(args.var_x)
        segment_compressed_y = r_low.read_step(args.var_y)
        segment_compressed_offset = r_low.read_step(args.var_offset)

        segment_uncompressed_x = r_high.read_step(args.var_x)
        segment_uncompressed_y = r_high.read_step(args.var_y)
        segment_uncompressed_offset = r_high.read_step(args.var_offset)

        segment_compressed_pairs = extract_streamlines_from_segments(
            segment_compressed_x, segment_compressed_y, segment_compressed_offset
        )

        segment_uncompressed_pair = extract_streamlines_from_segments(
            segment_uncompressed_x, segment_uncompressed_y, segment_uncompressed_offset
        )

        distance = frdist(segment_compressed_pairs, segment_uncompressed_pair)
        print(f"Distance between segments: {distance}")
        RK_visualization(
            segment_compressed_pairs,
            segment_uncompressed_pair,
            distance,
            step=current_step,
        )

        r_low.end_step()
        r_high.end_step()

    r_low.close()
    r_high.close()
    print("Finished ErrorStream.py successfully!")


if __name__ == "__main__":
    main()
    install()
