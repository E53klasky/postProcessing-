from adios2 import bindings
from frechetdist import frdist
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from rich.traceback import install
from ReaderClass import Reader


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
        ax.set_xlabel("Point Index (All Streamlines)", fontsize=12)
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


def RK_visualization(segment_compressed, segment_uncompressed, distance, step=None):

    errors = plot_pointwise_errors(segment_compressed, segment_uncompressed, step=step)
    # fix this to be two other plots one of the segments plottwed both of them and one from the errors two soerate plots

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    streamline_filename = "highlighted_Lower_Resolution_streamline.png"
    if step is not None:
        streamline_filename = (
            f"highlighted_Lower_Resolution_streamline_step_{step:04d}.png"
        )
    streamline_path = os.path.join(output_dir, streamline_filename)
    print(f"Saving streamline image to: {streamline_path}")

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(range(len(errors)), errors, marker="o", linestyle="-", color="b")
    plt.yscale("log")
    plt.title(f"RK Error Plot {distance}", fontsize=12)
    plt.xlabel("RK steps", fontsize=12)
    plt.ylabel("Error Magnitude", fontsize=12)

    plt.tick_params(axis="both", labelsize=12)

    plt.grid(True, which="both")
    plt.tight_layout()

    errorplot_filename = "distance_error_plot.png"
    if step is not None:
        errorplot_filename = f"distance_error_plot_step_{step:04d}.png"
    errorplot_path = os.path.join(output_dir, errorplot_filename)
    print(f"Saving error plot image to: {errorplot_path}")

    plt.savefig(errorplot_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    # ----------------------------------------------------
    segment_compressed = np.array(segment_compressed)
    segment_uncompressed = np.array(segment_uncompressed)
    # ----------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.plot(
        segment_compressed[:, 0],
        segment_compressed[:, 1],
        linestyle="-",
        color="red",
        label="Lower Res",
    )
    ax3.plot(
        segment_uncompressed[:, 0],
        segment_uncompressed[:, 1],
        linestyle="--",
        color="green",
        label="Higher res",
    )

    ax3.set_xlabel("X", fontsize=12)
    ax3.set_ylabel("Y", fontsize=12)
    if step is not None:
        ax3.set_title(
            f"Lower Resolution vs Higher Resolution Streamlines (Step {step:04d})",
            fontsize=12,
        )
    else:
        ax3.set_title("Lower Resolution vs Higher Resolution Streamlines", fontsize=12)
    ax3.legend(fontsize=12)
    ax3.tick_params(axis="both", labelsize=12)
    ax3.grid(True)

    streamline_comparison_filename = "streamline_comparison.png"
    if step is not None:
        streamline_comparison_filename = f"streamline_comparison_step_{step:04d}.png"
    streamline_comparison_path = os.path.join(
        output_dir, streamline_comparison_filename
    )
    print(f"Saving streamline comparison image to: {streamline_comparison_path}")

    fig3.savefig(
        streamline_comparison_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig3)


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
            step=r_low.current_step,
        )

        r_low.end_step()
        r_high.end_step()

    r_low.close()
    r_high.close()
    print("Finished ErrorStream.py successfully!")


if __name__ == "__main__":
    main()
    install()
