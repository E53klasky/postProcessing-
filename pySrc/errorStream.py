from adios2 import Adios, Stream, bindings
from frechetdist import frdist 
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from rich.traceback import install
from ReaderClass import Reader


# make it work with the offsets for mutle streamlines
# make cleaner/DONE
def RK_visualization(segment_compressed, segment_uncompressed, distance, step=None):
    errors = np.linalg.norm(segment_compressed - segment_uncompressed, axis=1)

    points = segment_compressed.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments, cmap="jet", norm=plt.Normalize(errors.min(), errors.max())
    )
    lc.set_array(errors[:-1])
    lc.set_linewidth(3)

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_aspect("equal")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    if step is not None:
        ax1.set_title(f"Lower Resolution Streamline Colored by Error (Step {step:04d})")
    else:
        ax1.set_title("Lower Resolution Streamline Colored by Error")
    ax1.grid(False)

    cbar = plt.colorbar(lc, ax=ax1)
    cbar.set_label("Error Magnitude")

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    streamline_filename = "highlighted_Lower_Resolution_streamline.png"
    if step is not None:
        streamline_filename = (
            f"highlighted_Lower_Resolution_streamline_step_{step:04d}.png"
        )
    streamline_path = os.path.join(output_dir, streamline_filename)
    print(f"Saving streamline image to: {streamline_path}")

    fig1.savefig(streamline_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(range(len(errors)), errors, marker="o", linestyle="-", color="b")
    plt.yscale("log")
    plt.title(f"Distance Error Plot {distance}")
    plt.xlabel("Point Index")
    plt.ylabel("Error Magnitude")
    plt.grid(True, which="both")
    plt.tight_layout()

    errorplot_filename = "distance_error_plot.png"
    if step is not None:
        errorplot_filename = f"distance_error_plot_step_{step:04d}.png"
    errorplot_path = os.path.join(output_dir, errorplot_filename)
    print(f"Saving error plot image to: {errorplot_path}")

    plt.savefig(errorplot_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

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
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    if step is not None:
        ax3.set_title(
            f"Lower Resolution vs Higher Resolution Streamlines (Step {step:04d})"
        )
    else:
        ax3.set_title("Lower Resolution vs Higher Resolution Streamlines")
    ax3.legend()
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

        segment_compressed_pairs = np.column_stack(
            (segment_compressed_x, segment_compressed_y)
        )
        segment_uncompressed_pairs = np.column_stack(
            (segment_uncompressed_x, segment_uncompressed_y)
        )
        distance = frdist(segment_compressed_pairs, segment_uncompressed_pairs)
        print(f"Distance between segments: {distance}")
        RK_visualization(
            segment_compressed_pairs,
            segment_uncompressed_pairs,
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
