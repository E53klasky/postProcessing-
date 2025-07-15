from adios2 import bindings
from frechetdist import frdist
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rich.traceback import install
from ReaderClass import Reader
from matplotlib.collections import LineCollection
from scipy.interpolate import splprep, splev


# make this work with 3d maybe
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

    L = np.max(offsets)
    n_points_per_streamline = L
    n_streamlines = len(x_coords) // n_points_per_streamline

    streamlines = []
    for i in range(n_streamlines):
        start = i * n_points_per_streamline
        end = start + n_points_per_streamline
        streamline = np.column_stack((x_coords[start:end], y_coords[start:end]))
        streamlines.append(streamline)

    return streamlines


def plot_pointwise_errors(
    segments_compressed, segments_uncompressed, step=None, spline_distances=None
):

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
        )
        ax.set_yscale("log")

        if spline_distances is not None and len(spline_distances) > 0:
            distances_str = ", ".join([f"{d:.6f}" for d in spline_distances])
            title = (
                f"RK step errors at (Step {step:04d}) - Spline Distances: [{distances_str}]"
                if step is not None
                else f"RK step errors - Spline Distances: [{distances_str}]"
            )
        else:
            title = (
                f"RK step errors at (Step {step:04d})"
                if step is not None
                else "RK step errors"
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
        ax.set_title("RK step errors (No Data)", fontsize=14)

    filename = (
        f"RK_step_erros_at_step_{step:04d}.png"
        if step is not None
        else "RK_step_erros.png"
    )
    filepath = os.path.join(output_dir, filename)
    print(f"Saving RK step errors plot to: {filepath}")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return errors


def RK_visualization(
    segments_compressed,
    segments_uncompressed,
    distances,
    step=None,
    spline_distances=None,
):
    errors = plot_pointwise_errors(
        segments_compressed,
        segments_uncompressed,
        step=step,
        spline_distances=spline_distances,
    )

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    fig_streamlines, ax_streamlines = plt.subplots(figsize=(12, 10))

    colors_comp = plt.cm.Reds(np.linspace(0.4, 0.9, len(segments_compressed)))
    colors_uncomp = plt.cm.Greens(np.linspace(0.4, 0.9, len(segments_uncompressed)))

    for i, segment in enumerate(segments_compressed):
        if len(segment) > 0:
            ax_streamlines.plot(
                segment[:, 0],
                segment[:, 1],
                linestyle="-",
                color=colors_comp[i],
                linewidth=2,
                label=f"low res {i+1}" if i < 5 else "",
            )

    for i, segment in enumerate(segments_uncompressed):
        if len(segment) > 0:
            ax_streamlines.plot(
                segment[:, 0],
                segment[:, 1],
                linestyle="--",
                color=colors_uncomp[i],
                linewidth=1.5,
                label=f"high res {i+1}" if i < 5 else "",
            )

    ax_streamlines.set_xlabel("X", fontsize=12)
    ax_streamlines.set_ylabel("Y", fontsize=12)
    title_str = (
        f"Streamlines Comparison - {len(segments_compressed)} streamlines (Step {step:04d})"
        if step is not None
        else f"Streamlines Comparison - {len(segments_compressed)} streamlines"
    )
    ax_streamlines.set_title(title_str, fontsize=14)
    ax_streamlines.legend(fontsize=10, loc="best")
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

    for idx, (comp_segment, uncomp_segment) in enumerate(
        zip(segments_compressed, segments_uncompressed)
    ):
        if len(comp_segment) > 1 and len(uncomp_segment) > 0:

            pointwise_errors = []
            for pt in comp_segment:
                dists = np.linalg.norm(uncomp_segment - pt, axis=1)
                pointwise_errors.append(np.min(dists))
            pointwise_errors = np.array(pointwise_errors)

            segments = [
                [comp_segment[i], comp_segment[i + 1]]
                for i in range(len(comp_segment) - 1)
            ]
            line_colors = pointwise_errors[:-1]

            fig, ax = plt.subplots(figsize=(10, 8))
            lc = LineCollection(segments, cmap="jet", array=line_colors, linewidths=3)
            line = ax.add_collection(lc)

            cbar = plt.colorbar(line, ax=ax)
            cbar.set_label("RK step Error", fontsize=12)

            ax.plot(
                uncomp_segment[:, 0],
                uncomp_segment[:, 1],
                linestyle="--",
                color="black",
                linewidth=1.5,
                label="High res Reference",
            )

            ax.set_xlim(
                np.min(comp_segment[:, 0]) - 0.1, np.max(comp_segment[:, 0]) + 0.1
            )
            ax.set_ylim(
                np.min(comp_segment[:, 1]) - 0.1, np.max(comp_segment[:, 1]) + 0.1
            )
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            title_str = (
                f"Streamline {idx+1} Highlighted by Error (Step {step:04d})"
                if step is not None
                else f"Streamline {idx+1} Highlighted by Error"
            )
            ax.set_title(title_str, fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True)
            plt.tight_layout()

            fname = (
                f"streamline_{idx+1}_highlighted_error_step_{step:04d}.png"
                if step is not None
                else f"streamline_{idx+1}_highlighted_error.png"
            )
            path = os.path.join(output_dir, fname)
            print(f"Saving error-highlighted streamline {idx+1} to: {path}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    return errors


def parse_arguments():
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

    parser.add_argument(
        "--var_offset", type=str, required=True, help="Variable name for offsets"
    )

    parser.add_argument(
        "--num_spline",
        "-N",
        default=1000,
        type=int,
        required=True,
        help="Number of spile points to interploate",
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

        num_streamlines = len(segment_uncompressed_pair)
        print(f"Number of streamlines: {num_streamlines}")

        spline_distances = []
        for i in range(num_streamlines):
            if (
                i < len(segment_compressed_pairs)
                and len(segment_compressed_pairs[i]) > 1
                and len(segment_uncompressed_pair[i]) > 1
            ):
                try:

                    comp_x = segment_compressed_pairs[i][:, 0]
                    comp_y = segment_compressed_pairs[i][:, 1]
                    uncomp_x = segment_uncompressed_pair[i][:, 0]
                    uncomp_y = segment_uncompressed_pair[i][:, 1]

                    tck0, u0 = splprep([uncomp_x, uncomp_y], s=0)
                    tck1, u1 = splprep([comp_x, comp_y], s=0)

                    N = args.num_spline
                    u_fine = np.linspace(0, 1, N)
                    x0_fine, y0_fine = splev(u_fine, tck0)
                    x1_fine, y1_fine = splev(u_fine, tck1)

                    diffx = x0_fine - x1_fine
                    diffy = y0_fine - y1_fine

                    diffx = diffx * diffx
                    diffy = diffy * diffy
                    d = np.sum(np.sqrt(diffx + diffy)) / float(N)
                    spline_distances.append(d)
                    print(f"Spline distance for streamline {i}: {d}")
                except Exception as e:
                    print(f"Error calculating spline distance for streamline {i}: {e}")
                    spline_distances.append(0.0)
            else:
                spline_distances.append(0.0)
                print(f"Insufficient data for spline calculation for streamline {i}")

        distances = []
        for i in range(num_streamlines):
            distance = 0
            distances.append(distance)
            print(f"Distance between streamline {i}: {distance}")

        error = RK_visualization(
            segment_compressed_pairs,
            segment_uncompressed_pair,
            distances,
            step=current_step,
            spline_distances=spline_distances,
        )

        r_low.end_step()
        r_high.end_step()

    r_low.close()
    r_high.close()
    print(f"Finished ErrorStream.py successfully!")


if __name__ == "__main__":
    install()
    main()