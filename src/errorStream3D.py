import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
import argparse
import os
from ReaderClass import Reader
from adios2 import bindings


def calculate_spline_distance(streamline1, streamline2, num_points=2000):
    try:
        x1, y1, z1 = streamline1[:, 0], streamline1[:, 1], streamline1[:, 2]
        x2, y2, z2 = streamline2[:, 0], streamline2[:, 1], streamline2[:, 2]

        if len(x1) < 4 or len(x2) < 4:
            print(
                f"Not enough points for spline interpolation (need ≥4, got {len(x1)}, {len(x2)})"
            )
            min_len = min(len(streamline1), len(streamline2))
            return np.mean(
                np.linalg.norm(streamline1[:min_len] - streamline2[:min_len], axis=1)
            )

        if (
            np.any(np.isnan(streamline1))
            or np.any(np.isinf(streamline1))
            or np.any(np.isnan(streamline2))
            or np.any(np.isinf(streamline2))
        ):
            print("NaN or infinite values detected in streamlines")
            min_len = min(len(streamline1), len(streamline2))
            return np.mean(
                np.linalg.norm(streamline1[:min_len] - streamline2[:min_len], axis=1)
            )

        def remove_duplicates(coords):
            unique_mask = np.ones(len(coords[0]), dtype=bool)
            for i in range(1, len(coords[0])):
                if np.allclose(
                    [coords[0][i], coords[1][i], coords[2][i]],
                    [coords[0][i - 1], coords[1][i - 1], coords[2][i - 1]],
                    rtol=1e-10,
                ):
                    unique_mask[i] = False
            return [coord[unique_mask] for coord in coords]

        coords1 = remove_duplicates([x1, y1, z1])
        coords2 = remove_duplicates([x2, y2, z2])

        if len(coords1[0]) < 4 or len(coords2[0]) < 4:
            print("Not enough unique points after removing duplicates")
            min_len = min(len(streamline1), len(streamline2))
            return np.mean(
                np.linalg.norm(streamline1[:min_len] - streamline2[:min_len], axis=1)
            )

        smoothing_factors = [0.0, 0.1, 0.5, 1.0, 5.0]

        for s_factor in smoothing_factors:
            try:
                tck1, _ = splprep(coords1, s=s_factor, k=3)
                tck2, _ = splprep(coords2, s=s_factor, k=3)

                u = np.linspace(0, 1, num_points)
                points1 = np.array(splev(u, tck1)).T
                points2 = np.array(splev(u, tck2)).T

                distances = np.linalg.norm(points1 - points2, axis=1)
                mean_distance = np.mean(distances)

                if not np.isnan(mean_distance) and not np.isinf(mean_distance):
                    print(
                        f"Spline interpolation successful with smoothing factor {s_factor}"
                    )
                    return mean_distance

            except Exception as spline_error:
                print(f"Spline failed with smoothing factor {s_factor}: {spline_error}")
                continue

        print("All spline interpolation attempts failed, using direct comparison")
        min_len = min(len(streamline1), len(streamline2))
        return np.mean(
            np.linalg.norm(streamline1[:min_len] - streamline2[:min_len], axis=1)
        )

    except Exception as e:
        print(f"Spline calculation failed: {e}")
        min_len = min(len(streamline1), len(streamline2))
        return np.mean(
            np.linalg.norm(streamline1[:min_len] - streamline2[:min_len], axis=1)
        )


def extract_streamlines(x_coords, y_coords, z_coords, offsets):
    streamlines = []

    if len(offsets) <= 2:
        streamlines.append(np.column_stack((x_coords, y_coords, z_coords)))
    else:

        offsets = np.sort(offsets)
        for i in range(len(offsets) - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            streamline = np.column_stack(
                (x_coords[start:end], y_coords[start:end], z_coords[start:end])
            )
            streamlines.append(streamline)

    return streamlines


def plot_rk_step_error(
    streamlines_low, streamlines_high, step=None, spline_distance=None
):

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    all_errors = []

    min_streams = min(len(streamlines_low), len(streamlines_high))

    for i in range(min_streams):
        low_stream = streamlines_low[i]
        high_stream = streamlines_high[i]

        if len(low_stream) == 0 or len(high_stream) == 0:
            continue

        min_points = min(len(low_stream), len(high_stream))

        for j in range(min_points):
            error = np.linalg.norm(low_stream[j] - high_stream[j])
            all_errors.append(error)

        if len(low_stream) > len(high_stream):
            for j in range(min_points, len(low_stream)):
                distances = np.linalg.norm(high_stream - low_stream[j], axis=1)
                min_error = np.min(distances)
                all_errors.append(min_error)
        elif len(high_stream) > len(low_stream):
            for j in range(min_points, len(high_stream)):
                distances = np.linalg.norm(low_stream - high_stream[j], axis=1)
                min_error = np.min(distances)
                all_errors.append(min_error)

    fig, ax = plt.subplots(figsize=(12, 8))

    if all_errors:
        ax.plot(range(len(all_errors)), all_errors, "o-", markersize=3, color="blue")
        ax.set_yscale("log")

        title = f"RK Step Errors"
        if step is not None:
            title += f" (Step {step})"
        if spline_distance is not None:
            title += f" - Spline Distance: {spline_distance:.6f}"

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("RK Steps", fontsize=12)
        ax.set_ylabel("Error Magnitude", fontsize=12)
        ax.grid(True)
    else:
        ax.text(
            0.5,
            0.5,
            "No error data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
        )
        ax.set_title("RK Step Errors (No Data)", fontsize=14)

    filename = (
        f"RK_step_errors_step_{step}.png" if step is not None else "RK_step_errors.png"
    )
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved RK step error plot to: {filepath}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors_comp = plt.cm.Reds(np.linspace(0.4, 0.9, len(streamlines_low)))
    colors_uncomp = plt.cm.Greens(np.linspace(0.4, 0.9, len(streamlines_high)))

    for i, segment in enumerate(streamlines_low):
        if len(segment) > 0:
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                linestyle="-",
                color=colors_comp[i],
                linewidth=2,
                label=f"low res {i+1}" if i < 5 else "",
            )

    for i, segment in enumerate(streamlines_high):
        if len(segment) > 0:
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                linestyle="--",
                color=colors_uncomp[i],
                linewidth=1.5,
                label=f"high res {i+1}" if i < 5 else "",
            )

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)

    title_str = (
        f"3D Streamlines Comparison - {len(streamlines_low)} streamlines (Step {step:04d})"
        if step is not None
        else f"3D Streamlines Comparison - {len(streamlines_high)} streamlines"
    )
    ax.set_title(title_str, fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True)

    plt.tight_layout()

    streamlines_fname = (
        f"3D_streamlines_comparison_step_{step:04d}.png"
        if step is not None
        else "3D_streamlines_comparison.png"
    )
    streamlines_path = os.path.join(output_dir, streamlines_fname)
    print(f"Saving 3D streamlines plot to: {streamlines_path}")
    fig.savefig(streamlines_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return all_errors


def main():
    parser = argparse.ArgumentParser(
        description="Simplified 3D streamline error analysis"
    )
    parser.add_argument("--file1", required=True, help="Low resolution file")
    parser.add_argument("--file2", required=True, help="High resolution file")
    parser.add_argument("--var_x", required=True, help="X coordinate variable")
    parser.add_argument("--var_y", required=True, help="Y coordinate variable")
    parser.add_argument("--var_z", required=True, help="Z coordinate variable")
    parser.add_argument("--var_offset", required=True, help="Offset variable")
    parser.add_argument("--IO_Name1", default="reader1", help="IO name for file1")
    parser.add_argument("--IO_Name2", default="reader2", help="IO name for file2")
    parser.add_argument("--xml", help="ADIOS2 XML config file")
    parser.add_argument(
        "--spline_points",
        type=int,
        default=2000,
        help="Number of points for spline interpolation (default: 2000)",
    )

    args = parser.parse_args()

    r_low = Reader(args.IO_Name1, args.file1, xml=args.xml)
    r_high = Reader(args.IO_Name2, args.file2, xml=args.xml)

    try:
        while True:
            status_low = r_low.begin_step()
            status_high = r_high.begin_step()

            if (
                status_low != bindings.StepStatus.OK
                or status_high != bindings.StepStatus.OK
            ):
                print("End of stream reached")
                break

            current_step = r_low.current_step()
            print(f"Processing step: {current_step}")

            vars_to_read = [args.var_x, args.var_y, args.var_z, args.var_offset]
            r_low.set_read_vars(vars_to_read)
            r_high.set_read_vars(vars_to_read)

            x_low = r_low.read_step(args.var_x)
            y_low = r_low.read_step(args.var_y)
            z_low = r_low.read_step(args.var_z)
            offset_low = r_low.read_step(args.var_offset)

            x_high = r_high.read_step(args.var_x)
            y_high = r_high.read_step(args.var_y)
            z_high = r_high.read_step(args.var_z)
            offset_high = r_high.read_step(args.var_offset)

            streamlines_low = extract_streamlines(x_low, y_low, z_low, offset_low)
            streamlines_high = extract_streamlines(x_high, y_high, z_high, offset_high)

            print(
                f"Found {len(streamlines_low)} low-res and {len(streamlines_high)} high-res streamlines"
            )

            if len(streamlines_low) > 0 and len(streamlines_high) > 0:
                low_bounds = np.array(
                    [
                        [
                            streamlines_low[0][:, 0].min(),
                            streamlines_low[0][:, 0].max(),
                        ],
                        [
                            streamlines_low[0][:, 1].min(),
                            streamlines_low[0][:, 1].max(),
                        ],
                        [
                            streamlines_low[0][:, 2].min(),
                            streamlines_low[0][:, 2].max(),
                        ],
                    ]
                )
                high_bounds = np.array(
                    [
                        [
                            streamlines_high[0][:, 0].min(),
                            streamlines_high[0][:, 0].max(),
                        ],
                        [
                            streamlines_high[0][:, 1].min(),
                            streamlines_high[0][:, 1].max(),
                        ],
                        [
                            streamlines_high[0][:, 2].min(),
                            streamlines_high[0][:, 2].max(),
                        ],
                    ]
                )

                print(f"Low-res streamline bounds:")
                print(f"  X: [{low_bounds[0,0]:.6f}, {low_bounds[0,1]:.6f}]")
                print(f"  Y: [{low_bounds[1,0]:.6f}, {low_bounds[1,1]:.6f}]")
                print(f"  Z: [{low_bounds[2,0]:.6f}, {low_bounds[2,1]:.6f}]")

                print(f"High-res streamline bounds:")
                print(f"  X: [{high_bounds[0,0]:.6f}, {high_bounds[0,1]:.6f}]")
                print(f"  Y: [{high_bounds[1,0]:.6f}, {high_bounds[1,1]:.6f}]")
                print(f"  Z: [{high_bounds[2,0]:.6f}, {high_bounds[2,1]:.6f}]")

                print(f"First few points from low-res streamline:")
                for i in range(min(5, len(streamlines_low[0]))):
                    pt = streamlines_low[0][i]
                    print(f"  Point {i}: ({pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f})")

                print(f"First few points from high-res streamline:")
                for i in range(min(5, len(streamlines_high[0]))):
                    pt = streamlines_high[0][i]
                    print(f"  Point {i}: ({pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f})")

            spline_distance = None
            if len(streamlines_low) > 0 and len(streamlines_high) > 0:
                spline_distance = calculate_spline_distance(
                    streamlines_low[0], streamlines_high[0], args.spline_points
                )
                print(f"Spline distance: {spline_distance:.6f}")

            errors = plot_rk_step_error(
                streamlines_low, streamlines_high, current_step, spline_distance
            )

            if errors:
                print(f"Error statistics:")
                print(f"  Number of error points: {len(errors)}")
                print(f"  Min error: {np.min(errors):.6f}")
                print(f"  Max error: {np.max(errors):.6f}")
                print(f"  Mean error: {np.mean(errors):.6f}")
                print(f"  Median error: {np.median(errors):.6f}")
            else:
                print("No errors calculated")

            r_low.end_step()
            r_high.end_step()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        r_low.close()
        r_high.close()

    print("Finished processing!")


if __name__ == "__main__":
    main()
