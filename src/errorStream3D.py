from adios2 import bindings
from frechetdist import frdist
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rich.traceback import install
from ReaderClass import Reader
from matplotlib.collections import LineCollection
from scipy.interpolate import splprep, splev


def calculate_spline_distance(comp_streamline, uncomp_streamline, num_spline_points=1000):
    try:
        comp_x, comp_y, comp_z = comp_streamline[:, 0], comp_streamline[:, 1], comp_streamline[:, 2]
        uncomp_x, uncomp_y, uncomp_z = uncomp_streamline[:, 0], uncomp_streamline[:, 1], uncomp_streamline[:, 2]
        
        if len(comp_x) < 4 or len(uncomp_x) < 4:
            print(f"Not enough points for spline interpolation (comp: {len(comp_x)}, uncomp: {len(uncomp_x)})")
            return calculate_direct_distance(comp_streamline, uncomp_streamline)
        
        for coords, name in [(uncomp_x, uncomp_y, uncomp_z, "uncompressed"), (comp_x, comp_y, comp_z, "compressed")]:
            ranges = [np.max(coord) - np.min(coord) for coord in coords]
            if all(r < 1e-6 for r in ranges):
                print(f"Insufficient coordinate variation in {name} streamline, using direct distance")
                return calculate_direct_distance(comp_streamline, uncomp_streamline)
        
        smoothing_factor = max(len(uncomp_x) * 0.001, 0.1)

        tck_uncomp, _ = splprep([uncomp_x, uncomp_y, uncomp_z], s=smoothing_factor)
        tck_comp, _ = splprep([comp_x, comp_y, comp_z], s=smoothing_factor)
        
        u_fine = np.linspace(0, 1, num_spline_points)
        
        x_uncomp, y_uncomp, z_uncomp = splev(u_fine, tck_uncomp)
        x_comp, y_comp, z_comp = splev(u_fine, tck_comp)
        

        distances = np.sqrt((x_uncomp - x_comp)**2 + (y_uncomp - y_comp)**2 + (z_uncomp - z_comp)**2)

        return np.mean(distances)
        
    except Exception as e:
        print(f"Spline fitting failed: {e}, using direct distance")
        return calculate_direct_distance(comp_streamline, uncomp_streamline)


def calculate_direct_distance(comp_streamline, uncomp_streamline):
    try:
        min_len = min(len(comp_streamline), len(uncomp_streamline))
        
        comp_points = comp_streamline[:min_len]
        uncomp_points = uncomp_streamline[:min_len]
        
        distances = np.linalg.norm(comp_points - uncomp_points, axis=1)
        
        return np.mean(distances)
        
    except Exception as e:
        print(f"Direct distance calculation failed: {e}")
        return 0.0


def calculate_all_spline_distances(segment_compressed_pairs, segment_uncompressed_pair, num_spline):
    num_streamlines = len(segment_uncompressed_pair)
    spline_distances = []
    
    for i in range(num_streamlines):
        if (i < len(segment_compressed_pairs) and 
            len(segment_compressed_pairs[i]) > 1 and 
            len(segment_uncompressed_pair[i]) > 1):
            
            distance = calculate_spline_distance(
                segment_compressed_pairs[i], 
                segment_uncompressed_pair[i], 
                num_spline
            )
            spline_distances.append(distance)
            print(f"3D Spline distance for streamline {i}: {distance:.6f}")
        else:
            spline_distances.append(0.0)
            print(f"Insufficient data for streamline {i}")
    
    return spline_distances


def extract_streamlines_from_segments_3d(x_coords, y_coords, z_coords, offsets):
    print(f"Raw data shapes: x={len(x_coords)}, y={len(y_coords)}, z={len(z_coords) if z_coords is not None else 'None'}")
    print(f"Offsets: {offsets}")
    print(f"Offsets type: {type(offsets)}")
    

    print(f"X coordinate statistics: min={np.min(x_coords):.6f}, max={np.max(x_coords):.6f}, mean={np.mean(x_coords):.6f}")
    print(f"Y coordinate statistics: min={np.min(y_coords):.6f}, max={np.max(y_coords):.6f}, mean={np.mean(y_coords):.6f}")
    if z_coords is not None:
        print(f"Z coordinate statistics: min={np.min(z_coords):.6f}, max={np.max(z_coords):.6f}, mean={np.mean(z_coords):.6f}")
    

    if len(offsets) == 2:
        start_idx = int(offsets[0])
        end_idx = int(offsets[1])
        
        print(f"Single streamline detected: points {start_idx} to {end_idx}")
        
        x_segment = x_coords[start_idx:end_idx]
        y_segment = y_coords[start_idx:end_idx]
        
        if z_coords is not None:
            z_segment = z_coords[start_idx:end_idx]
        else:
            z_segment = np.zeros(len(x_segment))
        
        streamline = np.column_stack((x_segment, y_segment, z_segment))
        streamlines = [streamline]
        print(f"  Streamline 0: shape {streamline.shape}")
        print(f"  X range: [{np.min(x_segment):.6f}, {np.max(x_segment):.6f}]")
        print(f"  Y range: [{np.min(y_segment):.6f}, {np.max(y_segment):.6f}]")
        if z_coords is not None:
            print(f"  Z range: [{np.min(z_segment):.6f}, {np.max(z_segment):.6f}]")
        
        return streamlines
    
    if len(offsets) <= 1:
        print(f"Warning: Only {len(offsets)} offset(s) found. Treating all data as one streamline.")
        if z_coords is not None:
            streamlines = [np.column_stack((x_coords, y_coords, z_coords))]
        else:

            z_coords_padded = np.zeros(len(x_coords))
            streamlines = [np.column_stack((x_coords, y_coords, z_coords_padded))]
        return streamlines

    offsets_sorted = np.sort(offsets)
    print(f"Sorted offsets: {offsets_sorted}")
    
    streamlines = []
    
    for i in range(len(offsets_sorted) - 1):
        start_idx = int(offsets_sorted[i])
        end_idx = int(offsets_sorted[i + 1])
        
        print(f"Extracting streamline {i}: points {start_idx} to {end_idx}")
        
        x_segment = x_coords[start_idx:end_idx]
        y_segment = y_coords[start_idx:end_idx]
        
        if z_coords is not None:
            z_segment = z_coords[start_idx:end_idx]
        else:
            z_segment = np.zeros(len(x_segment))
        
        streamline = np.column_stack((x_segment, y_segment, z_segment))
        streamlines.append(streamline)
        print(f"  Streamline {i}: shape {streamline.shape}")
        print(f"  X range: [{np.min(x_segment):.6f}, {np.max(x_segment):.6f}]")
        print(f"  Y range: [{np.min(y_segment):.6f}, {np.max(y_segment):.6f}]")
        if z_coords is not None:
            print(f"  Z range: [{np.min(z_segment):.6f}, {np.max(z_segment):.6f}]")

    return streamlines


def plot_pointwise_errors_3d(
    segments_compressed, segments_uncompressed, step=None, spline_distances=None
):
    """Calculate and plot pointwise errors for 3D streamlines."""
    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    if len(segments_compressed) == 0 or len(segments_uncompressed) == 0:
        print("Warning: No streamlines to calculate point-wise errors")
        return []

    fig, ax = plt.subplots(figsize=(12, 8))

    all_errors = []
    all_rk_steps = []
    
    for i, (seg_comp, seg_uncomp) in enumerate(
        zip(segments_compressed, segments_uncompressed)
    ):
        if len(seg_comp) == 0 or len(seg_uncomp) == 0:
            continue

        print(f"\nStreamline {i}:")
        print(f"  Compressed points: {len(seg_comp)}")
        print(f"  Uncompressed points: {len(seg_uncomp)}")

        errors = []
        for j, point in enumerate(seg_comp):
            distances_to_points = np.linalg.norm(seg_uncomp - point, axis=1)
            min_error = np.min(distances_to_points)
            errors.append(min_error)

        all_errors.extend(errors)
        all_rk_steps.extend(range(len(all_rk_steps), len(all_rk_steps) + len(errors)))

    if all_errors:
        ax.plot(
            all_rk_steps,
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
                f"3D RK step errors at (Step {step:04d}) - Spline Distances: [{distances_str}]"
                if step is not None
                else f"3D RK step errors - Spline Distances: [{distances_str}]"
            )
        else:
            title = (
                f"3D RK step errors at (Step {step:04d})"
                if step is not None
                else "3D RK step errors"
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
        ax.set_title("3D RK step errors (No Data)", fontsize=14)

    filename = (
        f"3D_RK_step_errors_at_step_{step:04d}.png"
        if step is not None
        else "3D_RK_step_errors.png"
    )
    filepath = os.path.join(output_dir, filename)
    print(f"Saving 3D RK step errors plot to: {filepath}")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return all_errors


def RK_visualization_3d(
    segments_compressed,
    segments_uncompressed,
    distances,
    step=None,
    spline_distances=None,
):
    """Create 3D visualization of streamlines and calculate errors."""
    errors = plot_pointwise_errors_3d(
        segments_compressed,
        segments_uncompressed,
        step=step,
        spline_distances=spline_distances,
    )

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors_comp = plt.cm.Reds(np.linspace(0.4, 0.9, len(segments_compressed)))
    colors_uncomp = plt.cm.Greens(np.linspace(0.4, 0.9, len(segments_uncompressed)))

    for i, segment in enumerate(segments_compressed):
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

    for i, segment in enumerate(segments_uncompressed):
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
        f"3D Streamlines Comparison - {len(segments_compressed)} streamlines (Step {step:04d})"
        if step is not None
        else f"3D Streamlines Comparison - {len(segments_compressed)} streamlines"
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

    return errors


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculating the error of 3D streamlines given the segments"
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
        help="Second Adios file with streamline segments (higher resolution)",
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
        help="IO Name for the second Adios file (default: reader2)",
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
        "--var_z", type=str, required=True, help="Variable name for z coordinates"
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
        help="Number of spline points to interpolate",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    r_low = Reader(args.IO_Name1, args.file1, xml=args.xml)
    r_high = Reader(args.IO_Name2, args.file2, xml=args.xml)

    try:
        while True:
            if r_low.Adios_reader is None or r_high.Adios_reader is None:
                print("One or both readers are closed. Exiting.")
                break
                
            status_low = r_low.begin_step()
            status_high = r_high.begin_step()

            if (
                bindings.StepStatus.OK != status_low
                or bindings.StepStatus.OK != status_high
            ):
                print(f"Step status: low={status_low}, high={status_high}")
                print("End of stream reached or error occurred. Exiting.")
                break
            
            current_step = r_low.current_step()
            print(f"Reading step: {int(current_step)}")

            try:
                r_high.set_read_vars([args.var_x, args.var_y, args.var_z, args.var_offset])
                r_low.set_read_vars([args.var_x, args.var_y, args.var_z, args.var_offset])

                if (
                    r_low.vars_Out.get(args.var_x) is None
                    or r_low.vars_Out.get(args.var_y) is None
                    or r_low.vars_Out.get(args.var_z) is None
                    or r_low.vars_Out.get(args.var_offset) is None
                ):
                    print("Variables not found in the low resolution stream.")
                    break

                segment_compressed_x = r_low.read_step(args.var_x)
                segment_compressed_y = r_low.read_step(args.var_y)
                segment_compressed_z = r_low.read_step(args.var_z)
                segment_compressed_offset = r_low.read_step(args.var_offset)

                segment_uncompressed_x = r_high.read_step(args.var_x)
                segment_uncompressed_y = r_high.read_step(args.var_y)
                segment_uncompressed_z = r_high.read_step(args.var_z)
                segment_uncompressed_offset = r_high.read_step(args.var_offset)

                segment_compressed_pairs = extract_streamlines_from_segments_3d(
                    segment_compressed_x, segment_compressed_y, segment_compressed_z, segment_compressed_offset
                )

                segment_uncompressed_pair = extract_streamlines_from_segments_3d(
                    segment_uncompressed_x, segment_uncompressed_y, segment_uncompressed_z, segment_uncompressed_offset
                )

                num_streamlines = len(segment_uncompressed_pair)
                print(f"Number of streamlines: {num_streamlines}")

                spline_distances = calculate_all_spline_distances(
                    segment_compressed_pairs, 
                    segment_uncompressed_pair, 
                    args.num_spline
                )

                distances = []
                for i in range(num_streamlines):
                    distance = 0
                    distances.append(distance)
                    print(f"Distance between streamline {i}: {distance}")

                error = RK_visualization_3d(
                    segment_compressed_pairs,
                    segment_uncompressed_pair,
                    distances,
                    step=current_step,
                    spline_distances=spline_distances,
                )

                print(f"Step {current_step} processed successfully.")
                
            except Exception as step_error:
                print(f"Error processing step {current_step}: {step_error}")

            finally:
                try:
                    r_low.end_step()
                    r_high.end_step()
                except Exception as end_error:
                    print(f"Error ending step: {end_error}")
                    break

    except Exception as main_error:
        print(f"Main loop error: {main_error}")
        
    finally:
        try:
            r_low.close()
        except:
            pass
        try:
            r_high.close()
        except:
            pass
        
    print(f"Finished 3D ErrorStream.py successfully!")


if __name__ == "__main__":
    install()
    main()