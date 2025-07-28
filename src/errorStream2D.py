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
import csv
from collections import defaultdict


def extract_streamlines_from_segments(x_coords, y_coords, offsets):

    if len(x_coords) == 0 or len(y_coords) == 0:
        return []
        offsets
    if len(offsets) - 1 <= 1:
        print(
            f"Warning: Only {len(offsets)} offset(s) found. Treating all data as one streamline."
        )
        if len(x_coords) > 0:
            streamlines = [np.column_stack((x_coords, y_coords))]
        else:
            streamlines = []
        return streamlines

    streamlines = []
    start_idx = 0

    sorted_offsets = np.sort(offsets)

    for i, end_idx in enumerate(sorted_offsets):
        if end_idx > start_idx and end_idx <= len(x_coords):
            streamline_x = x_coords[start_idx:end_idx]
            streamline_y = y_coords[start_idx:end_idx]

            if len(streamline_x) > 0:
                streamline = np.column_stack((streamline_x, streamline_y))
                streamlines.append(streamline)
                print(
                    f"Streamline {i}: {len(streamline)} points, range x=[{np.min(streamline_x):.6f}, {np.max(streamline_x):.6f}], y=[{np.min(streamline_y):.6f}, {np.max(streamline_y):.6f}]"
                )

            start_idx = end_idx

    print(f"Extracted {len(streamlines)} streamlines")
    return streamlines


def calculate_streamline_statistics(errors):
    if len(errors) == 0:
        return {
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'q1': 0.0,
            'q3': 0.0,
            'count': 0
        }
    
    errors_array = np.array(errors)
    return {
        'min': np.min(errors_array),
        'max': np.max(errors_array),
        'median': np.median(errors_array),
        'mean': np.mean(errors_array),
        'std': np.std(errors_array),
        'q1': np.percentile(errors_array, 25),
        'q3': np.percentile(errors_array, 75),
        'count': len(errors_array)
    }


def plot_pointwise_errors_separate(
    segments_compressed, segments_uncompressed, step=None, spline_distances=None, 
    accumulated_errors=None
):

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    if len(segments_compressed) == 0 or len(segments_uncompressed) == 0:
        print("Warning: No streamlines to calculate point-wise errors")
        return [], []

    all_streamline_errors = []
    streamline_statistics = []

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

        all_streamline_errors.append(errors)
        
        if accumulated_errors is not None:
            accumulated_errors[i].extend(errors)
        
        stats = calculate_streamline_statistics(errors)
        stats['streamline_id'] = i + 1
        stats['time_step'] = step
        streamline_statistics.append(stats)
        
        if errors:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            ax.plot(
                range(len(errors)),
                errors,
                marker="o",
                markersize=3,
                linestyle="-",
                color="blue",
            )
            ax.set_yscale("log")

            stats_text = f"Min: {stats['min']:.2e}, Max: {stats['max']:.2e}\n"
            stats_text += f"Mean: {stats['mean']:.2e}, Median: {stats['median']:.2e}\n"
            stats_text += f"Q1: {stats['q1']:.2e}, Q3: {stats['q3']:.2e}\n"
            stats_text += f"Std Dev: {stats['std']:.2e}, Count: {stats['count']}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            if spline_distances is not None and len(spline_distances) > i:
                spline_dist_str = f"{spline_distances[i]:.6f}"
                title = (
                    f"RK step errors for Streamline {i+1} (Step {step:04d}) - Spline Distance: {spline_dist_str}"
                    if step is not None
                    else f"RK step errors for Streamline {i+1} - Spline Distance: {spline_dist_str}"
                )
            else:
                title = (
                    f"RK step errors for Streamline {i+1} (Step {step:04d})"
                    if step is not None
                    else f"RK step errors for Streamline {i+1}"
                )
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("RK steps", fontsize=12)
            ax.set_ylabel("Error Magnitude", fontsize=12)
            ax.grid(True, which="both")

            filename = (
                f"RK_step_errors_streamline_{i+1}_step_{step:04d}.png"
                if step is not None
                else f"RK_step_errors_streamline_{i+1}.png"
            )
            filepath = os.path.join(output_dir, filename)
            print(f"Saving RK step errors plot for streamline {i+1} to: {filepath}")
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)

    return all_streamline_errors, streamline_statistics


def save_per_timestep_statistics(streamline_statistics, step=None, spline_distances=None):

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)
    
    individual_filename = (
        f"streamline_statistics_step_{step:04d}.csv"
        if step is not None
        else "streamline_statistics.csv"
    )
    individual_filepath = os.path.join(output_dir, individual_filename)
    
    with open(individual_filepath, 'w', newline='') as csvfile:
        fieldnames = ['streamline_id', 'time_step', 'min', 'max', 'median', 'mean', 'std', 'q1', 'q3', 'count']
        if spline_distances is not None:
            fieldnames.append('spline_distance')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, stats in enumerate(streamline_statistics):
            row = stats.copy()
            if spline_distances is not None and i < len(spline_distances):
                row['spline_distance'] = spline_distances[i]
            writer.writerow(row)
    
    print(f"Time step {step} statistics saved to: {individual_filepath}")


def save_comprehensive_statistics(accumulated_errors, accumulated_spline_distances):

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("CALCULATING COMPREHENSIVE STATISTICS ACROSS ALL TIME STEPS")
    print("="*60)
    
    streamline_comprehensive_stats = []
    all_errors_combined = []
    
    for streamline_id, errors in accumulated_errors.items():
        if len(errors) > 0:
            stats = calculate_streamline_statistics(errors)
            stats['streamline_id'] = streamline_id + 1
            stats['total_time_steps'] = len([e for e in errors if e > 0])  
            
            if streamline_id in accumulated_spline_distances:
                spline_dists = accumulated_spline_distances[streamline_id]
                if spline_dists:
                    stats['avg_spline_distance'] = np.mean(spline_dists)
                    stats['spline_distance_std'] = np.std(spline_dists)
                else:
                    stats['avg_spline_distance'] = 0.0
                    stats['spline_distance_std'] = 0.0
            
            streamline_comprehensive_stats.append(stats)
            all_errors_combined.extend(errors)
            
           
            print(f"Streamline {streamline_id + 1}:")
            print(f"  Total RK steps across all time steps: {stats['count']}")
            print(f"  Min error: {stats['min']:.6e}")
            print(f"  Max error: {stats['max']:.6e}")
            print(f"  Mean error: {stats['mean']:.6e}")
            print(f"  Median error: {stats['median']:.6e}")
            print(f"  Q1 error: {stats['q1']:.6e}")
            print(f"  Q3 error: {stats['q3']:.6e}")
            print(f"  Std Dev: {stats['std']:.6e}")
            if 'avg_spline_distance' in stats:
                print(f"  Avg Spline Distance: {stats['avg_spline_distance']:.6e}")
            print()
    

    if streamline_comprehensive_stats:
        comprehensive_filename = "comprehensive_streamline_statistics_all_timesteps.csv"
        comprehensive_filepath = os.path.join(output_dir, comprehensive_filename)
        
        fieldnames = ['streamline_id', 'min', 'max', 'median', 'mean', 'std', 'q1', 'q3', 'count', 'total_time_steps']
        if any('avg_spline_distance' in stats for stats in streamline_comprehensive_stats):
            fieldnames.extend(['avg_spline_distance', 'spline_distance_std'])
        
        with open(comprehensive_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for stats in streamline_comprehensive_stats:
                writer.writerow(stats)
        
        print(f"Comprehensive per-streamline statistics saved to: {comprehensive_filepath}")
    

    if all_errors_combined:
        overall_stats = calculate_streamline_statistics(all_errors_combined)
        overall_stats['total_streamlines'] = len(streamline_comprehensive_stats)
        overall_stats['total_rk_steps_all_streamlines'] = len(all_errors_combined)
        

        if streamline_comprehensive_stats:
            streamline_mins = [s['min'] for s in streamline_comprehensive_stats]
            streamline_maxs = [s['max'] for s in streamline_comprehensive_stats]
            streamline_means = [s['mean'] for s in streamline_comprehensive_stats]
            streamline_medians = [s['median'] for s in streamline_comprehensive_stats]
            streamline_stds = [s['std'] for s in streamline_comprehensive_stats]
            streamline_q1s = [s['q1'] for s in streamline_comprehensive_stats]
            streamline_q3s = [s['q3'] for s in streamline_comprehensive_stats]
            
            overall_stats.update({
                'min_of_streamline_mins': np.min(streamline_mins),
                'max_of_streamline_maxs': np.max(streamline_maxs),
                'mean_of_streamline_means': np.mean(streamline_means),
                'median_of_streamline_medians': np.median(streamline_medians),
                'mean_of_streamline_stds': np.mean(streamline_stds),
                'mean_of_streamline_q1s': np.mean(streamline_q1s),
                'mean_of_streamline_q3s': np.mean(streamline_q3s)
            })
        
        overall_filename = "overall_statistics_all_streamlines_all_timesteps.csv"
        overall_filepath = os.path.join(output_dir, overall_filename)
        
        with open(overall_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=overall_stats.keys())
            writer.writeheader()
            writer.writerow(overall_stats)
        
        print(f"Overall statistics across all streamlines and time steps saved to: {overall_filepath}")
        
      
        print("="*60)
        print("OVERALL STATISTICS SUMMARY (ALL STREAMLINES, ALL TIME STEPS)")
        print("="*60)
        print(f"Total Streamlines: {overall_stats['total_streamlines']}")
        print(f"Total RK Steps (all streamlines, all time steps): {overall_stats['total_rk_steps_all_streamlines']}")
        print(f"Overall Min Error: {overall_stats['min']:.6e}")
        print(f"Overall Max Error: {overall_stats['max']:.6e}")
        print(f"Overall Mean Error: {overall_stats['mean']:.6e}")
        print(f"Overall Median Error: {overall_stats['median']:.6e}")
        print(f"Overall Q1 Error: {overall_stats['q1']:.6e}")
        print(f"Overall Q3 Error: {overall_stats['q3']:.6e}")
        print(f"Overall Std Dev: {overall_stats['std']:.6e}")
        
        if 'min_of_streamline_mins' in overall_stats:
            print("\nMeta-Statistics (Statistics of Streamline Statistics):")
            print(f"Min of Streamline Mins: {overall_stats['min_of_streamline_mins']:.6e}")
            print(f"Max of Streamline Maxs: {overall_stats['max_of_streamline_maxs']:.6e}")
            print(f"Mean of Streamline Means: {overall_stats['mean_of_streamline_means']:.6e}")
            print(f"Median of Streamline Medians: {overall_stats['median_of_streamline_medians']:.6e}")
            print(f"Mean of Streamline Std Devs: {overall_stats['mean_of_streamline_stds']:.6e}")
            print(f"Mean of Streamline Q1s: {overall_stats['mean_of_streamline_q1s']:.6e}")
            print(f"Mean of Streamline Q3s: {overall_stats['mean_of_streamline_q3s']:.6e}")
        
        print("="*60 + "\n")


def RK_visualization(
    segments_compressed,
    segments_uncompressed,
    distances,
    step=None,
    spline_distances=None,
    accumulated_errors=None
):
 
    all_errors, streamline_stats = plot_pointwise_errors_separate(
        segments_compressed,
        segments_uncompressed,
        step=step,
        spline_distances=spline_distances,
        accumulated_errors=accumulated_errors
    )
    

    save_per_timestep_statistics(streamline_stats, step=step, spline_distances=spline_distances)

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    fig_streamlines, ax_streamlines = plt.subplots(figsize=(12, 10))

    n_streamlines = max(len(segments_compressed), len(segments_uncompressed))
    colors = plt.cm.tab10(np.linspace(0, 1, n_streamlines))

    for i, segment in enumerate(segments_compressed):
        if len(segment) > 0:
            ax_streamlines.plot(
                segment[:, 0],
                segment[:, 1],
                linestyle="-",
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.8,
                label=f"Low res streamline {i+1}" if i < 5 else "",
            )

    for i, segment in enumerate(segments_uncompressed):
        if len(segment) > 0:
            ax_streamlines.plot(
                segment[:, 0],
                segment[:, 1],
                linestyle="--",
                color=colors[i % len(colors)],
                linewidth=1.5,
                alpha=0.6,
                label=f"High res streamline {i+1}" if i < 5 else "",
            )

    ax_streamlines.set_xlabel("X", fontsize=12)
    ax_streamlines.set_ylabel("Y", fontsize=12)
    title_str = (
        f"Streamlines Comparison - {len(segments_compressed)} streamlines (Step {step:04d})"
        if step is not None
        else f"Streamlines Comparison - {len(segments_compressed)} streamlines"
    )
    ax_streamlines.set_title(title_str, fontsize=14)
    if n_streamlines <= 5:
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

    min_streamlines = min(len(segments_compressed), len(segments_uncompressed))
    for idx in range(min_streamlines):
        comp_segment = segments_compressed[idx]
        uncomp_segment = segments_uncompressed[idx]

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

            if len(segments) > 0 and len(line_colors) > 0:
                lc = LineCollection(
                    segments, cmap="jet", array=line_colors, linewidths=3
                )
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

            all_x = np.concatenate([comp_segment[:, 0], uncomp_segment[:, 0]])
            all_y = np.concatenate([comp_segment[:, 1], uncomp_segment[:, 1]])
            x_margin = (np.max(all_x) - np.min(all_x)) * 0.05
            y_margin = (np.max(all_y) - np.min(all_y)) * 0.05

            ax.set_xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
            ax.set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)

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


    flattened_errors = []
    for errors in all_errors:
        flattened_errors.extend(errors)
    
    return flattened_errors


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

 
    accumulated_errors = defaultdict(list) 
    accumulated_spline_distances = defaultdict(list)  

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

        print(
            f"Data shapes - Low res: x={len(segment_compressed_x)}, y={len(segment_compressed_y)}, offsets={segment_compressed_offset}"
        )
        print(
            f"Data shapes - High res: x={len(segment_uncompressed_x)}, y={len(segment_uncompressed_y)}, offsets={segment_uncompressed_offset}"
        )

        segment_compressed_pairs = extract_streamlines_from_segments(
            segment_compressed_x, segment_compressed_y, segment_compressed_offset
        )

        segment_uncompressed_pair = extract_streamlines_from_segments(
            segment_uncompressed_x, segment_uncompressed_y, segment_uncompressed_offset
        )

        num_streamlines = len(segment_uncompressed_pair)
        print(f"Number of streamlines: {num_streamlines}")

        spline_distances = []
        min_streamlines = min(
            len(segment_compressed_pairs), len(segment_uncompressed_pair)
        )

        for i in range(min_streamlines):
            if (
                len(segment_compressed_pairs[i]) > 1
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
                    d = np.mean(np.sqrt(diffx * diffx + diffy * diffy))
                    spline_distances.append(d)
                    

                    accumulated_spline_distances[i].append(d)
                    
                    print(f"Spline distance for streamline {i}: {d}")
                except Exception as e:
                    print(f"Error calculating spline distance for streamline {i}: {e}")
                    spline_distances.append(0.0)
                    accumulated_spline_distances[i].append(0.0)
            else:
                spline_distances.append(0.0)
                accumulated_spline_distances[i].append(0.0)
                print(f"Insufficient data for spline calculation for streamline {i}")

        distances = [0.0] * num_streamlines

        error = RK_visualization(
            segment_compressed_pairs,
            segment_uncompressed_pair,
            distances,
            step=current_step,
            spline_distances=spline_distances,
            accumulated_errors=accumulated_errors
        )

        r_low.end_step()
        r_high.end_step()

    save_comprehensive_statistics(accumulated_errors, accumulated_spline_distances)

    r_low.close()
    r_high.close()
    print(f"Finished ErrorStream.py successfully!")


if __name__ == "__main__":
    main()
    install()