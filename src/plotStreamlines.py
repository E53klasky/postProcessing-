from adios2 import bindings
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plotting streamlines")

    parser.add_argument(
        "--input",
        "-in",
        type=str,
        required=True,
        help="Adios file with streamline segments",
    )

    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        required=False,
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
        "--velocity_file",
        "-vf",
        type=str,
        required=True,
        help="Adios file with velocity data",
    )
    
    parser.add_argument(
        "--velocity_readIO",
        "-vrio",
        type=str,
        default="reader2",
        required=False,
        help="IO Name for the velocity Adios file (default: reader2)",
    )
    
    parser.add_argument(
        "--var_u", type=str, required=True, help="Variable name for u velocity component"
    )
    parser.add_argument(
        "--var_v", type=str, required=True, help="Variable name for v velocity component"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    r_segments = Reader(args.readIO, args.input, xml=args.xml)
    r_velocity = Reader(args.velocity_readIO, args.velocity_file, xml=args.xml)

    while True:
        status_segments = r_segments.begin_step()
        if bindings.StepStatus.OK != status_segments:
            break
            
        status_velocity = r_velocity.begin_step()
        if bindings.StepStatus.OK != status_velocity:
            break

        current_step = r_segments.current_step()
        print(f"Reading step: {int(current_step)}")

        r_segments.set_read_vars([args.var_x, args.var_y, args.var_offset])
        
        r_velocity.set_read_vars([args.var_u, args.var_v])

        if (
            r_segments.vars_Out.get(args.var_x) is None
            or r_segments.vars_Out.get(args.var_y) is None
            or r_segments.vars_Out.get(args.var_offset) is None
        ):
            print("Segment variables not found in the stream.")
            break
            
        if (
            r_velocity.vars_Out.get(args.var_u) is None
            or r_velocity.vars_Out.get(args.var_v) is None
        ):
            print("Velocity variables not found in the stream.")
            break

        x_vals = r_segments.read_step(args.var_x)
        y_vals = r_segments.read_step(args.var_y)
        offsets = r_segments.read_step(args.var_offset)
        

        u_vals = r_velocity.read_step(args.var_u)
        v_vals = r_velocity.read_step(args.var_v)
        
        if len(u_vals.shape) == 3 and u_vals.shape[0] == 1:
            u_vals = u_vals[0,:,:]
            v_vals = v_vals[0,:,:]

        streamlines = extract_streamlines_from_segments(x_vals, y_vals, offsets)
        

        normalized_streamlines = streamlines

        output_dir = "../RESULTS"
        os.makedirs(output_dir, exist_ok=True)

        
        plt.figure(figsize=(10, 8))
        for idx, streamline in enumerate(normalized_streamlines):
            if len(streamline) < 4:
                print(f"Skipping streamline {idx} (too few points)")
                continue

            x = streamline[:, 0]
            y = streamline[:, 1]
            plt.plot(x, y, label=f"Streamline {idx}")

            plt_individual = plt.figure(figsize=(8, 6))
            plt.plot(x, y, color="red", label="RK4 Points")
            plt.title(f"Streamline {idx} (Step {current_step}) - Normalized")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis("equal")
            plt.xlabel("x (normalized)")
            plt.ylabel("y (normalized)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/streamline_{idx:03d}_step_{current_step}.png")
            print(f"plot saved to {output_dir}/streamline_{idx:03d}_step_{current_step}.png")
            plt.close(plt_individual)

        plt.title(f"All Streamlines (Step {current_step}) - Normalized")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis("equal")
        plt.xlabel("x (normalized)")
        plt.ylabel("y (normalized)")
        plt.grid(True)
        plt.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_streamlines_step_{current_step}.png")
        print(f"plot saved to {output_dir}/all_streamlines_step_{current_step}.png")
        plt.close()

        # Plot vector field
        plt.figure(figsize=(10, 8))

        if len(u_vals.shape) == 1:
            grid_size = int(np.sqrt(len(u_vals)))
            if grid_size * grid_size == len(u_vals):
                nx, ny = grid_size, grid_size
            else:
                nx = int(np.sqrt(len(u_vals)))
                ny = len(u_vals) // nx
            
            U = u_vals[:nx*ny].reshape(ny, nx)
            V = v_vals[:nx*ny].reshape(ny, nx)
        else:
            U, V = u_vals, v_vals
            ny, nx = U.shape
        
        x_grid = np.linspace(0, 1, nx)
        y_grid = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        skip = max(1, min(nx, ny) // 15) 
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip], 
                  angles='xy', alpha=0.7, color='blue', width=0.003)
        
        plt.title(f"Vector Field (Step {current_step}) - Normalized")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis("equal")
        plt.xlabel("x (normalized)")
        plt.ylabel("y (normalized)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vector_field_step_{current_step}.png")
        print(f"plot saved to {output_dir}/vector_field_step_{current_step}.png")
        plt.close()


        plt.figure(figsize=(12, 10))
        

        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip], 
                  angles='xy', alpha=0.4, color='lightblue', width=0.002, label='Vector Field')
        

        for idx, streamline in enumerate(normalized_streamlines):
            if len(streamline) < 4:
                continue
            
            x = streamline[:, 0]
            y = streamline[:, 1]
            plt.plot(x, y, linewidth=2, label=f"Streamline {idx}")
        
        plt.title(f"Vector Field with Streamlines (Step {current_step}) - Normalized")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis("equal")
        plt.xlabel("x (normalized)")
        plt.ylabel("y (normalized)")
        plt.grid(True)
        plt.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vector_field_with_streamlines_step_{current_step}.png")
        print(f"plot saved to {output_dir}/vector_field_with_streamlines_step_{current_step}.png")
        plt.close()

        r_segments.end_step()
        r_velocity.end_step()


if __name__ == "__main__":
    install()
    main()