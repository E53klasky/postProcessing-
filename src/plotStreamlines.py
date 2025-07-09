from adios2 import bindings
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rich.traceback import install
from ReaderClass import Reader
from scipy.interpolate import CubicSpline, Akima1DInterpolator

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
        "--IO_Name1",
        type=str,
        default="reader1",
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


def plotStreamlines():
    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)
    
def main():
    args = parse_arguments()
    r = Reader(args.IO_Name1, args.file1, xml=args.xml)

    while True:
        status = r.begin_step()

        if bindings.StepStatus.OK != status:
            break

        current_step = r.current_step()
        print(f"Reading step: {int(current_step)}")

        r.set_read_vars([args.var_x, args.var_y, args.var_offset])

        if (
            r.vars_Out.get(args.var_x) is None
            or r.vars_Out.get(args.var_y) is None
            or r.vars_Out.get(args.var_offset) is None
        ):
            print("Variables not found in the stream.")
            break


        x_vals = r.read_step(args.var_x)
        y_vals = r.read_step(args.var_y)
        offsets = r.read_step(args.var_offset)

        streamlines = extract_streamlines_from_segments(x_vals, y_vals, offsets)


        for idx, streamline in enumerate(streamlines):
            if len(streamline) < 4:
                print(f"Skipping streamline {idx} (too few points)")
                continue

            x = streamline[:, 0]
            y = streamline[:, 1]

  
            dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            arc_len = np.concatenate([[0], np.cumsum(dist)])

            cubic_x = CubicSpline(arc_len, x)
            cubic_y = CubicSpline(arc_len, y)

            akima_x = Akima1DInterpolator(arc_len, x)
            akima_y = Akima1DInterpolator(arc_len, y)

            interp_points = np.linspace(arc_len[0], arc_len[-1], args.num_spline)

            plt.figure(figsize=(8, 6))
       
            # plt.plot(cubic_x(interp_points), cubic_y(interp_points), label="Cubic Spline", color='blue')
            # plt.plot(akima_x(interp_points), akima_y(interp_points), label="Akima", color='green')
            plt.plot(x, y, color='red', label='RK4 Points')
            plt.legend()
            plt.title(f"Streamline {idx} (Step {current_step})")
            plt.axis('equal')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.tight_layout()

            output_dir = "../RESULTS"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/streamline_{idx:03d}_step_{current_step}.png")
            plt.close()

        r.end_step()
if __name__ == "__main__":
    main()
    install()
