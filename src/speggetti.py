from adios2 import bindings
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rich.traceback import install
from ReaderClass import Reader


def extract_streamlines(x_coords, y_coords, offsets):
    streamlines = []
    for i in range(len(offsets) - 1):
        start = int(offsets[i])
        end   = int(offsets[i + 1])
        if end > start:
            streamlines.append(np.column_stack((x_coords[start:end], y_coords[start:end])))
    return streamlines


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot ensemble (spaghetti) streamlines")

    parser.add_argument("--input",      "-in",  type=str, required=True,
        help="ADIOS2 BP file with streamline segments (output of createStreamlines)")
    parser.add_argument("--readIO",     "-rio", type=str, default="reader1",
        help="IO name for the reader (default: reader1)")
    parser.add_argument("--xml",        "-x",   type=str, default=None,
        help="ADIOS2 XML config file (optional)")
    parser.add_argument("--var_x",      type=str, default="coords_x",
        help="Variable name for x coordinates (default: coords_x)")
    parser.add_argument("--var_y",      type=str, default="coords_y",
        help="Variable name for y coordinates (default: coords_y)")
    parser.add_argument("--var_offset", type=str, default="offsets",
        help="Variable name for offsets (default: offsets)")
    parser.add_argument("--alpha",      type=float, default=0.25,
        help="Line transparency for ensemble lines (default: 0.25). Lower = more transparent.")
    parser.add_argument("--linewidth",  type=float, default=1,
        help="Line width for ensemble lines (default: 1)")
    parser.add_argument("--color",      type=str, default="blue",
        help="Line color for ensemble lines (default: blue)")
    parser.add_argument("--output_dir", type=str, default="../RESULTS",
        help="Directory to save plots (default: ../RESULTS)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    reader = Reader(args.readIO, args.input, xml=args.xml)

    while True:
        status = reader.begin_step()
        if status != bindings.StepStatus.OK:
            break

        current_step = reader.current_step()
        print(f"Reading step: {int(current_step)}")

        reader.set_read_vars([args.var_x, args.var_y, args.var_offset])

        if (
            reader.vars_Out.get(args.var_x)      is None or
            reader.vars_Out.get(args.var_y)      is None or
            reader.vars_Out.get(args.var_offset) is None
        ):
            print("Variables not found in stream.")
            break

        x_vals  = reader.read_step(args.var_x)
        y_vals  = reader.read_step(args.var_y)
        offsets = reader.read_step(args.var_offset)

        print(f"  Total points : {len(x_vals)}")
        print(f"  Num offsets  : {len(offsets)}  ->  {len(offsets)-1} streamlines")

        streamlines = extract_streamlines(x_vals, y_vals, offsets)
        print(f"  Extracted {len(streamlines)} streamlines")

  
        fig, ax = plt.subplots(figsize=(10, 8))

        for sl in streamlines:
            if len(sl) < 2:
                continue
            ax.plot(sl[:, 0], sl[:, 1],
                    color=args.color,
                    linewidth=args.linewidth,
                    alpha=args.alpha)

   
        if streamlines:
            seed_x = streamlines[0][0, 0]
            seed_y = streamlines[0][0, 1]
            ax.scatter(seed_x, seed_y, color="red", s=40, zorder=5, label="Seed point")
            ax.legend()

        ax.set_title(f"Ensemble Streamlines — Step {current_step}  ({len(streamlines)} lines)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        fig.tight_layout()

        fname = f"{args.output_dir}/ensemble_streamlines_step_{current_step}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")

        reader.end_step()

    reader.close()
    print("Done.")


if __name__ == "__main__":
    install()
    main()
