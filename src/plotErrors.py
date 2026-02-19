import matplotlib.pyplot as plt
import argparse
import adios2
import os
import numpy as np
import ReaderClass
from rich.traceback import install

RESOLUTIONS = ["129-257", "257-513", "513-1025", "1025-2049"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


LINE_FRACTION = 0.5


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot truncation errors along a single line across resolutions per time step."
    )
    parser.add_argument(
        "--var",
        "-v",
        type=str,
        required=True,
        help="Variable prefix (e.g. phi01, ux, uy, pp)",
    )
    parser.add_argument(
        "--re",
        "-re",
        type=str,
        required=True,
        help="Reynolds number directory name (e.g. RE112676)",
    )
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        required=False,
        help="IO Name for the Adios reader (default: reader1)",
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        help="Optional ADIOS2 XML configuration",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    var_name = f"{args.var}_truncation_error"
    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    all_data = {}
    for res in RESOLUTIONS:
        bp_path = os.path.join(
            ".", "errors", res, args.re, f"{args.var}_diff_{args.re}.bp"
        )
        if not os.path.exists(bp_path):
            print(f"WARNING: BP file not found: {bp_path}, skipping.")
            continue

        print(f"Reading {res} -> {bp_path}")
        r = ReaderClass.Reader(args.readIO, bp_path, xml=args.xml)
        steps = []
        while True:
            status = r.begin_step()
            if status != adios2.bindings.StepStatus.OK:
                break
            r.set_read_vars([var_name])
            data = r.read_step(var_name)
            if data is not None:
                data = data.squeeze()
                steps.append(data)
            else:
                print(f"  Variable '{var_name}' not found at step {len(steps)}")
                steps.append(None)
            r.end_step()

        all_data[res] = steps
        print(f"  Read {len(steps)} steps.")

    if not all_data:
        print("No data found. Check your --var and --re arguments.")
        return

    ref_res = RESOLUTIONS[0]
    if ref_res not in all_data:
        ref_res = next(iter(all_data))

    ref_shape = all_data[ref_res][0].shape
    ny_ref, nx_ref = ref_shape
    print(f"Reference (coarsest) grid: {ny_ref} x {nx_ref} ({ref_res})")

    ref_row = int(LINE_FRACTION * (ny_ref - 1))
    print(f"Line index on reference grid: row={ref_row} (y ~ {LINE_FRACTION:.0%})")

    n_steps = len(next(iter(all_data.values())))

    for step_idx in range(n_steps):
        fig, ax = plt.subplots(figsize=(12, 5))

        for res, color in zip(RESOLUTIONS, COLORS):
            if res not in all_data:
                continue
            data = all_data[res][step_idx]
            if data is None:
                continue

            ny_high, nx_high = data.shape

            skip_y = (ny_high - 1) / (ny_ref - 1)
            skip_x = (nx_high - 1) / (nx_ref - 1)

            row_idx = int(round(ref_row * skip_y))
            row_idx = min(row_idx, ny_high - 1)

            skip_x_int = max(1, int(round(skip_x)))
            x_indices = np.arange(0, nx_high, skip_x_int)
            # take np.abs

            line_data = np.abs(data[row_idx, x_indices])

            x_norm = np.arange(len(x_indices))

            ax.scatter(
                x_norm,
                line_data,
                s=4,
                color=color,
                label=res,
                alpha=0.7,
                rasterized=True,
            )

            print(
                f"  Step {step_idx} | {res}: row={row_idx}, skip_x={skip_x_int}, nx={len(x_indices)}"
            )

        ax.set_xlabel("Points")
        ax.set_ylabel(var_name)
        ax.set_title(
            f"{var_name} | {args.re} | y ~ {LINE_FRACTION:.0%} | Step {step_idx}"
        )
        ax.legend(markerscale=3, loc="upper right")

        fig.tight_layout()
        plot_filename = f"{args.var}_{args.re}_line_step_{step_idx}.png"
        fig.savefig(os.path.join(output_dir, plot_filename), dpi=150)
        plt.close(fig)
        print(f"Saved: {plot_filename}")

    print("plotErrors completed successfully.")


if __name__ == "__main__":
    install()
    main()
