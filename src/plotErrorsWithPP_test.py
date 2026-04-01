import matplotlib.pyplot as plt
import argparse
import adios2
import os
import numpy as np
import ReaderClass
from rich.traceback import install


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot pp middle column with truncation error bars for a single resolution."
    )
    parser.add_argument(
        "--var",
        "-v",
        type=str,
        default="pp",
        help="Variable name (default: pp)",
    )
    parser.add_argument(
        "--re",
        "-re",
        type=str,
        required=True,
        help="Reynolds number string (e.g. RE28169)",
    )
    parser.add_argument(
        "--res",
        type=str,
        required=True,
        help="Resolution string (e.g. 1025-2049)",
    )
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        help="ReaderClass IO name (default: reader1)",
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        help="Optional ADIOS2 XML config file",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="../my_data",
        help="Base path for original BP files (default: ../my_data)",
    )
    parser.add_argument(
        "--compressed_bp",
        type=str,
        default=None,
        help="Full path to the compressed BP file",
    )
    parser.add_argument(
        "--error_path",
        type=str,
        default="./errors",
        help="Base path for truncation error BP files (default: ./errors)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../RESULTS",
        help="Directory to save plots (default: ../RESULTS)",
    )
    parser.add_argument(
        "--n_errorbars",
        type=int,
        default=7,
        help="Number of evenly-spaced error bar positions to show (default: 7)",
    )
    return parser.parse_args()


def find_bp_file(directory, re_string):
    if not os.path.exists(directory):
        return None
    for f in os.listdir(directory):
        if re_string in f and (f.endswith(".bp5") or f.endswith(".bp")):
            return os.path.join(directory, f)
    return None


def read_middle_column_steps(io_name, bp_file, var_name, xml=None):
    """Read all steps from a BP file, returning the middle column at each step."""
    reader = ReaderClass.Reader(io_name, bp_file, xml=xml)
    steps = []
    while True:
        status = reader.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break
        reader.set_read_vars([var_name])
        data = reader.read_step(var_name)
        if data is not None:
            data = np.squeeze(data)
            if data.ndim == 2:
                mid_col = data[:, data.shape[1] // 2]
            elif data.ndim == 1:
                mid_col = data
            else:
                mid_col = None
            steps.append(mid_col)
        else:
            steps.append(None)
        reader.end_step()
    reader.close()
    return steps


def main():
    args = parse_arguments()

    error_var = f"{args.var}_truncation_error"
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Find and read original pp ---
    res_dir = os.path.join(args.base_path, args.res)
    pp_bp = find_bp_file(res_dir, args.re)
    if pp_bp is None:
        print(f"ERROR: No pp BP file found in {res_dir} for {args.re}")
        return
    print(f"Reading pp         -> {pp_bp}")

    pp_steps = read_middle_column_steps(args.readIO, pp_bp, args.var, xml=args.xml)
    print(f"  Read {len(pp_steps)} pp steps.")

    # --- Read compressed pp ---
    comp_steps = None
    if args.compressed_bp is None:
        print("WARNING: No --compressed_bp provided. Compressed line will be skipped.")
    elif not os.path.exists(args.compressed_bp):
        print(f"WARNING: Compressed BP file not found at {args.compressed_bp}. Compressed line will be skipped.")
    else:
        print(f"Reading compressed -> {args.compressed_bp}")
        comp_steps = read_middle_column_steps(args.readIO + "_comp", args.compressed_bp, args.var, xml=args.xml)
        print(f"  Read {len(comp_steps)} compressed steps.")

    # --- Read error steps (track both max and min absolute error) ---
    err_dir = os.path.join(args.error_path, args.res, args.re)
    err_bp = os.path.join(err_dir, f"{args.var}_diff_{args.re}.bp")
    if not os.path.exists(err_bp):
        print(f"ERROR: No error BP file found at {err_bp}")
        return
    print(f"Reading error      -> {err_bp}")

    r_err = ReaderClass.Reader(args.readIO + "_err", err_bp, xml=args.xml)
    err_steps_max = []
    err_steps_min = []
    while True:
        status = r_err.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break
        r_err.set_read_vars([error_var])
        err_data = r_err.read_step(error_var)
        if err_data is not None:
            err_data = err_data.squeeze()
            abs_err = np.abs(err_data)
            err_steps_max.append(np.max(abs_err))
            err_steps_min.append(np.min(abs_err))
        else:
            err_steps_max.append(0.0)
            err_steps_min.append(0.0)
        r_err.end_step()
    r_err.close()
    print(f"  Read {len(err_steps_max)} error steps.")

    n_steps = len(pp_steps)

    for step_idx in range(n_steps):
        pp_line = pp_steps[step_idx]
        if pp_line is None:
            print(f"  Skipping step {step_idx}: no pp data.")
            continue

        err_max = err_steps_max[step_idx] if step_idx < len(err_steps_max) else 0.0
        err_min = err_steps_min[step_idx] if step_idx < len(err_steps_min) else 0.0

        ny = len(pp_line)
        y = np.linspace(0, 1.0, ny)

        n_eb = args.n_errorbars
        eb_indices = np.linspace(0, ny - 1, n_eb, dtype=int)
        y_eb = y[eb_indices]
        pp_eb = pp_line[eb_indices]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Original pp line
        ax.plot(
            y,
            pp_line,
            color="tab:red",
            linewidth=1.2,
            alpha=0.85,
            label=f"{args.var} (original)",
        )

        # Compressed pp line (if available)
        if comp_steps is not None and step_idx < len(comp_steps):
            comp_line = comp_steps[step_idx]
            if comp_line is not None:
                # Interpolate onto same y grid if sizes differ
                if len(comp_line) != ny:
                    comp_y = np.linspace(0, 1.0, len(comp_line))
                    comp_line = np.interp(y, comp_y, comp_line)
                ax.plot(
                    y,
                    comp_line,
                    color="tab:orange",
                    linewidth=1.2,
                    alpha=0.85,
                    linestyle="--",
                    label=f"{args.var} (compressed)",
                )

        # Error bars
        ax.errorbar(
            y_eb,
            pp_eb,
            yerr=np.full(n_eb, abs(err_max)),
            fmt="none",
            ecolor="tab:blue",
            elinewidth=0.9,
            capsize=3,
            label=f"max err {abs(err_max):.2e}",
        )

        ax.errorbar(
            y_eb,
            pp_eb,
            yerr=np.full(n_eb, abs(err_min)),
            fmt="none",
            ecolor="tab:green",
            elinewidth=0.9,
            capsize=3,
            label=f"min err {abs(err_min):.2e}",
        )

        ax.set_xlabel("value range [0, 1]")
        ax.set_ylabel(f"{args.var}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"{args.var}_{args.re}_{args.res}_errorbar_step_{step_idx}.png"
        fig.savefig(os.path.join(args.output_dir, filename), dpi=150)
        plt.close(fig)
        print(f"  Saved: {filename}")

    print("Done.")


if __name__ == "__main__":
    install()
    main()
