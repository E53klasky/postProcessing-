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
    return parser.parse_args()


def find_bp_file(directory, re_string):
    if not os.path.exists(directory):
        return None
    for f in os.listdir(directory):
        if re_string in f and (f.endswith(".bp5") or f.endswith(".bp")):
            return os.path.join(directory, f)
    return None


def main():
    args = parse_arguments()

    error_var = f"{args.var}_truncation_error"
    os.makedirs(args.output_dir, exist_ok=True)

    res_dir = os.path.join(args.base_path, args.res)
    pp_bp = find_bp_file(res_dir, args.re)
    if pp_bp is None:
        print(f"ERROR: No pp BP file found in {res_dir} for {args.re}")
        return
    print(f"Reading pp -> {pp_bp}")

    err_dir = os.path.join(args.error_path, args.res, args.re)
    err_bp = os.path.join(err_dir, f"{args.var}_diff_{args.re}.bp")
    if not os.path.exists(err_bp):
        print(f"ERROR: No error BP file found at {err_bp}")
        return
    print(f"Reading error -> {err_bp}")

    r_pp = ReaderClass.Reader(args.readIO, pp_bp, xml=args.xml)
    pp_steps = []
    while True:
        status = r_pp.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break
        r_pp.set_read_vars([args.var])
        data = r_pp.read_step(args.var)
        if data is not None:
            data = np.squeeze(data)
            if data.ndim == 2:
                mid_col = data[:, data.shape[1] // 2]  # middle column
            elif data.ndim == 1:
                mid_col = data
            else:
                mid_col = None
            pp_steps.append(mid_col)
        else:
            pp_steps.append(None)
        r_pp.end_step()
    r_pp.close()
    print(f"  Read {len(pp_steps)} pp steps.")

    # --- Read error steps ---
    r_err = ReaderClass.Reader(args.readIO + "_err", err_bp, xml=args.xml)
    err_steps = []
    while True:
        status = r_err.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break
        r_err.set_read_vars([error_var])
        err_data = r_err.read_step(error_var)
        if err_data is not None:
            err_data = err_data.squeeze()
            max_err = np.max(np.abs(err_data))
            err_steps.append(max_err)
        else:
            err_steps.append(0.0)
        r_err.end_step()
    r_err.close()
    print(f"  Read {len(err_steps)} error steps.")

    n_steps = len(pp_steps)

    for step_idx in range(n_steps):
        pp_line = pp_steps[step_idx]
        if pp_line is None:
            print(f"  Skipping step {step_idx}: no pp data.")
            continue

        err_val = err_steps[step_idx] if step_idx < len(err_steps) else 0.0

        ny = len(pp_line)
        y = np.linspace(0, 1.0, ny)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(
            y,  # x: spatial coordinate 0 -> 1
            pp_line,  # y: pp values
            yerr=err_val,  # vertical  bars = max abs truncation error this step
            color="tab:red",
            ecolor="tab:blue",
            linewidth=1.2,
            elinewidth=0.8,
            capsize=2,
            alpha=0.85,
            label=f"{args.res}  err=±{err_val:.2e}",
        )

        ax.set_xlabel("value range [0, 1]")
        ax.set_ylabel(f"{args.var}")
        ax.set_title(f"{args.var} | {args.re} | {args.res} | Step {step_idx}")
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
