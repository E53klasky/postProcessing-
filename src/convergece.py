import matplotlib.pyplot as plt
import argparse
import adios2
import os
import numpy as np
import ReaderClass
from rich.traceback import install

RESOLUTIONS = ["129-257", "257-513", "513-1025", "1025-2049", "2049-4097"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

Ly_default = 1.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot vertical mean profiles across resolutions per time step."
    )
    parser.add_argument(
        "--var",
        "-v",
        type=str,
        required=True,
        help="Variable name (e.g. pp, ux, phi01)",
    )
    parser.add_argument(
        "--re",
        "-re",
        type=str,
        required=True,
        help="Reynolds number string pattern (e.g. RE112676)",
    )
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        help="ReaderClass name (default: reader1)",
    )
    parser.add_argument("--xml", "-x", type=str, default=None, help="Optional XML file")
    return parser.parse_args()


def find_bp_file(res_dir, re_string):
    files = os.listdir(res_dir)
    for f in files:
        if re_string in f and f.endswith(".bp5"):
            return os.path.join(res_dir, f)
    return None


def main():
    args = parse_arguments()

    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    all_data = {}
    base_path = "../my_data"

    for res in RESOLUTIONS:
        res_path = os.path.join(base_path, res)
        bp_path = find_bp_file(res_path, args.re)

        if bp_path is None:
            print(f"WARNING: No bp file found in {res} for {args.re}")
            continue

        print(f"Reading {res} -> {bp_path}")
        r = ReaderClass.Reader(args.readIO, bp_path, xml=args.xml)

        steps = []
        while True:
            status = r.begin_step()
            if status != adios2.bindings.StepStatus.OK:
                break

            r.set_read_vars([args.var])
            data = r.read_step(args.var)

            if data is not None:
                data = data.squeeze()

                if data.ndim != 2:
                    raise ValueError(f"Expected 2D data, got shape {data.shape}")

                mean_profile = np.mean(data, axis=1)
                steps.append(mean_profile)
            else:
                steps.append(None)

            r.end_step()

        all_data[res] = steps
        print(f"  Read {len(steps)} steps.")

    if not all_data:
        print("No data found.")
        return

    n_steps = len(next(iter(all_data.values())))

    for step_idx in range(n_steps):
        fig, ax = plt.subplots(figsize=(6, 8))

        for res, color in zip(RESOLUTIONS, COLORS):
            if res not in all_data:
                continue

            profile = all_data[res][step_idx]
            if profile is None:
                continue

            ny = len(profile)
            y = np.linspace(0, Ly_default, ny)

            ax.plot(profile, y, color=color, label=res)

        ax.set_xlabel(f"Mean {args.var}")
        ax.set_ylabel("y")
        ax.set_title(f"{args.var} | {args.re} | Step {step_idx}")
        ax.legend()

        fig.tight_layout()

        filename = f"{args.var}_{args.re}_mean_step_{step_idx}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

        print(f"Saved: {filename}")

    print("Plotting completed successfully.")


if __name__ == "__main__":
    install()
    main()
