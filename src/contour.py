import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
from adios2 import bindings
import os
from rich.traceback import install
from ReaderClass import Reader


def parse_arguments():
    install()
    parser = argparse.ArgumentParser(
        description="Contour plot: GT vs lower-res, with optional Monte Carlo truncation error ensemble"
    )
    parser.add_argument("--bpfile", "-f", required=True, help="Low-res BP file")
    parser.add_argument(
        "--bpfile_hires",
        "-fh",
        default=None,
        help="High-res / ground-truth BP file (optional)",
    )
    parser.add_argument(
        "--bpfile2", "-f2", default=None, help="Second BP file to overlay (optional)"
    )
    parser.add_argument(
        "--error_bp",
        "-e",
        default=None,
        help="Truncation error BP file (optional, enables MC ensemble)",
    )
    parser.add_argument("--Declare_Read_Io", "-d", required=True)
    parser.add_argument("--xml", "-x", default=None)
    parser.add_argument("--vars", "-v", required=True)
    parser.add_argument(
        "--vars_hires",
        "-vh",
        default=None,
        help="Var name(s) in hi-res file (defaults to same as --vars)",
    )
    parser.add_argument(
        "--vars2",
        "-v2",
        default=None,
        help="Vars to read from bpfile2 (defaults to same as --vars)",
    )
    parser.add_argument(
        "--bpfile_compressed",
        "-fc",
        default=None,
        help="Compressed BP file to overlay (optional)",
    )
    parser.add_argument(
        "--vars_compressed",
        "-vc",
        default=None,
        help="Var name(s) in compressed file (defaults to same as --vars)",
    )
    parser.add_argument("--error_var", "-ev", default=None)
    parser.add_argument("--level", "-l", type=float, default=-0.02)
    parser.add_argument("--output_dir", "-o", default="../RESULTS")
    parser.add_argument(
        "--num_ensemble",
        "-ne",
        type=int,
        default=100,
        help="Number of Monte Carlo samples (default: 100). "
        "Only used when --error_bp is supplied.",
    )
    return parser.parse_args()


def get_2d_data(data):
    if data is None:
        return None
    if len(data.shape) == 2:
        return data
    elif len(data.shape) == 3 and data.shape[0] == 1:
        return np.squeeze(data)
    return None


def make_mesh(arr):
    """Return X, Y meshgrid spanning [0,1]x[0,1] for the given 2D array."""
    ny, nx = arr.shape
    return np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))


def main():
    install()
    args = parse_arguments()

    vars_list = args.vars.split(",")
    vars_hires_list = args.vars_hires.split(",") if args.vars_hires else vars_list
    vars2_list = args.vars2.split(",") if args.vars2 else vars_list
    vars_compressed_list = (
        args.vars_compressed.split(",") if args.vars_compressed else vars_list
    )
    level = args.level
    num_ensemble = args.num_ensemble
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Readers ---
    r = Reader(args.Declare_Read_Io, args.bpfile, args.xml)
    r_hi = (
        Reader(args.Declare_Read_Io + "_hi", args.bpfile_hires, args.xml)
        if args.bpfile_hires
        else None
    )
    r2 = (
        Reader(args.Declare_Read_Io + "_2", args.bpfile2, args.xml)
        if args.bpfile2
        else None
    )
    r_err = (
        Reader(args.Declare_Read_Io + "_err", args.error_bp, args.xml)
        if args.error_bp
        else None
    )
    r_compressed = (
        Reader(args.Declare_Read_Io + "_comp", args.bpfile_compressed, args.xml)
        if args.bpfile_compressed
        else None
    )

    while True:
        status = r.begin_step()
        if status != bindings.StepStatus.OK:
            break

        status_hi = r_hi.begin_step() if r_hi else None
        status2 = r2.begin_step() if r2 else None
        status_err = r_err.begin_step() if r_err else None
        status_compressed = r_compressed.begin_step() if r_compressed else None

        r.set_read_vars(vars_list)
        step = r.current_step()

        if r_hi and status_hi == bindings.StepStatus.OK:
            r_hi.set_read_vars(vars_hires_list)
        if r2 and status2 == bindings.StepStatus.OK:
            r2.set_read_vars(vars2_list)
        if r_compressed and status_compressed == bindings.StepStatus.OK:
            r_compressed.set_read_vars(vars_compressed_list)

        for idx, var in enumerate(vars_list):

            # ------------------------------------------------------------------
            # READ FIELDS
            # Each field lives on its own native grid. No resampling needed —
            # all grids span [0,1]x[0,1] so contours overlay correctly.
            # ------------------------------------------------------------------

            # Low-res (required)
            arr = get_2d_data(r.read_step(var))
            if arr is None:
                continue

            # Hi-res / GT (optional)
            arr_hi = None
            if r_hi and status_hi == bindings.StepStatus.OK:
                var_hi = vars_hires_list[idx] if idx < len(vars_hires_list) else var
                arr_hi = get_2d_data(r_hi.read_step(var_hi))

            # Second field (optional)
            arr2 = None
            if r2 and status2 == bindings.StepStatus.OK:
                var2 = vars2_list[idx] if idx < len(vars2_list) else var
                arr2 = get_2d_data(r2.read_step(var2))

            # Compressed field (optional)
            arr_compressed = None
            if r_compressed and status_compressed == bindings.StepStatus.OK:
                var_c = (
                    vars_compressed_list[idx]
                    if idx < len(vars_compressed_list)
                    else var
                )
                arr_compressed = get_2d_data(r_compressed.read_step(var_c))

            # Truncation error field (optional, same grid as low-res)
            err_field = None
            if r_err and status_err == bindings.StepStatus.OK:
                error_var = (
                    args.error_var if args.error_var else f"{var}_truncation_error"
                )
                r_err.set_read_vars([error_var])
                err_field = get_2d_data(r_err.read_step(error_var))

            # ------------------------------------------------------------------
            # FIGURE
            # ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            legend_handles = []

            # ------------------------------------------------------------------
            # MONTE CARLO SPAGHETTI
            # One scalar per sample broadcast over err_field grid:
            #   perturbed[i,j] = arr[i,j] + scale * err_field[i,j]
            # ------------------------------------------------------------------
            if err_field is not None and num_ensemble > 1:
                print(f"  MC {num_ensemble} samples — {var} step {int(step)}")
                X_lo, Y_lo = make_mesh(arr)
                for _ in range(num_ensemble):
                    scale = 2.0 * np.random.rand() - 1.0  # one scalar in [-1, 1]
                    perturbed = arr + scale * err_field
                    try:
                        ax.contour(
                            X_lo,
                            Y_lo,
                            perturbed,
                            levels=[level],
                            colors=["gold"],
                            linewidths=0.8,
                            alpha=0.4,
                        )
                    except Exception:
                        pass
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="gold",
                        lw=1.5,
                        alpha=0.8,
                        label=f"MC ensemble (N={num_ensemble})",
                    )
                )

            # ------------------------------------------------------------------
            # LOW-RES CONTOUR — bold blue
            # ------------------------------------------------------------------
            try:
                X_lo, Y_lo = make_mesh(arr)
                ax.contour(
                    X_lo, Y_lo, arr, levels=[level], colors="blue", linewidths=2.5
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="blue",
                        lw=2.5,
                        label=f"{var} (low-res  {arr.shape[1]}x{arr.shape[0]})",
                    )
                )
            except Exception as e:
                print(f"  WARNING low-res contour failed: {e}")

            # ------------------------------------------------------------------
            # HI-RES CONTOUR — bold red dashed, its own native meshgrid
            # ------------------------------------------------------------------
            if arr_hi is not None:
                try:
                    X_hi, Y_hi = make_mesh(arr_hi)
                    var_hi_label = (
                        vars_hires_list[idx] if idx < len(vars_hires_list) else var
                    )
                    ax.contour(
                        X_hi,
                        Y_hi,
                        arr_hi,
                        levels=[level],
                        colors="red",
                        linewidths=2.5,
                        linestyles="--",
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="red",
                            lw=2.5,
                            ls="--",
                            label=f"{var_hi_label} (hi-res  {arr_hi.shape[1]}x{arr_hi.shape[0]})",
                        )
                    )
                except Exception as e:
                    print(f"  WARNING hi-res contour failed: {e}")
            else:
                print(f"  WARNING arr_hi is None — hi-res not read or shape mismatch")

            # ------------------------------------------------------------------
            # SECOND FIELD CONTOUR — dashed green, its own native meshgrid
            # ------------------------------------------------------------------
            if arr2 is not None:
                try:
                    X2, Y2 = make_mesh(arr2)
                    var2_label = vars2_list[idx] if idx < len(vars2_list) else var
                    ax.contour(
                        X2,
                        Y2,
                        arr2,
                        levels=[level],
                        colors=["green"],
                        linewidths=2.0,
                        linestyles="--",
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="green",
                            lw=2.0,
                            ls="--",
                            label=f"{var2_label} (file2  {arr2.shape[1]}x{arr2.shape[0]})",
                        )
                    )
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # COMPRESSED CONTOUR — purple dashed, its own native meshgrid
            # ------------------------------------------------------------------
            if arr_compressed is not None:
                try:
                    X_c, Y_c = make_mesh(arr_compressed)
                    var_c_label = (
                        vars_compressed_list[idx]
                        if idx < len(vars_compressed_list)
                        else var
                    )
                    ax.contour(
                        X_c,
                        Y_c,
                        arr_compressed,
                        levels=[level],
                        colors=["purple"],
                        linewidths=2.0,
                        linestyles="--",
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="purple",
                            lw=2.0,
                            ls="--",
                            label=f"{var_c_label} (compressed  {arr_compressed.shape[1]}x{arr_compressed.shape[0]})",
                        )
                    )
                except Exception as e:
                    print(f"  WARNING compressed contour failed: {e}")

            # ------------------------------------------------------------------
            # AXES + LABELS
            # ------------------------------------------------------------------
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)

            parts = [var, f"step {int(step)}", f"level={level}"]
            if arr_hi is not None:
                parts.append("+hi-res")
            if arr2 is not None:
                parts.append("+file2")
            if arr_compressed is not None:
                parts.append("+compressed")
            if err_field is not None:
                parts.append(f"N={num_ensemble} MC")
            ax.set_title("  |  ".join(parts), fontsize=12)

            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", framealpha=0.8)

            # ------------------------------------------------------------------
            # SAVE
            # ------------------------------------------------------------------
            mc_suffix = f"_ne{num_ensemble}" if err_field is not None else ""
            hi_suffix = "_hires" if arr_hi is not None else ""
            f2_suffix = "_f2" if arr2 is not None else ""
            comp_suffix = "_compressed" if arr_compressed is not None else ""
            fname = f"{var}_step_{int(step)}_level_{level}{hi_suffix}{f2_suffix}{comp_suffix}{mc_suffix}.png"
            fpath = os.path.join(args.output_dir, fname)
            plt.tight_layout()
            plt.savefig(fpath, dpi=150)
            plt.close()
            print("Saved:", fpath)

        r.end_step()
        if r_hi and status_hi == bindings.StepStatus.OK:
            r_hi.end_step()
        if r2 and status2 == bindings.StepStatus.OK:
            r2.end_step()
        if r_compressed and status_compressed == bindings.StepStatus.OK:
            r_compressed.end_step()
        if r_err and status_err == bindings.StepStatus.OK:
            r_err.end_step()

    r.close()
    if r_hi:
        r_hi.close()
    if r2:
        r2.close()
    if r_compressed:
        r_compressed.close()
    if r_err:
        r_err.close()
    print("Finished")


if __name__ == "__main__":
    main()
