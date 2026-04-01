import numpy as np
import argparse
import adios2
from rich.traceback import install
from scipy.interpolate import RegularGridInterpolator
import ReaderClass
import WrighterClass
import re
import math


def rk4_streamline_from_grid(
    x0,
    y0,
    z0,
    vx,
    vy,
    vz=None,
    vx_trunc=None,
    vy_trunc=None,
    vz_trunc=None,
    num_ensemble=1,
    max_len=10.0,
    dt=0.01,
    max_steps=1000,
    xlim=None,
    ylim=None,
    zlim=None,
):
    is_3d = vz is not None and z0 is not None
    use_trunc = (vx_trunc is not None) and (num_ensemble > 1)

    if is_3d:
        zgrid = np.linspace(0, 1.0, vz.shape[0])
        ygrid = np.linspace(0, 1.0, vy.shape[1])
        xgrid = np.linspace(0, 1.0, vx.shape[2])

        assert (
            vz.shape[0] == vy.shape[0] == vx.shape[0]
        ), "Mismatch in z-dimension (axis 0)"
        assert (
            vz.shape[1] == vy.shape[1] == vx.shape[1]
        ), "Mismatch in y-dimension (axis 1)"
        assert (
            vz.shape[2] == vy.shape[2] == vx.shape[2]
        ), "Mismatch in x-dimension (axis 2)"

        interp_vx = RegularGridInterpolator((zgrid, ygrid, xgrid), vx, method="cubic")
        interp_vy = RegularGridInterpolator((zgrid, ygrid, xgrid), vy, method="cubic")
        interp_vz = RegularGridInterpolator((zgrid, ygrid, xgrid), vz, method="cubic")

        if use_trunc:
            ygrid_e = np.linspace(0, 1.0, vx_trunc.shape[0])
            xgrid_e = np.linspace(0, 1.0, vx_trunc.shape[1])
            interp_ex = RegularGridInterpolator(
                (ygrid_e, xgrid_e), vx_trunc, method="cubic"
            )
            interp_ey = RegularGridInterpolator(
                (ygrid_e, xgrid_e), vy_trunc, method="cubic"
            )
            interp_ez = (
                RegularGridInterpolator((ygrid_e, xgrid_e), vz_trunc, method="cubic")
                if vz_trunc is not None
                else None
            )

        def vector_field(x, y, z, pu=0.0, pv=0.0, pw=0.0):
            point = np.array([z, y, x])
            try:
                u = float(interp_vx(point).item()) + pu
                v = float(interp_vy(point).item()) + pv
                w = float(interp_vz(point).item()) + pw

                norm = np.linalg.norm([u, v, w])
                if norm < 1e-8:
                    return np.array([0.0, 0.0, 0.0])
                return np.array([u, v, w]) / norm
            except ValueError:
                return np.array([0.0, 0.0, 0.0])

    else:
        z0 = np.zeros(x0.shape[0])

        ygrid = np.linspace(0, 1, vy.shape[0])
        xgrid = np.linspace(0, 1, vx.shape[1])

        assert vy.shape[0] == vx.shape[0], "Mismatch in y-dimension (axis 0)"
        assert vy.shape[1] == vx.shape[1], "Mismatch in x-dimension (axis 1)"

        interp_vx = RegularGridInterpolator((ygrid, xgrid), vx, method="cubic")
        interp_vy = RegularGridInterpolator((ygrid, xgrid), vy, method="cubic")

        if use_trunc:
            ygrid_e = np.linspace(0, 1.0, vx_trunc.shape[0])
            xgrid_e = np.linspace(0, 1.0, vx_trunc.shape[1])
            interp_ex = RegularGridInterpolator(
                (ygrid_e, xgrid_e), vx_trunc, method="cubic"
            )
            interp_ey = RegularGridInterpolator(
                (ygrid_e, xgrid_e), vy_trunc, method="cubic"
            )

        def vector_field(x, y, pu=0.0, pv=0.0):
            point = np.array([y, x])
            try:
                u = float(interp_vx(point).item()) + pu
                v = float(interp_vy(point).item()) + pv

                norm = np.hypot(u, v)
                if norm < 1e-8:
                    return np.array([0.0, 0.0])
                return np.array([u, v]) / norm
            except ValueError:
                return np.array([0.0, 0.0])

    coords_x = []
    coords_y = []
    coords_z = []
    offsets = [0]

    for i in range(len(x0)):
        for _ensemble in range(num_ensemble):
            scale = (2 * np.random.rand() - 1) if use_trunc else 0.0

            x = x0[i]
            y = y0[i]
            z = z0[i]
            arc_len = 0

            coords_x.append(x)
            coords_y.append(y)
            if is_3d:
                coords_z.append(z)

            for _ in range(max_steps):

                if is_3d:
                    if use_trunc:
                        point_2d = np.array([y, x])
                        ex = float(interp_ex(point_2d).item())
                        ey = float(interp_ey(point_2d).item())
                        ez = (
                            float(interp_ez(point_2d).item())
                            if interp_ez is not None
                            else 0.0
                        )
                        pu = scale * ex
                        pv = scale * ey
                        pw = scale * ez
                    else:
                        pu, pv, pw = 0.0, 0.0, 0.0

                    k1 = vector_field(x, y, z, pu, pv, pw)
                    k2 = vector_field(
                        x + dt * k1[0] / 2,
                        y + dt * k1[1] / 2,
                        z + dt * k1[2] / 2,
                        pu,
                        pv,
                        pw,
                    )
                    k3 = vector_field(
                        x + dt * k2[0] / 2,
                        y + dt * k2[1] / 2,
                        z + dt * k2[2] / 2,
                        pu,
                        pv,
                        pw,
                    )
                    k4 = vector_field(
                        x + dt * k3[0], y + dt * k3[1], z + dt * k3[2], pu, pv, pw
                    )

                    dx, dy, dz = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    x_prev, y_prev, z_prev = x, y, z
                    x += dx
                    y += dy
                    z += dz
                    arc_len += np.sqrt(
                        (x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2
                    )

                else:
                    if use_trunc:
                        point = np.array([y, x])
                        ex = float(interp_ex(point).item())
                        ey = float(interp_ey(point).item())
                        pu = scale * ex
                        pv = scale * ey
                    else:
                        pu, pv = 0.0, 0.0

                    k1 = vector_field(x, y, pu, pv)
                    k2 = vector_field(x + dt * k1[0] / 2, y + dt * k1[1] / 2, pu, pv)
                    k3 = vector_field(x + dt * k2[0] / 2, y + dt * k2[1] / 2, pu, pv)
                    k4 = vector_field(x + dt * k3[0], y + dt * k3[1], pu, pv)

                    dx, dy = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    x_prev, y_prev = x, y
                    x += dx
                    y += dy
                    arc_len += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

                if xlim and (x < xlim[0] or x > xlim[1]):
                    break
                if ylim and (y < ylim[0] or y > ylim[1]):
                    break
                if is_3d and zlim and (z < zlim[0] or z > zlim[1]):
                    break

                coords_x.append(x)
                coords_y.append(y)
                if is_3d:
                    coords_z.append(z)

                if max_len > 0 and arc_len >= max_len:
                    break

            offsets.append(len(coords_x))

    return (
        np.array(offsets),
        np.array(coords_x),
        np.array(coords_y),
        np.array(coords_z) if is_3d else None,
    )


def rk4_2D(
    x0, y0, vx, vy, vx_trunc, vy_trunc, num_ensemble, max_len, dt, max_steps, xlim, ylim
):
    return rk4_streamline_from_grid(
        x0=x0,
        y0=y0,
        z0=None,
        vx=vx,
        vy=vy,
        vz=None,
        vx_trunc=vx_trunc,
        vy_trunc=vy_trunc,
        vz_trunc=None,
        num_ensemble=num_ensemble,
        max_len=max_len,
        dt=dt,
        max_steps=max_steps,
        xlim=xlim,
        ylim=ylim,
        zlim=None,
    )


def rk4_3D(
    x0,
    y0,
    z0,
    vx,
    vy,
    vz,
    vx_trunc,
    vy_trunc,
    vz_trunc,
    num_ensemble,
    max_len,
    dt,
    max_steps,
    xlim,
    ylim,
    zlim,
):
    return rk4_streamline_from_grid(
        x0=x0,
        y0=y0,
        z0=z0,
        vx=vx,
        vy=vy,
        vz=vz,
        vx_trunc=vx_trunc,
        vy_trunc=vy_trunc,
        vz_trunc=vz_trunc,
        num_ensemble=num_ensemble,
        max_len=max_len,
        dt=dt,
        max_steps=max_steps,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )


def parse_seed_points(seed_str, num_dims=2):
    if num_dims == 2:
        matches = re.findall(r"\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)", seed_str)
        if not matches:
            raise ValueError("Invalid 2D seed format. Use: '(0.1,0.5),(0.4,0.4)'")
        x_vals, y_vals = zip(*[(float(x), float(y)) for x, y in matches])
        return np.array(x_vals), np.array(y_vals), None
    elif num_dims == 3:
        matches = re.findall(
            r"\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)", seed_str
        )
        if not matches:
            raise ValueError(
                "Invalid 3D seed format. Use: '(0.1,0.5,0.3),(0.4,0.4,0.6)'"
            )
        x_vals, y_vals, z_vals = zip(
            *[(float(x), float(y), float(z)) for x, y, z in matches]
        )
        return np.array(x_vals), np.array(y_vals), np.array(z_vals)
    else:
        raise ValueError("Only 2D or 3D seed points supported.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate streamline plots from ADIOS2 BP files"
    )

    # ---- main velocity file ----
    parser.add_argument(
        "--input",
        "-in",
        type=str,
        required=True,
        help="Path to the main BP file containing velocity fields (REQUIRED)",
    )
    parser.add_argument(
        "--xml", "-x", type=str, default=None, help="ADIOS2 XML config file (optional)"
    )
    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Velocity variable names, comma-separated e.g. ux,uy or ux,uy,uz (REQUIRED)",
    )

    # ---- truncation error files: you explicitly give the BP path and var name for each ----
    parser.add_argument(
        "--trunc_file_x",
        "-tfx",
        type=str,
        default=None,
        help=(
            "Path to the BP file holding the ux truncation error. "
            "Example: errors/1025-2049/RE112676/ux_diff_RE112676.bp"
        ),
    )
    parser.add_argument(
        "--trunc_var_x",
        "-tvx",
        type=str,
        default="ux_truncation_error",
        help="Variable name for ux truncation error inside --trunc_file_x (default: ux_truncation_error)",
    )

    parser.add_argument(
        "--trunc_file_y",
        "-tfy",
        type=str,
        default=None,
        help=(
            "Path to the BP file holding the uy truncation error. "
            "Example: errors/1025-2049/RE112676/uy_diff_RE112676.bp"
        ),
    )
    parser.add_argument(
        "--trunc_var_y",
        "-tvy",
        type=str,
        default="uy_truncation_error",
        help="Variable name for uy truncation error inside --trunc_file_y (default: uy_truncation_error)",
    )

    parser.add_argument(
        "--trunc_file_z",
        "-tfz",
        type=str,
        default=None,
        help="Path to the BP file holding the uz truncation error (3D only, optional)",
    )
    parser.add_argument(
        "--trunc_var_z",
        "-tvz",
        type=str,
        default="uz_truncation_error",
        help="Variable name for uz truncation error inside --trunc_file_z (default: uz_truncation_error)",
    )

    # ---- ensemble ----
    parser.add_argument(
        "--num_ensemble",
        "-ne",
        type=int,
        default=1,
        help=(
            "Number of perturbed streamlines per seed (default: 1 = deterministic). "
            "Set e.g. to 1000 for a spaghetti plot. "
            "Requires --trunc_file_x and --trunc_file_y (and --trunc_file_z for 3D)."
        ),
    )

    # ---- seeds ----
    parser.add_argument(
        "--seeds_points",
        "-s",
        type=str,
        default=None,
        help="Seed points: 2D '(x1,y1),(x2,y2)' or 3D '(x1,y1,z1),(x2,y2,z2)'",
    )
    parser.add_argument(
        "--num_random_seeds",
        "-nrs",
        type=int,
        default=0,
        help="Number of random seeds to generate (default: 0)",
    )

    # ---- IO names ----
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        help="IO name for the main velocity reader (default: reader1)",
    )
    parser.add_argument(
        "--WrightIO",
        "-wio",
        type=str,
        default="writer1",
        help="IO name for the output writer (default: writer1)",
    )

    # ---- output ----
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="segments.bp",
        help="Output file name (default: segments.bp)",
    )

    # ---- RK parameters ----
    parser.add_argument(
        "--step_size",
        "-dh",
        type=np.float64,
        default=0.001,
        help="RK step size (default: 0.001)",
    )
    parser.add_argument(
        "--num_RK_steps",
        "-step",
        type=int,
        default=4500,
        help="Number of RK steps per streamline (default: 4500)",
    )
    parser.add_argument(
        "--max_length",
        "-MLen",
        type=float,
        default=4.5,
        help="Max streamline arc length (default: 4.5)",
    )

    return parser.parse_args()


def check_bounds(name, arr):
    if not np.all((0.0 <= np.array(arr)) & (np.array(arr) <= 1.0)):
        raise ValueError(f"All values in {name} must be between 0 and 1. Got: {arr}")


def read_all_steps_from_bp(bp_file, var_name, io_name):
    """
    Open a BP file, read every step of one variable, return as a list of 2-D numpy arrays.
    Each array is squeezed so shape is e.g. (1025, 2049) not (1, 1025, 2049).
    """
    reader = ReaderClass.Reader(IO_Name=io_name, bp_file=bp_file, xml=None)
    steps = []
    while True:
        status = reader.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break
        reader.set_read_vars([var_name])
        arr = reader.read_step(var_name)
        if arr is not None:
            arr = arr.squeeze()
        steps.append(arr)
        reader.end_step()
    reader.close()
    return steps


def main():
    args = parse_arguments()

    var_names = [v.strip() for v in args.vars.split(",")]
    is_3d = len(var_names) == 3

    need_z_trunc = is_3d
    use_ensemble = (
        args.num_ensemble > 1
        and args.trunc_file_x is not None
        and args.trunc_file_y is not None
        and (not need_z_trunc or args.trunc_file_z is not None)
    )

    if args.num_ensemble > 1 and not use_ensemble:
        missing = []
        if args.trunc_file_x is None:
            missing.append("--trunc_file_x")
        if args.trunc_file_y is None:
            missing.append("--trunc_file_y")
        if need_z_trunc and args.trunc_file_z is None:
            missing.append("--trunc_file_z")
        print(
            f"WARNING: --num_ensemble={args.num_ensemble} but the following truncation "
            f"error files are missing: {missing}. Falling back to deterministic."
        )

    if use_ensemble:
        print(
            f"Ensemble mode ON  ->  {args.num_ensemble} perturbed streamlines per seed\n"
            f"  ux error : {args.trunc_file_x}  (var: {args.trunc_var_x})\n"
            f"  uy error : {args.trunc_file_y}  (var: {args.trunc_var_y})"
        )
        if is_3d:
            print(f"  uz error : {args.trunc_file_z}  (var: {args.trunc_var_z})")
    else:
        print("Ensemble mode OFF -> deterministic streamlines (no perturbation)")

    trunc_steps_x, trunc_steps_y, trunc_steps_z = None, None, None
    if use_ensemble:
        print("Pre-loading truncation error fields from separate BP files...")
        trunc_steps_x = read_all_steps_from_bp(
            args.trunc_file_x, args.trunc_var_x, io_name="trunc_reader_x"
        )
        trunc_steps_y = read_all_steps_from_bp(
            args.trunc_file_y, args.trunc_var_y, io_name="trunc_reader_y"
        )
        if is_3d:
            trunc_steps_z = read_all_steps_from_bp(
                args.trunc_file_z, args.trunc_var_z, io_name="trunc_reader_z"
            )
        print(f"  Loaded {len(trunc_steps_x)} truncation-error steps.")

    x_seeds, y_seeds, z_seeds = None, None, None

    if args.seeds_points:
        x_seeds, y_seeds, z_seeds = parse_seed_points(
            args.seeds_points, num_dims=3 if is_3d else 2
        )
        check_bounds("x_seeds", x_seeds)
        check_bounds("y_seeds", y_seeds)
        if is_3d:
            check_bounds("z_seeds", z_seeds)

    elif args.num_random_seeds > 0:
        rand_seeds = []
        for _ in range(args.num_random_seeds):
            x = np.random.rand()
            y = np.random.rand()
            if is_3d:
                rand_seeds.append((x, y, np.random.rand()))
            else:
                rand_seeds.append((x, y))
        print(f"Generated {args.num_random_seeds} random seeds.")

        if is_3d:
            x_seeds, y_seeds, z_seeds = map(np.array, zip(*rand_seeds))
            print(f"Random seeds: {list(zip(x_seeds, y_seeds, z_seeds))}")
        else:
            x_seeds, y_seeds = map(np.array, zip(*rand_seeds))
            print(f"Random seeds: {list(zip(x_seeds, y_seeds))}")

        check_bounds("x_seeds", x_seeds)
        check_bounds("y_seeds", y_seeds)
        if is_3d:
            check_bounds("z_seeds", z_seeds)
    else:
        raise ValueError(
            "You must specify either --seeds_points or a positive --num_random_seeds."
        )

    reader = ReaderClass.Reader(IO_Name=args.readIO, bp_file=args.input, xml=args.xml)
    wrigher = WrighterClass.Writer(
        IO_Name=args.WrightIO, bp_file=args.output, xml=args.xml
    )

    print("Making streamlines now...")
    step_idx = 0

    while True:
        status = reader.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break

        current_step = reader.current_step()
        print(f"Reading step: {int(current_step)}")
        if step_idx == 19:
            wrigher.begin_step()

            reader.set_read_vars(var_names)
            if (
                reader.vars_Out.get(var_names[0]) is None
                or reader.vars_Out.get(var_names[1]) is None
            ):
                print("Velocity variables not found in the stream.")
                break

            data = []
            for name in var_names:
                arr = reader.read_step(name)
                if len(arr.shape) == 3 and arr.shape[0] == 1:
                    arr = np.squeeze(arr)
                data.append(arr)

            vx_trunc, vy_trunc, vz_trunc = None, None, None
            if use_ensemble:
                if step_idx < len(trunc_steps_x):
                    vx_trunc = trunc_steps_x[step_idx]
                    vy_trunc = trunc_steps_y[step_idx]
                    if is_3d:
                        vz_trunc = trunc_steps_z[step_idx]
                else:
                    print(
                        f"  WARNING: step {step_idx} has no matching truncation error "
                        f"(only {len(trunc_steps_x)} error steps available). "
                        "Running deterministic for this step."
                    )

            ne = args.num_ensemble if (use_ensemble and vx_trunc is not None) else 1

            if not is_3d:
                offsets, coords_x, coords_y, coords_z = rk4_2D(
                    x_seeds,
                    y_seeds,
                    data[0],
                    data[1],
                    vx_trunc,
                    vy_trunc,
                    ne,
                    max_len=args.max_length,
                    dt=args.step_size,
                    max_steps=args.num_RK_steps,
                    xlim=None,
                    ylim=None,
                )
                coords_x = np.ascontiguousarray(
                    np.array(coords_x, dtype=np.float64)
                ).flatten()
                coords_y = np.ascontiguousarray(
                    np.array(coords_y, dtype=np.float64)
                ).flatten()
                offsets = np.ascontiguousarray(
                    np.array(offsets, dtype=np.int32)
                ).flatten()

                wrigher.write("coords_x", coords_x)
                wrigher.write("coords_y", coords_y)
                wrigher.write("offsets", offsets)

            else:
                offsets, coords_x, coords_y, coords_z = rk4_3D(
                    x_seeds,
                    y_seeds,
                    z_seeds,
                    data[0],
                    data[1],
                    data[2],
                    vx_trunc,
                    vy_trunc,
                    vz_trunc,
                    ne,
                    max_len=args.max_length,
                    dt=args.step_size,
                    max_steps=args.num_RK_steps,
                    xlim=None,
                    ylim=None,
                    zlim=None,
                )
                coords_x = np.ascontiguousarray(np.array(coords_x, dtype=np.float64))
                coords_y = np.ascontiguousarray(np.array(coords_y, dtype=np.float64))
                coords_z = np.ascontiguousarray(np.array(coords_z, dtype=np.float64))
                offsets = np.ascontiguousarray(np.array(offsets, dtype=np.int32))

                wrigher.write("coords_x", coords_x)
                wrigher.write("coords_y", coords_y)
                wrigher.write("coords_z", coords_z)
                wrigher.write("offsets", offsets)

            wrigher.end_step()
        reader.end_step()
        step_idx += 1

    reader.close()
    wrigher.close()
    print(f"All streamline segments saved to ./{args.output}!")


if __name__ == "__main__":
    install()
    main()
