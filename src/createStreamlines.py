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
    max_len=10.0,
    dt=0.01,
    max_steps=1000,
    xlim=None,
    ylim=None,
    zlim=None,
):
    is_3d = vz is not None and z0 is not None

    if is_3d:
        # zgrid = np.linspace(0, 1, vx.shape[0])
        # ygrid = np.linspace(0, 1, vx.shape[1])
        # xgrid = np.linspace(0, 1, vx.shape[2])

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

        def vector_field(x, y, z):
            point = np.array([z, y, x])
            try:
                u = float(interp_vx(point).item())
                v = float(interp_vy(point).item())
                w = float(interp_vz(point).item())
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

        assert vy.shape[0] == vx.shape[0], "Mismatch in z-dimension (axis 0)"
        assert vy.shape[1] == vx.shape[1], "Mismatch in y-dimension (axis 1)"

        interp_vx = RegularGridInterpolator((ygrid, xgrid), vx, method="cubic")
        interp_vy = RegularGridInterpolator((ygrid, xgrid), vy, method="cubic")

        def vector_field(x, y):
            point = np.array([y, x])
            try:
                u = float(interp_vx(point).item())
                v = float(interp_vy(point).item())
                norm = np.hypot(u, v)
                if norm < 1e-8:
                    return np.array([0.0, 0.0])
                return np.array([u, v]) / norm
            except ValueError:
                return np.array([0.0, 0.0])

    paths = []
    coords_x = []
    coords_y = []
    coords_z = []
    offsets = [0]
    cnt = 0

    for i in range(len(x0)):
        x = x0[i]
        y = y0[i]
        z = z0[i]
        cnt = 0
        arc_len = 0
        path = [(x, y, z)] if is_3d else [(x, y)]

        coords_x.append(x)
        coords_y.append(y)
        if is_3d:
            coords_z.append(z)

        for _ in range(max_steps):
            cnt += 1

            if is_3d:
                k1 = vector_field(x, y, z)
                k2 = vector_field(
                    x + dt * k1[0] / 2, y + dt * k1[1] / 2, z + dt * k1[2] / 2
                )
                k3 = vector_field(
                    x + dt * k2[0] / 2, y + dt * k2[1] / 2, z + dt * k2[2] / 2
                )
                k4 = vector_field(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
                dx, dy, dz = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                x_prev, y_prev, z_prev = x, y, z
                x += dx
                y += dy
                z += dz
                arc_len += np.sqrt(
                    (x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2
                )
            else:
                k1 = vector_field(x, y)
                k2 = vector_field(x + dt * k1[0] / 2, y + dt * k1[1] / 2)
                k3 = vector_field(x + dt * k2[0] / 2, y + dt * k2[1] / 2)
                k4 = vector_field(x + dt * k3[0], y + dt * k3[1])
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
            if cnt > 1:
                d = math.sqrt((x0[i] - x) ** 2 + (y0[i] - y) ** 2 + (z0[i] - z) ** 2)
                # comment out for
                # if d < 0.0001:
                #     print(d,x,y)
                #     break

            if is_3d:
                path.append((x, y, z))
                coords_z.append(z)
            else:
                path.append((x, y))

            coords_x.append(x)
            coords_y.append(y)

            if max_len > 0 and arc_len >= max_len:
                break

        paths.append(path)

        offsets.append(len(coords_x))

    return (
        np.array(offsets),
        np.array(coords_x),
        np.array(coords_y),
        np.array(coords_z) if is_3d else None,
    )


def rk4_2D(x0, y0, vx, vy, max_len, dt, max_steps, xlim, ylim):
    return rk4_streamline_from_grid(
        x0=x0,
        y0=y0,
        z0=None,
        vx=vx,
        vy=vy,
        vz=None,
        max_len=max_len,
        dt=dt,
        max_steps=max_steps,
        xlim=xlim,
        ylim=ylim,
        zlim=None,
    )


def rk4_3D(x0, y0, z0, vx, vy, vz, max_len, dt, max_steps, xlim, ylim, zlim):
    return rk4_streamline_from_grid(
        x0=x0,
        y0=y0,
        z0=z0,
        vx=vx,
        vy=vy,
        vz=vz,
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
            raise ValueError(
                "Invalid 2D seed format. Use format like: '(0.1,0.5),(0.4,0.4)'"
            )
        x_vals, y_vals = zip(*[(float(x), float(y)) for x, y in matches])
        return np.array(x_vals), np.array(y_vals), None
    elif num_dims == 3:
        matches = re.findall(
            r"\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)", seed_str
        )
        if not matches:
            raise ValueError(
                "Invalid 3D seed format. Use format like: '(0.1,0.5,0.3),(0.4,0.4,0.6)'"
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
    parser.add_argument(
        "--input",
        "-in",
        type=str,
        required=True,
        help="Path to the BP file to process (REQUIRED)",
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        required=False,
        help="ADIOS2 XML config file default: None (optional)",
    )

    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Velocity variable names (order matters), separated by commas, e.g., vx,vy,vz (REQUIRED)",
    )

    parser.add_argument(
        "--seeds_points",
        "-s",
        type=str,
        required=False,
        default=None,
        help="Comma-separated list of seed points: 2D '(x1,y1),(x2,y2)' or 3D '(x1,y1,z1),(x2,y2,z2)' (REQUIRED)",
    )

    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        required=False,
        default="reader1",
        help="IO Name for the first Adios file (default: reader1)",
    )
    parser.add_argument(
        "--WrightIO",
        "-wio",
        type=str,
        default="writer1",
        required=False,
        help="IO Name for the second Adios file (default: writer1)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="segments.bp",
        required=False,
        help="Output file name (default: segments.bp) (optional)",
    )

    parser.add_argument(
        "--step_size",
        "-dh",
        required=False,
        type=np.float64,
        default=0.001,
        help="step size for the rk steps dh (defulat: 0.001)",
    )
    parser.add_argument(
        "--num_RK_steps",
        "-step",
        required=False,
        type=int,
        default=4500,
        help="Number of RK steps to take (default: 4500)",
    )

    parser.add_argument(
        "--max_length",
        "-MLen",
        required=False,
        default=4.5,
        type=float,
        help="The max length you want the streamlines to be (default: 4.5)",
    )
    
    parser.add_argument(
        "--num_random_seeds",
        "-nrs",
        required=False,
        type=int,
        default=0,
        help="Number of random seeds to generate (default: 0, no random seeds)",
    )

    return parser.parse_args()


def check_bounds(name, arr):
    if not np.all((0.0 <= np.array(arr)) & (np.array(arr) <= 1.0)):
        raise ValueError(f"All values in {name} must be between 0 and 1. Got: {arr}")


def main():
    args = parse_arguments()

    bp_file = args.input
    xml_file = args.xml
    io_name = args.readIO
    io_write_name = args.WrightIO
    var_names = [v.strip() for v in args.vars.split(",")]
    is_3d = len(var_names) == 3

    x_seeds, y_seeds, z_seeds = None, None, None

    if args.seeds_points:
        x_seeds, y_seeds, z_seeds = parse_seed_points(args.seeds_points, num_dims=3 if is_3d else 2)
        check_bounds("x_seeds", x_seeds)
        check_bounds("y_seeds", y_seeds)
        if is_3d:
            check_bounds("z_seeds", z_seeds)

    elif args.num_random_seeds > 0:
        rand_seeds = []
        for i in range(args.num_random_seeds):
            x = np.random.rand()
            y = np.random.rand()
            if is_3d:
                z = np.random.rand()
                rand_seeds.append((x, y, z))
            else:
                rand_seeds.append((x, y))
        print(f"Generated {args.num_random_seeds} random seeds.")

        if is_3d:
            x_seeds, y_seeds, z_seeds = zip(*rand_seeds)
            x_seeds = np.array(x_seeds)
            y_seeds = np.array(y_seeds)
            z_seeds = np.array(z_seeds)
        else:
            x_seeds, y_seeds = zip(*rand_seeds)
            x_seeds = np.array(x_seeds)
            y_seeds = np.array(y_seeds)

        check_bounds("x_seeds", x_seeds)
        check_bounds("y_seeds", y_seeds)
        if is_3d:
            check_bounds("z_seeds", z_seeds)

        if is_3d:
            print(f"Random seeds: {list(zip(x_seeds, y_seeds, z_seeds))}")
        else:
            print(f"Random seeds: {list(zip(x_seeds, y_seeds))}")
    else:
        raise ValueError("You must specify either seed points with --seeds_points or a positive number of random seeds with --num_ranndom_seeds.")

    
    output_file = args.output
    dt = args.step_size
    num_rk_steps = args.num_RK_steps

    reader = ReaderClass.Reader(IO_Name=io_name, bp_file=bp_file, xml=xml_file)

    wrigher = WrighterClass.Writer(
        IO_Name=io_write_name, bp_file=output_file, xml=xml_file
    )

    print("Making streamlines Now")

    while True:
        status = reader.begin_step()

        if status != adios2.bindings.StepStatus.OK:
            break

        current_step = reader.current_step()
        print(f"Reading step: {int(current_step)}")
        wrigher.begin_step()

        reader.set_read_vars(var_names)
        if (
            reader.vars_Out.get(var_names[0]) is None
            or reader.vars_Out.get(var_names[1]) is None
        ):
            print("Variables not found in the stream.")
            break
        data = []
        for i in range(len(var_names)):
            data.append(reader.read_step(var_names[i]))
            if len(data[i].shape) == 3 and data[i].shape[0] == 1:
                data[i] = np.squeeze(data[i])

        if len(data) == 2:
            offsets, coords_x, coords_y, coords_z = rk4_2D(
                x_seeds,
                y_seeds,
                data[0],
                data[1],
                max_len=args.max_length,
                dt=dt,
                max_steps=num_rk_steps,
                xlim=None,
                ylim=None,
            )
            coords_x = np.ascontiguousarray(np.array(coords_x, dtype=np.float64))
            coords_y = np.ascontiguousarray(np.array(coords_y, dtype=np.float64))
            offsets = np.ascontiguousarray(np.array(offsets, dtype=np.int32))

            coords_x = coords_x.flatten()
            coords_y = coords_y.flatten()
            offsets = offsets.flatten()

            wrigher.write("coords_x", coords_x)
            wrigher.write("coords_y", coords_y)
            wrigher.write("offsets", offsets)
            wrigher.end_step()
            reader.end_step()

        else:
            offsets, coords_x, coords_y, coords_z = rk4_3D(
                x_seeds,
                y_seeds,
                z_seeds,
                data[0],
                data[1],
                data[2],
                max_len=args.max_length,
                dt=dt,
                max_steps=num_rk_steps,
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

    reader.close()
    wrigher.close()
    print(f"All streamline segments saved to ./{output_file}!")


if __name__ == "__main__":
    install()
    main()
