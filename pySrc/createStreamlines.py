import numpy as np
import argparse
import adios2
from rich.traceback import install
from scipy.interpolate import RegularGridInterpolator
import ReaderClass
import WrighterClass
import re
from mpi4py import MPI
import sys


# clean up code and make this work for 3d
def rk4_streamline_from_grid(
    x0, y0, vx, vy, max_len=3.0, dt=0.01, max_steps=1000, xlim=None, ylim=None
):
    xgrid = np.linspace(0, 1, vx.shape[1])
    ygrid = np.linspace(0, 1, vx.shape[0])

    interp_vx = RegularGridInterpolator((ygrid, xgrid), vx)
    interp_vy = RegularGridInterpolator((ygrid, xgrid), vy)

    def vector_field(x, y):
        point = np.array([y, x])
        u = (
            interp_vx(point)[0]
            if isinstance(interp_vx(point), np.ndarray)
            else float(interp_vx(point))
        )
        v = (
            interp_vy(point)[0]
            if isinstance(interp_vy(point), np.ndarray)
            else float(interp_vy(point))
        )
        norm = np.hypot(u, v)
        if norm < 1e-8:
            return np.array([0.0, 0.0])
        return np.array([u, v]) / norm

    paths = []
    coords_x = []
    coords_y = []
    offsets = []
    offset = 0
    for i in range(len(x0)):
        x = x0[i]
        y = y0[i]
        cnt = 0
        arc_len = 0
        path = [(x, y)]
        path_x = [x]
        path_y = [y]
        offsets.append(offset)
        for _ in range(max_steps):
            cnt += 1
            offset += 1
            k1 = vector_field(x, y)
            k2 = vector_field(x + dt * k1[0] / 2, y + dt * k1[1] / 2)
            k3 = vector_field(x + dt * k2[0] / 2, y + dt * k2[1] / 2)
            k4 = vector_field(x + dt * k3[0], y + dt * k3[1])
            dx, dy = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x_prev = x
            y_prev = y

            x += dx
            y += dy
            arc_len += np.sqrt(pow((x - x_prev), 2) + pow((y - y_prev), 2))

            if xlim and (x < xlim[0] or x > xlim[1]):
                print("xlim")
                break
            if ylim and (y < ylim[0] or y > ylim[1]):
                print("ylim")
                break
            path.append((x, y))
            path_x.append(x)
            path_y.append(y)

            if max_len > 0 and arc_len >= max_len:
                print("arc length")
                break

        print(f"Arc length: {arc_len}")
        print("--" * 60)
        print(f"Number of RK4 steps: {cnt}")
        print("--" * 60)
        print(f"Number of points in streamline segments: {len(path)}")
        paths.append(path)
        coords_x.append(path_x)
        coords_y.append(path_y)

    return (np.array(coords_x), np.array(coords_y), np.array(offsets))


def parse_seed_points(seed_str):
    matches = re.findall(r"\(([^,]+),([^)]+)\)", seed_str)
    if not matches:
        raise ValueError("Invalid seed format. Use format like: '(0.1,0.5),(0.4,0.4)'")
    x_vals = []
    y_vals = []
    for x, y in matches:
        x_vals.append(float(x.strip()))
        y_vals.append(float(y.strip()))
    return np.array(x_vals), np.array(y_vals)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate streamline plots from ADIOS2 BP files"
    )
    parser.add_argument(
        "--file",
        "-f",
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
        required=True,
        help="Comma-separated list of seed points in the format '(x1,y1),(x2,y2)' (REQUIRED)",
    )
    parser.add_argument(
        "--io_read_name",
        "-ior",
        type=str,
        required=True,
        help="Name you want to declare the io name as (if you are using the xml this must match) (REQUIRED)",
    )
    parser.add_argument(
        "--io_write_name",
        "-iow",
        type=str,
        required=True,
        help="Name you want to declare the io name as for writing (REQUIRED)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="segments.bp",
        required=False,
        help="Output file name default: segments.bp (optional)",
    )

    parser.add_argument(
        "--step_size",
        "-dh",
        required=False,
        type=np.float64,
        default=0.001,
        help="step size for the rk steps dh",
    )
    parser.add_argument(
        "--num_RK_steps",
        "-step",
        required=False,
        type=int,
        default=4500,
        help="Number of RK steps to take",
    )

    return parser.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        print("Only works with 1 ranks")
        sys.exit()
    args = parse_arguments()

    bp_file = args.file
    xml_file = args.xml
    io_name = args.io_read_name
    io_write_name = args.io_write_name
    var_names = [v.strip() for v in args.vars.split(",")]
    x_seeds, y_seeds = parse_seed_points(args.seeds_points)
    output_file = args.output
    dt = args.step_size
    num_rk_steps = args.num_RK_steps
    adios_obj = adios2.Adios()
    reader = ReaderClass.Reader(
        IO_Name=io_name, bp_file=bp_file, xml=xml_file, comm=comm
    )

    wrigher = WrighterClass.Writer(
        IO_Name=io_write_name, bp_file=output_file, xml=xml_file, comm=comm
    )

    print("Making streamlines Now")

    not_defined = True
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

        coords_x, coords_y, offsets = rk4_streamline_from_grid(
            x_seeds,
            y_seeds,
            data[0],
            data[1],
            max_len=1000,
            dt=dt,
            max_steps=num_rk_steps,
        )

        # How to get this to work in 3d???????????????????????????
        coords_x = np.ascontiguousarray(np.array(coords_x, dtype=np.float64))
        coords_y = np.ascontiguousarray(np.array(coords_y, dtype=np.float64))
        offsets = np.ascontiguousarray(np.array(offsets, dtype=np.int32))

        coords_x = coords_x.flatten()
        coords_y = coords_y.flatten()
        offsets = offsets.flatten()

        # I would have
        wrigher.write("coords_x", coords_x)
        wrigher.write("coords_y", coords_y)
        wrigher.write("offsets", offsets)
        wrigher.end_step()
        reader.end_step()

    reader.close()
    wrigher.close()
    print(f"All streamline segments saved to ./{output_file}!")


if __name__ == "__main__":
    install()
    main()
