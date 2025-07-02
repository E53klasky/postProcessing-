import numpy as np
import adios2
import argparse
import sys
from rich.traceback import install
import ReaderClass
import WrighterClass
from mpi4py import MPI


def curl_2d(vx, vy):
    return np.gradient(vy, axis=1, edge_order=2) - np.gradient(vx, axis=0, edge_order=2)


def curl_3d(vx, vy, vz):
    curl_x = np.gradient(vz, axis=1, edge_order=2) - np.gradient(
        vy, axis=0, edge_order=2
    )
    curl_y = np.gradient(vx, axis=0, edge_order=2) - np.gradient(
        vz, axis=2, edge_order=2
    )
    curl_z = np.gradient(vy, axis=2, edge_order=2) - np.gradient(
        vx, axis=1, edge_order=2
    )
    return curl_x, curl_y, curl_z


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate divergence and curl from ADIOS2 BP5 velocity files"
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
        "--writeIO",
        "-wio",
        type=str,
        required=True,
        help="IO Name for the output Adios file",
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        help="Path to ADIOS2 XML configuration file (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="div_curl.bp",
        help="Output file name (default: div_curl.bp)",
    )
    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Velocity variable names (comma-separated, e.g., vx,vy,vz)",
    )
    return parser.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_arguments()
    var_names = [v.strip() for v in args.vars.split(",")]
    if len(var_names) < 2:
        print("At least two velocity variables are required (vx, vy[, vz])")
        sys.exit(1)

    reader = ReaderClass.Reader(args.IO_Name1, args.file1, xml=args.xml, comm=comm)
    writer = WrighterClass.Writer(
        args.writeIO, bp_file=args.output, xml=args.xml, comm=comm
    )

    while True:
        status = reader.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break

        current_step = reader.current_step()
        print(f"Reading step: {int(current_step)}")
        writer.begin_step()

        reader.set_read_vars(var_names)

        vx = reader.read_step(var_names[0])
        vy = reader.read_step(var_names[1])
        vz = reader.read_step(var_names[2]) if len(var_names) == 3 else None

        if any(v is None for v in (vx, vy, vz)):
            break

        if vx.ndim == 3 and vx.shape[0] == 1:
            vx = np.squeeze(vx)
            vy = np.squeeze(vy)
            vz = np.squeeze(vz) if vz is not None else None
            div = np.gradient(vx, axis=1, edge_order=2) + np.gradient(
                vy, axis=0, edge_order=2
            )
            curl_z = curl_2d(vx, vy)
            writer.write("Div", div)
            writer.write("Curl_Z", curl_z)
            writer.write("Curl_Z", curl_z)

        elif vx.ndim == 2:
            div = np.gradient(vx, axis=1, edge_order=2) + np.gradient(
                vy, axis=0, edge_order=2
            )
            curl_z = curl_2d(vx, vy)
            writer.write("Div", div)
            writer.write("Curl_Z", curl_z)
            writer.write("Curl_Z", curl_z)

        else:
            div = (
                np.gradient(vx, axis=2, edge_order=2)
                + np.gradient(vy, axis=1, edge_order=2)
                + np.gradient(vz, axis=0, edge_order=2)
            )
            curl_x, curl_y, curl_z = curl_3d(vx, vy, vz)
            writer.write("Div", div)
            writer.write("Curl_x", curl_x)
            writer.write("Curl_y", curl_y)
            writer.write("Curl_z", curl_z)

        reader.end_step()
        writer.end_step()

    reader.close()
    writer.close()
    print(f"DivCurl finished successfully and saved to ./{args.output}")


if __name__ == "__main__":
    install()
    main()
