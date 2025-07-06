import numpy as np
import adios2
import argparse
import sys
from rich.traceback import install
import ReaderClass
import WrighterClass
from mpi4py import MPI
import time


def exchange_ghost_cells(data, comm, axis=2, ghost_width=1):

    if comm is None or comm.Get_size() == 1:
        return data, (0, data.shape[axis])

    rank = comm.Get_rank()
    size = comm.Get_size()

    left_rank = rank - 1 if rank > 0 else MPI.PROC_NULL
    right_rank = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    def make_slice(idx_slice, axis, ndim):
        s = [slice(None)] * ndim
        s[axis] = idx_slice
        return tuple(s)

    left_boundary = data[make_slice(slice(0, ghost_width), axis, data.ndim)]
    right_boundary = data[make_slice(slice(-ghost_width, None), axis, data.ndim)]

    send_requests = []
    if left_rank != MPI.PROC_NULL:
        req = comm.Isend(left_boundary.copy(), dest=left_rank, tag=1)
        send_requests.append(req)
    if right_rank != MPI.PROC_NULL:
        req = comm.Isend(right_boundary.copy(), dest=right_rank, tag=2)
        send_requests.append(req)

    recv_requests = []
    left_ghost = None
    right_ghost = None

    if left_rank != MPI.PROC_NULL:
        left_ghost = np.empty_like(left_boundary)
        req = comm.Irecv(left_ghost, source=left_rank, tag=2)
        recv_requests.append(req)

    if right_rank != MPI.PROC_NULL:
        right_ghost = np.empty_like(right_boundary)
        req = comm.Irecv(right_ghost, source=right_rank, tag=1)
        recv_requests.append(req)

    MPI.Request.waitall(send_requests + recv_requests)

    extended_shape = list(data.shape)
    total_ghost_width = 0
    if left_ghost is not None:
        total_ghost_width += ghost_width
    if right_ghost is not None:
        total_ghost_width += ghost_width

    extended_shape[axis] += total_ghost_width
    extended_data = np.empty(extended_shape, dtype=data.dtype)

    start_idx = 0
    if left_ghost is not None:
        extended_data[make_slice(slice(0, ghost_width), axis, data.ndim)] = left_ghost
        start_idx = ghost_width

    end_idx = start_idx + data.shape[axis]
    extended_data[make_slice(slice(start_idx, end_idx), axis, data.ndim)] = data

    if right_ghost is not None:
        extended_data[make_slice(slice(-ghost_width, None), axis, data.ndim)] = (
            right_ghost
        )

    return extended_data, (start_idx, end_idx)


def extract_original_data(extended_data, slice_info, axis=2):
    start_idx, end_idx = slice_info
    s = [slice(None)] * extended_data.ndim
    s[axis] = slice(start_idx, end_idx)
    return extended_data[tuple(s)]


def compute_gradient_with_ghosts(data, comm, axis, edge_order=2):
    if axis == 2:
        ghost_width = 2 if edge_order == 2 else 1
        extended_data, slice_info = exchange_ghost_cells(
            data, comm, axis=2, ghost_width=ghost_width
        )

        grad_extended = np.gradient(extended_data, axis=axis, edge_order=edge_order)

        gradient = extract_original_data(grad_extended, slice_info, axis=2)
    else:
        gradient = np.gradient(data, axis=axis, edge_order=edge_order)

    return gradient


def curl_2d_with_ghosts(vx, vy, comm):
    grad_vy_x = compute_gradient_with_ghosts(vy, comm, axis=1, edge_order=2)
    grad_vx_y = compute_gradient_with_ghosts(vx, comm, axis=0, edge_order=2)
    return grad_vy_x - grad_vx_y


def curl_3d_with_ghosts(vx, vy, vz, comm):

    curl_x = compute_gradient_with_ghosts(
        vz, comm, axis=1, edge_order=2
    ) - compute_gradient_with_ghosts(vy, comm, axis=0, edge_order=2)

    curl_y = compute_gradient_with_ghosts(
        vx, comm, axis=0, edge_order=2
    ) - compute_gradient_with_ghosts(vz, comm, axis=2, edge_order=2)

    curl_z = compute_gradient_with_ghosts(
        vy, comm, axis=2, edge_order=2
    ) - compute_gradient_with_ghosts(vx, comm, axis=1, edge_order=2)

    return curl_x, curl_y, curl_z


def divergence_with_ghosts(vx, vy, vz, comm):
    if vz is None:
        div_x = compute_gradient_with_ghosts(vx, comm, axis=1, edge_order=2)
        div_y = compute_gradient_with_ghosts(vy, comm, axis=0, edge_order=2)
        return div_x + div_y
    else:
        div_x = compute_gradient_with_ghosts(vx, comm, axis=2, edge_order=2)
        div_y = compute_gradient_with_ghosts(vy, comm, axis=1, edge_order=2)
        div_z = compute_gradient_with_ghosts(vz, comm, axis=0, edge_order=2)
        return div_x + div_y + div_z


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
    program_start = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_arguments()
    var_names = [v.strip() for v in args.vars.split(",")]
    if len(var_names) < 2:
        print("At least two velocity variables are required (vx, vy[, vz])")
        sys.exit(1)

    if rank == 0:
        times_file = open("div_curl_times.txt", "w")
        times_file.write(f"Program started at {program_start:.6f}\n")
    else:
        times_file = None

    reader = ReaderClass.Reader(args.IO_Name1, args.file1, xml=args.xml, comm=comm)
    writer = WrighterClass.Writer(
        args.writeIO, bp_file=args.output, xml=args.xml, comm=comm
    )

    while True:
        step_start = time.time()
        status = reader.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break

        current_step = reader.current_step()
        if rank == 0:
            times_file.write(f"\nReading step: {int(current_step)}\n")
        print(f"Reading step: {int(current_step)}")
        writer.begin_step()

        reader.set_read_vars(var_names)

        vx_read_start = time.time()
        vx = reader.read_step(var_names[0])
        vx_read_end = time.time()
        
        vy_read_start = time.time()
        vy = reader.read_step(var_names[1])
        vy_read_end = time.time()
        
        vz = None
        vz_read_time = 0.0
        if len(var_names) == 3:
            vz_read_start = time.time()
            vz = reader.read_step(var_names[2])
            vz_read_end = time.time()
            vz_read_time = vz_read_end - vz_read_start

        if vx is None or vy is None:
            print(f"Rank {rank}: Failed to read velocity components")
            break

        if len(var_names) == 3 and vz is None:
            print(f"Rank {rank}: Failed to read third velocity component")
            break

        if rank == 0:
            times_file.write(f"Variable: {var_names[0]}, Read time: {vx_read_end - vx_read_start:.6f} s\n")
            times_file.write(f"Variable: {var_names[1]}, Read time: {vy_read_end - vy_read_start:.6f} s\n")
            if len(var_names) == 3:
                times_file.write(f"Variable: {var_names[2]}, Read time: {vz_read_time:.6f} s\n")

        if vx.ndim == 3 and vx.shape[0] == 1:
            vx = np.squeeze(vx)
            vy = np.squeeze(vy)
            vz = np.squeeze(vz) if vz is not None else None

            div_start = time.time()
            div = divergence_with_ghosts(vx, vy, None, comm)
            div_end = time.time()
            
            curl_start = time.time()
            curl_z = curl_2d_with_ghosts(vx, vy, comm)
            curl_end = time.time()
            
            write_start = time.time()
            writer.write("Div", div)
            writer.write("Curl_Z", curl_z)
            write_end = time.time()

            if rank == 0:
                times_file.write(
                    f"Divergence calculation time: {div_end - div_start:.6f} s\n"
                    f"Curl calculation time: {curl_end - curl_start:.6f} s\n"
                    f"Write time: {write_end - write_start:.6f} s\n"
                )

        elif vx.ndim == 2:
            div_start = time.time()
            div = divergence_with_ghosts(vx, vy, None, comm)
            div_end = time.time()
            
            curl_start = time.time()
            curl_z = curl_2d_with_ghosts(vx, vy, comm)
            curl_end = time.time()
            
            write_start = time.time()
            writer.write("Div", div)
            writer.write("Curl_Z", curl_z)
            write_end = time.time()

            if rank == 0:
                times_file.write(
                    f"Divergence calculation time: {div_end - div_start:.6f} s\n"
                    f"Curl calculation time: {curl_end - curl_start:.6f} s\n"
                    f"Write time: {write_end - write_start:.6f} s\n"
                )

        else:
            div_start = time.time()
            div = divergence_with_ghosts(vx, vy, vz, comm)
            div_end = time.time()
            
            curl_start = time.time()
            curl_x, curl_y, curl_z = curl_3d_with_ghosts(vx, vy, vz, comm)
            curl_end = time.time()
            
            write_start = time.time()
            writer.write("Div", div)
            writer.write("Curl_x", curl_x)
            writer.write("Curl_y", curl_y)
            writer.write("Curl_z", curl_z)
            write_end = time.time()

            if rank == 0:
                times_file.write(
                    f"Divergence calculation time: {div_end - div_start:.6f} s\n"
                    f"Curl calculation time: {curl_end - curl_start:.6f} s\n"
                    f"Write time: {write_end - write_start:.6f} s\n"
                )

        reader.end_step()
        writer.end_step()
        step_end = time.time()
        
        if rank == 0:
            times_file.write(f"Step time: {step_end - step_start:.6f} s\n")

    reader.close()
    writer.close()
    program_end = time.time()
    
    if rank == 0:
        times_file.write(f"\nProgram ended at {program_end:.6f}\n")
        times_file.write(f"Total program time: {program_end - program_start:.6f} s\n")
        times_file.close()
    
    print(f"DivCurl finished successfully and saved to ./{args.output}")


if __name__ == "__main__":
    install()
    main()