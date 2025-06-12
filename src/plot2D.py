import adios2
import matplotlib.pyplot as plt
import argparse
import os
from mpi4py import MPI
import numpy as np

def parser_arguments():
    parser = argparse.ArgumentParser(description="Simple 2D plot")
    parser.add_argument('input_file', type=str, help='Path to the BP file to process (REQUIRED)')
    parser.add_argument('--xml', '-x', type=str, default=None, help='ADIOS2 XML config file (optional)')
    parser.add_argument('--vars', '-v', type=str, required=True, help='Variables to plot, separated by commas (REQUIRED)')
    parser.add_argument('--max_steps', '-n', type=int, required=True, help='Maximum number of timesteps to process')
    return parser.parse_args()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parser_arguments()
    input_file = args.input_file
    adios2_xml = args.xml
    var_list = args.vars.split(',')
    max_steps = args.max_steps

    if adios2_xml:
        adios = adios2.Adios(adios2_xml, comm)
    else:
        adios = adios2.Adios(comm)

    io = adios.declare_io("ReadIO")
    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    with adios2.Stream(io, input_file, 'r', comm) as stream:
        for step_count, _ in enumerate(stream):
            if step_count >= max_steps:
                break

            for var in var_list:
                if var not in stream.available_variables():
                    if rank == 0:
                        print(f"Variable {var} not found.")
                    continue

                var_in = io.inquire_variable(var)
                shape = var_in.shape()  

                if not shape or len(shape) != 3 or shape[0] != 1:
                    if rank == 0:
                        print(f"Skipping variable {var} due to unexpected shape {shape}")
                    continue

                Y, Z = shape[1], shape[2]

                base = Z // size
                rem = Z % size
                local_z = base + 1 if rank < rem else base
                local_start = rank * base + min(rank, rem)

                count = [1, Y, local_z]
                start = [0, 0, local_start]
                var_in.set_selection((start, count))

                local_data = stream.read(var)[0]  

                sendbuf = local_data.flatten()
                recvcounts = comm.gather(sendbuf.size, root=0)

                if rank == 0:
                    total_size = sum(recvcounts)
                    recvbuf = np.empty(total_size, dtype=sendbuf.dtype)
                else:
                    recvbuf = None

                comm.Gatherv(sendbuf, (recvbuf, recvcounts), root=0)

                if rank == 0:
                    full_data = np.empty((Y, Z), dtype=sendbuf.dtype)
                    offset = 0
                    for i in range(size):
                        z_len = base + 1 if i < rem else base
                        chunk = recvbuf[offset:offset + Y * z_len].reshape((Y, z_len))
                        z_start = i * base + min(i, rem)
                        full_data[:, z_start:z_start + z_len] = chunk
                        offset += Y * z_len

                    plt.imshow(full_data, cmap='inferno', aspect='auto')
                    plt.title(f"{var} at step {step_count}")
                    plt.colorbar()
                    plt.savefig(os.path.join(output_dir, f"{var}_step_{step_count}.png"))
                    plt.close()

    if rank == 0:
        print(f"Output saved to {output_dir}")

if __name__ == "__main__":
    main()
