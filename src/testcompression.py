import argparse
import adios2
import ReaderClass
import WrighterClass
from Reader3DClass import Reader3D
from Writer3DClass import Writer3D
import numpy as np
from mpi4py import MPI
import time
from rich.traceback import install


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compression code to compress the data")
    parser.add_argument(
        "--input",
        "-in",
        type=str,
        required=True,
        help="Adios file to compress",
    )
    parser.add_argument(
        "--readIO",
        "--ReadIO",
        "-rio",
        type=str,
        default="reader1",
        required=False,
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
        "--error_bound",
        "-eb",
        type=float,
        required=True,
        default=None,
        help="Error bound for compression in xml file (required)",
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        required=True,
        help="Path to ADIOS2 XML configuration file (required)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="compressed.bp",
        help="Output file name (default: compressed.bp)",
    )
    parser.add_argument(
        "--use_3d",
        "--3d",
        "-3d",
        action="store_true",
        help="Use 3D domain decomposition instead of last dimension decomposition",
    )
    return parser.parse_args()


def main():
    program_start = time.time()
    parser = parse_arguments()
    input_file = parser.input
    readIO = parser.readIO
    WrightIO = parser.WrightIO
    xml = parser.xml
    output = parser.output
    use_3d = parser.use_3d
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        times_file = open("compress_times.txt", "w")
        times_file.write(f"Program started at {program_start:.6f}\n")
        if use_3d:
            times_file.write("Using 3D domain decomposition\n")
        else:
            times_file.write("Using last dimension decomposition\n")
    else:
        times_file = None

    if use_3d:
        r = Reader3D(readIO, input_file, xml=xml, comm=comm)
        w = Writer3D(WrightIO, output, xml=xml, comm=comm)
        if rank == 0:
            print("Using 3D domain decomposition classes")
    else:
        r = ReaderClass.Reader(readIO, input_file, xml=xml, comm=comm)
        w = WrighterClass.Writer(WrightIO, output, xml=xml, comm=comm)
        if rank == 0:
            print("Using standard (last dimension) decomposition classes")

    flag = True
    while True:
        step_start = time.time()
        status = r.begin_step()

        if status != adios2.bindings.StepStatus.OK:
            break

        current_step = r.current_step()
        if rank == 0:
            times_file.write(f"\nReading step: {int(current_step)}\n")
        w.begin_step()

        for name, info in r.Adios_reader.available_variables().items():
            read_start = time.time()
            r.set_read_vars([name])

            data = r.read_step(name)
            read_end = time.time()

            if data is not None:
                data = np.array(data, dtype=np.float64)
                
                write_start = time.time()
                w.write(name, data)
                if flag:
                    w.write("error_bound", np.array([parser.error_bound], dtype=np.float64))
                    flag = False
                write_end = time.time()

                if rank == 0:
                    times_file.write(
                        f"Variable: {name}, Shape: {data.shape}, Read time: {read_end - read_start:.6f} s, "
                        f"Write time: {write_end - write_start:.6f} s\n"
                    )
            else:
                if rank == 0:
                    print(f"Warning: No data read for variable {name} on rank {rank}")

        r.end_step()
        w.end_step()
        step_end = time.time()
        if rank == 0:
            times_file.write(f"Step time: {step_end - step_start:.6f} s\n")

    r.close()
    w.close()
    program_end = time.time()
    if rank == 0:
        times_file.write(f"\nProgram ended at {program_end:.6f}\n")
        times_file.write(f"Total program time: {program_end - program_start:.6f} s\n")
        times_file.close()
    
    decomp_type = "3D domain" if use_3d else "last dimension"
    print(f"Compression completed using {decomp_type} decomposition. Output written to {output}.")


if __name__ == "__main__":
    main()