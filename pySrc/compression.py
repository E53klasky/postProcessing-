import argparse
import adios2
import ReaderClass
import WrighterClass
import numpy as np
from mpi4py import MPI
import time


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate streamline plots from ADIOS2 BP5 files"
    )
    parser.add_argument(
        "--bpfile1",
        "--bp1",
        type=str,
        required=True,
        help="First Adios file with streamline segments (lower resolution/compressed)",
    )
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        help="IO Name for the first Adios file (default: reader1)",
    )
    parser.add_argument(
        "--WrightIO",
        "-wio",
        type=str,
        default="writer1",
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
        help="Path to ADIOS2 XML configuration file (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="compressed.bp",
        help="Output file name (default: compressed.bp)",
    )
    return parser.parse_args()


def main():
    program_start = time.time()
    parser = parse_arguments()
    bpfile1 = parser.bpfile1
    readIO = parser.readIO
    WrightIO = parser.WrightIO
    xml = parser.xml
    output = parser.output
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        times_file = open("compress_times.txt", "w")
        times_file.write(f"Program started at {program_start:.6f}\n")
    else:
        times_file = None

    r = ReaderClass.Reader(readIO, bpfile1, xml=xml, comm=comm)
    w = WrighterClass.Writer(WrightIO, output, xml=xml, comm=comm)

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

            data = np.array(r.read_step(name), dtype=np.float64)
            w.write(name, data)


            write_start = time.time()
            w.write(name, data)
            if flag:
                w.write("error_bound", np.array([parser.error_bound], dtype=np.float64))
                flag = False
            write_end = time.time()

            if rank == 0:
                times_file.write(
                    f"Variable: {name}, Read time: {read_end - read_start:.6f} s, "
                    f"Write time: {write_end - write_start:.6f} s\n"
                )

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
    print(f"Compression completed. Output written to {output}.")


if __name__ == "__main__":
    main()
