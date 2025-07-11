import argparse
import adios2
import ReaderClass
import WrighterClass
import numpy as np
from mpi4py import MPI
import time
from rich.traceback import install


def parse_arguments():
    parser = argparse.ArgumentParser(description="Copy code to compress the data")
    parser.add_argument(
        "--input",
        "-in",
        type=str,
        required=True,
        help="Adios file to compress",
    )
    parser.add_argument(
        "--readIO",
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
        "--xml",
        "-x",
        type=str,
        default=None,
        required=False,
        help="Path to ADIOS2 XML configuration file (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        default="copied.bp",
        help="Output file name (default: copied.bp)",
    )
    
    parser.add_argument(
        "--sleep",
        "-s",
        type=int,
        required=False,
        default=1,
        help="sleep time in seconds",
    )
    
    return parser.parse_args()


def main():
    program_start = time.time()
    
    
    parser = parse_arguments()
    
    sleep_time = parser.sleep
    input = parser.input
    readIO = parser.readIO
    WrightIO = parser.WrightIO
    xml = parser.xml
    output = parser.output
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        times_file = open("copied_times.txt", "w")
        times_file.write(f"Program started at {program_start:.6f}\n")
    else:
        times_file = None

    r = ReaderClass.Reader(readIO, input, xml=xml, comm=comm)
    w = WrighterClass.Writer(WrightIO, output, xml=xml, comm=comm)

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

            write_start = time.time()
            time.sleep(sleep_time)
            w.write(name, data)

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
    print(f"Copier completed. Output written to {output}.")


if __name__ == "__main__":
    install()
    main()
