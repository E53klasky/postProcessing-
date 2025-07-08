import argparse
import adios2
import ReaderClass
import WrighterClass
import numpy as np
from mpi4py import MPI


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
    parser = parse_arguments()
    bpfile1 = parser.bpfile1
    readIO = parser.readIO
    WrightIO = parser.WrightIO
    xml = parser.xml
    output = parser.output
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    r = ReaderClass.Reader(readIO, bpfile1, xml=xml, comm=comm)
    w = WrighterClass.Writer(WrightIO, output, xml=xml, comm=comm)

    flag = True
    while True:
        status = r.begin_step()

        if status != adios2.bindings.StepStatus.OK:
            break

        current_step = r.current_step()
        print(f"Reading step: {int(current_step)}")
        w.begin_step()

        for name, info in r.Adios_reader.available_variables().items():
            r.set_read_vars([name])
            data = np.array(r.read_step(name), dtype=np.float64)
            w.write(name, data)

            if flag:
                w.write("error_bound", np.array([parser.error_bound], dtype=np.float64))
                flag = False

        r.end_step()
        w.end_step()
    r.close()
    w.close()
    print(f"Compression completed. Output written to {output}.")


if __name__ == "__main__":
    main()
