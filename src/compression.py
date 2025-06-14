import sys
import argparse
from adios2 import Adios, Stream
from mpi4py import MPI

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate streamline plots from ADIOS2 BP5 files')

    parser.add_argument('path',
                        type=str,
                        help='Path to the BP5 file to process (REQUIRED)')

    parser.add_argument('--errorBound', '-eb',
                        type=float,
                        default=0,
                        help='Error bound for compression (default: 0). No compression if 0. If using XML, it will take settings from there (optional)')

    parser.add_argument('max_steps',
                        type=int,
                        help='Maximum number of time steps to process (REQUIRED)')

    parser.add_argument('--xml', '-x',
                        type=str,
                        default=None,
                        help='Path to ADIOS2 XML configuration file (optional)')

    parser.add_argument('--output', '-o',
                        type=str,
                        default='compressed.bp',
                        help='Output file name (default: compressed.bp)')

    parser.add_argument('--compres_step', '-c',
                        type=int,
                        default=None,
                        help="If provided, compress only this specific step (optional)")

    return parser.parse_args()


def adios2_reader(bp_file, xml_file, error_bound, max_steps, compress_step, output_file="compressed.bp"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if xml_file is None:
        adios = Adios(comm)
        op = adios.define_operator("CompressMGARD", "mgard", {"tolerance": str(error_bound)})
        if rank == 0:
            print("Using MGARD compression with error bound =", error_bound)
            print("Code doesn't work without XML yet")
        sys.exit(1)
    else:
        adios = Adios(xml_file, comm)
        op = None
        if rank == 0:
            print("Using compression settings from XML")

    Rio = adios.declare_io("ReadIOCompressed")
    Wio = adios.declare_io("WriteIOCompressed")

    if rank == 0:
        print(f"Opening input file: {bp_file}")

    with Stream(Rio, bp_file, "r", comm) as s, Stream(Wio, output_file, "w", comm) as w:
        variables_defined = False

        for step in s:
            status = s.begin_step()
            if not status:
                break
            
            current = s.current_step()
            comm.Barrier()

            if rank == 0:
                print(f"Processing step {current}")

            if compress_step is None or compress_step == current:
                w.begin_step()
                comm.Barrier()

                if not variables_defined:
                    for name, info in s.available_variables().items():
                        var_in = Rio.inquire_variable(name)
                        shape = var_in.shape()

                        if not shape or len(shape) < 3:
                            continue

                        total_slices = shape[2]
                        base = total_slices // size
                        rem = total_slices % size
                        local_count_2 = base + 1 if rank < rem else base
                        local_start_2 = rank * base + min(rank, rem)

                        start = [0, 0, local_start_2] + [0] * (len(shape) - 3)
                        count = list(shape)
                        count[2] = local_count_2

                        var_in.set_selection((start, count))
                        data = s.read(var_in)

                        var_out = Wio.define_variable(name, data, shape, start, count)
                        if op:
                            var_out.add_operation(op, {})

                    variables_defined = True

                for name, info in s.available_variables().items():
                    var_in = Rio.inquire_variable(name)
                    shape = var_in.shape()

                    if not shape or len(shape) < 3:
                        continue

                    total_slices = shape[2]
                    base = total_slices // size
                    rem = total_slices % size
                    local_count_2 = base + 1 if rank < rem else base
                    local_start_2 = rank * base + min(rank, rem)

                    start = [0, 0, local_start_2] + [0] * (len(shape) - 3)
                    count = list(shape)
                    count[2] = local_count_2

                    var_in.set_selection((start, count))
                    data = s.read(var_in)
                    w.write(name, data)

                w.end_step()
                comm.Barrier()

            if not status or current >= max_steps - 1:
                if rank == 0:
                    print(f"Reached max_steps = {max_steps}")
                break


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running with {size} MPI processes")

    args = parse_arguments()

    input_file = args.path
    adios2_xml = args.xml
    error_bound = args.errorBound
    max_steps = args.max_steps
    output_file = args.output
    compress_step = args.compres_step

    if max_steps <= 0:
        if rank == 0:
            print("Error: max_steps must be a positive integer.")
        sys.exit(1)

    if rank == 0:
        print(f"Input file: {input_file}")
        print(f"ADIOS2 XML file: {adios2_xml}")
        print(f"Max steps: {max_steps}")
        print(f"Output file: {output_file}")
        if compress_step is not None:
            print(f"Compressing only step {compress_step}")
        else:
            print("Compressing every step")

    adios2_reader(
        bp_file=input_file,
        xml_file=adios2_xml,
        error_bound=error_bound,
        max_steps=max_steps,
        compress_step=compress_step,
        output_file=output_file
    )


if __name__ == "__main__":
    main()
