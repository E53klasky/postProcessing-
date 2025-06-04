import sys
import argparse
from adios2 import Adios, Stream
from mpi4py import MPI

# Does not work without an XML file yet — to be fixed soon

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate streamline plots from ADIOS2 BP5 files')
    
    parser.add_argument('path', 
                        type=str, 
                        help='Path to the BP5 file to process (REQUIRED)')
    
    parser.add_argument("--errorBound", "-eb", 
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
    
    return parser.parse_args()


def adios2_reader(bp_file, xml_file, error_bound, max_steps, output_file="compressed.bp"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if xml_file == "no xml file provided":
        adios = Adios(comm)  
        op = adios.define_operator("CompressMGARD", "mgard", {"tolerance": str(error_bound)})
        print("Using MGARD with error bound =", error_bound)
        print("This code does not work without an XML file right now — coming soon ...")
        sys.exit(1)
    else:
        adios = Adios(xml_file, comm)  
        op = None
        print("Using compression settings from XML")

    Rio = adios.declare_io("ReadIOCompressed")
    Wio = adios.declare_io("WriteIOCompressed")
    
    print(f"Opening input file: {bp_file}")
    
    with Stream(Rio, bp_file, "r") as s, Stream(Wio, output_file, "w") as w:
        variables_defined = False

        for step in s:
            status = s.begin_step()
            if status:
                print(f"Processing step {s.current_step()}")
                w.begin_step()

                for name, info in s.available_variables().items():
                    var_in = Rio.inquire_variable(name)
                    shape = var_in.shape()

                    if not shape:
                        print(f"No shape info for variable {name}")
                        sys.exit(1)

                    if len(shape) < 3:
                        print(f"Skipping variable {name} — expected at least 3 dimensions, got {len(shape)}")
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

                    if not variables_defined:
                        Wio.define_variable(name, data, shape, start, count)

                    w.write(name, data)

                variables_defined = True
                w.end_step()

                if s.current_step() >= max_steps - 1:
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
    adios2_xml = args.xml if args.xml else "no xml file provided"     
    error_bound = args.errorBound
    max_steps = args.max_steps
    output_file = args.output

    if max_steps <= 0:
        if rank == 0:
            print("Error: max_steps must be a positive integer.")
        sys.exit(1)
        
    if rank == 0:
        print(f"Input file: {input_file}")
        print(f"ADIOS2 XML file: {adios2_xml}")
        print(f"Max steps: {max_steps}")
        print(f"Output file: {output_file}")
        
    adios2_reader(input_file, adios2_xml, error_bound, max_steps, output_file)


if __name__ == "__main__":
    main()
