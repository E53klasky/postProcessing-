import sys
import argparse
from adios2 import Adios, Stream
from mpi4py import MPI

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
                        help='Path to ADIOS2 XML configuration file (optional) ')
    
    parser.add_argument('--output', '-o',
                        type=str,
                        default='compressed.bp',
                        help='Output file name default: compressed.bp (optional)')
    
    return parser.parse_args()


def adios2_reader(bp_file, xml_file, error_bound, max_steps, output_file="compressed.bp"):
    if xml_file == "no xml file provided":
        adios = Adios()
        op = adios.define_operator("CompressMGARD", "mgard", {"tolerance": str(error_bound)})
        print("Using mgard with error bound =", error_bound)
        print("This code does not work with out an XML right now comming soon ...")
        sys.exit(1)
    else:
        adios = Adios(xml_file)
        op = None  
        print("Using compression settings from XML")

    Rio = adios.declare_io("ReadIOCompressed")
    Wio = adios.declare_io("WriteIOCompressed")
    
    print(f"Opening input file: {bp_file}")
    
    with Stream(Rio, bp_file, "r") as s, Stream(Wio, output_file, "w") as w:
        for step in s:
            status = s.begin_step()
            print(f"Processing step {s.current_step()}")

            for name, info in s.available_variables().items():
                # print(f"Variable: {name}")
                # shape = info.get("Shape", [])
                # dtype = info.get("Type", "")
                # print(f"\tShape: {shape}, Type: {dtype}")
                
                data = s.read(name)
   
                w.write(name, data, shape=data.shape, start=[0] * len(data.shape), count=data.shape)

                # if op is not None:
                #     var = w._io.define_variable(name, data, shape, start, count)
                #     var.add_operation(op)
                # else:
                #     w.write(name, data, shape, start, count)


            w.end_step()

            if s.current_step() >= max_steps - 1:
                print(f"Reached max_steps = {max_steps}")
                break


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    args = parse_arguments()
    
    input_file = args.path
    adios2_xml = args.xml if args.xml else "no xml file provided"     
    error_bound = args.errorBound
    max_steps = args.max_steps
    output_file = args.output

    if max_steps < 0:
        print("Error: max_steps must be a non-negative integer.")
        sys.exit(1)
        
    if rank == 0:
        print(f"Input file: {input_file}")
        print(f"ADIOS2 XML file: {adios2_xml}")
        print(f"Max steps: {max_steps}")
        print(f"Output file: {output_file}")
        
    adios2_reader(input_file, adios2_xml, error_bound, max_steps, output_file)


if __name__ == "__main__":
    main()
