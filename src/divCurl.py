from mpi4py import MPI
import numpy as np
from adios2 import Adios, Stream
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate divergence and curl from ADIOS2 BP5 velocity files')
    
    parser.add_argument('input_file', 
                       type=str, 
                       help='Path to the input ADIOS2 BP5 file (REQUIRED)')
    
    parser.add_argument('--xml', 
                        '-x',
                       type=str,
                       default=None,
                       help='Path to ADIOS2 XML configuration file (optional)')
    
    parser.add_argument('--output', 
                        '-o',
                       type=str,
                       default='div_curl.bp',
                       help='Output file name default: div_curl.bp (optional))')
    
    parser.add_argument('max_steps', 
                        type=int, 
                        help='Maximum number of time steps to process (REQUIRED)')
   
    return parser.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running with {size} MPI processes")

    args = parse_arguments()
    
    input_file = args.input_file
    adios2_xml = args.xml if args.xml else "no xml file provided"
    output_file = args.output
    max_steps = args.max_steps

    if rank == 0:
        print(f"Input file: {input_file}")
        print(f"ADIOS2 XML file: {adios2_xml}")
        print(f"Output file: {output_file}")

    if max_steps <= 0:
        if rank == 0:
            print("Error: max_steps must be a non-negative integer.")
        sys.exit(1)
    
    if "no xml file provided" == adios2_xml:
        adios_obj = Adios(comm)
    else:
        adios_obj = Adios(adios2_xml, comm)
        
    Rio = adios_obj.declare_io("readerIO")
    Wio = adios_obj.declare_io("WriteIO")

    with Stream(Rio, input_file, 'r', comm) as s, Stream(Wio, output_file, "w", comm) as w:
        variables_defined = False
        
        for step in s:            
            status = s.begin_step()
            if not status:
                break
                
            current_step = s.current_step()
            comm.Barrier()
            
            if rank == 0:
                print(f"Processing step {current_step}")
            
            w.begin_step()
            comm.Barrier()

            if not variables_defined:                                
                uxR = Rio.inquire_variable('ux')
                uyR = Rio.inquire_variable('uy')
                uzR = Rio.inquire_variable('uz')

                global_shape = uxR.shape() 
                if rank == 0:
                    print(f"Global shape: {global_shape}")
                
                if not global_shape:
                    if rank == 0:
                        print(f"No shape info for variable ux")
                    sys.exit(1)

                total_slices = global_shape[2]
                base = total_slices // size
                rem = total_slices % size
                local_count_2 = base + 1 if rank < rem else base
                local_start_2 = rank * base + min(rank, rem)

                start = [0, 0, local_start_2] + [0] * (len(global_shape) - 3)
                count = list(global_shape)
                count[2] = local_count_2

                if rank == 0:
                    print(f"Rank {rank}: start={start}, count={count}")

                uxR.set_selection((start, count))
                uyR.set_selection((start, count))
                uzR.set_selection((start, count))
                
 
                ux = s.read(uxR)
                uy = s.read(uyR)
                uz = s.read(uzR)

                comm.Barrier()
                if rank == 0:
                    print(f"Read data shapes: ux={ux.shape}, uy={uy.shape}, uz={uz.shape}")

                if len(global_shape) == 3 and global_shape[0] != 1:
                    div = (np.gradient(ux, axis=2, edge_order=2) +
                           np.gradient(uy, axis=1, edge_order=2) +
                           np.gradient(uz, axis=0, edge_order=2))
                                   
                    curl_x = np.gradient(uz, axis=1, edge_order=2) - np.gradient(uy, axis=0, edge_order=2)
                    curl_y = np.gradient(ux, axis=0, edge_order=2) - np.gradient(uz, axis=2, edge_order=2)
                    curl_z = np.gradient(uy, axis=2, edge_order=2) - np.gradient(ux, axis=1, edge_order=2)
                    
                elif len(global_shape) == 3 and global_shape[0] == 1: 
                    div = (np.gradient(ux, axis=2, edge_order=2) +   
                           np.gradient(uy, axis=1, edge_order=2))   

                    curl_z = np.gradient(uy, axis=2, edge_order=2) - np.gradient(ux, axis=1, edge_order=2)
                    curl_x = np.zeros_like(ux) 
                    curl_y = np.zeros_like(uy)

                var_div = Wio.define_variable('Div', div, global_shape, start, count)
                var_curlx = Wio.define_variable('Curl_x', curl_x, global_shape, start, count)
                var_curly = Wio.define_variable('Curl_y', curl_y, global_shape, start, count)
                var_curlz = Wio.define_variable('Curl_z', curl_z, global_shape, start, count)
                
                variables_defined = True
                
            else:
                uxR = Rio.inquire_variable('ux')
                uyR = Rio.inquire_variable('uy')
                uzR = Rio.inquire_variable('uz')
                
                global_shape = uxR.shape()
                total_slices = global_shape[2]
                base = total_slices // size
                rem = total_slices % size
                local_count_2 = base + 1 if rank < rem else base
                local_start_2 = rank * base + min(rank, rem)

                start = [0, 0, local_start_2] + [0] * (len(global_shape) - 3)
                count = list(global_shape)
                count[2] = local_count_2

                uxR.set_selection((start, count))
                uyR.set_selection((start, count))
                uzR.set_selection((start, count))
                
                ux = s.read(uxR)
                uy = s.read(uyR)
                uz = s.read(uzR)

                if len(global_shape) == 3 and global_shape[0] != 1:
                    div = (np.gradient(ux, axis=2, edge_order=2) +
                           np.gradient(uy, axis=1, edge_order=2) +
                           np.gradient(uz, axis=0, edge_order=2))
                                   
                    curl_x = np.gradient(uz, axis=1, edge_order=2) - np.gradient(uy, axis=0, edge_order=2)
                    curl_y = np.gradient(ux, axis=0, edge_order=2) - np.gradient(uz, axis=2, edge_order=2)
                    curl_z = np.gradient(uy, axis=2, edge_order=2) - np.gradient(ux, axis=1, edge_order=2)
                    
                elif len(global_shape) == 3 and global_shape[0] == 1:
                    div = (np.gradient(ux, axis=2, edge_order=2) +   
                           np.gradient(uy, axis=1, edge_order=2))   

                    curl_z = np.gradient(uy, axis=2, edge_order=2) - np.gradient(ux, axis=1, edge_order=2)
                    curl_x = np.zeros_like(ux) 
                    curl_y = np.zeros_like(uy)

            w.write('Div', div)
            w.write('Curl_x', curl_x)
            w.write('Curl_y', curl_y)
            w.write('Curl_z', curl_z)
        
            w.end_step()
            comm.Barrier()
                                                            
            if current_step >= max_steps - 1:
                if rank == 0:
                    print(f"Reached max_steps = {max_steps}")
                break

    if rank == 0:
        print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()