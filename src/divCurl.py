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
        import pdb; pdb.set_trace()

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
    
    if  "no xml file provided" == adios2_xml:
        adios_obj = Adios(comm)
    else:
        adios_obj = Adios(adios2_xml, comm)
    Rio = adios_obj.declare_io("readerIO")
    Wio = adios_obj.declare_io("WriteIO")

    try:
        with Stream(Rio, input_file, 'r') as reader, Stream(Wio, output_file, "w", comm) as w:
            variables_defined = False
            for _ in reader:
                    
                status = reader.begin_step()
                if status:
                    print(f"Processing step {reader.current_step()}")
                    w.begin_step()


                step = reader.current_step()
                                                
                uxR = Rio.inquire_variable('ux')
                uyR = Rio.inquire_variable('uy')
                uzR = Rio.inquire_variable('uz')
                shape = uxR.shape()
                            
                if not shape:
                    print(f"No shape info for variable ux")
                    sys.exit(1)
                            
                            
                total_slices = shape[2]
                base = total_slices // size
                rem = total_slices % size
                local_count_2 = base + 1 if rank < rem else base
                local_start_2 = rank * base + min(rank, rem)

                start = [0, 0, local_start_2] + [0] * (len(shape) - 3)
                count = list(shape)
                count[2] = local_count_2
                uxR.set_selection((start, count))
                uyR.set_selection((start, count))
                uzR.set_selection((start, count))
                            
                            
                ux = reader.read(uxR)
                uy = reader.read(uyR)
                uz = reader.read(uzR)

                if rank == 0:
                    print(f"Read data shapes: ux={ux.shape}, uy={uy.shape}, uz={uz.shape}")

                            # if len(ux.shape) == 4 and ux.shape[0] == 1:
                            #     ux = ux[0, :, :, :]
                            #     uy = uy[0, :, :, :]
                            #     uz = uz[0, :, :, :]
                            # elif len(ux.shape) == 3 and ux.shape[0] == 1:
                            #     ux = ux[0, :, :]
                            #     uy = uy[0, :, :]
                            #     uz = uz[0, :, :]

                            # if rank == 0:
                            #     print(f"After squeezing: ux={ux.shape}, uy={uy.shape}, uz={uz.shape}")

                            
                if len(ux.shape) == 3 and ux.shape[0] != 1:
                        
                    div = (np.gradient(ux, axis=2,  edge_order=2) +
                            np.gradient(uy, axis=1,  edge_order=2) +
                            np.gradient(uz, axis=0,  edge_order=2))
                               
                    curl_x = np.gradient(uz, axis=1,  edge_order=2) - np.gradient(uy, axis=0, edge_order=2)
                    curl_y = np.gradient(ux, axis=0,  edge_order=2) - np.gradient(uz, axis=2,  edge_order=2)
                    curl_z = np.gradient(uy, axis=2,  edge_order=2) - np.gradient(ux, axis=1,  edge_order=2)
                                
                           
                                
                elif len(ux.shape) == 3 and ux.shape[0] == 1: # this might be wrong?????
                    shape = ux.shape
                    if shape[0] == 1:
                        div = (np.gradient(ux, axis=2, edge_order=2) +   
                                np.gradient(uy, axis=1, edge_order=2))   

                            # Only curl_z is meaningful in 2D
                        curl_z = np.gradient(uy, axis=2, edge_order=2) - np.gradient(ux, axis=1, edge_order=2)
                        curl_x = np.zeros_like(ux) 
                        curl_y = np.zeros_like(uy)  #- np.gradient(uz, axis=2, edge_order=2)
                        
                        # elif shape[1] == 1:
                        #     # 2D in (Z, X), Y is trivial
                        #     div = (np.gradient(ux, axis=2, edge_order=2) +   # d(ux)/dx
                        #         np.gradient(uz, axis=0, edge_order=2))    # d(uz)/dz

                        #     curl_y = np.gradient(ux, axis=0, edge_order=2) - np.gradient(uz, axis=2, edge_order=2)
                        #     curl_z = np.zeros_like(ux)
                        #     curl_x = np.zeros_like(ux)

                        # elif shape[2] == 1:
                        #     # 2D in (Z, Y), X is trivial
                        #     div = (np.gradient(uy, axis=1, edge_order=2) +   # d(uy)/dy
                        #         np.gradient(uz, axis=0, edge_order=2))    # d(uz)/dz

                        #     curl_x = np.gradient(uz, axis=1, edge_order=2) - np.gradient(uy, axis=0, edge_order=2)
                        #     curl_z = np.zeros_like(ux)
                        #     curl_y = np.zeros_like(ux)
                            


                            
                    # else:
                    #     if rank == 0:
                    #         print(f"Unsupported data dimensionality: {ux.shape}")
                    #     continue

                    # if rank == 0:
                    #     print(f"Calculated divergence shape: {div.shape}")
                    #     if len(ux.shape) == 2:
                    #         print(f"Calculated curl (2D) shape: {curl_z.shape}")
                    #     else:
                    #         print(f"Calculated curl components shape: {curl_x.shape}")
                            
                        
                            # if (div.ndim == 3):
                                
                    
                if not variables_defined:
                    Div = Wio.define_variable("div", div,  shape, start, count)
                    
                    Curl_x = Wio.define_variable("curl_x", curl_x, shape, start, count)
                    Curl_y = Wio.define_variable("curl_y", curl_y, shape, start, count)
                    Curl_z = Wio.define_variable("curl_z", curl_z, shape, start, count)
                             
                      
                w.write(Div, div)
                  
                w.write(Curl_x, curl_x)
                w.write(Curl_y, curl_y)
                w.write(Curl_z, curl_z)
                variables_defined = True
                w.end_step()
                    
                                
                            # else:
                                
                            #     writer.write("div", div,
                            #            shape=div.shape,
                            #            start=[0] * len(div.shape),
                            #            count=div.shape)

                            #     writer.write("curl_x", curl_x,
                            #                shape=curl_x.shape,
                            #                start=[0] * len(curl_x.shape),
                            #                count=curl_x.shape)
                            #     writer.write("curl_y", curl_y,
                            #                shape=curl_y.shape,
                            #                start=[0] * len(curl_y.shape),
                            #                count=curl_y.shape)
                            #     writer.write("curl_z", curl_z,
                            #                shape=curl_z.shape,
                            #                start=[0] * len(curl_z.shape),
                            #                count=curl_z.shape)    
                            
                if step >= max_steps-1:
                    print(f"End of steps reached or error reading step {step}")
                    break

    except Exception as e:
        if rank == 0:
            print(f"Error opening files: {e}")
        return
    
    if rank == 0:
        print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
