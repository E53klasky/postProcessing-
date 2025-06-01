from mpi4py import MPI
import numpy as np
from adios2 import Adios, Stream, bindings
import os
import argparse

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

    if not os.path.exists(input_file):
        if rank == 0:
            print(f"Error: File {input_file} does not exist.")
        return
    if  "no xml file provided" == adios2_xml:
        adios_obj = Adios(comm)
    else:
        adios_obj = Adios(adios2_xml, comm)
    Rio = adios_obj.declare_io("readerIO")
    Wio = adios_obj.declare_io("WriteIO")

    try:
        with Stream(Rio, input_file, 'r') as reader:
            with Stream(Wio, output_file, "w") as writer:

                for _ in reader:
                    status = reader.begin_step()
                    step = reader.current_step()
                    if rank == 0:
                        print(f"Processing step {step}")

                    try:
                        ux = reader.read('ux')
                        uy = reader.read('uy')
                        uz = reader.read('uz')

                        if rank == 0:
                            print(f"Read data shapes: ux={ux.shape}, uy={uy.shape}, uz={uz.shape}")

                        if len(ux.shape) == 4 and ux.shape[0] == 1:
                            ux = ux[0, :, :, :]
                            uy = uy[0, :, :, :]
                            uz = uz[0, :, :, :]
                        elif len(ux.shape) == 3 and ux.shape[0] == 1:
                            ux = ux[0, :, :]
                            uy = uy[0, :, :]
                            uz = uz[0, :, :]

                        if rank == 0:
                            print(f"After squeezing: ux={ux.shape}, uy={uy.shape}, uz={uz.shape}")

                        if len(ux.shape) == 3:
                            div = (np.gradient(ux, axis=2,  edge_order=2) +
                                   np.gradient(uy, axis=1,  edge_order=2) +
                                   np.gradient(uz, axis=0,  edge_order=2))

                            curl_x = np.gradient(uz, axis=1,  edge_order=2) - np.gradient(uy, axis=0, edge_order=2)
                            curl_y = np.gradient(ux, axis=0,  edge_order=2) - np.gradient(uz, axis=2,  edge_order=2)
                            curl_z = np.gradient(uy, axis=2,  edge_order=2) - np.gradient(ux, axis=1,  edge_order=2)

                        elif len(ux.shape) == 2:
                            div = (np.gradient(ux, axis=1,  edge_order=2) +
                                   np.gradient(uy, axis=0,  edge_order=2))

                            curl_z = np.gradient(uy, axis=1, edge_order=2) - np.gradient(ux, axis=0, edge_order=2)
                            curl_x = np.gradient(uz, axis=0,  edge_order=2)
                            curl_y = -np.gradient(uz, axis=1,  edge_order=2)

                        else:
                            if rank == 0:
                                print(f"Unsupported data dimensionality: {ux.shape}")
                            continue

                        if rank == 0:
                            print(f"Calculated divergence shape: {div.shape}")
                            if len(ux.shape) == 2:
                                print(f"Calculated curl (2D) shape: {curl_z.shape}")
                            else:
                                print(f"Calculated curl components shape: {curl_x.shape}")

                        writer.write("div", div,
                                   shape=div.shape,
                                   start=[0] * len(div.shape),
                                   count=div.shape)

                        writer.write("curl_x", curl_x,
                                       shape=curl_x.shape,
                                       start=[0] * len(curl_x.shape),
                                       count=curl_x.shape)
                        writer.write("curl_y", curl_y,
                                       shape=curl_y.shape,
                                       start=[0] * len(curl_y.shape),
                                       count=curl_y.shape)
                        writer.write("curl_z", curl_z,
                                       shape=curl_z.shape,
                                       start=[0] * len(curl_z.shape),
                                       count=curl_z.shape)
                        
                        if step >= max_steps-1:
                            if rank == 0:
                                print(f"End of steps reached or error reading step {step}")
                            writer.end_step()
                            break

                        writer.end_step()

                    except Exception as e:
                        if rank == 0:
                            print(f"Error processing step {step}: {e}")
                            try:
                                available_vars = reader.available_variables()
                                print(f"Available variables: {list(available_vars.keys())}")
                            except:
                                print("Could not retrieve available variables")
                        break

                if rank == 0:
                    print(f"Successfully processed {step} steps")

    except Exception as e:
        if rank == 0:
            print(f"Error opening files: {e}")
        return
    
    if rank == 0:
        print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
