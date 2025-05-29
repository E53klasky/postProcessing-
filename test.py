from mpi4py import MPI
import numpy as np 
from adios2 import Adios, Stream
import os
import sys


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running with {size} MPI processes")
    
    if(len(sys.argv) < 3):
        if rank == 0:
            print("Usage: python div_curl.py <PATH/TO/ADIOS2/FILE.bp5> <PATH/TO/ADIOS2/xml> [output_file]")
        return
    
    input_file = sys.argv[1]
    adios2_xml = sys.argv[2]
    
    if rank == 0:
        print(f"Input file: {input_file}")
        print(f"ADIOS2 XML file: {adios2_xml}")
    
    if not os.path.exists(input_file):
        if rank == 0:
            print(f"Error: File {input_file} does not exist.")
        return
    
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_div_curl.bp5"
    
    if rank == 0:
        print(f"Output file: {output_file}")
        
    adios_obj = Adios(adios2_xml)
    Rio = adios_obj.declare_io("readerIO")
    Wio = adios_obj.declare_io("WriteIO")
    
    try:
        with Stream(Rio, input_file, 'r') as reader:
            with Stream(Wio, output_file, "w") as writer:
                
                step_count = 0
                for step_data in reader:
                    if rank == 0:
                        print(f"Processing step {step_count}")
                    
                    try:
                        ux = step_data.read('ux')
                        uy = step_data.read('uy')
                        uz = step_data.read('uz')
                        
                        if rank == 0:
                            print(f"Read data shapes: ux={ux.shape}, uy={uy.shape}, uz={uz.shape}")
                        
                        original_shape = ux.shape
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
                            div = (np.gradient(ux, axis=2) +  
                                   np.gradient(uy, axis=1) +  
                                   np.gradient(uz, axis=0))   
                            
                            curl_x = np.gradient(uz, axis=1) - np.gradient(uy, axis=0)  
                            curl_y = np.gradient(ux, axis=0) - np.gradient(uz, axis=2)  
                            curl_z = np.gradient(uy, axis=2) - np.gradient(ux, axis=1)  
                            
                        elif len(ux.shape) == 2:  
                            div = (np.gradient(ux, axis=1) +  
                                   np.gradient(uy, axis=0))   
                            
  
                            curl_z = np.gradient(uy, axis=1) - np.gradient(ux, axis=0) 
                            

                            curl_x = np.gradient(uz, axis=0) 
                            curl_y = -np.gradient(uz, axis=1)     
                            
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
                        
                        if len(ux.shape) == 2: 
                            writer.write("curl_z", curl_z, 
                                       shape=curl_z.shape, 
                                       start=[0] * len(curl_z.shape), 
                                       count=curl_z.shape)
                            writer.write("curl_x", curl_x, 
                                       shape=curl_x.shape, 
                                       start=[0] * len(curl_x.shape), 
                                       count=curl_x.shape)
                            writer.write("curl_y", curl_y, 
                                       shape=curl_y.shape, 
                                       start=[0] * len(curl_y.shape), 
                                       count=curl_y.shape)
                        else: 
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
                        
                        writer.end_step()
                        step_count += 1
                        
                    except Exception as e:
                        if rank == 0:
                            print(f"Error processing step {step_count}: {e}")
                            try:
                                available_vars = step_data.available_variables()
                                print(f"Available variables: {list(available_vars.keys())}")
                            except:
                                print("Could not retrieve available variables")
                        break
                
                if rank == 0:
                    print(f"Successfully processed {step_count} steps")
                    
    except Exception as e:
        if rank == 0:
            print(f"Error opening files: {e}")
        return
    if rank == 0:
        print(f"Output written to {output_file}")

if __name__ == "__main__":
    main()
