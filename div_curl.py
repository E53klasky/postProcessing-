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

    if len(sys.argv) < 3:
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
        output_file = "div_curl_mpi.bp"

    if rank == 0:
        print(f"Output file: {output_file}")

    try:
        adios_obj = Adios(adios2_xml)
        Rio = adios_obj.DeclareIO("readerIO")
        Wio = adios_obj.DeclareIO("WriteIO")

        if rank == 0:
            with Stream(Rio, input_file, "r",comm) as reader:
                with Stream(Wio, output_file, "w", comm) as writer:

                    step_count = 0
                    for _ in reader:
                        print(f"Processing step {step_count}")

                        try:
                            ux_var = reader.inquire_variable('ux')
                            uy_var = reader.inquire_variable('uy')
                            uz_var = reader.inquire_variable('uz')

                            print(f"Read variable shapes: ux={ux_var.shape()}, uy={uy_var.shape()}, uz={uz_var.shape()}")

                            # Read actual numpy arrays from variables
                            ux = reader.read(ux_var)
                            uy = reader.read(uy_var)
                            uz = reader.read(uz_var)

                            # Squeeze data if needed and determine dimensionality flag
                            if len(ux_var.shape()) == 4 and ux_var.shape()[0] == 1:
                                ux = ux[0, :, :, :]
                                uy = uy[0, :, :, :]
                                uz = uz[0, :, :, :]
                                flag = False  # 3D data after squeezing
                            elif len(ux_var.shape()) == 3 and ux_var.shape()[0] == 1:
                                ux = ux[0, :, :]
                                uy = uy[0, :, :]
                                uz = uz[0, :, :]
                                flag = True  # 2D data after squeezing
                            elif len(ux_var.shape()) == 3:
                                flag = False
                            elif len(ux_var.shape()) == 2:
                                flag = True
                            else:
                                print(f"Unsupported data dimensionality: {ux_var.shape()}")
                                break

                            print(f"After processing: ux={ux.shape}, uy={uy.shape}, uz={uz.shape}")
                            print(f"Flag (3D mode): {flag}")

                            # Compute divergence and curl
                            if len(ux.shape) == 3:  # 3D case
                                print("Computing 3D divergence and curl...")
                                div = (np.gradient(ux, axis=2, edge_order=2) +
                                       np.gradient(uy, axis=1, edge_order=2) +
                                       np.gradient(uz, axis=0, edge_order=2))

                                curl_x = np.gradient(uz, axis=1, edge_order=2) - np.gradient(uy, axis=0, edge_order=2)
                                curl_y = np.gradient(ux, axis=0, edge_order=2) - np.gradient(uz, axis=2, edge_order=2)
                                curl_z = np.gradient(uy, axis=2, edge_order=2) - np.gradient(ux, axis=1, edge_order=2)

                            elif len(ux.shape) == 2:  # 2D case
                                print("Computing 2D divergence and curl...")
                                div = (np.gradient(ux, axis=1, edge_order=2) +
                                       np.gradient(uy, axis=0, edge_order=2))

                                curl_z = np.gradient(uy, axis=1, edge_order=2) - np.gradient(ux, axis=0, edge_order=2)
                                curl_x = np.gradient(uz, axis=0, edge_order=2)
                                curl_y = -np.gradient(uz, axis=1, edge_order=2)

                            else:
                                print(f"Unsupported data shape for computation: {ux.shape}")
                                break

                            print(f"Calculated divergence shape: {div.shape}")
                            if len(ux.shape) == 2:
                                print(f"Calculated curl (2D) shape: {curl_z.shape}")
                            else:
                                print(f"Calculated curl components shape: {curl_x.shape}")

                            print("Starting to write data...")

                            # Write data to output file
                            if flag:  # 2D case
                                writer.write("div", div, shape=div.shape, start=[0, 0], count=div.shape)
                                writer.write("curl_x", curl_x, shape=curl_x.shape, start=[0, 0], count=curl_x.shape)
                                writer.write("curl_y", curl_y, shape=curl_y.shape, start=[0, 0], count=curl_y.shape)
                                writer.write("curl_z", curl_z, shape=curl_z.shape, start=[0, 0], count=curl_z.shape)
                            else:  # 3D case
                                writer.write("div", div, shape=div.shape, start=[0, 0, 0], count=div.shape)
                                writer.write("curl_x", curl_x, shape=curl_x.shape, start=[0, 0, 0], count=curl_x.shape)
                                writer.write("curl_y", curl_y, shape=curl_y.shape, start=[0, 0, 0], count=curl_y.shape)
                                writer.write("curl_z", curl_z, shape=curl_z.shape, start=[0, 0, 0], count=curl_z.shape)

                            writer.end_step()

                            print(f"Finished writing data for step {step_count}")
                            step_count += 1

                        except Exception as e:
                            print(f"Error processing step {step_count}: {e}")
                            try:
                                available_vars = reader.available_variables()
                                print(f"Available variables: {list(available_vars.keys())}")
                            except Exception:
                                print("Could not retrieve available variables")
                            break

                    print(f"Successfully processed {step_count} steps")

        else:
            # Other ranks just wait for rank 0 to finish
            if rank == 1:
                print(f"Rank {rank} waiting for rank 0 to complete...")

    except Exception as e:
        if rank == 0:
            print(f"Error opening files: {e}")
        return

    if rank == 0:
        print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
