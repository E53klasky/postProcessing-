import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from adios2 import Adios, Stream
from mpi4py import MPI 



# eventaully add choose compression
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate streamline plots from ADIOS2 BP5 files')
    
    parser.add_argument('path', type=str, help='Path to the BP5 file to process')
    parser.add_argument('max_steps', type=int, help='Maximum number of time steps to process')
    parser.add_argument("--errrorBound", "-eb", type=float, default=0, help='Error bound for compression (default: 0) No compression if 0')
    parser.add_argument('--xml', '-x', type=str, default=None,
                        help='Path to ADIOS2 XML configuration file (optional)')
    
    return parser.parse_args()


# make this read in ever varible to a container then you will to make a method for compression to compression by the error bound
# if there is an xml file set then it is done through the xml file and wrights out if not then compress it 
# calculate error OPTION i think not sure 
def adios2_reader(bp_file, max_steps, xml_file):
    
    if xml_file == "no xml file provided":
        adios = Adios()
        adios.Operator("mgard","mgard")
        print("Using mgard as default for now ")
    else:
        adios = Adios(xml_file)
        print("takeing error bound from xml_file")
    Rio = adios.declare_io("ReadIO")
    Wio = adios.declare_io("WriteIO")
    avalid_variables = Rio.available_variables()
    print(f"Available variables in {avalid_variables}:")
    print(f"available variables in data type : {type(avalid_variables)}")
    with Stream(bp_file, "r") as s:
        for  _  in enumerate(s.steps()):
           
            print(f"Processing step {s.current_step()}")

            print("Available Variables:")
            for name, info in s.available_variables().items():
                print(f"Variable: {name}")
                for key, value in info.items():
                    print(f"\t{key}: {value}")
                    
            if s.current_step() >= max_steps-1:
                print(f"Reached maximum steps: {max_steps}. Stopping processing.")
                break
 


           


def main():
    print(f"hello world")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    args = parse_arguments()
    
    input_file = args.path
    max_steps = args.max_steps
    adios2_xml = args.xml if args.xml else "no xml file provided"     
    error_bound = args.errrorBound
    
    if max_steps <= 0:
        if rank == 0:
            print("Error: max_steps must be a positive integer.")
        sys.exit(1)
        
    # NOT sure if this will work with the SST engine 
    # if not os.path.exists(bp_file):
    #     print(f"Error: File {bp_file} does not exist.")
    #     sys.exit(1)
    if rank == 0:
        print(f"Input file: {input_file}")
        print(f"ADIOS2 XML file: {adios2_xml}")
        print(f"Number of steps to process: {max_steps}")
    adios2_reader(input_file, max_steps, adios2_xml)
    
    
    
    
    





# sys.exit(1)

# fname1 = sys.argv[1]
# var1 = sys.argv[2]
# varx = sys.argv[3]
# fname2 = sys.argv[4]
# var2 = sys.argv[5]
# output_file = sys.argv[6]

# adios = Adios("adios2.xml")
# io1 = adios.declare_io("WriteIO")
# io2 = adios.declare_io("CopierOutput")
# print(f"Subtract: open stream 1 {fname1}")
# f1 = Stream(io1, fname1, "r")
# print(f"Subtract: open stream 2 {fname2}")
# f2 = Stream(io2, fname2, "r")
# ioOut = adios.declare_io("uncompressed error")
# print(f"Subtract: create output stream {output_file}")
# fout = Stream(ioOut, output_file, "w")

# step = 0
# while True:
#     print(f"Subtract: Step {step}")
#     status = f1.begin_step()
#     if status != bindings.StepStatus.OK:
#         print(f"Subtract: No more steps or error reading first stream: {fname1}")
#         break
#     v1 = f1.inquire_variable(var1)
#     vx = f1.inquire_variable(varx)
#     shape1 = v1.shape()
#     data1 = f1.read(v1)
#     datax = f1.read(vx)
#     step1 = f1.current_step()
#     print(f"Subtract:    Data from stream 1, step = {step1} shape = {shape1}")
#     f1.end_step()

#     status = f2.begin_step()
#     if status != bindings.StepStatus.OK:
#         print(f"Subtract: No more steps or error reading second stream: {fname2}")
#         break
#     v2 = f2.inquire_variable(var2)
#     shape2 = v2.shape()
#     data2 = f2.read(v2)
#     step2 = f2.current_step()
#     print(f"Subtract:    Data from stream 2, step = {step2} shape = {shape2}")
#     f2.end_step()

#     if shape1 != shape2:
#         print(f"Subtract: The shape of the two variables differ! {shape1}  and  {shape2}")
#         break
#     x = datax
#     diff = data1 - data2
 
#     fout.begin_step()
#     start = np.zeros(3, dtype=np.int64)

#     fout.write("diff", diff, [len(diff),len(diff),len(diff)], [0,0,0], [len(diff),len(diff),len(diff)])
#     fout.write("x", x, [len(x)], [0], [len(x)])
#     fout.end_step()

# print("Subtract: Completed")
# f1.close()
# f2.close()
# fout.close()

# # If you want to create a plot
# plt.hist(diff, bins=50, alpha=0.5, label='Difference')
# plt.legend(loc='upper right')
# plt.title("Histogram of Differences")
# plt.xlabel("Difference")
# plt.ylabel("Frequency")
# plt.show()



if __name__ == "__main__":
    main()
