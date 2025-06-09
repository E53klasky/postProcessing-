import adios2
import matplotlib.pyplot as plt
import argparse
import os

def parser_arguments():
    parser = argparse.ArgumentParser(description="Simple 2D plot")
    parser.add_argument('input_file', type=str, help='Path to the BP file to process (REQUIRED)')
    parser.add_argument('--xml', '-x', type=str, default=None, help='ADIOS2 XML config file default: None (optional)')
    parser.add_argument('--vars', '-v', type=str, required=True, help='Variables to plot, separated by commas (REQUIRED)')
    parser.add_argument('--max_steps', '-n', type=int, required=True, help='Maximum number of timesteps to process')
    return parser.parse_args()

def main():
    args = parser_arguments()
    max_steps = args.max_steps
    input_file = args.input_file
    adios2_xml = args.xml
    vars = args.vars.split(',')
    

    if adios2_xml is not None:
        adios = adios2.Adios(adios2_xml)
    else:
        adios = adios2.Adios()

    Rio = adios.declare_io("ReadIO")
    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    with adios2.Stream(Rio, input_file, 'r') as s:
        for step_count, _ in enumerate(s):
            if step_count >= max_steps:
                break
            print(f"Processing step {step_count}")
            for var in vars:
                if var in s.available_variables():
                    data = s.read(var)
                    if  len(data.shape) == 3 and data.shape[0] == 1:
                        data = data[0,:,:]
                    
                    plt.imshow(data, cmap='inferno')
                    plt.title(f"{var} at step {step_count}")
                    plt.colorbar()
                    plt.savefig(os.path.join(output_dir, f"{var}_step_{step_count}.png"))
                    plt.close()
                else:
                    print(f"Variable {var} not found in the input file.")
    print("output saved to ../RESULTS")
if __name__ == "__main__":
    main()
