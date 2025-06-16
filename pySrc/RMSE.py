import adios2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from rich.traceback import install

def RMSE(GT, E, step, var_NAME="Variable", skip_factor=2):
    install()
    error = np.zeros_like(E)
    diff_sq = 0
    count = 0

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            gt_value = GT[i * skip_factor, j * skip_factor]
            e_value = E[i, j]
            error[i, j] = gt_value - e_value
            diff_sq += (error[i, j]) ** 2
            count += 1

    rmse = np.sqrt(diff_sq / count)
    print("=" * 60)
    print(f"The RMSE for the ground truth {var_NAME} is: {rmse}")
    print()
    return rmse, error



def parse_arguments():
    install()
    parser = argparse.ArgumentParser(description="Compute RMSE from ADIOS2 files")
    parser.add_argument("--lowres", required=True, help="Path to the lower resolution ADIOS2 file")
    parser.add_argument("--highres", required=True, help="Path to the ground truth (high resolution) ADIOS2 file")
    parser.add_argument("--var", required=True, help="Variable name to read from the files")
    parser.add_argument("--max_steps", type=int, required=True, help="Maximum number of steps to process")
    parser.add_argument("--skip_factor", type=int, required=True, help="The skip factor for the higher resolution")
    return parser.parse_args()

def main():
    install()
    args = parse_arguments()

    adios = adios2.Adios()
    RLio = adios.declare_io("readerIOLow")
    RHio = adios.declare_io("readerIOHigh")

    with adios2.Stream(RLio, args.lowres, 'r') as rl:
        with adios2.Stream(RHio, args.highres, 'r') as rh:
            for _ in rl:
                for _ in rh:
                    statusL = rl.begin_step()
                    statusR = rh.begin_step()
                    stepl = rl.current_step()
                    steh = rh.current_step()
                    print(f"Processing step {steh}")

                    varl = rl.read(args.var)
                    varh = rh.read(args.var)

                    if len(varl.shape) == 3 and varl.shape[0] == 1:
                        varl = varl[0, :, :]
                    if len(varh.shape) == 3 and varh.shape[0] == 1:
                        varh = varh[0, :, :]

                    rmse= RMSE(varh, varl, steh, args.var, args.skip_factor)
                    

                    if not statusL or not statusR or stepl >= args.max_steps - 1 or steh >= args.max_steps - 1:
                        print(f"Reached max_steps = {args.max_steps}")
                        break

if __name__ == "__main__":
    install()
    main()
