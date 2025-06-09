import numpy as np
import argparse
from adios2 import Adios, Stream, bindings

# same size for now
def parse_arguments():
    parser = argparse.ArgumentParser(description="Subtract variables from two ADIOS2 files and write the difference.")
    parser.add_argument("bpfile1", help="First input BP file lower Res") 
    parser.add_argument("--var1", help="Variable name from the first file")
    parser.add_argument("bpfile2", help="Second input BP file") 
    parser.add_argument("--var2", help="Variable name from the second file higher res")
    parser.add_argument("--output_file",default='subtract.bp' ,help="Output BP file for the result")
    parser.add_argument("--xml", default=None, help="Optional ADIOS2 XML configuration (default: adios2.xml)")
    parser.add_argument("--max_steps", default=None, help="The number of max time steps")
    parser.add_argument("--tolerance",default=None, help="Tolerance level of the error this will show 0 if it is <= the tolerance" )
    parser.add_argument("--skip", type=int, default=0, help="number of points to skip for the higher resolution")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.xml is not None:
        adios = Adios(args.xml)
    else:
        adios = Adios()
    io1 = adios.declare_io("ReadIO1")
    io2 = adios.declare_io("ReadIO2")
    io_out = adios.declare_io("OutputIO")
    skip_factor = args.skip
    GT = args.bpfile2
    E = args.bpfile1
    print(f"Opening input streams: {E} and {GT}")
    with Stream(io1, E, "r") as f1, Stream(io2, GT, "r") as f2, Stream(io_out, args.output_file, "w") as fout:
        step = 0
        while True:
            print(f"\n--- Step {step} ---")

           
            status1 = f1.begin_step()
            if status1 != bindings.StepStatus.OK:
                print("End of stream or error in first file.")
                break

            e = f1.inquire_variable(args.var1)
            error = f1.read(e)
            error_shape = e.shape()
            f1.end_step()
            print(f"Read {args.var1}  from {E}, shape = {error_shape}")

           
            status2 = f2.begin_step()
            if status2 != bindings.StepStatus.OK:
                print("End of stream or error in second file.")
                break

            v2 = f2.inquire_variable(args.var2)
            groud_truth = f2.read(v2)
            GT_shape = v2.shape()
            f2.end_step()
            print(f"Read {args.var2} from {GT}, shape = {GT_shape}")


            diff = np.zeros_like(error)
            for i in range(error.shape[1]):  # assuming shape = [1, H, W]
                for j in range(error.shape[2]):
                    gt_i = int(i * skip_factor)
                    gt_j = int(j * skip_factor)
                    gt_value = groud_truth[0, gt_i, gt_j]
                    e_value = error[0, i, j]
                    diff[0, i, j] = np.abs(gt_value - e_value)

            # not important right now 
            # if args.tolerance is not None:
            #     tol = float(args.tolerance)
            # diff[diff <= tol] = 0.0
            
            fout.begin_step()
            fout.write(f"{args.var1} error.bp", diff, error_shape, [0] * len(error_shape), error_shape)
            fout.end_step()

            step += 1
            if step == args.max_steps:
                break

    print("\nSubtraction completed and written to", args.output_file)


if __name__ == "__main__":
    main()
