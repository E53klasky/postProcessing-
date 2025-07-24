import argparse
import adios2
import Reader3DClass
import Writer3DClass
import numpy as np
from mpi4py import MPI
import time
from rich.traceback import install


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="3D domain decomposition copier for ADIOS2 data"
    )
    parser.add_argument(
        "--input",
        "-in",
        type=str,
        required=True,
        help="Input ADIOS2 file to copy",
    )
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader3d",
        required=False,
        help="IO Name for the reader (default: reader3d)",
    )
    parser.add_argument(
        "--writeIO",
        "-wio",
        type=str,
        default="writer3d",
        required=False,
        help="IO Name for the writer (default: writer3d)",
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        required=False,
        help="Path to ADIOS2 XML configuration file (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        default="copied3d.bp",
        help="Output file name (default: copied3d.bp)",
    )
    parser.add_argument(
        "--sleep",
        "-s",
        type=int,
        required=False,
        default=0,
        help="Sleep time in seconds between operations (default: 0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main():
    program_start = time.time()

    args = parse_arguments()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Starting 3D copier with {size} ranks")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        times_file = open("copied3d_times.txt", "w")
        times_file.write(f"Program started at {program_start:.6f}\n")
        times_file.write(f"Using {size} MPI ranks\n")
    else:
        times_file = None

    # Initialize reader and writer with 3D classes
    try:
        reader = Reader3DClass.Reader3D(
            args.readIO, args.input, xml=args.xml, comm=comm
        )
        writer = Writer3DClass.Writer3D(
            args.writeIO, args.output, xml=args.xml, comm=comm
        )

        if rank == 0:
            print("Reader and Writer initialized successfully")
    except Exception as e:
        if rank == 0:
            print(f"Error initializing reader/writer: {e}")
        return

    step_count = 0
    total_vars_processed = 0

    try:
        while True:
            step_start = time.time()

            # Begin reading step
            status = reader.begin_step()

            if status != adios2.bindings.StepStatus.OK:
                if rank == 0:
                    print("No more steps available or error in reading")
                break

            current_step = reader.current_step()
            step_count += 1

            if rank == 0:
                print(f"\nProcessing step: {current_step}")
                if times_file:
                    times_file.write(f"\nStep: {current_step}\n")

            # Begin writing step
            writer.begin_step()

            # Get all available variables
            available_vars = reader.Adios_reader.available_variables()
            vars_in_step = list(available_vars.keys())

            if rank == 0:
                print(f"Variables in step {current_step}: {vars_in_step}")
                if args.verbose:
                    for name, info in available_vars.items():
                        print(f"  {name}: {info}")

            # Set variables for reading
            reader.set_read_vars(vars_in_step)

            # Process each variable
            for var_name in vars_in_step:
                var_start = time.time()

                try:
                    # Read the variable data
                    read_start = time.time()
                    data = reader.read_step(var_name)
                    read_end = time.time()

                    if data is None:
                        if rank == 0:
                            print(f"Warning: No data returned for variable {var_name}")
                        continue

                    # Convert to numpy array if needed
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)

                    # Optional sleep (for testing/debugging)
                    if args.sleep > 0:
                        time.sleep(args.sleep)

                    # Write the variable data
                    write_start = time.time()
                    writer.write(var_name, data)
                    write_end = time.time()

                    var_end = time.time()
                    total_vars_processed += 1

                    if rank == 0 and args.verbose:
                        print(
                            f"  Processed {var_name}: shape={data.shape}, "
                            f"read={read_end-read_start:.4f}s, write={write_end-write_start:.4f}s"
                        )

                    if times_file and rank == 0:
                        times_file.write(
                            f"Variable: {var_name}, Shape: {data.shape}, "
                            f"Read time: {read_end - read_start:.6f} s, "
                            f"Write time: {write_end - write_start:.6f} s, "
                            f"Total var time: {var_end - var_start:.6f} s\n"
                        )

                except Exception as e:
                    if rank == 0:
                        print(f"Error processing variable {var_name}: {e}")
                    continue

            # End steps
            reader.end_step()
            writer.end_step()

            step_end = time.time()

            if rank == 0:
                print(f"Step {current_step} completed in {step_end - step_start:.4f}s")
                if times_file:
                    times_file.write(f"Step time: {step_end - step_start:.6f} s\n")

    except KeyboardInterrupt:
        if rank == 0:
            print("\nCopying interrupted by user")
    except Exception as e:
        if rank == 0:
            print(f"Error during copying: {e}")
    finally:
        # Clean up
        try:
            reader.close()
            writer.close()
        except:
            pass

        program_end = time.time()

        if rank == 0:
            print(f"\nCopying completed!")
            print(f"Steps processed: {step_count}")
            print(f"Variables processed: {total_vars_processed}")
            print(f"Total time: {program_end - program_start:.4f}s")
            print(f"Output written to: {args.output}")

            if times_file:
                times_file.write(f"\nProgram ended at {program_end:.6f}\n")
                times_file.write(
                    f"Total program time: {program_end - program_start:.6f} s\n"
                )
                times_file.write(f"Steps processed: {step_count}\n")
                times_file.write(f"Variables processed: {total_vars_processed}\n")
                times_file.close()


if __name__ == "__main__":
    install()
    main()
