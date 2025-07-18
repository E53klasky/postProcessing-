import argparse
import numpy as np
import os
import sys
from pathlib import Path
from rich.traceback import install
import adios2


from ReaderClass import Reader
from WrighterClass import Writer

# this may work try it with dumy data
# All datasets used in this work are stored in NumPy .npz format and follow a standardized 5D tensor structure: [variable, n_samples, T, H, W]


# Variable: number of physical quantities


# Sections: number of independent spatial samples
# Frames: number of time steps per sample TIME STESP for nbp
# H/W: spatial resolution (height × width)
# np.savez("path.npz", data=your_data)
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert between ADIOS2 .bp and .npz formats"
    )
    parser.add_argument(
        "--input",
        "-in",
        type=str,
        required=True,
        help="Input file path (.bp or .npz)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (.npz or .bp)",
    )
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader_io",
        required=False,
        help="IO Name for reading (default: reader_io)",
    )
    parser.add_argument(
        "--writeIO",
        "-wio",
        type=str,
        default="writer_io",
        required=False,
        help="IO Name for writing (default: writer_io)",
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        required=False,
        help="Path to ADIOS2 XML configuration file (optional)",
    )

    return parser.parse_args()


def bp_to_npz(input_file, output_file, read_io_name, xml=None):
    print(f"Converting {input_file} to {output_file}")

    reader = Reader(read_io_name, input_file, xml=xml)

    all_data = {}
    step_count = 0

    try:
        while True:

            status = reader.begin_step()

            if status != adios2.bindings.StepStatus.OK:
                print(f"No more steps available or error occurred. Status: {status}")
                break

            current_step = reader.current_step()
            print(f"Reading step: {current_step}")

            try:
                available_variables = reader.Adios_reader.available_variables()
                print(f"Available variables: {list(available_variables.keys())}")
            except Exception as e:
                print(f"Error getting available variables: {e}")
                reader.end_step()
                continue

            if not available_variables:
                print("No variables found in this step.")
                reader.end_step()
                continue

            step_data = {}
            for var_name, var_info in available_variables.items():
                try:
                    print(f"  Attempting to read variable: {var_name}")
                    reader.set_read_vars([var_name])
                    var_data = reader.read_step(var_name)
                    step_data[var_name] = var_data
                    print(
                        f"    Successfully read '{var_name}' with shape: {var_data.shape}"
                    )
                except Exception as e:
                    print(f"    Warning: Could not read variable '{var_name}': {e}")
                    continue

            if step_data:
                all_data[f"step_{current_step}"] = step_data
                step_count += 1

            reader.end_step()

    except Exception as e:
        print(f"Error during reading: {e}")
    finally:
        reader.close()

    if all_data:

        npz_data = {}
        npz_data["num_steps"] = step_count

        var_names = set()
        for step_data in all_data.values():
            var_names.update(step_data.keys())

        npz_data["variable_names"] = list(var_names)

        for step_key, step_data in all_data.items():
            for var_name, var_data in step_data.items():
                npz_key = f"{step_key}_{var_name}"
                npz_data[npz_key] = var_data

        np.savez_compressed(output_file, **npz_data)
        print(f"Successfully saved {len(npz_data)} arrays to {output_file}")
        print(f"Total steps processed: {step_count}")
        print(f"Variables found: {var_names}")
    else:
        print("No data was read from the BP file.")


def npz_to_bp(input_file, output_file, write_io_name, xml=None):
    print(f"Converting {input_file} to {output_file}")

    try:
        npz_data = np.load(input_file)
        print(f"Loaded NPZ file with {len(npz_data.files)} arrays")
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return

    writer = Writer(write_io_name, output_file, xml=xml)

    try:

        num_steps = int(npz_data.get("num_steps", 1))
        print(f"Number of steps to write: {num_steps}")

        var_names = npz_data.get("variable_names", [])
        if len(var_names) > 0:
            print(f"Variables to write: {var_names}")

        steps_data = {}
        for key in npz_data.files:
            if key in ["num_steps", "variable_names"]:
                continue

            if key.startswith("step_"):
                parts = key.split("_", 2)
                if len(parts) >= 3:
                    step_num = int(parts[1])
                    var_name = "_".join(parts[2:])

                    if step_num not in steps_data:
                        steps_data[step_num] = {}
                    steps_data[step_num][var_name] = npz_data[key]

        for step_num in sorted(steps_data.keys()):
            print(f"Writing step {step_num}")
            writer.begin_step()

            step_data = steps_data[step_num]
            for var_name, var_data in step_data.items():
                print(f"  Writing variable '{var_name}' with shape: {var_data.shape}")
                writer.write(var_name, var_data)

            writer.end_step()

        print(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        print(f"Error during conversion: {e}")
    finally:
        writer.close()


def get_file_extension(filename):
    """Get file extension in lowercase, handling .bp5 files"""
    path = Path(filename)
    ext = path.suffix.lower()

    if ext == ".bp5":
        return ".bp"

    if path.is_dir():

        if any(f.endswith(".bp") or f.endswith(".bp5") for f in os.listdir(filename)):
            return ".bp"

    return ext


def main():
    install()

    args = parse_arguments()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    if os.path.isdir(args.input):
        print(f"Input is a directory: {args.input}")

        bp_files = [
            f for f in os.listdir(args.input) if f.endswith(".bp") or f.endswith(".bp5")
        ]
        if bp_files:
            print(f"Found BP files in directory: {bp_files}")
            print("Please specify the exact BP file path, not the directory.")
            sys.exit(1)

    if args.xml and not os.path.exists(args.xml):
        print(f"Error: XML file '{args.xml}' does not exist.")
        sys.exit(1)

    input_ext = get_file_extension(args.input)
    output_ext = get_file_extension(args.output)

    print(f"Input file: {args.input} ({input_ext})")
    print(f"Output file: {args.output} ({output_ext})")
    if args.xml:
        print(f"XML config: {args.xml}")

    if input_ext == ".bp" and output_ext == ".npz":

        print(f"Using read IO: {args.readIO}")
        bp_to_npz(args.input, args.output, args.readIO, args.xml)
    elif input_ext == ".npz" and output_ext == ".bp":

        print(f"Using write IO: {args.writeIO}")
        npz_to_bp(args.input, args.output, args.writeIO, args.xml)
    else:
        print(f"Error: Unsupported conversion from {input_ext} to {output_ext}")
        print("Supported conversions:")
        print("  .bp/.bp5 -> .npz")
        print("  .npz -> .bp")
        sys.exit(1)


if __name__ == "__main__":
    main()
