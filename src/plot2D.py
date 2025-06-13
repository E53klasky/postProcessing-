import adios2
import matplotlib.pyplot as plt
import argparse
import os
from mpi4py import MPI
import numpy as np
from PIL import Image

def parser_arguments():
    parser = argparse.ArgumentParser(description="Parallel 2D plot with consistent color scale")
    parser.add_argument('input_file', type=str, help='Path to the BP file to process (REQUIRED)')
    parser.add_argument('--xml', '-x', type=str, default=None, help='ADIOS2 XML config file (optional)')
    parser.add_argument('--vars', '-v', type=str, required=True, help='Variables to plot, separated by commas (REQUIRED)')
    parser.add_argument('--max_steps', '-n', type=int, required=True, help='Maximum number of timesteps to process')
    return parser.parse_args()

def save_rank_image(local_data, rank, var, step, vmin, vmax):
    plt.imshow(local_data[0], cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax)
    plt.axis('off')
    fname = f"tmp_{var}_rank{rank:03d}_step{step:04d}.png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()
    return fname

def stitch_images_horizontally(image_paths, output_path):
    images = [Image.open(p) for p in image_paths]
    heights = [img.height for img in images]
    widths = [img.width for img in images]

    total_width = sum(widths)
    max_height = max(heights)

    stitched = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.width

    stitched.save(output_path)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parser_arguments()
    input_file = args.input_file
    adios2_xml = args.xml
    var_list = args.vars.split(',')
    max_steps = args.max_steps

    if adios2_xml:
        adios = adios2.Adios(adios2_xml, comm)
    else:
        adios = adios2.Adios(comm)

    io = adios.declare_io("ReadIO")
    output_dir = "../RESULTS"
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    with adios2.Stream(io, input_file, 'r', comm) as stream:
        for step_count, _ in enumerate(stream):
            status = stream.begin_step()
            if status and step_count >= max_steps:
                break

            for var in var_list:
                if var not in stream.available_variables():
                    if rank == 0:
                        print(f"Variable {var} not found.")
                    continue

                var_in = io.inquire_variable(var)
                shape = var_in.shape()

                if not shape or len(shape) != 3 or shape[0] != 1:
                    if rank == 0:
                        print(f"Skipping variable {var} due to unexpected shape {shape}")
                    continue

                Y, Z = shape[1], shape[2]
                base = Z // size
                rem = Z % size
                local_z = base + 1 if rank < rem else base
                local_start = rank * base + min(rank, rem)

                count = [1, Y, local_z]
                start = [0, 0, local_start]
                var_in.set_selection((start, count))

                local_data = stream.read(var)

                local_min = float(np.min(local_data))
                local_max = float(np.max(local_data))
                global_min = comm.allreduce(local_min, op=MPI.MIN)
                global_max = comm.allreduce(local_max, op=MPI.MAX)

                img_path = save_rank_image(local_data, rank, var, step_count, global_min, global_max)
                all_img_paths = comm.gather(img_path, root=0)

                if rank == 0:
                    final_path = os.path.join(output_dir, f"{var}_step_{step_count:04d}.png")
                    stitch_images_horizontally(all_img_paths, final_path)
                    print(f"Saved stitched image: {final_path}")

                    for p in all_img_paths:
                        os.remove(p)

    if rank == 0:
        print(f"\nAll output saved in {output_dir}/")

if __name__ == "__main__":
    main()
