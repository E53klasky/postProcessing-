import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from adios2 import Adios, Stream
from mpi4py import MPI 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def read_adios2_velocity(bp_dir, max_steps=None, xml_file=None):
    adios_obj = Adios(xml_file) if xml_file else Adios()
    io = adios_obj.declare_io("readerIO")

    with Stream(io, bp_dir, 'r') as f:
        step_count = 0
        for _ in f.steps():
            if max_steps is not None and step_count >= max_steps:
                print(f"Reached maximum steps limit ({max_steps}), stopping.")
                break
            
            step = f.current_step()
            print(f"Reading step {step}")
            
            ux = f.read('ux')
            uy = f.read('uy')
            
            if len(ux.shape) == 3 and ux.shape[0] == 1:
                ux = ux[0, :, :]
                uy = uy[0, :, :]
                
            yield step, ux, uy
            step_count += 1

def calculate_global_velocity_range(bp_file, max_steps=None, xml_file=None):
    global_min = float('inf')
    global_max = float('-inf')
    
    print("Calculating global velocity magnitude range...")
    print(f"Scanning {os.path.basename(bp_file)}...")

    try:
        for step, ux, uy in read_adios2_velocity(bp_file, max_steps, xml_file):
            magnitude = np.sqrt(ux**2 + uy**2)
            current_min = np.min(magnitude)
            current_max = np.max(magnitude)
            
            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)
            
            print(f"  Step {step}: min={current_min:.6f}, max={current_max:.6f}")
    except Exception as e:
        print(f"Error scanning {bp_file}: {str(e)}")
        sys.exit(1)
    
    print(f"Global velocity magnitude range: [{global_min:.6f}, {global_max:.6f}]")
    return global_min, global_max

def plot_streamlines(bp_dir, vmin, vmax, max_steps=None, xml_file=None):
    print(f"Processing BP5 directory: {bp_dir}")
    base_filename = os.path.basename(bp_dir).split('.')[0]
    
    try:
        for step, ux, uy in read_adios2_velocity(bp_dir, max_steps, xml_file):
            ny, nx = ux.shape
            x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
            
            plt.figure(figsize=(10, 8))
            magnitude = np.sqrt(ux**2 + uy**2)
            
            plt.streamplot(x, y, ux, uy, color=magnitude, cmap='jet', density=1.5)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f"{os.path.basename(bp_dir)} - Streamlines at Step {step}")
            
            cb = plt.colorbar(label="Velocity magnitude")
            cb.mappable.set_clim(vmin, vmax)
            
            dir_parts = os.path.basename(bp_dir).split('.')
            resolution_info = '_'.join(dir_parts[1:4]) if len(dir_parts) >= 4 else 'unknown'
            
            output_filename = f"{base_filename}_{resolution_info}_streamlines_step{step:04d}.png"
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Step {step} processed and saved as {output_filename}")
    except Exception as e:
        print(f"Error processing {bp_dir}: {str(e)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate streamline plots from ADIOS2 BP5 files')
    
    parser.add_argument('path', type=str, help='Path to the BP5 file to process')
    parser.add_argument('max_steps', type=int, help='Maximum number of time steps to process')
    
    parser.add_argument('--workers', '-w', type=int, default=multiprocessing.cpu_count(),
                        help=f'Number of worker processes (default: {multiprocessing.cpu_count()})')
    
    parser.add_argument('--xml', '-x', type=str, default=None,
                        help='Path to ADIOS2 XML configuration file (optional)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if not (os.path.isfile(args.path) or (os.path.isdir(args.path) and args.path.endswith('.bp5'))):
        print(f"Error: Path '{args.path}' is not a valid BP5 file.")
        sys.exit(1)

    bp_file = args.path
    max_steps = args.max_steps
    xml_file = args.xml

    # NOT sure if this will work with the SST engine 
    # if not os.path.exists(bp_file):
    #     print(f"Error: File {bp_file} does not exist.")
    #     sys.exit(1)

    print(f"Processing BP5 file: {bp_file}")
    print(f"Will process maximum of {max_steps} time steps.")
    
    if xml_file:
        print(f"Using XML configuration: {xml_file}")
    
    vmin, vmax = calculate_global_velocity_range(bp_file, max_steps, xml_file)
    print(f"Using color scale range: [{vmin:.6f}, {vmax:.6f}]")
    
    plot_streamlines(bp_file, vmin, vmax, max_steps, xml_file)
    
    print("Streamline images saved with consistent color scale!")

if __name__ == "__main__":
    main()
