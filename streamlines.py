import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime
from adios2 import Adios, Stream

def find_bp5_directories(directory="."):
    """Find all BP5 directories starting with cavity2D in the specified directory"""
    bp_dirs = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith("cavity2D") and item.endswith(".bp5"):
            bp_dirs.append(item_path)
    return bp_dirs

def read_adios2_velocity(bp_dir):
    """Read velocity components from ADIOS2 BP5 directory using the Stream API"""
    # Create ADIOS object
    adios_obj = Adios()
    
    # Declare IO object
    io = adios_obj.declare_io("reader")
    
    # Open the file using Stream API
    with Stream(io, bp_dir, 'r') as f:
        for _ in f.steps():
            step = f.current_step()
            print(f"Reading step {step}")
            
            # Read velocity components
            ux = f.read('ux')
            uy = f.read('uy')
            
            # If 3D data with singleton first dimension, extract 2D slice
            if len(ux.shape) == 3 and ux.shape[0] == 1:
                ux = ux[0, :, :]
                uy = uy[0, :, :]
                
            yield step, ux, uy

def calculate_global_velocity_range(bp_dirs):
    """Calculate the global min/max velocity magnitudes across all BP5 directories and steps"""
    global_min = float('inf')
    global_max = float('-inf')
    
    print("Calculating global velocity magnitude range...")
    
    for bp_dir in bp_dirs:
        print(f"Scanning {os.path.basename(bp_dir)}...")
        try:
            for step, ux, uy in read_adios2_velocity(bp_dir):
                magnitude = np.sqrt(ux**2 + uy**2)
                current_min = np.min(magnitude)
                current_max = np.max(magnitude)
                
                global_min = min(global_min, current_min)
                global_max = max(global_max, current_max)
                
                print(f"  Step {step}: min={current_min:.6f}, max={current_max:.6f}")
        except Exception as e:
            print(f"Error scanning {bp_dir}: {str(e)}")
            continue
    
    print(f"Global velocity magnitude range: [{global_min:.6f}, {global_max:.6f}]")
    return global_min, global_max

def plot_streamlines(bp_dir, vmin, vmax):
    """Plot streamlines for all steps in the given BP5 directory with fixed color scale"""
    print(f"Processing BP5 directory: {bp_dir}")
    
    # Get the directory name for output naming
    base_filename = os.path.basename(bp_dir).split('.')[0]
    
    try:
        for step, ux, uy in read_adios2_velocity(bp_dir):
            # Create coordinate meshgrid based on velocity field dimensions
            ny, nx = ux.shape
            x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
            
            # Plot streamlines
            plt.figure(figsize=(10, 8))
            # Compute velocity magnitude for color
            magnitude = np.sqrt(ux**2 + uy**2)
            
            # Create streamlines with fixed color scale
            plt.streamplot(x, y, ux, uy, color=magnitude, cmap='jet', density=1.5)
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f"{os.path.basename(bp_dir)} - Streamlines at Step {step}")
            
            # Set fixed color scale range
            cb = plt.colorbar(label="Velocity magnitude")
            cb.mappable.set_clim(vmin, vmax)
            
            # Create a more unique filename with directory info to prevent overwriting
            # Extract resolution info from the directory name
            dir_parts = os.path.basename(bp_dir).split('.')
            resolution_info = '_'.join(dir_parts[1:4]) if len(dir_parts) >= 4 else 'unknown'
            
            # Add timestamp for uniqueness
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create unique filename
            output_filename = f"{base_filename}_{resolution_info}_streamlines_step{step:04d}_{timestamp}.png"
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Step {step} processed and saved as {output_filename}")
    except Exception as e:
        print(f"Error processing {bp_dir}: {str(e)}")

def main():
    """Main function to process all BP5 directories in the current directory"""
    # Find all BP5 directories in the current directory
    bp_dirs = find_bp5_directories()
    
    if not bp_dirs:
        print("No BP5 directories starting with 'cavity2D' found in the current directory.")
        sys.exit(1)
    
    print(f"Found {len(bp_dirs)} BP5 directories to process.")
    
    # Calculate global velocity magnitude range
    vmin, vmax = calculate_global_velocity_range(bp_dirs)
    
    # Add a small buffer to the range for better visualization
    range_buffer = (vmax - vmin) * 0.02  # 2% buffer
    vmin_plot = max(0, vmin - range_buffer)  # Don't go below 0 for velocity magnitude
    vmax_plot = vmax + range_buffer
    
    print(f"Using color scale range: [{vmin_plot:.6f}, {vmax_plot:.6f}]")
    
    # Process each BP5 directory with fixed color scale
    for bp_dir in bp_dirs:
        plot_streamlines(bp_dir, vmin_plot, vmax_plot)
    
    print(f"All streamline images saved with consistent color scale!")

if __name__ == "__main__":
    main()
