import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from adios2 import Adios, Stream
import mpi4py as MPI


# You need to do the same with 3d also take in the seeds as a param
# save segments and write out and save seed points 
def calculate_global_velocity_range(all_data, is_3d=False):
    """Calculate global velocity range for consistent coloring"""
    global_min = float('inf')
    global_max = float('-inf')
    
    print("Calculating global velocity magnitude range...")
    
    for step, ux, uy, uz in all_data:
        if is_3d and uz is not None:
            magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
        else:
            magnitude = np.sqrt(ux**2 + uy**2)
        
        current_min = np.min(magnitude)
        current_max = np.max(magnitude)
        # for now
        # Using color scale range: [0.000000, 0.591330]
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        
        #global_min = 0
        #global_max = 0.591330
        
        print(f"  Step {step}: min={current_min:.6f}, max={current_max:.6f}")
    
    print(f"Global velocity magnitude range: [{global_min:.6f}, {global_max:.6f}]")
    return global_min, global_max

def plot_streamlines_2d(ux, uy, step, base_filename, vmin, vmax):
    if len(ux.shape) == 3:
        mid_slice = ux.shape[2] // 2
        ux_2d = ux[:, :, mid_slice]
        uy_2d = uy[:, :, mid_slice]
        print(f"Using middle slice {mid_slice} for 2D visualization")
    else:
        ux_2d = ux
        uy_2d = uy

    ny, nx = ux_2d.shape
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    magnitude = np.sqrt(ux_2d**2 + uy_2d**2)

    # ------------------------------
    # Plot and save all streamlines
    # ------------------------------
    
    fig_all, ax_all = plt.subplots(figsize=(10, 8))
    strm_all = ax_all.streamplot(x, y, ux_2d, uy_2d, color=magnitude, cmap='jet', density=1.5, maxlength= 0.5)
    
    ax_all.set_xlabel('X')
    ax_all.set_ylabel('Y')
    ax_all.set_title(f"{base_filename} - 2D Streamlines - Step {step}")
    cb_all = fig_all.colorbar(strm_all.lines, ax=ax_all, label="Velocity magnitude")
    cb_all.mappable.set_clim(vmin, vmax)
    output_filename_all = f"{base_filename}_2d_all_streamlines_step{step:04d}.png"
    output_path_all = os.path.join(output_dir, output_filename_all)
    fig_all.savefig(output_path_all, dpi=300, bbox_inches='tight')
    print(f"Saved all streamlines plot: {output_path_all}")

    # ---------------------------------------------
    # Plot and save a single streamline from a point
    # ---------------------------------------------
    fig_one, ax_one = plt.subplots(figsize=(10, 8))
    seed_points = np.array([[0.5], [0.1]])
    strm_one = ax_one.streamplot(x, y, ux_2d, uy_2d, color=magnitude, cmap='jet', density=1.5, start_points=seed_points.T, maxlength=10, integration_direction='forward')
    segments_list = strm_one.lines.get_segments()
    segments = np.concatenate(segments_list, axis=0).flatten() if segments_list else np.array([], dtype=float)
    seed_points_linear = seed_points.flatten() if seed_points.size else np.array([], dtype=float)
    ad = Adios()
    Wio = ad.declare_io("WriteIO")
    variables_defined = False
    
    #NOTE ONLY SAVES THE LAST SEGMENTS STEPS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    with Stream(Wio, 'segments.bp', 'w') as w:
        if not variables_defined:
            seg_shape = [segments.size]
            seg_start = [0]
            seg_count = [segments.size]
            var_seg = Wio.define_variable('segments', segments, seg_shape, seg_start, seg_count)
            seed_shape = [seed_points_linear.size]
            seed_start = [0]
            seed_count = [seed_points_linear.size]
            var_seed = Wio.define_variable('seeds', seed_points_linear, seed_shape, seed_start, seed_count)
            
        w.begin_step()
        w.write('segments', segments)
        w.write('seeds', seed_points_linear)
        w.end_step()

        
    print('='*60)
    print(segments_list)
    for segments in segments_list:
        ax_one.plot(segments[:, 0], segments[:, 1], color='red', linewidth=0.5)

    # plt.show()
    # sys.exit(1)
    print('='*60)
    ax_one.set_xlabel('X')
    ax_one.set_ylabel('Y')
    ax_one.set_title(f"{base_filename} - 2D Single Streamline - Step {step}")
    cb_one = fig_one.colorbar(strm_one.lines, ax=ax_one, label="Velocity magnitude")
    cb_one.mappable.set_clim(vmin, vmax)
    output_filename_one = f"{base_filename}_2d_single_streamline_step{step:04d}.png"
    output_path_one = os.path.join(output_dir, output_filename_one)
    fig_one.savefig(output_path_one, dpi=300, bbox_inches='tight')
    print(f"Saved single streamline plot: {output_path_one}")
    plt.close()
    
    return output_filename_one

def plot_streamlines_3d(ux, uy, uz, step, base_filename, vmin, vmax, var_1, var_2, slice_idx):
    """Plot 3D streamlines by extracting 2D slice"""
    if len(ux.shape) != 3:
        print(f"Warning: Expected 3D data, got {ux.shape}")
        return None
    
    if slice_idx >= ux.shape[2]:
        print(f"Warning: Slice {slice_idx} exceeds data size {ux.shape[2]}")
        slice_idx = ux.shape[2] // 2
        print(f"Using middle slice: {slice_idx}")
    

    vel_dict = {
        'ux': ux, 'uy': uy, 'uz': uz,
        'vx': ux, 'vy': uy, 'vz': uz
    }
    
    u = vel_dict[var_1][:, :, slice_idx]
    v = vel_dict[var_2][:, :, slice_idx]
    
    ny, nx = u.shape
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    output_dir = "../RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    magnitude = np.sqrt(u**2 + v**2)
    plt.streamplot(x, y, u, v, color=magnitude, cmap='jet', density=1.5)
    plt.xlabel(var_1.upper())
    plt.ylabel(var_2.upper())
    plt.title(f"{base_filename} - 3D Streamlines ({var_1} vs {var_2}) - Step {step}, Slice {slice_idx}")
    
    cb = plt.colorbar(label="Velocity magnitude")
    cb.mappable.set_clim(vmin, vmax)
    
    output_filename = f"{base_filename}_3d_{var_1}{var_2}_slice{slice_idx}_step{step:04d}.png"
    full_path = os.path.join(output_dir, output_filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_filename


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate streamline plots from ADIOS2 BP files')
    parser.add_argument('path', 
                        type=str, 
                        help='Path to the BP file to process (REQUIRED)')
    parser.add_argument('--xml', 
                        '-x', type=str, 
                        default=None, 
                        help='ADIOS2 XML config file default: None (optional)')
    parser.add_argument('--mode',
                        '-m', 
                        type=str, 
                        choices=['2d', '3d'], 
                        default='2d', 
                        help='2D or 3D mode default: 2d (optional)') 
    parser.add_argument('--var1', 
                        type=str, 
                        default='ux', 
                        help='First velocity component (3D mode) (REQUIRED)')
    parser.add_argument('--var2', 
                        type=str, 
                        default='uy', 
                        help='Second velocity component (3D mode) (REQUIRED)')
    parser.add_argument('--slice', 
                        '-s', 
                        type=int, 
                        default=16, 
                        help='Slice index (3D mode) default: 16 (optional)')
    parser.add_argument('max_steps', 
                        type=int, 
                        help='Maximum number of time steps to process (REQUIRED)')
    # add a point here
    return parser.parse_args()

def main():

    
    args = parse_arguments()
    
    # if not (os.path.isfile(args.path) or (os.path.isdir(args.path) and args.path.endswith('.bp5'))):
    #     print(f"Error: Path '{args.path}' is not a valid BP5 file.")
    #     sys.exit(1)

    bp_file = args.path
    xml_file = args.xml
    is_3d = (args.mode == '3d')
    if is_3d == '3d':
        print("This code does not work in 3D mode yet, comming soon ...")
        sys.exit(1)
    var_1 = args.var1.lower()
    var_2 = args.var2.lower()
    slice_idx = args.slice
    max_steps = args.max_steps
    
    if max_steps <= 0:
        print("Error: max_steps must be a non-negative integer.")
        sys.exit(1)

    print(f"Processing BP5 file: {bp_file}")
    print(f"Mode: {'3D' if is_3d else '2D'}")
    if is_3d:
        print(f"Variables: {var_1} vs {var_2}, slice: {slice_idx}")
    
    if xml_file:
        adios_obj = Adios(xml_file)
    else:
        adios_obj = Adios()
    
    io = adios_obj.declare_io("readerIO")
    base_filename = os.path.basename(bp_file).split('.bp')[0]
    
    all_data = []
    
    print("First pass: Reading all data...")
    with Stream(io, bp_file, 'r') as reader:
        step_count = 0
        for _ in reader:

            status = reader.begin_step()
            step = reader.current_step()
            print(f"Reading step {step}")
            
            try:
                ux = reader.read('ux')
                uy = reader.read('uy')
                
                try:
                    uz = reader.read('uz')
                except:
                    uz = None
                
                if len(ux.shape) == 4 and ux.shape[0] == 1:
                    ux = ux[0, :, :, :]
                    uy = uy[0, :, :, :]
                    if uz is not None:
                        uz = uz[0, :, :, :]
                elif len(ux.shape) == 3 and ux.shape[0] == 1:
                    ux = ux[0, :, :]
                    uy = uy[0, :, :]
                    if uz is not None:
                        uz = uz[0, :, :]
                
                print(f"Data shapes: ux={ux.shape}, uy={uy.shape}" + (f", uz={uz.shape}" if uz is not None else ""))
                
                all_data.append((step, ux, uy, uz))
                step_count += 1
                
                if not status or step_count >= max_steps:
                    print(f"End of steps reached or error reading step {step}")
                    break
                else:
                    continue
            except Exception as e:
                print(f"Error reading step {step}: {e}")
                try:
                    available_vars = reader.available_variables()
                    print(f"Available variables: {list(available_vars.keys())}")
                except:
                    print("Could not retrieve available variables")
                break
    
    if not all_data:
        print("No data was read successfully!")
        sys.exit(1)
    
    vmin, vmax = calculate_global_velocity_range(all_data, is_3d)
    
    print("Second pass: Generating plots...")
    for step, ux, uy, uz in all_data:
        print(f"Processing step {step}")
        
        try:
            if is_3d and uz is not None:
                output_filename = plot_streamlines_3d(ux, uy, uz, step, base_filename, 
                                                    vmin, vmax, var_1, var_2, slice_idx)
            else:
                output_filename = plot_streamlines_2d(ux, uy, step, base_filename, vmin, vmax)
            
            if output_filename:
                print(f"Saved: {output_filename}")
                
        except Exception as e:
            print(f"Error processing step {step}: {e}")
            continue
    
    print("All streamline plots completed!")
    print("Please check the ../RESULTS")

if __name__ == "__main__":
    main()
