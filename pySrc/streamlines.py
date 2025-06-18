import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from adios2 import Adios, Stream
import mpi4py as MPI
from rich.traceback import install
from scipy.interpolate import RegularGridInterpolator
from matplotlib.collections import LineCollection

# dt is the physical step size change for each 
# 257 -> 0.002, 515 -> 0.0005, 1025-> 0.0001, 2049-> 0.00005, 4097 -> 0.000025
# how to tell the all to be the same length full streamline then same steps then different dt
def rk4_streamline_from_grid(x0, y0, vx, vy, max_len=-1, dt=0.002, max_steps=10000, xlim=None, ylim=None):
    xgrid = np.linspace(0, 1, vx.shape[1])  
    ygrid = np.linspace(0, 1, vx.shape[0]) 

    interp_vx = RegularGridInterpolator((ygrid, xgrid), vx)
    interp_vy = RegularGridInterpolator((ygrid, xgrid), vy)
    def vector_field(x, y):
        point = np.array([y, x]) 
        u = interp_vx(point)[0] if isinstance(interp_vx(point), np.ndarray) else float(interp_vx(point))
        v = interp_vy(point)[0] if isinstance(interp_vy(point), np.ndarray) else float(interp_vy(point))
        norm = np.hypot(u, v)
        if norm < 1e-8:
            return np.array([0.0, 0.0])
        return np.array([u, v]) / norm

    path = [(x0, y0)]
    x, y = x0, y0

    arc_len = 0 
    cnt = 0 
    for _ in range(max_steps):
        cnt +=1
        k1 = vector_field(x, y)
        k2 = vector_field(x + dt * k1[0] / 2, y + dt * k1[1] / 2)
        k3 = vector_field(x + dt * k2[0] / 2, y + dt * k2[1] / 2)
        k4 = vector_field(x + dt * k3[0], y + dt * k3[1])
        dx, dy = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_prev = x
        y_prev = y 
        
        x += dx
        y += dy
        arc_len +=np.sqrt(pow((x-x_prev),2) +  pow((y-y_prev),2) )
        
        if xlim and (x < xlim[0] or x > xlim[1]):
            print('xlim')
            break
        if ylim and (y < ylim[0] or y > ylim[1]):
            print('ylim')
            break
        path.append((x, y))
        
        if max_len > 0 and arc_len >= max_len:
            print('arc length')            
            break
    print(arc_len)
    print("--"*60)
    print(cnt)
    print('--'*60)
    return np.array(path)



# You need to do the same with 3d also take in the seeds as a param
# save segments and write out and save seed points 
def calculate_global_velocity_range(all_data, is_3d=False):
    install()
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

# make seeds in parallel 
def plot_streamlines_2d(ux, uy, step, base_filename, vmin, vmax, save_fig, streamline_writer):
    install()
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
    if save_fig:
        fig_all.savefig(output_path_all, dpi=300, bbox_inches='tight')
        print(f"Saved all streamlines plot: {output_path_all}")
    plt.close(fig_all)

    # ---------------------------------------------
    # Plot and save a single streamline from a point
    # ---------------------------------------------
    fig_one, ax_one = plt.subplots(figsize=(10, 8))
    
    streamline_path = rk4_streamline_from_grid(0.5, 0.1, ux_2d, uy_2d, max_len=1000)
    
    # Plot the background velocity field for context
    strm_bg = ax_one.streamplot(x, y, ux_2d, uy_2d, color='lightgray', density=0.5)
    # Set alpha on the line collection
    strm_bg.lines.set_alpha(0.3)
    
    # Plot the custom streamline
    ax_one.plot(streamline_path[:, 0], streamline_path[:, 1], 'red', linewidth=2, 
                label='Custom RK4 Streamline (0.5, 0.1)')
    ax_one.plot(streamline_path[0, 0], streamline_path[0, 1], 'go', markersize=8, 
                label='Start Point')
    ax_one.plot(streamline_path[-1, 0], streamline_path[-1, 1], 'ro', markersize=8, 
                label='End Point')

    ax_one.set_xlabel('X')
    ax_one.set_ylabel('Y')
    ax_one.set_title(f"{base_filename} - 2D Custom RK4 Streamline - Step {step}")
    ax_one.legend()
    ax_one.grid(True, alpha=0.3)
    ax_one.set_aspect('equal')
    
    # ---------------------------------------------
    # Third plot: Just the RK4 streamline, colored by velocity magnitude
    # ---------------------------------------------
    fig_three, ax_three = plt.subplots(figsize=(10, 8))
    
    ux_interp = RegularGridInterpolator((np.linspace(0, 1, ny), np.linspace(0, 1, nx)), ux_2d)
    uy_interp = RegularGridInterpolator((np.linspace(0, 1, ny), np.linspace(0, 1, nx)), uy_2d)
    magnitudes = []
    for pt in streamline_path:
        u_val = ux_interp((pt[1], pt[0]))
        v_val = uy_interp((pt[1], pt[0]))
        mag = np.hypot(u_val, v_val)
        magnitudes.append(mag)
    magnitudes = np.array(magnitudes)
    
    points = streamline_path.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Use the average magnitude on each segment for color mapping
    segment_mags = 0.5 * (magnitudes[:-1] + magnitudes[1:])
    
    lc = LineCollection(segments, cmap='jet', 
                        norm=plt.Normalize(vmin=segment_mags.min(), vmax=segment_mags.max()))
    lc.set_array(segment_mags)
    lc.set_linewidth(2)
    ax_three.add_collection(lc)
    
    ax_three.set_xlim(x.min(), x.max())
    ax_three.set_ylim(y.min(), y.max())
    ax_three.set_xlabel('X')
    ax_three.set_ylabel('Y')
    ax_three.set_title(f"{base_filename} - 2D RK4 Streamline Colored by Velocity Magnitude - Step {step}")
    ax_three.set_aspect('equal')
    cbar = fig_three.colorbar(lc, ax=ax_three, label="Velocity Magnitude")
    
    output_filename_third = f"{base_filename}_2d_rk4_streamline_step{step:04d}.png"
    output_path_third = os.path.join(output_dir, output_filename_third)
    if save_fig:
        fig_three.savefig(output_path_third, dpi=300, bbox_inches='tight')
        print(f"Saved RK4 streamline only plot: {output_path_third}")
    plt.close(fig_three)

    
    segments = streamline_path.flatten()
    seed_points = np.array([0.5, 0.1])
    
    streamline_writer.begin_step()
    print("="*60)
    print(segments)
    print("="*60)
    streamline_writer.write('segments', segments)
    streamline_writer.end_step()
    
    print(f"Written streamline data for step {step} to segments.bp")

    output_filename_one = f"{base_filename}_2d_single_streamline_step{step:04d}.png"
    output_path_one = os.path.join(output_dir, output_filename_one)
    if save_fig:
        fig_one.savefig(output_path_one, dpi=300, bbox_inches='tight')
        print(f"Saved single streamline plot: {output_path_one}")
    plt.close(fig_one)
    
    return output_filename_one

def plot_streamlines_3d(ux, uy, uz, step, base_filename, vmin, vmax, var_1, var_2, slice_idx):
    """Plot 3D streamlines by extracting 2D slice"""
    install()
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
    install()
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
    
    parser.add_argument('--save_fig',
                        type=bool,
                        default=False,
                        help='save images')
    # add a point here
    
    return parser.parse_args()

def main():
    install()
    
    args = parse_arguments()
    
    # if not (os.path.isfile(args.path) or (os.path.isdir(args.path) and args.path.endswith('.bp5'))):
    #     print(f"Error: Path '{args.path}' is not a valid BP5 file.")
    #     sys.exit(1)

    bp_file = args.path
    xml_file = args.xml
    save_fig = args.save_fig
    print(f"saving figures = {save_fig}")
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
    
    write_io = adios_obj.declare_io("WriteStreamlineIO")
    streamline_output_file = 'segments.bp'
    
    print("Second pass: Generating plots and writing streamline data...")
    
    first_step, first_ux, first_uy, first_uz = all_data[0]
    if len(first_ux.shape) == 3:
        mid_slice = first_ux.shape[2] // 2
        first_ux_2d = first_ux[:, :, mid_slice]
        first_uy_2d = first_uy[:, :, mid_slice]
    else:
        first_ux_2d = first_ux
        first_uy_2d = first_uy
    
    sample_streamline = rk4_streamline_from_grid(0.5, 0.1, first_ux_2d, first_uy_2d, max_len=1000)
    sample_segments = sample_streamline.flatten()
    sample_seeds = np.array([0.5, 0.1])
    
    seg_shape = [sample_segments.size]
    var_segments = write_io.define_variable('segments', sample_segments, seg_shape, [0], seg_shape)
    
    seed_shape = [sample_seeds.size]
    var_seeds = write_io.define_variable('seeds', sample_seeds, seed_shape, [0], seed_shape)
    
    with Stream(write_io, streamline_output_file, 'w') as streamline_writer:
        
        for step, ux, uy, uz in all_data:
            print(f"Processing step {step}")
            
            try:
                if is_3d and uz is not None:
                    output_filename = plot_streamlines_3d(ux, uy, uz, step, base_filename, 
                                                        vmin, vmax, var_1, var_2, slice_idx)
                else:
                    output_filename = plot_streamlines_2d(ux, uy, step, base_filename, vmin, vmax, save_fig, streamline_writer)
                
                if output_filename:
                    print(f"Saved: {output_filename}")
                    
            except Exception as e:
                print(f"Error processing step {step}: {e}")
                continue
    
    print("All streamline plots completed!")
    print(f"Streamline data saved to: {streamline_output_file}")
    print("Please check the ../RESULTS")

if __name__ == "__main__":
    install()
    main()
