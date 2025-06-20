import numpy as np
import os
import sys
import argparse
import adios2 
from rich.traceback import install
from scipy.interpolate import RegularGridInterpolator
from matplotlib.collections import LineCollection
import ReaderClass 
import re

def rk4_streamline_from_grid(x0, y0, vx, vy, max_len=3.0, dt=0.01, max_steps=1000, xlim=None, ylim=None):
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

    paths = []
    coords_x = []
    coords_y = []
    offsets = []
    offset = 0
    for i in range(len(x0)):
        x = x0[i]
        y = y0[i]
        cnt = 0
        arc_len = 0
        path = [(x,y)]
        path_x = [x]
        path_y = [y]
        offsets.append(offset)
        for _ in range(max_steps):
            cnt +=1
            offset +=1 
            k1 = vector_field(x, y)
            k2 = vector_field(x + dt * k1[0] / 2, y + dt * k1[1] / 2)
            k3 = vector_field(x + dt * k2[0] / 2, y + dt * k2[1] / 2)
            k4 = vector_field(x + dt * k3[0], y + dt * k3[1])
            dx, dy = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x_prev = x
            y_prev = y 
            
            x += dx
            y += dy
            arc_len +=np.sqrt(pow((x-x_prev),2) + pow((y-y_prev),2) )
            
            if xlim and (x < xlim[0] or x > xlim[1]):
                print('xlim')
                break
            if ylim and (y < ylim[0] or y > ylim[1]):
                print('ylim')
                break
            path.append((x, y))
            path_x.append(x)
            path_y.append(y)
            
            if max_len > 0 and arc_len >= max_len:
                print('arc length')            
                break

        print(arc_len)
        print("--"*60)
        print(cnt)
        print('--'*60)
        print(len(path))
        paths.append(path)
        coords_x.append(path_x)
        coords_y.append(path_y)
    
        
    
    return  (np.array(coords_x), np.array(coords_y), np.array(offsets))  


def parse_seed_points(seed_str):
    matches = re.findall(r'\(([^,]+),([^)]+)\)', seed_str)
    if not matches:
        raise ValueError("Invalid seed format. Use format like: '(0.1,0.5),(0.4,0.4)'")
    x_vals = []
    y_vals = []
    for x, y in matches:
        x_vals.append(float(x.strip()))
        y_vals.append(float(y.strip()))
    return np.array(x_vals), np.array(y_vals)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate streamline plots from ADIOS2 BP files')
    parser.add_argument('--file',
                        '-f', 
                        type=str,
                        required=True, 
                        help='Path to the BP file to process (REQUIRED)')
    parser.add_argument('--xml', 
                        '-x', type=str, 
                        default=None, 
                        required=False,
                        help='ADIOS2 XML config file default: None (optional)')

    parser.add_argument('--vars', 
                        '-v', 
                        type=str, 
                        required=True,
                        help='Velocity variable names (order matters), separated by commas, e.g., vx,vy,vz (REQUIRED)')

    parser.add_argument('--seeds_points',
                        '-s',
                        type=str,
                        required=True,
                         help="Comma-separated list of seed points in the format '(x1,y1),(x2,y2)' (REQUIRED)")
    parser.add_argument('--io_name',
                        '-io',
                        type=str,
                        required=True,
                        help='Name you want to declare the io name as (if you are using the xml this must match) (REQUIRED)')
    parser.add_argument('--output', 
                        '-o',
                        type=str,
                        default='segments.bp',
                        required=False,
                        help='Output file name default: segments.bp (optional)')
    # take in dt, lenght, num steps as  optional 
        
    return parser.parse_args()

def main():
    args = parse_arguments()

    bp_file = args.file
    xml_file = args.xml
    io_name = args.io_name
    var_names = [v.strip() for v in args.vars.split(',')]
    x_seeds, y_seeds = parse_seed_points(args.seeds_points)
    output_file = args.output
    reader = ReaderClass.Reader(bp_file, io_name, xml=xml_file)
    vars = reader.reader(var_names)


    adios_obj = adios2.Adios()
    write_io = adios_obj.declare_io("WriteStreamlineIO")
    
    print("Making streamlines Now")
    
    not_defined = True

    for i, v in enumerate(vars):
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[0] == 1:
            vars[i] = v[0]
    num_steps = int(len(vars) / len(var_names))
    wrighter = adios2.Stream(write_io, output_file, "w")

    
    for step in range(num_steps):
        print(f"Wrighting step: {step}")
        vx = vars[step * len(var_names) + 0]
        vy = vars[step * len(var_names) + 1]

        coords_x, coords_y, offsets = rk4_streamline_from_grid(x_seeds, y_seeds, vx, vy, max_len=1000)
        coords_x = np.ascontiguousarray(np.array(coords_x, dtype=np.float64))
        coords_y = np.ascontiguousarray(np.array(coords_y, dtype=np.float64))
        offsets  = np.ascontiguousarray(np.array(offsets, dtype=np.int32)) 
   
        
        coords_x = coords_x.flatten()
        coords_y = coords_y.flatten()
        offsets = offsets.flatten()

        shape = [coords_x.size]
        OShape = [offsets.size]
        if not_defined:
            # how to fix ?????????????????? when more than one thing
            var_coords_x = write_io.define_variable('coords_x', coords_x, shape, [0], shape)
            var_coords_y = write_io.define_variable('coords_y', coords_y, shape, [0], shape)
            var_offsets = write_io.define_variable('offsets', offsets , OShape, [0], OShape)
            not_defined = False
        wrighter.begin_step()
        wrighter.write('coords_x', coords_x)
        wrighter.write('coords_y', coords_y)
        wrighter.write('offsets', offsets)
        wrighter.end_step()
    
    wrighter.close()
    
    print(f"All streamline segments saved to {output_file}!")

if __name__ == "__main__":
    install()
    main()
