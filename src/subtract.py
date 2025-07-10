import numpy as np
import argparse
from adios2 import bindings
from rich.traceback import install
from ReaderClass import Reader
from WrighterClass import Writer
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator


# this will not scale 
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Subtract variables from two ADIOS2 files and write the difference."
    )
    parser.add_argument(
        "--bpfile1", required=True, help="Lower-resolution input BP file"
    )
    parser.add_argument(
        "--bpfile2", required=True, help="Higher-resolution input BP file"
    )
    parser.add_argument(
        "--Declare_Read_Io1", required=True, help="IO name for lower-resolution input"
    )
    parser.add_argument(
        "--Declare_Read_Io2", required=True, help="IO name for higher-resolution input"
    )
    parser.add_argument(
        "--Declare_Write_IO", required=True, help="IO name for writing output"
    )
    parser.add_argument("--var", required=True, help="Variable name to subtract")
    parser.add_argument(
        "--output_file", "-o", default="subtract.bp", help="Output BP file"
    )
    parser.add_argument("--xml", default=None, help="Optional ADIOS2 XML config")
    parser.add_argument(
        "--tolerance", type=float, default=None, help="Tolerance for subtraction"
    )
    return parser.parse_args()


def get_global_array_info(local_data, comm):

    local_shape = np.array(local_data.shape)

    all_shapes = comm.allgather(local_shape)
    
    global_shape = local_shape.copy()
    global_shape[-1] = sum(shape[-1] for shape in all_shapes)
    
    rank = comm.Get_rank()
    local_offset = np.zeros_like(local_shape)
    local_offset[-1] = sum(all_shapes[i][-1] for i in range(rank))
    
    return global_shape, local_offset, local_shape


def create_local_interpolator(global_low_shape, global_high_shape, local_high_offset, local_high_shape):

    if len(global_low_shape) == 2:
        low_y = np.linspace(0, 1, global_low_shape[0])
        low_x = np.linspace(0, 1, global_low_shape[1])
        
        high_y = np.linspace(0, 1, global_high_shape[0])
        high_x = np.linspace(0, 1, global_high_shape[1])
        
        local_high_y = high_y[local_high_offset[0]:local_high_offset[0] + local_high_shape[0]]
        local_high_x = high_x[local_high_offset[1]:local_high_offset[1] + local_high_shape[1]]
        
        xv, yv = np.meshgrid(local_high_x, local_high_y)
        points = np.stack((yv.ravel(), xv.ravel()), axis=-1)
        
        return points, (low_y, low_x), local_high_shape
        
    elif len(global_low_shape) == 3:
        low_y = np.linspace(0, 1, global_low_shape[0])
        low_x = np.linspace(0, 1, global_low_shape[1])
        low_z = np.linspace(0, 1, global_low_shape[2])
        
        high_y = np.linspace(0, 1, global_high_shape[0])
        high_x = np.linspace(0, 1, global_high_shape[1])
        high_z = np.linspace(0, 1, global_high_shape[2])
        
        local_high_y = high_y[local_high_offset[0]:local_high_offset[0] + local_high_shape[0]]
        local_high_x = high_x[local_high_offset[1]:local_high_offset[1] + local_high_shape[1]]
        local_high_z = high_z[local_high_offset[2]:local_high_offset[2] + local_high_shape[2]]
        
        xv, yv, zv = np.meshgrid(local_high_x, local_high_y, local_high_z, indexing='ij')
        points = np.stack((yv.ravel(), xv.ravel(), zv.ravel()), axis=-1)
        
        return points, (low_y, low_x, low_z), local_high_shape
    
    else:
        raise ValueError(f"Unsupported dimensionality: {len(global_low_shape)}")


def parallel_upscale_and_subtract(local_low_res, local_high_res, tolerance, comm):

    rank = comm.Get_rank()
    
    global_low_shape, local_low_offset, local_low_shape = get_global_array_info(local_low_res, comm)
    global_high_shape, local_high_offset, local_high_shape = get_global_array_info(local_high_res, comm)
    
    if rank == 0:
        print(f"Global low shape: {global_low_shape}, Global high shape: {global_high_shape}")
    
    # assuming small data set is small enough to fit in memory for each process 
    all_low_data = comm.allgather(local_low_res)

    if len(global_low_shape) == 2:
        full_low_res = np.concatenate(all_low_data, axis=-1)
    elif len(global_low_shape) == 3:
        full_low_res = np.concatenate(all_low_data, axis=-1)
    else:
        raise ValueError(f"Unsupported dimensionality: {len(global_low_shape)}")
    
    if len(global_low_shape) == 2:
        low_y = np.linspace(0, 1, global_low_shape[0])
        low_x = np.linspace(0, 1, global_low_shape[1])
        interp = RegularGridInterpolator((low_y, low_x), full_low_res, bounds_error=False, fill_value=0)
    elif len(global_low_shape) == 3:
        low_y = np.linspace(0, 1, global_low_shape[0])
        low_x = np.linspace(0, 1, global_low_shape[1])
        low_z = np.linspace(0, 1, global_low_shape[2])
        interp = RegularGridInterpolator((low_y, low_x, low_z), full_low_res, bounds_error=False, fill_value=0)
    
    points, _, target_shape = create_local_interpolator(
        global_low_shape, global_high_shape, local_high_offset, local_high_shape
    )

    local_upsampled = interp(points).reshape(target_shape)

    local_diff = np.abs(local_high_res - local_upsampled)
    
    if tolerance is not None:
        local_diff = np.where(local_diff <= tolerance, 0.0, local_diff)
    
    return local_diff


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = parse_arguments()

    r_low = Reader(args.Declare_Read_Io1, args.bpfile1, comm=comm)
    r_high = Reader(args.Declare_Read_Io2, args.bpfile2, comm=comm)
    w = Writer(args.Declare_Write_IO, args.output_file, comm=comm)

    var = args.var
    tolerance = args.tolerance

    while True:
        status_low = r_low.begin_step()
        status_high = r_high.begin_step()
        
        if (bindings.StepStatus.OK != status_low or 
            bindings.StepStatus.OK != status_high):
            break

        current_step = r_low.current_step()
        if rank == 0:
            print(f"Processing step {int(current_step)}")

        w.begin_step()
        r_low.set_read_vars([var])
        r_high.set_read_vars([var])

        low_res = r_low.read_step(var)
        ground_truth = r_high.read_step(var)

        if rank == 0:
            print(f"Rank {rank}: Low res shape: {low_res.shape}, Ground truth shape: {ground_truth.shape}")

        if len(low_res.shape) == 3 and low_res.shape[0] == 1:
            low_res = low_res[0, :, :]
            ground_truth = ground_truth[0, :, :]

        try:
            diff = parallel_upscale_and_subtract(low_res, ground_truth, tolerance, comm)
        except Exception as e:
            if rank == 0:
                print(f"Error in parallel processing: {e}")
            break

        w.write(f"diff_{var}", diff)
        w.end_step()
        
        r_low.end_step()
        r_high.end_step()
        

    r_low.close()
    r_high.close()
    w.close()

    if rank == 0:
        print(f"Output written to {args.output_file}")


if __name__ == "__main__":
    install()
    main()