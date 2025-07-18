import numpy as np
import argparse
from adios2 import bindings
from rich.traceback import install
from ReaderClass import Reader
from WrighterClass import Writer
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
import math


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Resample data to nearest power of two dimensions for decimation preprocessing."
    )
    parser.add_argument(
        "--input_file", 
        "-in",
        help="Input BP file to resample", 
        required=True
    )
    parser.add_argument(
        "--input_io", 
        help="IO name for input file", 
        required=False,
        default="reaio"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="resampled.bp",
        help="Output BP file for resampled data",
    )
    parser.add_argument(
        "--output_io", 
        help="IO name for output file", 
        required=False,
        default="wio"
    )
    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=True,
        help="Variable names to resample (order matters), separated by commas, e.g., temperature,pressure,velocity (REQUIRED)",
    )
    parser.add_argument(
        "--xml", 
        default=None, 
        help="Optional ADIOS2 XML configuration (default: None)"
    )
    parser.add_argument(
        "--method",
        choices=['nearest', 'upsample', 'downsample'],
        default='nearest',
        help="Resampling method: 'nearest' (default), 'upsample', or 'downsample'"
    )
    
    return parser.parse_args()


def find_nearest_power_of_two_forced(n, method):
    """
    Find the nearest power of two for a given number, forcing change for upsample/downsample.
    
    Args:
        n: Input number
        method: 'upsample' or 'downsample'
    
    Returns:
        Power of two (forced to be different from input if already power of 2)
    """
    if n <= 0:
        return 1
    
    # Check if n is already a power of 2
    is_power_of_2 = (n & (n - 1)) == 0
    
    log_n = math.log2(n)
    
    if method == 'upsample':
        if is_power_of_2:
            # If already power of 2, go to next higher power
            power = int(log_n) + 1
        else:
            # Round up to next power of 2
            power = math.ceil(log_n)
    elif method == 'downsample':
        if is_power_of_2:
            # If already power of 2, go to next lower power
            power = max(0, int(log_n) - 1)
        else:
            # Round down to previous power of 2
            power = math.floor(log_n)
    
    return 2 ** power


def find_nearest_power_of_two(n, method='nearest'):
    """
    Find the nearest power of two for a given number.
    
    Args:
        n: Input number
        method: 'nearest', 'upsample', or 'downsample'
    
    Returns:
        Nearest power of two
    """
    if n <= 0:
        return 1
    
    # Find the power of two
    log_n = math.log2(n)
    
    if method == 'nearest':
        # Round to nearest power of two
        power = round(log_n)
    elif method == 'upsample':
        # Always round up
        power = math.ceil(log_n)
    elif method == 'downsample':
        # Always round down
        power = math.floor(log_n)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result = 2 ** power
    
    return result


def resample_to_power_of_two(data, target_shape):
    """
    Resample data to target shape using interpolation.
    
    Args:
        data: Input array
        target_shape: Target shape (tuple)
    
    Returns:
        Resampled array
    """
    if data.shape == target_shape:
        return data
    
    # Create coordinate arrays for the original data
    original_shape = data.shape
    original_axes = [np.linspace(0, 1, s) for s in original_shape]
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        original_axes, 
        data.astype(np.float64),
        method='cubic',
        bounds_error=False,
        fill_value=0
    )
    
    # Create coordinate arrays for the target shape
    target_axes = [np.linspace(0, 1, s) for s in target_shape]
    mesh = np.meshgrid(*target_axes, indexing='ij')
    points = np.stack([axis.ravel() for axis in mesh], axis=-1)
    
    # Interpolate to new shape
    resampled = interpolator(points).reshape(target_shape)
    
    return resampled.astype(data.dtype)


def calculate_target_shape(original_shape, method='nearest'):
    """
    Calculate target shape with power-of-two dimensions.
    Each dimension is resampled independently based on the method.
    
    Args:
        original_shape: Original data shape
        method: Resampling method
    
    Returns:
        Target shape tuple
    """
    target_shape = []
    for dim in original_shape:
        if method == 'upsample':
            # Always go to next higher power of 2
            target_dim = find_nearest_power_of_two_forced(dim, 'upsample')
        elif method == 'downsample':
            # Always go to next lower power of 2
            target_dim = find_nearest_power_of_two_forced(dim, 'downsample')
        else:  # nearest
            # Round to nearest power of 2
            target_dim = find_nearest_power_of_two(dim, 'nearest')
        target_shape.append(target_dim)
    
    return tuple(target_shape)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    args = parse_arguments()
    
    # Initialize readers and writers
    reader = Reader(args.input_io, args.input_file, args.xml, comm=comm)
    writer = Writer(args.output_io, args.output_file, args.xml, comm=comm)
    
    # Parse comma-separated variables
    variables = [var.strip() for var in args.vars.split(',')]
    step_count = 0
    
    if rank == 0:
        print(f"Processing variables: {variables}")
        print(f"Number of variables: {len(variables)}")
    
    try:
        while True:
            # Begin step for both reader and writer
            status = reader.begin_step()
            
            if status != bindings.StepStatus.OK:
                break
            
            current_step = reader.current_step()
            if rank == 0:
                print(f"Processing step: {int(current_step)}")
            
            # Read all variables
            if rank == 0:
                print(f"  Setting read variables: {variables}")
            reader.set_read_vars(variables)
            
            writer.begin_step()
            
            # Process each variable
            for var in variables:
                if rank == 0:
                    print(f"  Processing variable: '{var}'")
                
                try:
                    data = reader.read_step(var)
                    
                    if data is None:
                        if rank == 0:
                            print(f"    Warning: Variable '{var}' returned None, skipping...")
                        continue
                        
                except Exception as e:
                    if rank == 0:
                        print(f"    Error reading variable '{var}': {e}")
                        print(f"    Skipping variable '{var}'")
                    continue
                
                # Handle 3D data with singleton dimension
                if len(data.shape) == 3 and data.shape[0] == 1:
                    data = data[0, :, :]
                    if rank == 0:
                        print(f"    Squeezed 3D data to 2D: {data.shape}")
                
                original_shape = data.shape
                
                # Calculate target shape
                target_shape = calculate_target_shape(
                    original_shape, 
                    args.method
                )
                
                if rank == 0:
                    print(f"    Original shape: {original_shape}")
                    print(f"    Target shape: {target_shape}")
                
                # Resample data if needed
                if original_shape != target_shape:
                    if rank == 0:
                        print(f"    Resampling from {original_shape} to {target_shape}")
                    
                    resampled_data = resample_to_power_of_two(data, target_shape)
                    
                    # Calculate compression ratio
                    original_size = np.prod(original_shape)
                    target_size = np.prod(target_shape)
                    ratio = target_size / original_size
                    
                    if rank == 0:
                        print(f"    Resampling ratio: {ratio:.4f}")
                        print(f"    Data range: [{np.min(resampled_data):.6f}, {np.max(resampled_data):.6f}]")
                else:
                    resampled_data = data
                    if rank == 0:
                        print("    No resampling needed - data already has power-of-two dimensions")
                
                # Write the resampled data
                writer.write(var, resampled_data)
            
            writer.end_step()
            reader.end_step()
            
            step_count += 1
            
    except Exception as e:
        if rank == 0:
            print(f"Error during processing: {e}")
        raise
    
    finally:
        # Clean up
        reader.close()
        writer.close()
    
    if rank == 0:
        print(f"\nResampling completed successfully!")
        print(f"Processed {step_count} steps")
        print(f"Variables processed: {variables}")
        print(f"Output written to: {args.output_file}")
        print(f"Resampling method: {args.method}")
        print(f"Data is now ready for decimation processing")


if __name__ == "__main__":
    install()
    main()