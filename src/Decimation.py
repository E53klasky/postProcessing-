import numpy as np
import matplotlib.pyplot as plt
from WrighterClass import Writer
from ReaderClass import Reader
from rich.traceback import install
import adios2
import argparse


def next_power_of_2(x):
    return 1 << (x - 1).bit_length()

def pad_to_power_of_2(data, min_size=8):
    n, m = data.shape
    new_n = max(next_power_of_2(n), min_size)
    new_m = max(next_power_of_2(m), min_size)
    
    if new_n == n and new_m == m:
        return data, (n, m)
    
    # Pad with edge values to maintain smoothness
    padded = np.pad(data, ((0, new_n - n), (0, new_m - m)), mode='edge')
    return padded, (n, m)

def build_progressive_array(data, min_size=8):
    assert len(data.shape) == 2, "Input must be 2D"
    
    # Pad to power of 2 if needed
    padded_data, original_shape = pad_to_power_of_2(data, min_size)
    n, m = padded_data.shape
    
    # Generate size sequences
    sizes_n = []
    sizes_m = []
    sn, sm = n, m
    
    while sn >= min_size:
        sizes_n.append(sn)
        sn //= 2
    while sm >= min_size:
        sizes_m.append(sm)
        sm //= 2
    
    # Reverse to go from coarse to fine
    sizes_n = sizes_n[::-1]
    sizes_m = sizes_m[::-1]
    
    # Make sure both sequences have the same length
    min_levels = min(len(sizes_n), len(sizes_m))
    sizes_n = sizes_n[:min_levels]
    sizes_m = sizes_m[:min_levels]
    
    output_chunks = []
    
    for i in range(len(sizes_n)):
        size_n = sizes_n[i]
        size_m = sizes_m[i]
        
        if i == 0:
            # First (coarsest) level - just downsample
            step_n = n // size_n
            step_m = m // size_m
            down = data[::step_n, ::step_m]
            output_chunks.append(down.flatten())
        else:
            # Subsequent levels - add new points
            step_n = n // size_n
            step_m = m // size_m
            full = data[::step_n, ::step_m]
            
            # Extract new points compared to previous level
            new_points = []
            # Odd rows, even cols
            new_points.append(full[1::2, ::2].flatten())
            # Even rows, odd cols  
            new_points.append(full[::2, 1::2].flatten())
            # Odd rows, odd cols
            new_points.append(full[1::2, 1::2].flatten())
            
            output_chunks.append(np.concatenate(new_points))
    
    # Return the progressive array along with metadata
    return np.concatenate(output_chunks), original_shape, (n, m)


def extract_level(progressive_array, target_size, min_size=8, full_resolution_shape=(128, 128), original_shape=None):
    n, m = full_resolution_shape
    target_n, target_m = target_size
    
    # Use original_shape if provided, otherwise use full_resolution_shape
    if original_shape is None:
        original_shape = full_resolution_shape
    
    assert (n & (n - 1)) == 0 and (m & (m - 1)) == 0, "Full resolution must be power of two in both dimensions"
    assert (target_n & (target_n - 1)) == 0 and (target_m & (target_m - 1)) == 0, "Target size must be power of two in both dimensions"
    assert target_n <= n and target_m <= m, "Target size must be ≤ full resolution"
    
    # Generate size sequences
    sizes_n = []
    sizes_m = []
    sn, sm = n, m
    
    while sn >= min_size:
        sizes_n.append(sn)
        sn //= 2
    while sm >= min_size:
        sizes_m.append(sm)
        sm //= 2
    
    sizes_n = sizes_n[::-1]  # From coarse to fine
    sizes_m = sizes_m[::-1]
    
    # Make sure both sequences have the same length
    min_levels = min(len(sizes_n), len(sizes_m))
    sizes_n = sizes_n[:min_levels]
    sizes_m = sizes_m[:min_levels]
    
    # Find target level
    target_level = None
    for i in range(len(sizes_n)):
        if sizes_n[i] == target_n and sizes_m[i] == target_m:
            target_level = i
            break
    
    if target_level is None:
        raise ValueError(f"Target size {target_size} not found in progressive array")
    
    # Calculate chunk sizes and positions
    chunks = []
    for i in range(len(sizes_n)):
        if i == 0:
            chunks.append(sizes_n[i] * sizes_m[i])
        else:
            prev_n, prev_m = sizes_n[i-1], sizes_m[i-1]
            curr_n, curr_m = sizes_n[i], sizes_m[i]
            added = curr_n * curr_m - prev_n * prev_m
            chunks.append(added)
    
    # Extract data for target level
    start = sum(chunks[:target_level])
    end = start + chunks[target_level]
    flat_data = progressive_array[start:end]
    
    if target_level == 0:
        # First level - just reshape
        return flat_data.reshape((target_n, target_m))
    
    # Reconstruct from previous level
    prev_n, prev_m = sizes_n[target_level - 1], sizes_m[target_level - 1]
    prev_data = extract_level(progressive_array, (prev_n, prev_m), min_size, full_resolution_shape, original_shape)
    
    # Initialize output array
    out = np.zeros((target_n, target_m))
    out[::2, ::2] = prev_data
    
    # Fill in new values
    half_n = target_n // 2
    half_m = target_m // 2
    
    i = 0
    # Odd rows, even cols
    out[1::2, ::2] = flat_data[i:i + half_n * half_m].reshape((half_n, half_m))
    i += half_n * half_m
    
    # Even rows, odd cols
    out[::2, 1::2] = flat_data[i:i + half_n * half_m].reshape((half_n, half_m))
    i += half_n * half_m
    
    # Odd rows, odd cols
    out[1::2, 1::2] = flat_data[i:i + half_n * half_m].reshape((half_n, half_m))
    
    result = out
    
    # Crop to original size if needed
    if original_shape is not None:
        orig_n, orig_m = original_shape
        if target_n == full_resolution_shape[0] and target_m == full_resolution_shape[1]:
            # Only crop at the finest level
            result = result[:orig_n, :orig_m]
    
    return result


def calculate_errors(original, reconstructed):
    """Calculate L1, L2, and L∞ errors"""
    diff = np.abs(original - reconstructed)
    
    l1_error = np.sum(diff)
    l2_error = np.sqrt(np.sum(diff**2))
    linf_error = np.max(diff)
    
    return l1_error, l2_error, linf_error


def find_best_level_for_error_bound(progressive_array, original_shape, padded_shape, error_bound, min_size=8):
  
    n, m = padded_shape
    
    # Generate possible sizes
    sizes_n = []
    sizes_m = []
    sn, sm = n, m
    
    while sn >= min_size:
        sizes_n.append(sn)
        sn //= 2
    while sm >= min_size:
        sizes_m.append(sm)
        sm //= 2
    
    sizes_n = sizes_n[::-1]
    sizes_m = sizes_m[::-1]
    min_levels = min(len(sizes_n), len(sizes_m))
    sizes_n = sizes_n[:min_levels]
    sizes_m = sizes_m[:min_levels]
    
    # Get original data (cropped to original size)
    original = extract_level(progressive_array, padded_shape, min_size, padded_shape, original_shape)
    
    best_level = None
    best_size = None
    best_errors = None
    
    # Test each level from coarsest to finest
    for i in range(len(sizes_n)):
        target_size = (sizes_n[i], sizes_m[i])
        reconstructed = extract_level(progressive_array, target_size, min_size, padded_shape, original_shape)
        
        # Upsample to original size for comparison if needed
        if reconstructed.shape != original.shape:
            # Simple upsampling using nearest neighbor
            scale_n = original.shape[0] / reconstructed.shape[0]
            scale_m = original.shape[1] / reconstructed.shape[1]
            
            # Create coordinate grids for upsampling
            old_n, old_m = reconstructed.shape
            new_n, new_m = original.shape
            
            # Use nearest neighbor upsampling
            row_indices = np.round(np.linspace(0, old_n-1, new_n)).astype(int)
            col_indices = np.round(np.linspace(0, old_m-1, new_m)).astype(int)
            
            reconstructed_upsampled = reconstructed[np.ix_(row_indices, col_indices)]
        else:
            reconstructed_upsampled = reconstructed
        
        l1, l2, linf = calculate_errors(original, reconstructed_upsampled)
        
        # Check if this level meets the error bound (using L2 error)
        if l2 <= error_bound:
            best_level = i
            best_size = target_size
            best_errors = (l1, l2, linf)
            break
    
    return best_level, best_size, best_errors


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Decimates a variable to be smaller by some error bound."
    )
    parser.add_argument(
        "--file1",
        type=str,
        required=True,
        help="First Adios file with data",
    )
    parser.add_argument(
        "--IO_Name1",
        type=str,
        default="reader1",
        help="IO Name for the first Adios file (default: reader1)",
    )
    parser.add_argument(
        "--var",
        "-v",
        type=str,
        required=True,
        help="Variable to decimate",
    )
    parser.add_argument(
        "--error_bound", 
        "-eb", 
        type=float,
        required=True, 
        help="Error tolerance level"
    )
    parser.add_argument(
        "--xml", 
        type=str, 
        default=None, 
        help="Optional ADIOS2 XML configuration"
    )
    parser.add_argument(
        "--Declare_Write_IO", 
        help="IO name for writing output", 
        required=True
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="Decimated.bp",
        help="Output BP file for the result",
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    var = args.var
    error_bound = args.error_bound
    
    r = Reader(args.IO_Name1, args.file1, args.xml)
    w = Writer(args.Declare_Write_IO, args.output_file, args.xml)
    
    while True:
        status = r.begin_step()
        
        if status != adios2.bindings.StepStatus.OK:
            break
        
        current_step = r.current_step()
        print(f"Reading step: {int(current_step)}")
        
        r.set_read_vars([var])
        data = r.read_step(var)
        
        # Ensure data is 2D
        if len(data.shape) == 3:
            # Handle 3D data by processing each slice
            print(f"Processing 3D data with shape: {data.shape}")
            decimated_slices = []
            all_l1_errors = []
            all_l2_errors = []
            all_linf_errors = []
            
            for slice_idx in range(data.shape[2]):
                slice_data = data[:, :, slice_idx]
                
                # Skip if slice is too small
                if slice_data.shape[0] < 8 or slice_data.shape[1] < 8:
                    print(f"Slice {slice_idx}: Too small, using original")
                    decimated_slices.append(slice_data)
                    all_l1_errors.append(0.0)
                    all_l2_errors.append(0.0)
                    all_linf_errors.append(0.0)
                    continue
                
                # Build progressive array for this slice
                progressive, orig_shape, padded_shape = build_progressive_array(slice_data)
                
                # Find best level for error bound
                best_level, best_size, best_errors = find_best_level_for_error_bound(
                    progressive, orig_shape, padded_shape, error_bound
                )
                
                if best_level is not None:
                    # Extract the decimated data
                    decimated = extract_level(progressive, best_size, 8, padded_shape, orig_shape)
                    decimated_slices.append(decimated)
                    
                    l1, l2, linf = best_errors
                    all_l1_errors.append(l1)
                    all_l2_errors.append(l2)
                    all_linf_errors.append(linf)
                    
                    print(f"Slice {slice_idx}: Best size {best_size}, L1: {l1:.6f}, L2: {l2:.6f}, L∞: {linf:.6f}")
                else:
                    print(f"Slice {slice_idx}: No level meets error bound, using original")
                    decimated_slices.append(slice_data)
                    all_l1_errors.append(0.0)
                    all_l2_errors.append(0.0)
                    all_linf_errors.append(0.0)
            
            # Stack the decimated slices back together
            output_data = np.stack(decimated_slices, axis=2)
            
            # Use average errors across slices
            l1 = np.mean(all_l1_errors)
            l2 = np.mean(all_l2_errors)
            linf = np.max(all_linf_errors)
            
        elif len(data.shape) == 2:
            # Handle 2D data
            print(f"Processing 2D data with shape: {data.shape}")
            
            # Check if data is too small
            if data.shape[0] < 8 or data.shape[1] < 8:
                print("Data too small for decimation, using original")
                output_data = data
                l1 = l2 = linf = 0.0
            else:
                # Build progressive array
                progressive, orig_shape, padded_shape = build_progressive_array(data)
                
                # Find best level for error bound
                best_level, best_size, best_errors = find_best_level_for_error_bound(
                    progressive, orig_shape, padded_shape, error_bound
                )
                
                if best_level is not None:
                    # Extract the decimated data
                    output_data = extract_level(progressive, best_size, 8, padded_shape, orig_shape)
                    l1, l2, linf = best_errors
                    print(f"Best size {best_size}, L1: {l1:.6f}, L2: {l2:.6f}, L∞: {linf:.6f}")
                else:
                    print("No level meets error bound, using original data")
                    output_data = data
                    l1 = l2 = linf = 0.0
        else:
            print(f"Unsupported data shape: {data.shape}")
            output_data = data
            l1 = l2 = linf = 0.0
        
        # Write the decimated data
        w.begin_step()
        w.write(var, output_data)
        w.write("l1_error", np.array([l1]))
        w.write("l2_error", np.array([l2]))
        w.write("linf_error", np.array([linf]))
        w.write("error_bound", np.array([error_bound]))
        w.end_step()
        
        r.end_step()
    
    r.close()
    w.close()
    print(f"Decimation completed. Output written to {args.output_file}")


if __name__ == "__main__":
    install()
    main()