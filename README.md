# postProcessing

This repository contains post-processing tools for CFD simulation data from xcompact3d. The tools process ADIOS2 BP5 files for divergence/curl calculations, streamline visualization, contour plotting, data compression, and RMSE analysis.

## Table of Contents

- [Setup](#setup)
- [Scripts Overview](#scripts-overview)
- [Usage Examples](#usage-examples)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Known Issues](#known-issues)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Setup

### System Dependencies

You'll need to install the following system-level dependencies:

1. **MPI** - Required for parallel execution

   - [MPICH Installation Guide](https://www.mpich.org/downloads/)
   - [OpenMPI Installation Guide](https://www.open-mpi.org/software/ompi/)

2. **ADIOS2** - For reading and writing .bp5 simulation files

   - [ADIOS2 Installation Guide](https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html)
   - **Important**: Make sure ADIOS2 is compiled with Python bindings and MPI support enabled
   - Update your `PYTHONPATH` to include ADIOS2 Python bindings

3. **MGARD** - For compression functionality
   - [MGARD Installation Guide](https://github.com/CODARcode/MGARD)
   - Required for compression operations in `compression.py`

### Python Dependencies

Install the required Python packages:

```bash
pip install numpy matplotlib numpy
```

For a clean setup with virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy matplotlib mpi4py adios2
```

## Scripts Overview

### divCurl.py

Calculate divergence and curl from velocity field data. Supports both 2D and 3D velocity fields.

**Status**: ⚠️ Parallel processing implementation is in development

### streamlines.py

Generate streamline visualizations from velocity field data.

**Status**:

- ✅ Works for 2D and 3D data
- ⚠️ Parallel processing implementation is in development

### contour.py

Generate contour plots for specified variables from simulation data.

**Status**:

- ✅ Works for both 2D and 3D modes
- ✅ Supports multiple variables
- ⚠️ Parallel processing implementation is in development

### compression.py

Compress ADIOS2 BP5 files using MGARD compression.

**Status**:

- ✅ Works with XML configuration files
- ⚠️ Standalone operation (without XML) is in development

### RMSE.py

Calculate Root Mean Square Error between two datasets (e.g., high-res vs low-res simulations).

**Status**:

- ✅ Functional for comparing simulation results
- ⚠️ Parallel processing implementation is in development

## Usage Examples

### divCurl.py - Divergence and Curl Calculation

```bash
# Basic usage
mpirun -np 4 python3 divCurl.py input_file.bp5 50

# With custom output file
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --output results.bp

# With XML configuration file
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --xml config.xml

# Complete example with all options
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --xml config.xml --output my_results.bp
```

**Arguments:**

- `input_file` (required): Path to input ADIOS2 BP5 file
- `max_steps` (required): Maximum number of time steps to process
- `--xml, -x` (optional): Path to ADIOS2 XML configuration file
- `--output, -o` (optional): Output file name (default: `div_curl.bp`)

### streamlines.py - Streamline Visualization

```bash
# Basic 2D usage
python3 streamlines.py input_file.bp5 20

# With XML configuration
python3 streamlines.py input_file.bp5 20 --xml config.xml

python3 streamlines.py input_file.bp5 20 --mode 3d --var1 ux --var2 uy --slice 16
```

**Arguments:**

- `path` (required): Path to BP5 file to process
- `max_steps` (required): Maximum number of time steps to process
- `--xml, -x` (optional): ADIOS2 XML config file
- `--mode, -m` (optional): Processing mode - `2d` (default) or `3d` (experimental)
- `--var1` (optional): First velocity component for 3D mode (default: `ux`)
- `--var2` (optional): Second velocity component for 3D mode (default: `uy`)
- `--slice, -s` (optional): Slice index for 3D mode (default: 16)

### contour.py - Contour Plotting

```bash
# 2D contour plots for multiple variables
python3 contour.py input_file.bp5 --vars ux,uy,pp --max_steps 10 --mode 2d

# 3D contour plots with specific slice
python3 contour.py input_file.bp5 --vars ux,uy --max_steps 10 --mode 3d --slice 20

# With XML configuration
python3 contour.py input_file.bp5 --vars ux,uy,pp --max_steps 10 --xml config.xml
```

**Arguments:**

- `input_file` (required): Path to input ADIOS2 BP5 file
- `--vars, -v` (required): Variables to plot, separated by commas
- `--max_steps, -n` (required): Maximum number of timesteps to process
- `--xml, -x` (optional): Path to ADIOS2 XML configuration file
- `--mode, -m` (optional): Processing mode - `2d` (default) or `3d`
- `--slice, -s` (optional): Slice index for 3D mode (default: 16)

### compression.py - Data Compression

```bash
# With XML configuration (recommended)
python3 compression.py input_file.bp5 50 --xml config.xml --output compressed.bp

# With specific error bound
python3 compression.py input_file.bp5 50 --xml config.xml --errorBound 0.001 --output compressed.bp
```

**Arguments:**

- `path` (required): Path to BP5 file to process
- `max_steps` (required): Maximum number of time steps to process
- `--xml, -x` (optional): Path to ADIOS2 XML configuration file
- `--errorBound, -eb` (optional): Error bound for compression (default: 0 - no compression)
- `--output, -o` (optional): Output file name (default: `compressed.bp`)

### RMSE.py - Root Mean Square Error Analysis

```bash
# Compare high-res and low-res simulations
python3 RMSE.py --highres ground_truth.bp5 --lowres compressed.bp5 --var ux --max_steps 10

# Compare different variables
python3 RMSE.py --highres reference.bp5 --lowres test.bp5 --var pp --max_steps 25
```

**Arguments:**

- `--highres` (required): Path to the ground truth (high resolution) ADIOS2 file
- `--lowres` (required): Path to the lower resolution ADIOS2 file
- `--var` (required): Variable name to compare
- `--max_steps` (required): Maximum number of steps to process

## Output Files

### divCurl.py Output

Creates a new BP5 file (default: `div_curl.bp`) containing:

- `div`: Divergence field
- `curl_x`, `curl_y`, `curl_z`: Curl components (3D) or `curl_z` only (2D)

### streamlines.py Output

Generates PNG images in `../RESULTS/` directory:

- **2D Mode**: `[base_name]_2d_streamlines_step####.png`
- **3D Mode**: `[base_name]_3d_[var1][var2]_slice[N]_step####.png`

All images use a consistent color scale based on global velocity magnitude.

### contour.py Output

Generates PNG images in `../RESULTS/` directory:

- **2D Mode**: `[variable]_step_[N].png`
- **3D Mode**: `[variable]_3d_slice_[mode]_idx[N].png`

### compression.py Output

Creates a compressed BP5 file with MGARD compression applied to all variables.

### RMSE.py Output

Prints RMSE values to console for each time step processed.

## Configuration

### ADIOS2 XML Configuration

If using ADIOS2 XML configuration files, ensure they include:

- Variable definitions
- Transport methods
- Compression settings (for compression.py)
- MPI settings for parallel processing

Example XML structure:

```xml
<?xml version="1.0"?>
<adios-config>
    <io name="ReadIO">
        <transport type="File"/>
    </io>
    <io name="WriteIO">
        <transport type="File"/>
    </io>
</adios-config>
```

## Known Issues & Development Status

- **Parallel Processing**: Currently being developed for all scripts
- **Compression**: Standalone operation (without XML) is being implemented
- **Error Handling**: Improved error handling and validation being added
- **MPI Integration**: Some scripts need better MPI error handling

## Troubleshooting

### Common Issues

1. **"Module not found" errors**

   - Ensure all dependencies are installed
   - Verify ADIOS2 Python bindings are accessible
   - Check your `PYTHONPATH` includes ADIOS2 installation

2. **MPI errors**

   - Verify MPI installation
   - Ensure mpi4py is compatible with your MPI version
   - Make sure ADIOS2 was compiled with MPI support (`-DADIOS2_USE_MPI=ON`)

3. **File access errors**

   - Check file paths and permissions
   - Ensure input files exist and are readable
   - Verify output directories are writable

4. **Memory issues**

   - Reduce the number of time steps (`max_steps`)
   - Use more MPI processes to distribute memory load
   - Check available system memory

5. **Compression errors**
   - Ensure MGARD is properly installed
   - Use XML configuration files for compression
   - Check error bound values are reasonable

### Performance Tips

- Use appropriate number of MPI processes for your system
- Process data in chunks if memory is limited
- Use XML configuration files for better performance
- Monitor system resources during processing

## Contributing

This project is under active development. Current development priorities:

1. **High Priority**:

   - Parallel processing implementation across all scripts
   - 3D streamline visualization fixes
   - Standalone compression functionality

2. **Medium Priority**:

   - Enhanced error handling and validation
   - Performance optimizations
   - Better documentation and examples

3. **Low Priority**:
   - Additional visualization options
   - More compression algorithms
   - GUI interface

### Development Guidelines

- Follow existing code style and structure
- Add appropriate error handling
- Include documentation for new features
- Test with various data sizes and MPI configurations
- Update README when adding new functionality

For bug reports or feature requests, please create an issue with detailed information about your system configuration and the specific problem encountered.
