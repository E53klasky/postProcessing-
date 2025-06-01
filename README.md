# postProcessing

This repository contains post-processing tools for CFD simulation data from xcompact3d. The tools process ADIOS2 BP5 files for divergence/curl calculations, streamline visualization, and data compression.

## Setup

### System Dependencies

You'll need to install the following system-level dependencies:

1. **MPI** - Required for parallel execution

   - [MPICH Installation Guide](https://www.mpich.org/downloads/)
   - [OpenMPI Installation Guide](https://www.open-mpi.org/software/ompi/)

2. **ADIOS2** - For reading and writing .bp5 simulation files

   - [ADIOS2 Installation Guide](https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html)
   - **Important**: Make sure ADIOS2 is compiled with Python bindings enabled

3. **MGARD** - For compression functionality
   - [MGARD Installation Guide](https://github.com/CODARcode/MGARD)
   - Required for compression operations in `compression.py`

### Python Dependencies

Install the required Python packages:

```bash
pip install numpy matplotlib mpi4py
```

For a clean setup with virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy matplotlib mpi4py
```

If you're using ADIOS2 with Python, make sure the Python bindings are properly installed and accessible.

## Scripts Overview

### divCurl.py

Calculate divergence and curl from velocity field data. Supports both 2D and 3D velocity fields.

**Current Status**: ⚠️ Parallel processing implementation is in development

### streamlines.py

Generate streamline visualizations from velocity field data.

**Current Status**:

- ✅ Works for 2D data
- ⚠️ 3D functionality is in development (known bugs)
- ⚠️ Parallel processing implementation is in development

### compression.py

Compress ADIOS2 BP5 files using MGARD compression.

**Current Status**:

- ✅ Works with XML configuration files
- ⚠️ Standalone operation (without XML) is in development
- ⚠️ Parallel processing implementation is in development

## How to Run

### divCurl.py

Calculate divergence and curl from velocity field data:

```bash
# Basic usage
mpirun -np 4 python3 divCurl.py input_file.bp5 50

# With custom output file
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --output results.bp

# With XML configuration file
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --xml config.xml

# With both optional arguments
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --xml config.xml --output my_results.bp
```

**Arguments:**

- `input_file` (required): Path to input ADIOS2 BP5 file
- `max_steps` (required): Maximum number of time steps to process
- `--xml, -x` (optional): Path to ADIOS2 XML configuration file
- `--output, -o` (optional): Output file name (default: `div_curl.bp`)

**Examples:**

```bash
mpirun -np 8 python3 divCurl.py ./data/cavity3D.bp5 25
mpirun -np 4 python3 divCurl.py /path/to/simulation.bp5 100 -x adios2_config.xml -o divergence_curl.bp
python3 divCurl.py cavity2D.bp5 10 --output test_output.bp
```

### streamlines.py

Generate streamline visualizations from 2D velocity field data:

```bash
# Basic usage (2D mode)
python3 streamlines.py input_file.bp5 20

# With XML configuration
python3 streamlines.py input_file.bp5 20 --xml config.xml

# 3D mode (experimental - has known bugs)
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

**Examples:**

```bash
python3 streamlines.py cavity2D.bp5 10
python3 streamlines.py ./results/cavity2D.bp5 50
python3 streamlines.py /home/user/sims/cavity2D_256x256.bp5 25
python3 streamlines.py test_data.bp5 3 --xml adios2_config.xml
```

### compression.py

Compress ADIOS2 BP5 files using MGARD compression:

```bash
# With XML configuration (recommended)
python3 compression.py input_file.bp5 50 --xml config.xml --output compressed.bp

# With error bound (requires XML currently)
python3 compression.py input_file.bp5 50 --xml config.xml --errorBound 0.001 --output compressed.bp
```

**Arguments:**

- `path` (required): Path to BP5 file to process
- `max_steps` (required): Maximum number of time steps to process
- `--xml, -x` (optional): Path to ADIOS2 XML configuration file
- `--errorBound, -eb` (optional): Error bound for compression (default: 0 - no compression)
- `--output, -o` (optional): Output file name (default: `compressed.bp`)

**Examples:**

```bash
python3 compression.py cavity3D.bp5 25 --xml adios2_config.xml
python3 compression.py simulation.bp5 100 --xml config.xml --errorBound 0.01 --output compressed_data.bp
```

## Output Files

### divCurl.py Output

Creates a new BP5 file (default: `div_curl.bp`) containing:

- `div`: Divergence field
- `curl_x`, `curl_y`, `curl_z`: Curl components (3D) or `curl_z` only (2D)

### streamlines.py Output

Generates PNG images for each time step:

- **2D Mode**: `[base_name]_2d_streamlines_step####.png`
- **3D Mode**: `[base_name]_3d_[var1][var2]_slice[N]_step####.png`

Example: `cavity2D_2d_streamlines_step0001.png`

All images use a consistent color scale based on global velocity magnitude.

### compression.py Output

Creates a compressed BP5 file with MGARD compression applied to all variables.

## Known Issues & Development Status

- **Parallel Processing**: Currently being developed for all scripts
- **3D Streamlines**: Known bugs in 3D mode, under active development
- **Compression**: Standalone operation (without XML) is being implemented
- **Error Handling**: Improved error handling and validation being added

## ADIOS2 XML Configuration

If you're using ADIOS2 XML configuration files, make sure to configure them appropriately with:

- Variable definitions
- Transport methods
- Compression settings (for compression.py)

## Troubleshooting

1. **"Module not found" errors**: Ensure all dependencies are installed and ADIOS2 Python bindings are accessible
2. **MPI errors**: Verify your MPI installation and that mpi4py is compatible with your MPI version also make you installed ADIOS2 with MPI ON and you updated your PYTHONPATH
3. **File access errors**: Check file paths and permissions for input/output files

## Contributing

This project is under active development. Known issues are being addressed in the following priority order:

1. Parallel processing implementation
2. 3D streamline visualization fixes
3. Standalone compression functionality
4. Enhanced error handling and validation
