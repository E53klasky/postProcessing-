# postProcessing

This is where I will be doing my post processing of the CFD simulation of xcompact3d

## Setup

If you are using it, add edits to adios2 xml config

## How to run

### div_curl

Calculate divergence and curl from velocity field data using MPI for parallel processing.

#### Basic Usage:

```bash
# Required arguments: input_file and num_steps
mpirun -np 4 python3 divCurl.py input_file.bp5 50

# With custom output file
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --output results.bp

# With XML configuration file
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --xml config.xml

# With both optional arguments
mpirun -np 4 python3 divCurl.py input_file.bp5 50 --xml config.xml --output my_results.bp
```

#### Examples:

```bash
# Process 25 time steps with 8 MPI processes
mpirun -np 8 python3 divCurl.py ./data/cavity3D.bp5 25

# Process with custom XML and output file
mpirun -np 4 python3 divCurl.py /path/to/simulation.bp5 100 -x adios2_config.xml -o divergence_curl.bp

# Single process run
python3 divCurl.py cavity2D.bp5 10 --output test_output.bp
```

### streamlines

Generate streamline visualizations from 2D velocity field data.

#### Basic Usage:

```bash
python3 streamlines.py input_file.bp5 20
python3 streamlines.py input_file.bp5 20 --workers 8
```

#### Examples:

```bash
python3 streamlines.py cavity2D.bp5 10
python3 streamlines.py ./results/cavity2D.bp5 50 --workers 4
python3 streamlines.py /home/user/sims/cavity2D_256x256.bp5 25
python3 streamlines.py test_data.bp5 3
```

## Output Files

### div_curl output:

- Creates a new BP5 file (default: `div_curl.bp`) containing:
  - `div`: Divergence field
  - `curl_x`, `curl_y`, `curl_z`: Curl components

### streamlines output:

- Creates PNG images for each time step:
  - Format: `[filename]_[resolution]_streamlines_step####.png`
  - Example: `cavity2D_128x128_streamlines_step0001.png`
  - All images use consistent color scaling based on velocity magnitude

## Command Line Arguments Summary

### divCurl.py

- **Required:** `input_file` `num_steps`
- **Optional:** `--xml/-x` `--output/-o`

### streamlines.py

- **Required:** `input_file` `max_steps`
- **Optional:** `--workers/-w`
