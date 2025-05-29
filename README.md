postProcessing
This is where I will be doing my post-processing of the CFD simulation of xcompact3d.

Setup
If you're using ADIOS2, make sure to configure your XML file appropriately (e.g., variable definitions, transport methods, etc.).

How to Run
divCurl.py
Calculate divergence and curl from velocity field data using MPI for parallel processing.

Basic Usage
bash
Copy
Edit

# Required arguments: input_file and num_steps

mpirun -np 4 python3 divCurl.py input_file.bp5 50

# With custom output file

mpirun -np 4 python3 divCurl.py input_file.bp5 50 --output results.bp

# With XML configuration file

mpirun -np 4 python3 divCurl.py input_file.bp5 50 --xml config.xml

# With both optional arguments

mpirun -np 4 python3 divCurl.py input_file.bp5 50 --xml config.xml --output my_results.bp
Examples
bash
Copy
Edit
mpirun -np 8 python3 divCurl.py ./data/cavity3D.bp5 25
mpirun -np 4 python3 divCurl.py /path/to/simulation.bp5 100 -x adios2_config.xml -o divergence_curl.bp
python3 divCurl.py cavity2D.bp5 10 --output test_output.bp
streamlines.py
Generate streamline visualizations from 2D velocity field data. Supports optional ADIOS2 XML configuration and parallel plotting.

Basic Usage
bash
Copy
Edit

# Basic run

python3 streamlines.py input_file.bp5 20

# Use multiple workers

python3 streamlines.py input_file.bp5 20 --workers 8

# With XML configuration

python3 streamlines.py input_file.bp5 20 --xml config.xml

# Combined usage

python3 streamlines.py input_file.bp5 20 --xml config.xml --workers 4
Examples
bash
Copy
Edit
python3 streamlines.py cavity2D.bp5 10
python3 streamlines.py ./results/cavity2D.bp5 50 --workers 4
python3 streamlines.py /home/user/sims/cavity2D_256x256.bp5 25
python3 streamlines.py test_data.bp5 3 --xml adios2_config.xml
Output Files
divCurl.py Output
Creates a new BP5 file (default: div_curl.bp) containing:

div: Divergence field

curl_x, curl_y, curl_z: Curl components

streamlines.py Output
Generates PNG images for each time step:

Filename format: [base_name]\_[resolution]\_streamlines_step####.png

Example: cavity2D_128x128_streamlines_step0001.png

All images use a consistent color scale based on global velocity magnitude

Command Line Arguments Summary
divCurl.py
Required: input_file num_steps

Optional:

--xml / -x : Path to ADIOS2 XML config

--output / -o : Output BP5 file name

streamlines.py
Required: input_file max_steps

Optional:

--workers / -w : Number of worker processes

--xml / -x : Path to ADIOS2 XML config
