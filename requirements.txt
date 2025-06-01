# Requirements for xcompact3d Post-Processing Tools

## System Dependencies

These must be installed at the system level before installing Python dependencies.

### 1. MPI (Message Passing Interface)
**Required for**: Parallel execution of divCurl.py and future parallel implementations

**Installation Options:**

**MPICH** (Recommended)
- Website: https://www.mpich.org/downloads/
- Ubuntu/Debian: `sudo apt-get install mpich libmpich-dev`
- CentOS/RHEL: `sudo yum install mpich mpich-devel` or `sudo dnf install mpich mpich-devel`
- macOS: `brew install mpich`

**OpenMPI** (Alternative)
- Website: https://www.open-mpi.org/software/ompi/
- Ubuntu/Debian: `sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev`
- CentOS/RHEL: `sudo yum install openmpi openmpi-devel` or `sudo dnf install openmpi openmpi-devel`
- macOS: `brew install open-mpi`

### 2. ADIOS2
**Required for**: Reading and writing .bp simulation files in all scripts

**Installation:**
- Website: https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html
- Documentation: https://adios2.readthedocs.io/

**Important Notes:**
- Must be compiled with Python bindings enabled
- Version 2.8+ recommended
- For Python bindings: ensure `-DADIOS2_USE_Python=ON -DADIOS2_USE_MPI=ON -DADIOS2_USE_MGARD=ON` during compilation

**Quick Install (if available via package manager):**
- Ubuntu/Debian: `sudo apt-get install adios2-tools libadios2-dev python3-adios2`
- Conda: `conda install -c conda-forge adios2`

### 3. Protocol Buffers (protobuf)
**Required for**: MGARD dependency
**Version**: >= 3.2.0 (3.6+ recommended)

**Installation:**
- Website: https://developers.google.com/protocol-buffers
- GitHub: https://github.com/protocolbuffers/protobuf

**Install via package manager:**
- Ubuntu/Debian: `sudo apt-get install protobuf-compiler libprotobuf-dev`
- CentOS/RHEL: `sudo dnf install protobuf-compiler protobuf-devel`
- macOS: `brew install protobuf`

### 4. MGARD
**Required for**: Compression functionality in compression.py
**Dependencies**: Requires protobuf >= 3.2.0

**Installation:**
- GitHub: https://github.com/CODARcode/MGARD
- Documentation: https://mgard.readthedocs.io/

**Build from source:**
```bash
# Ensure protobuf is installed first
git clone https://github.com/CODARcode/MGARD.git
cd MGARD
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make -j4
make install
```

**Note**: If protobuf is installed in a non-standard location, you may need to specify:
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install -DProtobufT=/path/to/protobuf
```

## Python Dependencies

Install these after system dependencies are properly configured:

### Required Packages
```bash
pip install numpy>=1.20.0 matplotlib>=3.5.0 mpi4py>=3.1.0
```

### Package Details

**numpy** (>=1.20.0)
- Purpose: Numerical computations, gradient calculations
- Used in: All scripts for array operations and mathematical functions

**matplotlib** (>=3.5.0)
- Purpose: Streamline plotting and visualization
- Used in: streamlines.py for generating PNG output files

**mpi4py** (>=3.1.0)
- Purpose: Python MPI bindings for parallel processing
- Used in: divCurl.py and compression.py for MPI communication
- Must be compatible with your system MPI installation

### Virtual Environment Setup (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install Python packages
pip install --upgrade pip
pip install numpy>=1.20.0 matplotlib>=3.5.0 mpi4py>=3.1.0
```

### Testing Your Installation

**Test protobuf:**
```bash
protoc --version  # Should show version >= 3.2.0
```

**Test MPI + mpi4py:**
```bash
mpirun -np 2 python3 -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}')"
```

**Test ADIOS2:**
```bash
python3 -c "import adios2; print('ADIOS2 version:', adios2.__version__)"
```

**Test other packages:**
```bash
python3 -c "import numpy, matplotlib; print('NumPy:', numpy.__version__, 'Matplotlib:', matplotlib.__version__)"
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install mpich libmpich-dev adios2-tools libadios2-dev python3-adios2 protobuf-compiler libprotobuf-dev

# Install Python dependencies
pip install numpy matplotlib mpi4py
```

### Linux (CentOS/RHEL/Rocky)
```bash
# Install system dependencies
sudo dnf install mpich mpich-devel protobuf-compiler protobuf-devel
# ADIOS2 may need to be built from source

# Install Python dependencies
pip install numpy matplotlib mpi4py
```

### macOS
```bash
# Install system dependencies
brew install mpich protobuf
# ADIOS2 and MGARD likely need to be built from source

# Install Python dependencies
pip install numpy matplotlib mpi4py
```

## Troubleshooting

### Common Issues

**mpi4py compilation errors:**
- Ensure MPI is properly installed and in PATH
- Try: `env MPICC=/path/to/mpicc pip install mpi4py`

**ADIOS2 import errors:**
- Verify ADIOS2 Python bindings are installed
- Check LD_LIBRARY_PATH includes ADIOS2 libraries
- Try: `export PYTHONPATH=/path/to/adios2/python/lib:$PYTHONPATH`

**Missing MGARD/protobuf errors:**
- Install protobuf >= 3.2.0 before building MGARD
- MGARD must be compiled and installed separately
- Ensure MGARD libraries are in LD_LIBRARY_PATH
- Check protobuf version: `protoc --version`

### Version Compatibility
- Python 3.7+ required
- Compatible MPI implementation with mpi4py
- ADIOS2 2.8+ recommended for best compatibility

## Optional Dependencies

**For development/testing:**
```bash
pip install pytest flake8 black
```

**For enhanced visualization (future features):**
```bash
pip install plotly scipy
```