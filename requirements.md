# Requirements and Installation Guide for Xcompact3d Post-Processing Tools

## System Dependencies

These must be installed at the system level before Python dependencies.

---

### 1. Protocol Buffers (protobuf)

**Purpose:** Used as a dependency for MGARD and other components.

- **Recommended version:** 3.2.0 or higher (3.6+ preferred)
- **Website:** [https://developers.google.com/protocol-buffers](https://developers.google.com/protocol-buffers)
- **GitHub:** [https://github.com/protocolbuffers/protobuf](https://github.com/protocolbuffers/protobuf)

**Example build:**

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.21.12
git submodule update --init --recursive
./autogen.sh
./configure --prefix=/your/install/path
make -j
make install
Make sure protoc is in your PATH and verify with:

bash
Copy
Edit
protoc --version
2. MPI (Message Passing Interface)
Purpose: Provides parallel execution support required by MPI-enabled components.

Options: MPICH, OpenMPI

Installation: Use your system package manager or build from source.

Example building MPICH from source:

bash
Copy
Edit
wget https://download.mpich.org/mpich/stable/mpich-4.1.1.tar.gz
tar -xzf mpich-4.1.1.tar.gz
cd mpich-4.1.1
./configure --prefix=/your/install/path
make -j
make install
Verify installation:

bash
Copy
Edit
mpirun --version
3. MGARD
Purpose: Compression backend required for compression functionality.

GitHub: https://github.com/CODARcode/MGARD

Dependencies: Requires protobuf installed first.

Example build:

bash
Copy
Edit
git clone https://github.com/CODARcode/MGARD.git
cd MGARD
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/your/install/path
make -j
make install
4. ADIOS2
Purpose: Reading and writing .bp files used in all scripts.

Docs: https://adios2.readthedocs.io/

GitHub: https://github.com/ornladios/ADIOS2

Important: Build with Python bindings, MPI, and MGARD enabled.

Example build:

bash
Copy
Edit
git clone https://github.com/ornladios/ADIOS2.git
cd ADIOS2
git checkout v2.8.0
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DADIOS2_USE_Python=ON \
  -DADIOS2_USE_MPI=ON \
  -DADIOS2_USE_MGARD=ON \
  -DCMAKE_INSTALL_PREFIX=/your/install/path
make -j
make install
Verify python import:

bash
Copy
Edit
python3 -c "import adios2; print(adios2.__version__)"
5. 2decomp-fft
Purpose: Library required by Xcompact3d for parallel 2D domain decomposition.

GitHub: https://github.com/cppla/2decomp_fft

Example build:

bash
Copy
Edit
git clone https://github.com/cppla/2decomp_fft.git
cd 2decomp_fft
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/your/install/path
make -j
make install
6. Xcompact3d
Purpose: Main post-processing tool to build after dependencies.

GitHub: https://github.com/cppla/Xcompact3d

Example build:

bash
Copy
Edit
git clone https://github.com/cppla/Xcompact3d.git
cd Xcompact3d
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/adios2;/path/to/2decomp_fft" \
  -DCMAKE_INSTALL_PREFIX=/your/install/path \
  -DIO_BACKEND=adios2
make -j
make install
Python Dependencies
After system libraries are installed, install these Python packages:

bash
Copy
Edit
pip install numpy matplotlib mpi4py black
Set up a virtual environment (recommended):

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib mpi4py black
Testing Your Setup
Protobuf:

bash
Copy
Edit
protoc --version
MPI + mpi4py:

bash
Copy
Edit
mpirun -np 2 python3 -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}')"
ADIOS2 Python binding:

bash
Copy
Edit
python3 -c "import adios2; print(adios2.__version__)"
```
