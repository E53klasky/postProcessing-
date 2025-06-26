# Xcompact3d Installation Guide

## Overview

Installation guide for Xcompact3d with ADIOS2 support.

**Build Order**: ADIOS2 → 2DECOMP&FFT → Xcompact3d

## Dependencies

Dependencies: You must have MPI (e.g., OpenMPI) and CMake ≥ 3.20 installed before proceeding.

Set up environment variables:

```bash
export FC=mpif90
export CC=mpicc
export CXX=mpicxx
export PREFIX=/usr/local
```

### 1. ADIOS2

**Repository**: https://github.com/ornladios/ADIOS2

```bash
git clone https://github.com/ornladios/ADIOS2.git
cd ADIOS2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DADIOS2_USE_MPI=ON   -DADIOS2_USE_Python=ON
make -j$(nproc) && sudo make install
```

### 2. 2DECOMP&FFT

**Repository**: https://github.com/2decomp-fft/2decomp-fft

```bash
git clone https://github.com/2decomp-fft/2decomp-fft.git
cd 2decomp-fft
git checkout v2.0.4
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DIO_BACKEND=adios2
make -j$(nproc) && sudo make install
```

### 3. Xcompact3d

**Repository**: https://github.com/xcompact3d/Incompact3d

```bash
git clone https://github.com/xcompact3d/Incompact3d.git
cd Incompact3d
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DIO_BACKEND=adios2 \
  -DUSE_ADIOS2=ON
make -j$(nproc)
```

**Note**: After building, search through the `.f90` source files and change any instances of `BP4` to `BP5` for better performance.

## Python Environment

```bash
python3 -m venv xcompact3d-env
source xcompact3d-env/bin/activate
pip install numpy matplotlib mpi4py
```

## Quick Test

```bash
mpirun -np 2 echo "MPI works"

cd Incompact3d/build
./bin/xcompact3d --help

python3 -c "import numpy; print('Python ready')"
```

## Install Script

```bash
set -e

INSTALL_DIR="$HOME/xcompact3d"
PREFIX="/usr/local"

export FC=mpif90 CC=mpicc CXX=mpicxx

mkdir -p $INSTALL_DIR && cd $INSTALL_DIR

echo "Building ADIOS2..."
git clone https://github.com/ornladios/ADIOS2.git
cd ADIOS2 && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DADIOS2_USE_MPI=ON
make -j$(nproc) && sudo make install
cd ../..

echo "Building 2DECOMP&FFT..."
git clone https://github.com/2decomp-fft/2decomp-fft.git
cd 2decomp-fft && git checkout v2.0.4
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_INSTALL_PREFIX="$PREFIX" -DIO_BACKEND=adios2
make -j$(nproc) && sudo make install
cd ../..

echo "Building Xcompact3d..."
git clone https://github.com/xcompact3d/Incompact3d.git
cd Incompact3d && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" -DIO_BACKEND=adios2 -DUSE_ADIOS2=ON
make -j$(nproc)

echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib mpi4py

echo "Executable: $INSTALL_DIR/Incompact3d/build/bin/xcompact3d"
```

## Common Issues

**Build fails**: Check that environment variables (`FC`, `CC`, `CXX`) are set and MPI is working with `mpicc --version`.

**Library not found**: Ensure `LD_LIBRARY_PATH` includes `/usr/local/lib` and run `sudo ldconfig`.

**Python import fails**: Make sure ADIOS2 was built with Python support and activate the virtual environment.

**CMake version**: Xcompact3d requires CMake >= 3.20. Update if needed.