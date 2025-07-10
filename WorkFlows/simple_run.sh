#!/bin/bash

EXE_PATH="../../build/bin/xcompact3d"
SIM_DIR="$HOME/Programs/Incompact3d/examples/Cylinder-wake"
PY_DIR="$HOME/Programs/postProcessing-/src"

cd ${SIM_DIR}

bpfile1="${SIM_DIR}/data"

mpirun -np 2 ${EXE_PATH} input_DNS300_LR_2D.i3d &

mpirun -np 2 python3 ${PY_DIR}/compression.py \
  --bpfile1 "${bpfile1}" \
  --error_bound 0.001 \
  --readIO ReadIOCompressed \
  --WrightIO WriteIOCompressed \
  --xml "${SIM_DIR}/adios2_config.xml" \
  --output C.bp &

python3 ${PY_DIR}/plot2D.py \
  --bpfile1 "${bpfile1}" \
  --read_io io \
  --xml "${SIM_DIR}/adios2_config.xml" \
  --vars ux,uy,critq,vort,pp &

wait
echo "All processes completed."
cd -
