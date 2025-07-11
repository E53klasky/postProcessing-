#!/bin/bash

EXE_PATH="../../build/bin/xcompact3d"
SIM_DIR="$HOME/Programs/Incompact3d/examples/Cylinder-wake"
PY_DIR="$HOME/Programs/postProcessing-/src"
HOME_DIR="$HOME/Programs/postProcessing-/"

cd "${SIM_DIR}" || exit 1

SIM_DATA="${SIM_DIR}/data"
COMPRES_DATA="${SIM_DIR}/C"
SEGMETS_DATA="${SIM_DIR}/seg"


mpirun -np 2 "${EXE_PATH}" input_DNS300_LR_2D.i3d &


mpirun -np 2 python3 "${PY_DIR}/compression.py" \
  --input "${SIM_DATA}" \
  --error_bound 0.001 \
  --readIO ReadIOCompressed \
  --WrightIO WriteIOCompressed \
  --xml "${SIM_DIR}/adios2_config.xml" \
  --output C &


python3 "${PY_DIR}/plot2D.py" \
  --input "${COMPRES_DATA}" \
  --readIO io \
  --xml "${SIM_DIR}/adios2_config.xml" \
  --vars ux,uy,critq,vort,pp &


python3 "${PY_DIR}/createStreamlines.py" \
  --input "${COMPRES_DATA}" \
  --readIO io \
  --WrightIO wio \
  --xml "${SIM_DIR}/adios2_config.xml" \
  --vars ux,uy \
  --seeds_points '(0.3,0.42)' \
  --step_size 0.0025 \
  --num_RK_steps 1800 \
  --output seg &


python3 "${PY_DIR}/plotStreamlines.py" \
  --input "${SEGMETS_DATA}" \
  --readIO io \
  --xml "${SIM_DIR}/adios2_config.xml" \
  --var_x coords_x \
  --var_y coords_y \
  --var_offset offsets &


wait 
echo "Completed workflow"
echo "All processes completed."


mv RESULTS "${HOME_DIR}/"

cd "${HOME_DIR}/RESULTS" 
ls
echo "Here are the results."