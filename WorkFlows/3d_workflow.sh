#!/bin/bash



# gen data  
mpirun -np 6 ../../build/bin/xcompact3d  input_ILES_Re5000.i3d

# work flow 1
mpirun -np 2 python3 compression.py  -in  ../../Incompact3d/examples/TGV-Taylor-Green-vortex/data.bp5/ -x ../XML/adios2_config.xml  -rio  ReadIOCompressed  -wio  WriteIOCompressed   -eb 0.0

mpirun -np 2 python3 divCurl.py   --file1 compressed.bp/  --IO_Name1 IO_NAME -wio wio  -o compressed_div_curl.bp  -v ux,uy,uz

python3 createStreamlines.py -in compressed.bp/ -v ux,uy,uz  -o seg_compressed -s '(0.75,0.85,0.3)'

python3 createStreamlines.py -in ../../Incompact3d/examples/TGV-Taylor-Green-vortex/data.bp5/  -v ux,uy,uz  -o seg -s '(0.75,0.85,0.3)'

python3 errorStream3D.py  --file1 seg_compressed/ --file2 seg/ --var_x coords_x --var_y coords_y  --var_z coords_z  --var_offset offsets

# work flow 2

python3 prep_desimateter.py -in ../../Incompact3d/examples/TGV-Taylor-Green-vortex/data.bp5/ -v ux,uy,uz,critq,vort --method upsample

python3 Decimation.py -in resampled.bp/ -v ux,uy,uz,vort,critq  -l 2

python3 createStreamlines.py -in Decimated.bp/ -v ux,uy,uz  -o seg_decimated -s '(0.75,0.85,0.3)'

python3 errorStream3D.py  --file1 seg_decimated/ --file2 seg/ --var_x coords_x --var_y coords_y  --var_z coords_z  --var_offset offsets

