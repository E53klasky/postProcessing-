#!/bin/bash

# note the errors are low because the data is comparing the compressed data with the original data
# the error bound is also 0.001 which is also why the errors are low
mpirun -np 4 ../../build/bin/xcompact3d  input.i3d

# workflow 1

mpirun -np 2 python3 compression.py  -in  ../../Incompact3d/examples/Cavity/data.bp5/ -x ../XML/ad.xml  -rio  ReadIOCompressed  -wio  WriteIOCompressed   -eb 0.001

python3 plot2D.py  -in compressed.bp/  -v ux,uy,pp,phi01
python3 createStreamlines.py -in compressed.bp/ -v ux,uy -s '(0.75,0.8)' -o segmetns_compressed
python3 plotStreamlines.py -in segmetns_compressed/  --var_x coords_x --var_y coords_y --var_offset offsets -vf compressed.bp/ --var_u ux --var_v uy

# workflow 2

mpirun -np 2 python3 compression.py  -in  ../../Incompact3d/examples/Cavity/data.bp5/ -x ../XML/ad.xml  -rio  ReadIOCompressed  -wio  WriteIOCompressed   -eb 0.001

python3 plot2D.py  -in compressed.bp/  -v ux,uy,pp,phi01
python3 createStreamlines.py -in compressed.bp/ -v ux,uy -s '(0.75,0.8)' -o segmetns_compressed
python3 plotStreamlines.py -in segmetns_compressed/  --var_x coords_x --var_y coords_y --var_offset offsets -vf compressed.bp/ --var_u ux --var_v uy
python3 contour.py  -f1 compressed.bp/  -v ux,uy,pp,phi01  -d1 io
mpirun -np 2 python3 divCurl.py --file1 compressed.bp/  -v ux,uy -wio wio
python3 plot2D.py  -in div_curl.bp/  -v   Curl_Z,Div_Z

# workflow 3
python3 plot2D.py  -in ../../Incompact3d/examples/Cavity/data.bp5/ -v ux,uy,pp,phi01
# note this will overwrite the previous
python3 contour.py  -f1 ../../Incompact3d/examples/Cavity/data.bp5/ -v ux,uy,pp,phi01  -d1 io

mpirun -np 2 python3 compression.py  -in  ../../Incompact3d/examples/Cavity/data.bp5/ -x ../XML/ad.xml  -rio  ReadIOCompressed  -wio  WriteIOCompressed   -eb 0.001


mpirun -np 2 python3 divCurl.py --file1 compressed.bp/  -v ux,uy -wio wio


python3 plot2D.py  -in compressed.bp/  -v ux,uy,pp,phi01
# note this will overwrite the previous
python3 contour.py  -f1 compressed.bp/  -v ux,uy,pp,phi01  -d1 io

python3 plot2D.py  -in div_curl.bp/  -v   Curl_Z,Div_Z


python3 createStreamlines.py -in ../../Incompact3d/examples/Cavity/data.bp5/ -v ux,uy -s '(0.75,0.8)'
python3 createStreamlines.py -in compressed.bp/ -v ux,uy -s '(0.75,0.8)' -o segmetns_compressed


python3 plotStreamlines.py -in segments.bp/  --var_x coords_x --var_y coords_y --var_offset offsets -vf ../../Incompact3d/examples/Cavity/data.bp5/ --var_u ux --var_v uy
# note this will overwrite the previous
python3 plotStreamlines.py -in segmetns_compressed/  --var_x coords_x --var_y coords_y --var_offset offsets -vf compressed.bp/ --var_u ux --var_v uy

python3 errorStream2D.py --file1 segmetns_compressed/ --file2 segments.bp/ --var_x coords_x --var_y coords_y --var_offset offsets -N 2000

# workflow 4 lots of streamlines
python3 plot2D.py  -in ../../Incompact3d/examples/Cavity/data.bp5/ -v ux,uy,pp,phi01
# note this will overwrite the previous
python3 contour.py  -f1 ../../Incompact3d/examples/Cavity/data.bp5/ -v ux,uy,pp,phi01  -d1 io

mpirun -np 2 python3 compression.py  -in  ../../Incompact3d/examples/Cavity/data.bp5/ -x ../XML/ad.xml  -rio  ReadIOCompressed  -wio  WriteIOCompressed   -eb 0.001


mpirun -np 2 python3 divCurl.py --file1 compressed.bp/  -v ux,uy -wio wio


python3 plot2D.py  -in compressed.bp/  -v ux,uy,pp,phi01
# note this will overwrite the previous
python3 contour.py  -f1 compressed.bp/  -v ux,uy,pp,phi01  -d1 io

python3 plot2D.py  -in div_curl.bp/  -v   Curl_Z,Div_Z


python3 createStreamlines.py -in ../../Incompact3d/examples/Cavity/data.bp5/ -v ux,uy -s '(0.75,0.8),(0.45,0.5),(0.25,0.3),(0.05,0.1),(0.15,0.2)' -dh 0.005 -step 900
python3 createStreamlines.py -in compressed.bp/ -v ux,uy -s '(0.75,0.8),(0.45,0.5),(0.25,0.3),(0.05,0.1),(0.15,0.2)' -o segmetns_compressed   -dh 0.005 -step 900

python3 plotStreamlines.py -in segments.bp/  --var_x coords_x --var_y coords_y --var_offset offsets -vf ../../Incompact3d/examples/Cavity/data.bp5/ --var_u ux --var_v uy
# note this will overwrite the previous
python3 plotStreamlines.py -in segmetns_compressed/  --var_x coords_x --var_y coords_y --var_offset offsets -vf compressed.bp/ --var_u ux --var_v uy

python3 errorStream2D.py --file1 segmetns_compressed/ --file2 segments.bp/ --var_x coords_x --var_y coords_y --var_offset offsets -N 2000

