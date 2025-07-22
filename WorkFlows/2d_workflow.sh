#!/bin/bash

 mpirun -np 4 ../../build/bin/xcompact3d  input.i3d

# workflow 1
python3 plot2D.py  -in ../../Incompact3d/examples/Cavity/data.bp5/ -v ux,uy,pp,phi01
# note this will overwrite the previous
python3 contour.py  -f1 ../../Incompact3d/examples/Cavity/data.bp5/ -v ux,uy,pp,phi01  -d1 io

mpirun -np 2 python3 compression.py  -in  ../../Incompact3d/examples/Cavity/data.bp5/ -x ../XML/ad.xml  -rio  ReadIOCompressed  -wio  WriteIOCompressed   -eb 0.001


python3 plot2D.py  -in compressed.bp/  -v ux,uy,pp,phi01
# note this will overwrite the previous
python3 contour.py  -f1 compressed.bp/  -v ux,uy,pp,phi01  -d1 io


