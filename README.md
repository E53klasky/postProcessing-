# postProcessing-
This is where i will be doing my post processing of the cfd simulation of xcompact3d



Add edits to adios2 xml config





How to run

div_curl 
mpirun -np 4 python3 div_curl.py ../Incompact3d/examples/Cavity/data.bp5/ ../Incompact3d/examples/Cavity/adios2_config.xml
