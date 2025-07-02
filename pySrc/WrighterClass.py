import adios2
from rich.traceback import install
from mpi4py import MPI

# works for now -> subtract, rmse | todo -> histagram
"""
Handles serial/parallel reading from ADIOS2 .bp files.
Initializes an ADIOS2 IO object with an optional XML and MPI configuration and manages data reading.
"""


class Writer:
    def __init__(self, IO_Name, bp_file="data.bp", xml=None, comm=None):
        install()

        self.comm = comm
        if xml is not None:
            self.adios_obj = adios2.Adios(xml, comm=self.comm)
        else:
            self.adios_obj = adios2.Adios(comm=self.comm)
        self.IO_Name = IO_Name
        self.Write_IO = self.adios_obj.declare_io(IO_Name)
        self.bp_file = bp_file
        self.Adios_writer = adios2.Stream(
            self.Write_IO, self.bp_file, "w", comm=self.comm
        )
        self.current_step = -1
        self.numRanks = 1
        self.rank = 0

        if self.comm:
            self.numRanks = self.comm.Get_size()
            self.rank = self.comm.Get_rank()

        self.state = False

    def begin_step(self):
        if self.state == False:
            self.state = True
        else:
            print("Error begin step called wihtout ending the step")
            self.close()
        status = self.Adios_writer.begin_step()
        self.current_step = self.Adios_writer.current_step()
        print(f"Writing step: {self.current_step}")

    def get_var_info(self, data):
        count = list(data.shape)
        global_count = count.copy()
        global_count[-1] = self.comm.allreduce(count[-1], op=MPI.SUM)
        offset = count.copy()
        offset[-1] = self.comm.exscan(count[-1]) or 0
        for i in range(len(offset) - 1):
            offset[i] = 0

        return (global_count, offset, count)

    def write(self, name, data):
        shape, offset, count = self.get_var_info(data)
        self.Adios_writer.write(name, data, shape, offset, count)

    def end_step(self):
        if self.state == True:
            self.state = False
        else:
            print("Error begin step called wihtout ending the step")
            self.close()
        self.Adios_writer.end_step()
        print(f"Step {self.current_step} written successfully.")

    def close(self):
        self.Adios_writer.close()
        print("Writer closed successfully.")


# === How to Use the Writer Class (Serial) ===

# import numpy as np
# var1 = np.arange(10, dtype=np.float64)
# var2 = np.linspace(0, 1, 10, dtype=np.float64)
# w = Writer(IO_Name="example_IO", bp_file="example.bp", xml=None)
# w.begin_step()
# w.write("var1", var1)
# w.write("var2", var2)
# w.end_step()
# w.close()

# === How to Use the Writer Class (Parallel with MPI) ===

# from mpi4py import MPI
# import numpy as np
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# # Each rank writes its own chunk of data
# local_size = 10
# var1 = np.arange(rank * local_size, (rank + 1) * local_size, dtype=np.float64)
# var2 = np.linspace(rank, rank + 1, local_size, dtype=np.float64)
# w = Writer(IO_Name="example_IO", bp_file="example.bp", xml=None, comm=comm)
# w.begin_step()
# w.write("var1", var1)
# w.write("var2", var2)
# w.end_step()
# w.close()
