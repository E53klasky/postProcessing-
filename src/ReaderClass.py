import adios2
from rich.traceback import install
import time

""" 
Handles serial/parallel reading from ADIOS2 .bp files.
Initializes an ADIOS2 IO object with an optional XML and MPI configuration and manages data reading.

NOTE parallel only works for 3d in the 0,1,2 in the 2 
"""


class Reader:
    def __init__(self, IO_Name, bp_file, xml=None, comm=None):
        install()

        self.comm = comm
        if xml is not None:
            self.adios_obj = adios2.Adios(xml, comm=self.comm)
        else:
            self.adios_obj = adios2.Adios(comm=self.comm)
        self.IO_Name = IO_Name
        self.Read_IO = self.adios_obj.declare_io(IO_Name)
        self.bp_file = bp_file
        self.Adios_reader = adios2.Stream(
            self.Read_IO, self.bp_file, "r", comm=self.comm
        )

        self.vars_Out = {}

        self.numRanks = 1
        self.rank = 0

        if self.comm:
            self.numRanks = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        self.state = False

    def begin_step(self):

        if self.state == True:

            print("Error begin step called wihtout ending the step")
            self.close()

        while True:
            status = self.Adios_reader.begin_step(timeout=0.1)
            if status == adios2.bindings.StepStatus.NotReady:
                time.sleep(0.1)
            else:
                break

        if status == adios2.bindings.StepStatus.OK:
            self.state = True
        return status

    def current_step(self):
        step = self.Adios_reader.current_step()
        return step

    def set_read_vars(self, vars):
        for var in vars:
            adios_var = self.Read_IO.inquire_variable(var)

            if adios_var is None:
                print(f"Variable '{var}' not found in the stream.")
            self.vars_Out[var] = adios_var

    def set_selection(self, data):
        shape = data.shape()

        if len(shape) == 3 and shape[0] == 1:
            total_slices = shape[1]
            base = total_slices // self.numRanks
            rem = total_slices % self.numRanks
            local_count_1 = base + 1 if self.rank < rem else base
            local_start_1 = self.rank * base + min(self.rank, rem)

            start = [0, local_start_1, 0]
            count = [1, local_count_1, shape[2]]
            data.set_selection((start, count))

        elif len(shape) == 3:
            total_slices = shape[0]
            base = total_slices // self.numRanks
            rem = total_slices % self.numRanks
            local_count_0 = base + 1 if self.rank < rem else base
            local_start_0 = self.rank * base + min(self.rank, rem)

            start = [local_start_0, 0, 0]
            count = [local_count_0, shape[1], shape[2]]
            data.set_selection((start, count))

        elif len(shape) == 2:
            print("skip")
        else:
            print("Skip")

    def read_step(self, var_name):

        adios_var = self.vars_Out.get(var_name)
        self.set_selection(adios_var)
        var_data = self.Adios_reader.read(adios_var)

        return var_data

    def end_step(self):
        if self.state == True:
            self.state = False
        else:
            print("Error begin step called wihtout ending the step")
            self.close()
        self.Adios_reader.end_step()
        print(f"Step {self.Adios_reader.current_step()} read successfully.")

    def close(self):
        self.Adios_reader.close()
        print("Reader closed successfully.")


# ===  How to Use the Reader Class ===
# r = Reader("example_IO", "example.bp", "example.xml")
# while True:
#     status = r.begin_step()
#     r.set_read_vars(["var1", "var2"])
#     if not status or adios2.bindings.StepStatus.OK != status:
#         break
#     data1 = r.read_step("var1")
#     data2 = r.read_step("var2")
#     print(f"Data1: {data1}, Data2: {data2}")
#     r.end_step()
# r.close()
