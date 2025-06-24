import adios2
from rich.traceback import install

""" 
Handles serial reading from ADIOS2 .bp files.
Initializes an ADIOS2 IO object with an optional XML configuration and manages data reading.
"""


class Reader:
    def __init__(self, IO_Name, bp_file, xml=None):
        install()
        if xml is not None:
            self.adios_obj = adios2.Adios(xml)
        else:
            self.adios_obj = adios2.Adios()
        self.IO_Name = IO_Name
        self.Read_IO = self.adios_obj.declare_io(IO_Name)
        self.bp_file = bp_file
        self.Adios_reader = adios2.Stream(self.Read_IO, self.bp_file, "r")
        self.current_step = -1
        self.vars_Out = {}

    def begin_step(self):
        status = self.Adios_reader.begin_step()
        self.current_step = self.Adios_reader.current_step()
        print(f"Reading step: {self.current_step}")
        return status

    def set_read_vars(self, vars):
        for var in vars:
            adios_var = self.Read_IO.inquire_variable(var)

            if adios_var is None:
                print(f"Variable '{var}' not found in the stream.")
            self.vars_Out[var] = adios_var

    def read_step(self, var_name):
        adios_var = self.vars_Out.get(var_name)
        var_data = self.Adios_reader.read(adios_var)
        return var_data

    def end_step(self):
        self.Adios_reader.end_step()
        print(f"Step {self.current_step} read successfully.")

    def close(self):
        self.Adios_reader.close()
        print("Reader closed successfully.")


# === ✅ How to Use the Reader Class ===
# r = Reader("example.bp", "example_IO", "example.xml")
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
