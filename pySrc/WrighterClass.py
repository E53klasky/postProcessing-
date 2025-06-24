import adios2
from rich.traceback import install


class Writer:
    def __init__(self, IO_Name, bp_file="data.bp", xml=None):
        install()
        if xml is not None:
            self.adios_obj = adios2.Adios(xml)
        else:
            self.adios_obj = adios2.Adios()
        self.IO_Name = IO_Name
        self.Write_IO = self.adios_obj.declare_io(IO_Name)
        self.bp_file = bp_file
        self.Adios_writer = adios2.Stream(self.Write_IO, self.bp_file, "w")
        self.current_step = -1
        self.vars_Out = {}

    def set_write_vars(self, var, name):
        global_size = var.shape
        local_size = var.shape
        start = [0] * len(global_size)

        adios_var = self.Write_IO.define_variable(
            name, var, global_size, start, local_size
        )
        self.vars_Out[name] = (adios_var, var)

    def begin_step(self):
        status = self.Adios_writer.begin_step()
        self.current_step = self.Adios_writer.current_step()
        print(f"Writing step: {self.current_step}")

    def write(self, name,data):
        self.Adios_writer.write(name, data)

    def end_step(self):
        self.Adios_writer.end_step()
        print(f"Step {self.current_step} written successfully.")

    def close(self):
        self.Adios_writer.close()
        print("Writer closed successfully.")



# === ✅ How to Use the Writer Class ===

# # 1. Prepare your data
# var1 = np.arange(10, dtype=np.float64)
# var2 = np.linspace(0, 1, 10, dtype=np.float64)

# # 2. Create a Writer object
# # Arguments: IO_Name, output file name (optional), XML config file (optional)
# w = Writer(IO_Name="example_IO", bp_file="example.bp", xml=None)
#  start step
# w.begin_step()

# # 3. Define variables you want to write (must be done before writing)
# this is onely done once per variable
# w.set_write_vars(var1, "var1")
# w.set_write_vars(var2, "var2")

# # 4. Write a single timestep (you can loop this for multiple steps)

# w.write("var1", var1)
# w.write("var2", var2)
# w.end_step()

# # 5. Always close the writer at the end
# w.close()