import adios2
from rich.traceback import install

class Writer:
    def __init__(self, IO_Name, bp_file='data.bp', xml=None):
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
        
        adios_var = self.Write_IO.define_variable(name, var, global_size, start, local_size)
        self.vars_Out[name] = (adios_var, var)

    def begin_step(self):
            status = self.Adios_writer.begin_step()
            self.current_step = self.Adios_writer.current_step()
            print(f"Writing step: {self.current_step}")
    
    def write(self):
        for name, (adios_var, data) in self.vars_Out.items():
            self.Adios_writer.write(name, data)
            
    def end_step(self):
            self.Adios_writer.end_step()
            print(f"Step {self.current_step} written successfully.")
    
    def close(self):
            self.Adios_writer.close()
            print("Writer closed successfully.")
            

# # === ✅ How to Use ===

# # Example data
# var1 = np.arange(10, dtype=np.float64)
# var2 = np.linspace(0, 1, 10, dtype=np.float64)

# # Create writer
# w = Wrigher("example.bp", "example_IO", "example.xml")

# # Set variables (you call this for each variable individually)
# w.set_write_vars(var1, "var1")
# w.set_write_vars(var2, "var2")

# # Write one timestep
# while True:
# w.begin_step()
# w.write()
# w.end_step()

# # Clean up
# w.close()
# #