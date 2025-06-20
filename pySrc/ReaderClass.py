import adios2 
from rich.traceback import install

""" 
Handles serial reading from ADIOS2 .bp files.
Initializes an ADIOS2 IO object with an optional XML configuration and manages data reading.
"""
class Reader:
    def __init__(self, bp_file, IO_Name, xml=None):
        """
        Initializes the Reader with a .bp file, an IO name, and an optional XML configuration file.
        """
        install()
        if xml is not None:
            self.adios_obj = adios2.Adios(xml)
        else:
            self.adios_obj = adios2.Adios()
        self.IO_Name = IO_Name
        self.Read_IO = self.adios_obj.declare_io(IO_Name) 
        self.bp_file = bp_file 

    def reader(self, vars):
        """
        Reads a list of variable names from the .bp file across all timesteps.
        Returns a list containing the data arrays of each variable read.
        Handles step-based reading and prints progress and errors if encountered.
        """
        vars_Out = []
        reader = adios2.Stream(self.Read_IO, self.bp_file, "r")
        while True:
            try:
                status = reader.begin_step()
                current_step = reader.current_step()
                print(f"Reading step: {current_step}")
                if status !=  adios2.bindings.StepStatus.OK:
                    print(f"No more steps or error reading first stream: {self.bp_file}")
                    break
                
                for var in vars:
                    var_to_read = self.Read_IO.inquire_variable(var)  
                    var_data = reader.read(var_to_read)
                    vars_Out.append(var_data)
                    
                reader.end_step()
            except Exception as e:
                print(f"Error reading step {current_step}: {e}")
                try:
                    available_vars = reader.available_variables()
                    print(f"Available variables: {list(available_vars.keys())}")
                except:
                    print("Could not retrieve available variables")
                    print("Stopping the reading")
                    break
        reader.close()
        print("Reading completed Successfully")
        return vars_Out
