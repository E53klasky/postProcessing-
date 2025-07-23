import adios2
import numpy as np
import time
from rich.traceback import install


class Reader3D:
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
        
        self._calculate_3d_grid()

    def _calculate_3d_grid(self):
        """Calculate optimal 3D processor grid dimensions"""
        if self.numRanks == 1:
            self.px, self.py, self.pz = 1, 1, 1
            self.rankx, self.ranky, self.rankz = 0, 0, 0
            return
            
        factors = []
        for i in range(1, int(np.sqrt(self.numRanks)) + 1):
            if self.numRanks % i == 0:
                factors.append(i)
                if i != self.numRanks // i:
                    factors.append(self.numRanks // i)
        factors.sort()
        
        best_diff = float('inf')
        self.px, self.py, self.pz = 1, 1, self.numRanks
        
        for px in factors:
            remaining = self.numRanks // px
            for py in factors:
                if remaining % py == 0:
                    pz = remaining // py
                    diff = max(px, py, pz) - min(px, py, pz)
                    if diff < best_diff:
                        best_diff = diff
                        self.px, self.py, self.pz = px, py, pz
        
        self.rankz = self.rank // (self.px * self.py)
        remaining = self.rank % (self.px * self.py)
        self.ranky = remaining // self.px
        self.rankx = remaining % self.px
        
        if self.rank == 0:
            print(f"3D Grid: {self.px}x{self.py}x{self.pz}, Total ranks: {self.numRanks}")

    def begin_step(self):
        if self.state == True:
            print("Error begin step called without ending the step")
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
                continue
            self.vars_Out[var] = adios_var

    def set_selection_3d(self, data):
        """Set 3D domain decomposition selection for reading"""
        shape = data.shape()
        
        if len(shape) < 3:
            self._set_selection_fallback(data)
            return
        
        if self.comm is None:
            start = [0] * len(shape)
            count = list(shape)
            data.set_selection((start, count))
            return
        
        start = [0] * len(shape)
        count = list(shape)
        
        if self.rank == 0:
            print(f"Original shape: {shape}")
            print(f"3D grid: {self.px}x{self.py}x{self.pz}")
        
        if shape[0] >= self.px:
            base_x = shape[0] // self.px
            rem_x = shape[0] % self.px
            count[0] = base_x + (1 if self.rankx < rem_x else 0)
            start[0] = self.rankx * base_x + min(self.rankx, rem_x)
        else:
            if self.rankx < shape[0]:
                count[0] = 1
                start[0] = self.rankx
            else:
                count[0] = 0
                start[0] = 0
        
        if shape[1] >= self.py:
            base_y = shape[1] // self.py
            rem_y = shape[1] % self.py
            count[1] = base_y + (1 if self.ranky < rem_y else 0)
            start[1] = self.ranky * base_y + min(self.ranky, rem_y)
        else:
            if self.ranky < shape[1]:
                count[1] = 1
                start[1] = self.ranky
            else:
                count[1] = 0
                start[1] = 0
        
        if shape[2] >= self.pz:
            base_z = shape[2] // self.pz
            rem_z = shape[2] % self.pz
            count[2] = base_z + (1 if self.rankz < rem_z else 0)
            start[2] = self.rankz * base_z + min(self.rankz, rem_z)
        else:
            if self.rankz < shape[2]:
                count[2] = 1
                start[2] = self.rankz
            else:
                count[2] = 0
                start[2] = 0
        
        for i in range(3, len(shape)):
            start[i] = 0
            count[i] = shape[i]
        
        if self.rank == 0:
            print(f"Rank {self.rank}: start={start}, count={count}")
        
        data.set_selection((start, count))

    def _set_selection_fallback(self, data):
        """Fallback to last dimension decomposition for non-3D data"""
        shape = data.shape()
        
        if self.comm is None:
            start = [0] * len(shape)
            count = list(shape)
            data.set_selection((start, count))
            return
        
        total_elements = shape[-1]
        base = total_elements // self.numRanks
        rem = total_elements % self.numRanks
        local_count = base + 1 if self.rank < rem else base
        local_start = self.rank * base + min(self.rank, rem)

        start = [0] * len(shape)
        start[-1] = local_start
        count = list(shape)
        count[-1] = local_count

        data.set_selection((start, count))

    def read_step(self, var_name):
        adios_var = self.vars_Out.get(var_name)
        if adios_var is None:
            print(f"Variable {var_name} not set for reading")
            return None
        self.set_selection_3d(adios_var)
        var_data = self.Adios_reader.read(adios_var)
        return var_data

    def end_step(self):
        if self.state == True:
            self.state = False
        else:
            print("Error end step called without beginning the step")
            self.close()
        self.Adios_reader.end_step()
        print(f"Step {self.Adios_reader.current_step()} read successfully.")

    def close(self):
        self.Adios_reader.close()
        print("Reader3D closed successfully.")