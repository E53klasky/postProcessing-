import adios2
import numpy as np
from mpi4py import MPI
from rich.traceback import install


class Writer3D:
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

        self._calculate_3d_grid()

    def _calculate_3d_grid(self):
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

    def begin_step(self):
        if self.state == False:
            self.state = True
        else:
            print("Error begin step called without ending the step")
            self.close()
        status = self.Adios_writer.begin_step()
        self.current_step = self.Adios_writer.current_step()
        print(f"Writing step: {self.current_step}")

    def get_var_info_3d(self, data):
        shape = list(data.shape)
        
        if len(shape) < 3:
            return self._get_var_info_fallback(data)
        
        if self.comm is None:
            global_count = shape
            offset = [0] * len(shape)
            count = shape
            return (global_count, offset, count)
        
        local_shape = np.array(shape, dtype=np.int64)

        all_shapes = None
        if self.rank == 0:
            all_shapes = np.zeros((self.numRanks, len(shape)), dtype=np.int64)
        
        self.comm.Gather(local_shape, all_shapes, root=0)
        
        global_shape = np.zeros(len(shape), dtype=np.int64)
        if self.rank == 0:
            global_shape = local_shape.copy()
            

            for dim in range(min(3, len(shape))):
                global_shape[dim] = 0
            
            for r in range(self.numRanks):
                rankz_r = r // (self.px * self.py)
                remaining_r = r % (self.px * self.py)
                ranky_r = remaining_r // self.px
                rankx_r = remaining_r % self.px
                
                if len(shape) >= 1 and rankx_r == 0:  
                    if ranky_r == 0 and rankz_r == 0:  
                        global_shape[0] += all_shapes[r, 0] * self.px
                        break
                        
            for r in range(self.numRanks):
                rankz_r = r // (self.px * self.py)
                remaining_r = r % (self.px * self.py)
                ranky_r = remaining_r // self.px
                rankx_r = remaining_r % self.px
                
                if len(shape) >= 2 and ranky_r == 0: 
                    if rankx_r == 0 and rankz_r == 0:  
                        global_shape[1] += all_shapes[r, 1] * self.py
                        break
                        
            for r in range(self.numRanks):
                rankz_r = r // (self.px * self.py)
                remaining_r = r % (self.px * self.py)
                ranky_r = remaining_r // self.px
                rankx_r = remaining_r % self.px
                
                if len(shape) >= 3 and rankz_r == 0:  
                    if rankx_r == 0 and ranky_r == 0: 
                        global_shape[2] += all_shapes[r, 2] * self.pz
                        break
            
 
            max_extents = np.zeros(len(shape), dtype=np.int64)
            for r in range(self.numRanks):
                rankz_r = r // (self.px * self.py)
                remaining_r = r % (self.px * self.py)
                ranky_r = remaining_r // self.px
                rankx_r = remaining_r % self.px
                

                if len(shape) >= 1:
                    base_x = all_shapes[r, 0]  
                    if ranky_r == 0 and rankz_r == 0:
                        max_extents[0] += all_shapes[r, 0]
                
                if len(shape) >= 2:
                    if rankx_r == 0 and rankz_r == 0:
                        max_extents[1] += all_shapes[r, 1]
                        
                if len(shape) >= 3:
                    if rankx_r == 0 and ranky_r == 0:
                        max_extents[2] += all_shapes[r, 2]
            
            global_shape[:min(3, len(shape))] = max_extents[:min(3, len(shape))]
            

            for i in range(3, len(shape)):
                global_shape[i] = all_shapes[0, i]
        
        self.comm.Bcast(global_shape, root=0)
        
        global_count = global_shape.tolist()
        offset = [0] * len(shape)
        count = shape.copy()
        

        if len(shape) >= 1:
            x_offset = 0
            for rx in range(self.rankx):
                target_rank = self.rankz * (self.px * self.py) + self.ranky * self.px + rx
                if target_rank < self.numRanks:
                    pass

            base_x = global_count[0] // self.px
            rem_x = global_count[0] % self.px
            offset[0] = self.rankx * base_x + min(self.rankx, rem_x)
        

        if len(shape) >= 2:
            base_y = global_count[1] // self.py
            rem_y = global_count[1] % self.py
            offset[1] = self.ranky * base_y + min(self.ranky, rem_y)
        
 
        if len(shape) >= 3:
            base_z = global_count[2] // self.pz
            rem_z = global_count[2] % self.pz
            offset[2] = self.rankz * base_z + min(self.rankz, rem_z)
        
        return (global_count, offset, count)
    
    def _get_var_info_fallback(self, data):
        """Fallback to last dimension decomposition for non-3D data"""
        count = list(data.shape)
        if self.comm:
            global_count = count.copy()
            global_count[-1] = self.comm.allreduce(count[-1], op=MPI.SUM)
            offset = count.copy()
            offset[-1] = self.comm.exscan(count[-1]) or 0
            for i in range(len(offset) - 1):
                offset[i] = 0
        else:
            global_count = count = list(data.shape)
            offset = [0] * len(data.shape)
        return (global_count, offset, count)

    def write(self, name, data):
        shape, offset, count = self.get_var_info_3d(data)
        
        if self.rank == 0:
            print(f"Writing {name}: global_shape={shape}, local_offset={offset}, local_count={count}")
        
        self.Adios_writer.write(name, data, shape, offset, count)

    def end_step(self):
        if self.state == True:
            self.state = False
        else:
            print("Error end step called without beginning the step")
            self.close()
        self.Adios_writer.end_step()
        print(f"Step {self.current_step} written successfully.")

    def close(self):
        self.Adios_writer.close()
        print("Writer3D closed successfully.")