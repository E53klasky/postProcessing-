import adios2 
import numpy as np



"""
REMBER TO CHANGE XML AND CODE A LITTLE

RMSE every step 
step 0 ux = 0.0011409772551731416
step 1 ux = 0.18360780968262544
step 2 ux = 0.19935939100421454
step 3 ux = 0.1551769769387862
step 4 ux = 0.1851770334122703

step 0 uy = 0.0011242040074490008
step 1 uy = 0.21242714002311616
step 2 uy = 0.20226947946705912
step 3 uy = 0.16006675943955836 
step 4 uy = 0.19107487253320435

step 0 pp = 0.00018643898700178768
step 1 pp = 0.04176714070153108
step 2 pp = 0.03348528355656824
step 3 pp = 0.038415066687282874
step 4 pp = 0.03662853431777195

step 0 phi01 = 0.00132457675310982
step 1 phi01 = 0.19699523217691003
step 2 phi01 = 0.1352433522312631
step 3 phi01 = 0.14905412988582592
step 4 phi01 = 0.1325432864450968

"""

# change to RMSE need to do for each var add var in there 
def RMSE(GT, E, var_NAME):
    skip_factor = 2
    diff_sq = 0
    count = 0

    for i in range(0, E.shape[0]):
        for j in range(0, E.shape[1]):
            gt_value = GT[i * skip_factor, j * skip_factor]
            e_value = E[i, j]
            diff_sq += (gt_value - e_value) ** 2
            count += 1

    rmse = np.sqrt(diff_sq / count)
    print("=" * 60)
    print(f"The RMSE for the ground truth {var_NAME} is: {rmse}")
    print()



def main():
    max_steps = 5
    adios = adios2.Adios()
    lower_res =  "../../Incompact3d/examples/Cavity/data/cavity2D.1025.513.1.0.0001.bp5"
    groud_truth = "../../Incompact3d/examples/Cavity/data/cavity2D.2049.1025.1.0.00005.bp5"
    RLio = adios.declare_io("readerIOLow")
    RHio = adios.declare_io("readerIOH")
    var_name = 'phi01'
    with adios2.Stream(RHio, lower_res, 'r') as rl:    
        with adios2.Stream(RLio, groud_truth, 'r') as rh:
            for _ in rl:
                for _ in rh:
                        stepl = rl.current_step()
                        steh = rh.current_step()
                        print(f"Processing step {steh}")
                        # change var each time 
                        varl = rl.read(var_name)
                        varh = rh.read(var_name)
                        
                        
                        if len(varl.shape) == 3 and varl.shape[0] == 1:
                            varl = varl[0, :, :]
                        if len(varh.shape) == 3  and varh.shape[0] == 1:
                            varh = varh[0, :, :]
                            
                        RMSE(varh, varl, var_name)    


                        if rl.current_step() >= max_steps - 1 or rh.current_step() >= max_steps -1:
                            print(f"Reached max_steps = {max_steps}")
                            break

if __name__ == "__main__":
    main()

