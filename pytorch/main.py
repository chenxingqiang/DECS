import torch

import scipy.io as sio

from pytorch import *

def main():
    # Load data (you need to replace this part with your own data loading code)
    data = sio.loadmat("datasets/enron.mat")
    W_Cube = data["W_Cube"]
    data = sio.loadmat("datasets/firststep_DYNMOGA_enron.mat")
    GT_Cube = data["dynMoeaResult"]

    # Parameter setting
    maxgen = 100  # The maximum number of iterations
    pop_size = 100  # The population size
    num_neighbor = 5  # The neighbor size for each subproblem in decomposition-based multi-objective optimization
    p_mutation = 0.2  # The mutation rate
    p_migration = 0.5  # The migration rate
    p_mu_mi = 0.5  # The parameter to organize the execution of mutation and migration
    PGLP_iter = 5  # The number of iterations in PGLP
    num_repeat = 5  # The number of repeated runs

    # Results at each time step
    dynMod = torch.zeros(
        (W_Cube.shape[1], num_repeat)
    )  # Modularity of detected community structure
    dynNmi = torch.zeros(
        (W_Cube.shape[1], num_repeat)
    )  # NMI between detected community structure and the ground truth
    dynPop = {}  # The population
    dynTime = torch.zeros((W_Cube.shape[1], num_repeat))  # The running time
    DECS_Result = {}  # The detected community structure

    for r in range(num_repeat):
        num_timestep = W_Cube.shape[
            1
        ]  # W_Cube contains several cells restoring temporal adjacent matrices

        # DECS only optimizes the modularity at the 1st time step
        timestep_num = 0
        (
            dynMod[timestep_num, r],
            dynPop[(timestep_num, r)],
            DECS_Result[(timestep_num, r)],
            dynTime[timestep_num, r],
        ) = DECS_1(
            W_Cube[0, timestep_num],
            maxgen,
            pop_size,
            p_mutation,
            p_migration,
            p_mu_mi,
            PGLP_iter,
        )
        dynNmi[timestep_num, r] = NMI(
            torch.tensor(GT_Cube[timestep_num][0]),
            torch.tensor(DECS_Result[(timestep_num, r)]),
        )

        print(
            f"timestep = {timestep_num}, Modularity = {dynMod[timestep_num, r]:.4f}, NMI = {dynNmi[timestep_num, r]:.4f}"
        )

        # DECS optimizes the modularity and NMI in the following time steps
        for timestep_num in range(1, num_timestep):
            (
                dynMod[timestep_num, r],
                dynPop[(timestep_num, r)],
                DECS_Result[(timestep_num, r)],
                dynTime[timestep_num, r],
            ) = DECS_2(
                W_Cube[0, timestep_num],
                maxgen,
                pop_size,
                p_mutation,
                p_migration,
                p_mu_mi,
                num_neighbor,
                DECS_Result[(timestep_num - 1, r)],
                PGLP_iter,
            )
            dynNmi[timestep_num, r] = NMI(
                torch.tensor(GT_Cube[timestep_num][0]),
                torch.tensor(DECS_Result[(timestep_num, r)]),
            )

            print(
                f"timestep = {timestep_num}, Modularity = {dynMod[timestep_num, r]:.4f}, NMI = {dynNmi[timestep_num, r]:.4f}"
            )

    avg_dynMod = torch.mean(dynMod, dim=1)
    avg_dynNmi = torch.mean(dynNmi, dim=1)


if __name__ == "__main__":
    main()
