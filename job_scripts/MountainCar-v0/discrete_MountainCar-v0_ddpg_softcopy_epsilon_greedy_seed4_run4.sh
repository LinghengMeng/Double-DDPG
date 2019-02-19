#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=10:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/discrete_MountainCar-v0_ddpg_softcopy_epsilon_greedy_seed4_run4_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env MountainCar-v0 --random-seed 4 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/discrete/MountainCar-v0/ddpg_softcopy_epsilon_greedy_seed4_run4  --double-ddpg-flag  

