#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=10:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_Pendulum-v0_ddpg_hardcopy_epsilon_greedy_seed5_run0_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env Pendulum-v0 --random-seed 5 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/Pendulum-v0/ddpg_hardcopy_epsilon_greedy_seed5_run0 --continuous-act-space-flag --double-ddpg-flag --target-hard-copy-flag 

