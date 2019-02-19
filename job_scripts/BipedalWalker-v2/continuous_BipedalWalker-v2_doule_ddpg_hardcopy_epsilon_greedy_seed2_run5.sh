#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_BipedalWalker-v2_doule_ddpg_hardcopy_epsilon_greedy_seed2_run5_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env BipedalWalker-v2 --random-seed 2 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/BipedalWalker-v2/doule_ddpg_hardcopy_epsilon_greedy_seed2_run5 --continuous-act-space-flag  --target-hard-copy-flag 

