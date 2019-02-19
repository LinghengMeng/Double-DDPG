#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=24:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_MountainCarContinuous-v0_ddpg_hardcopy_epsilon_greedy_seed4_run5_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env MountainCarContinuous-v0 --random-seed 4 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/MountainCarContinuous-v0/ddpg_hardcopy_epsilon_greedy_seed4_run5 --continuous-act-space-flag --double-ddpg-flag --target-hard-copy-flag 

