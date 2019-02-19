#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_BipedalWalkerHardcore-v2_doule_ddpg_softcopy_epsilon_greedy_seed2_run1_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env BipedalWalkerHardcore-v2 --random-seed 2 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/BipedalWalkerHardcore-v2/doule_ddpg_softcopy_epsilon_greedy_seed2_run1 --continuous-act-space-flag   

