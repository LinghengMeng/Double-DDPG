#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_RoboschoolHalfCheetah-v1_doule_ddpg_hardcopy_epsilon_greedy_seed1_run9_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolHalfCheetah-v1 --random-seed 1 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolHalfCheetah-v1/doule_ddpg_hardcopy_epsilon_greedy_seed1_run9 --continuous-act-space-flag  --target-hard-copy-flag 

