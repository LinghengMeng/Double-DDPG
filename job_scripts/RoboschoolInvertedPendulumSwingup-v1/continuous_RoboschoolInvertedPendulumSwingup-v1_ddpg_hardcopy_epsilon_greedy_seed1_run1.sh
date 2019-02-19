#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_RoboschoolInvertedPendulumSwingup-v1_ddpg_hardcopy_epsilon_greedy_seed1_run1_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolInvertedPendulumSwingup-v1 --random-seed 1 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolInvertedPendulumSwingup-v1/ddpg_hardcopy_epsilon_greedy_seed1_run1 --continuous-act-space-flag --double-ddpg-flag --target-hard-copy-flag 

