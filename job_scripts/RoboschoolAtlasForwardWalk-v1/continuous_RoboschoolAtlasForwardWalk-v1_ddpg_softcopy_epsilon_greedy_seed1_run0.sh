#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_RoboschoolAtlasForwardWalk-v1_ddpg_softcopy_epsilon_greedy_seed1_run0_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolAtlasForwardWalk-v1 --random-seed 1 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolAtlasForwardWalk-v1/ddpg_softcopy_epsilon_greedy_seed1_run0 --continuous-act-space-flag --double-ddpg-flag  

