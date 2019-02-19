#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_LunarLanderContinuous-v2_doule_ddpg_softcopy_action_noise_seed1_run6_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env LunarLanderContinuous-v2 --random-seed 1 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/continuous/LunarLanderContinuous-v2/doule_ddpg_softcopy_action_noise_seed1_run6 --continuous-act-space-flag   

