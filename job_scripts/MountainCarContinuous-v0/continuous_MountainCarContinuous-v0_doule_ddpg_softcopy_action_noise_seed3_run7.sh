#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=24:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_MountainCarContinuous-v0_doule_ddpg_softcopy_action_noise_seed3_run7_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env MountainCarContinuous-v0 --random-seed 3 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/continuous/MountainCarContinuous-v0/doule_ddpg_softcopy_action_noise_seed3_run7 --continuous-act-space-flag   

