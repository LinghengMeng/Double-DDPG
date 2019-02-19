#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=10:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/discrete_MountainCar-v0_doule_ddpg_hardcopy_action_noise_seed1_run2_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env MountainCar-v0 --random-seed 1 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/discrete/MountainCar-v0/doule_ddpg_hardcopy_action_noise_seed1_run2   --target-hard-copy-flag 

