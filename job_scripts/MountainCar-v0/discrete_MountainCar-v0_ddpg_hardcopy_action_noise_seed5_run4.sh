#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=10:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/discrete_MountainCar-v0_ddpg_hardcopy_action_noise_seed5_run4_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env MountainCar-v0 --random-seed 5 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/discrete/MountainCar-v0/ddpg_hardcopy_action_noise_seed5_run4  --double-ddpg-flag --target-hard-copy-flag 

