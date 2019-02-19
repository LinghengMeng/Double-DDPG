#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_RoboschoolHumanoidFlagrun-v1_ddpg_hardcopy_action_noise_seed4_run3_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolHumanoidFlagrun-v1 --random-seed 4 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolHumanoidFlagrun-v1/ddpg_hardcopy_action_noise_seed4_run3 --continuous-act-space-flag --double-ddpg-flag --target-hard-copy-flag 

