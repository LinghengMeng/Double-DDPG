#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=24:00:00           # time (DD-HH:MM)
#SBATCH --output=/project/6001934/lingheng/Double_DDPG_Job_output/continuous_MountainCarContinuous-v0_doule_ddpg_hardcopy_action_noise_seed3_run0_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0 cuda cudnn
source ~/tf_cpu/bin/activate

python ./ddpg_discrete_action.py --env MountainCarContinuous-v0 --random-seed 3 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/continuous/MountainCarContinuous-v0/doule_ddpg_hardcopy_action_noise_seed3_run0 --continuous-act-space-flag  --target-hard-copy-flag 

