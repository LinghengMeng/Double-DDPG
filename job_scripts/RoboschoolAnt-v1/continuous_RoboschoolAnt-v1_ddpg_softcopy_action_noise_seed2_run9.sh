#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_RoboschoolAnt-v1_ddpg_softcopy_action_noise_seed2_run9_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolAnt-v1 --random-seed 2 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolAnt-v1/ddpg_softcopy_action_noise_seed2_run9 --continuous-act-space-flag --double-ddpg-flag  

