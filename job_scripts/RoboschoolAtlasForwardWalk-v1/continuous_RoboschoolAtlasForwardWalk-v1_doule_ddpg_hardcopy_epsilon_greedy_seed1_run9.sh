#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=/project/6001934/lingheng/Double_DDPG_Job_output/continuous_RoboschoolAtlasForwardWalk-v1_doule_ddpg_hardcopy_epsilon_greedy_seed1_run9_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0 cuda cudnn
source ~/tf_cpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolAtlasForwardWalk-v1 --random-seed 1 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolAtlasForwardWalk-v1/doule_ddpg_hardcopy_epsilon_greedy_seed1_run9 --continuous-act-space-flag  --target-hard-copy-flag 

