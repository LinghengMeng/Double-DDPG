#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=/project/6001934/lingheng/Double_DDPG_Job_output/continuous_RoboschoolWalker2d-v1_doule_ddpg_softcopy_epsilon_greedy_seed5_run5_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0 cuda cudnn
source ~/tf_cpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolWalker2d-v1 --random-seed 5 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolWalker2d-v1/doule_ddpg_softcopy_epsilon_greedy_seed5_run5 --continuous-act-space-flag   

