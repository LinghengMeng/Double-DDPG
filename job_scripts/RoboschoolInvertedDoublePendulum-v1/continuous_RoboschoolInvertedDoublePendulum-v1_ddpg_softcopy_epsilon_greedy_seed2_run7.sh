#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=/project/6001934/lingheng/Double_DDPG_Job_output/continuous_RoboschoolInvertedDoublePendulum-v1_ddpg_softcopy_epsilon_greedy_seed2_run7_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0 cuda cudnn
source ~/tf_cpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolInvertedDoublePendulum-v1 --random-seed 2 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolInvertedDoublePendulum-v1/ddpg_softcopy_epsilon_greedy_seed2_run7 --continuous-act-space-flag --double-ddpg-flag  

