#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=10:00:00           # time (DD-HH:MM)
#SBATCH --output=/project/6001934/lingheng/Double_DDPG_Job_output/discrete_CartPole-v1_ddpg_softcopy_action_noise_seed3_run2_%N-%j.out  # %N for node name, %j for jobID

module load qt/5.9.6 python/3.6.3 nixpkgs/16.09  gcc/7.3.0 boost/1.68.0 cuda cudnn
source ~/tf_cpu/bin/activate

python ./ddpg_discrete_action.py --env CartPole-v1 --random-seed 3 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/discrete/CartPole-v1/ddpg_softcopy_action_noise_seed3_run2  --double-ddpg-flag  

