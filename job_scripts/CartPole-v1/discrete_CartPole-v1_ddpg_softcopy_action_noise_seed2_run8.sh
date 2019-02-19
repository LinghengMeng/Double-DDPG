#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=10:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/discrete_CartPole-v1_ddpg_softcopy_action_noise_seed2_run8_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env CartPole-v1 --random-seed 2 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/discrete/CartPole-v1/ddpg_softcopy_action_noise_seed2_run8  --double-ddpg-flag  

