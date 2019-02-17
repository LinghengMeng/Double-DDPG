#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:0       # request GPU generic resource
#SBATCH --cpus-per-task=1  #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=27:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_BipedalWalker-v2_doule_ddpg_softcopy_epsilon_greedy_seed4_run2_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env BipedalWalker-v2 --random-seed 4 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/BipedalWalker-v2/doule_ddpg_softcopy_epsilon_greedy_seed4_run2 --continuous-act-space-flag   

