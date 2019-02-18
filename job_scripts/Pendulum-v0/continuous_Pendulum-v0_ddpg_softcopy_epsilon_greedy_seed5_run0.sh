#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:0       # request GPU generic resource
#SBATCH --cpus-per-task=1  #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=27:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_Pendulum-v0_ddpg_softcopy_epsilon_greedy_seed5_run0_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env Pendulum-v0 --random-seed 5 --exploration-strategy epsilon_greedy --summary-dir ../Double_DDPG_Results_no_monitor/continuous/Pendulum-v0/ddpg_softcopy_epsilon_greedy_seed5_run0 --continuous-act-space-flag --double-ddpg-flag  

