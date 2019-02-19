#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --mem=8000M            # memory per node
#SBATCH --time=23:00:00           # time (DD-HH:MM)
#SBATCH --output=../Double_DDPG_Job_output/continuous_RoboschoolInvertedPendulum-v1_doule_ddpg_softcopy_action_noise_seed5_run1_%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn
source ~/tf_gpu/bin/activate

python ./ddpg_discrete_action.py --env RoboschoolInvertedPendulum-v1 --random-seed 5 --exploration-strategy action_noise --summary-dir ../Double_DDPG_Results_no_monitor/continuous/RoboschoolInvertedPendulum-v1/doule_ddpg_softcopy_action_noise_seed5_run1 --continuous-act-space-flag   

