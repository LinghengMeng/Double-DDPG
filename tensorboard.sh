#!/bin/bash
#SBATCH --gres=gpu:0        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M        # memory per node
#SBATCH --time=00:20:00      # time (hours:minutes:seconds)

source ~/tf_gpu/bin/activate
tensorboard --logdir=../Double_DDPG_Results --host 0.0.0.0