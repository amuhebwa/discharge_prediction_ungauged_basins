#!/bin/bash
#
#BATCH --job-name=rivers_training
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amuhebwa@umass.edu
#SBATCH -p titanx-long
#SBATCH --mem-per-cpu=10G
#module load cudnn/7.6-cuda_9.2

# srun -n1 python3 heldout_prediction_atStation.py
# srun -n1 python3 heldout_prediction_lumped.py
# srun -n1 python3 heldout_prediction_distributed.py
