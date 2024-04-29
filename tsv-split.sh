#!/bin/bash

#SBATCH --job-name=split
#SBATCH --output=split_%j.out
#SBATCH --error=split_%j.err
#SBATCH --time=90:00:00 
#SBATCH --partition=mundus,all
#SBATCH --gpus-per-node=a100-10:1

# Load module necessary for the initial environment
module load python/3.8
module load cuda/11.8

# Activate the Conda environment
source /mundus/abadawi696/miniconda3/etc/profile.d/conda.sh
conda activate slm

cd /mundus/abadawi696/slm_project/slm-60k

echo "spliting..."

/mundus/abadawi696/miniconda3/envs/slm/bin/python train-splits.py

# Deactivate Conda environment
conda deactivate

echo "done"
