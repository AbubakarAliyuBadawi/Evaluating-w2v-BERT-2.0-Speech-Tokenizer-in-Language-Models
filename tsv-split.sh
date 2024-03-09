#!/bin/bash

#SBATCH --job-name=tsv-split
#SBATCH --output=slm_emb_%j.out
#SBATCH --error=slm_emb_%j.err
#SBATCH --time=30:00:00 
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=mundus-mir-2

# Load module necessary for the initial environment
module load python/3.8
module load cuda/11.0

# Activate the Conda environment
source /mundus/abadawi696/miniconda3/etc/profile.d/conda.sh
conda activate slm

# Increase NumExpr max threads, if applicable
export NUMEXPR_MAX_THREADS=16

# Change directory to your project directory
cd /mundus/abadawi696/slm_project/slm-100h

# Run your Python script
/mundus/abadawi696/miniconda3/envs/slm/bin/python tsv-split-100h.py

# Deactivate Conda environment
conda deactivate