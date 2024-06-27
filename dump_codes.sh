#!/bin/bash

#SBATCH --job-name=dump_w2vbert-codes
#SBATCH --output=dump_w2vbert-codes_%j.out
#SBATCH --error=dump_w2vbert-codes_%j.err
#SBATCH --time=30:00:00 
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=mundus-mir-3

# Load module necessary for the initial environment
module load python/3.8

# Activate the Conda environment
source /mundus/abadawi696/miniconda3/etc/profile.d/conda.sh
conda activate slm

# Increase NumExpr max threads, if applicable
export NUMEXPR_MAX_THREADS=16

# Change directory to your project directory
cd /mundus/abadawi696/slm_project/slm-100h

/mundus/abadawi696/slm_project/slm-100h/train.tsv

echo "Extracting the codes..."
!python dump_w2vbert_codes.py "/mundus/abadawi696/slm_project/slm-100h/train.tsv" --split "train" --km_path "/mundus/abadawi696/slm_project/slm-100h/features/kmeans" --save-dir "/mundus/abadawi696/slm_project/slm-100h/codes" --layer 16 --nshard 1 --rank 0

# Deactivate Conda environment
conda deactivate

echo "Codes extraction completed."