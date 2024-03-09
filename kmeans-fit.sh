#!/bin/bash

#SBATCH --job-name=kmeans-fit
#SBATCH --output=kmeans-fit_%j.out
#SBATCH --error=kmeans-fit_%j.err
#SBATCH --time=40:00:00 
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=mundus-mir-3

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

# Adjust the paths to your specific needs for train and test TSV files and output directories
TRAIN_PATH="/mundus/abadawi696/slm_project/slm-100h/features"
TEST_PATH="/test"
KM_DIR="/mundus/abadawi696/slm_project/slm-100h/features/kmeans"

# Fitting the kmeans algorithm
echo "Fitting the KMeans algorithm..."

/mundus/abadawi696/miniconda3/envs/slm/bin/python learn_kmeans.py "$TRAIN_PATH" "$TEST_PATH" "$KM_DIR" --percent -1

# Deactivate Conda environment
conda deactivate

echo "kmeans completed."
