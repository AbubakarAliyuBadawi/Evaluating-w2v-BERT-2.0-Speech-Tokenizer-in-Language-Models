#!/bin/bash

#SBATCH --job-name=w2vbert-emb
#SBATCH --output=w2vbert-emb_%j.out
#SBATCH --error=w2vbert-emb_%j.err
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

# Adjust the paths to your specific needs for train and test TSV files and output directories
TRAIN_TSV="/mundus/abadawi696/slm_project/slm-100h/train.tsv"
TEST_TSV="/mundus/abadawi696/slm_project/slm-100h/test.tsv"
TRAIN_OUTPUT="/train"
TEST_OUTPUT="/test"
FEATURES_DIR="/mundus/abadawi696/slm_project/slm-100h/features"

echo "Extracting train embeddings..."
/mundus/abadawi696/miniconda3/envs/slm/bin/python dump_w2vbert_feature.py "$TRAIN_TSV" "$TRAIN_OUTPUT" "$FEATURES_DIR"

echo "Extracting test embeddings..."
/mundus/abadawi696/miniconda3/envs/slm/bin/python dump_w2vbert_feature.py "$TEST_TSV" "$TEST_OUTPUT" "$FEATURES_DIR"

# Deactivate Conda environment
conda deactivate

echo "Embeddings extraction completed."
