#!/bin/bash
#SBATCH --job-name=combo_hotfix
#SBATCH --output=combo_hotfix_%j.out
#SBATCH --error=combo_hotfix_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # JAX usa una sola GPU di default
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.0
source ../.venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

python combo.py
