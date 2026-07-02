#!/bin/bash
#SBATCH --job-name=qspoof
#SBATCH --output=qspoof_%j.out
#SBATCH --error=qspoof_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4           # Richiede 4 GPU
#SBATCH --mem=128G             # Sufficiente grazie al chunking
#SBATCH --cpus-per-task=8      # Scalato per alimentare le 4 GPU

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.0.0

source ../.venv/bin/activate

# Lancia il processo multi-GPU. 
# --nproc_per_node deve coincidere con il numero di --gres=gpu
torchrun --standalone --nproc_per_node=4 neural_network2.py