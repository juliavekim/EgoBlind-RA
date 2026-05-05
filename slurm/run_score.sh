#!/bin/bash
#SBATCH --job-name=score
#SBATCH --partition=mit_normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/score_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/score_%j.err

source ~/.bashrc
conda activate egoblind
python ~/egoblind-ra/scripts/compute_all_methods_loss.py
