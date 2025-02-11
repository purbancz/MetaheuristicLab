#!/bin/bash
#SBATCH --job-name=pso_parameters_tuning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid-now
#SBATCH --account=plglscclass24-cpu
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load modules and activate the conda environment
module load miniconda3
conda init
eval "$(conda shell.bash hook)"
conda activate jmetal

echo "PYTHON SCRIPT IS BEING EXECUTED"

# Run the Python script
python $HOME/GA-PSO_Hybrid/test/tuning_framework.py

echo "PSO tuning completed successfully."