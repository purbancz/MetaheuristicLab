#!/bin/bash
#SBATCH --job-name=pso_parameters_tuning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=plgrid-now
#SBATCH --account=plglscclass24-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


# Load modules and activate the conda environment
module load miniconda3
conda init
eval "$(conda shell.bash hook)"
conda activate jmetal

# Log start time
START_TIME=$(date +%s)
echo "Job started at: $(date -d @$START_TIME)"

# Run the Python script
echo "PYTHON SCRIPT IS BEING EXECUTED"
export PYTHONPATH="$HOME/GA-PSO_Hybrid:$PYTHONPATH"
python -u $HOME/GA-PSO_Hybrid/test/ray_tune.py
echo "Tuning completed successfully."

# Log end time
END_TIME=$(date +%s)
echo "Job finished at: $(date)"
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "Total execution time: $(date -u -d @$EXECUTION_TIME +%H:%M:%S)"