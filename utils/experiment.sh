#!/bin/bash
#SBATCH --job-name=PSO_Shifted_Rotated_Weierstrass_1000dim_adaptive
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL


# Load modules and activate the conda environment
module load miniconda3
conda init
eval "$(conda shell.bash hook)"
conda activate jmetal12

# Log start time
START_TIME=$(date +%s)
echo "Job started at: $(date -d @$START_TIME)"

# Run the Python script
echo "PYTHON SCRIPT IS BEING EXECUTED"
export PYTHONPATH="$HOME/GA-PSO_Hybrid:$PYTHONPATH"
python -u $HOME/GA-PSO_Hybrid/main.py
#python -u $HOME/GA-PSO_Hybrid/utils/plot_benchmarks.py
echo "Swarming completed successfully."

# Log end time
END_TIME=$(date +%s)
echo "Job finished at: $(date)"
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "Total execution time: $(date -u -d @$EXECUTION_TIME +%H:%M:%S)"