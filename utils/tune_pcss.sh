#!/bin/bash
#SBATCH --job-name=irace_tune_universal
#SBATCH --output=slurm_tune_logs/%x_%j.out
#SBATCH --error=slurm_tune_logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=standard          # standard/fast/long/tesla
#SBATCH --account=pl0590-01
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=ALL

mkdir -p slurm_tune_logs

# Load modules and activate the conda environment
source ~/.bashrc
conda activate jmetal12

#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1

# Log start time
START_TIME=$(date +%s)
echo "Job started at: $(date -d @$START_TIME)"

# Run the Python script
echo "PYTHON SCRIPT IS BEING EXECUTED"
export PYTHONPATH="$HOME/GA-PSO_Hybrid:$PYTHONPATH"
python -u $HOME/GA-PSO_Hybrid/optimization/irace_tune_universal.py
echo "Tuning completed successfully."

# Log end time
END_TIME=$(date +%s)
echo "Job finished at: $(date)"
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "Total execution time: $(date -u -d @$EXECUTION_TIME +%H:%M:%S)"
