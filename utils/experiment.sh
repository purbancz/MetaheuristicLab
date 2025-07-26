#!/bin/bash
#SBATCH --job-name=CMAES_100dim
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid-now
#SBATCH --account=plglscclass24-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

echo "--- Slurm Environment Variables (from bash) ---"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_MEM_PER_CPU: $SLURM_MEM_PER_CPU"
echo "-------------------------------------------"


# Load modules and activate the conda environment
module load miniconda3
conda init
eval "$(conda shell.bash hook)"
conda activate jmetal12

# Set threading vars to 1 if using multiprocessing pool extensively
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Log start time
START_TIME=$(date +%s)
echo "Job started at: $(date -d @$START_TIME)"

# Run the Python script
echo "PYTHON SCRIPT IS BEING EXECUTED"
export PYTHONPATH="$HOME/GA-PSO_Hybrid:$PYTHONPATH"
srun python -u $HOME/GA-PSO_Hybrid/main.py
#python -u $HOME/GA-PSO_Hybrid/utils/plot_benchmarks.py
echo "Swarming completed successfully."

# Log end time
END_TIME=$(date +%s)
echo "Job finished at: $(date)"
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "Total execution time: $(date -u -d @$EXECUTION_TIME +%H:%M:%S)"