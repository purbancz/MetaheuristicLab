#!/bin/bash
#SBATCH --job-name=new_algorithms_all_problems_100dim
#SBATCH --output=slurm_logs/%x_%j.out # Log directory
#SBATCH --error=slurm_logs/%x_%j.err  # Log directory
#SBATCH --time=72:00:00           # Adjust time as needed
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu # Your account

# --- Request 1 task with multiple CPUs ---
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # Request only ONE main task (your python script)
#SBATCH --cpus-per-task=25         # Give that task ALL 48 cores requested
#SBATCH --mem=8G                  # Total memory for the single task (adjust if needed)
#SBATCH --mail-type=ALL

# --- Create Log Directory ---
mkdir -p slurm_logs

# --- Environment Setup ---
echo "--- Slurm Environment Variables (from bash) ---"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "-------------------------------------------"

echo "Loading modules..."
module load miniconda3

echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate jmetal12 # Your environment name
echo "Python executable: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# --- Set Threading Limits (Crucial for multiprocessing!) ---
# Prevent libraries like NumPy/MKL from oversubscribing cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Set PYTHONPATH ---
export PYTHONPATH="$HOME/GA-PSO_Hybrid:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# --- Run the Python Script ---
START_TIME=$(date +%s)
echo "Job started at: $(date -d @$START_TIME)"
echo "PYTHON SCRIPT IS BEING EXECUTED (Single Instance with $SLURM_CPUS_PER_TASK cores available)"

# *** srun to launch the single task (still good practice) ***
# or just run python directly here, as ntasks=1
# python -u $HOME/GA-PSO_Hybrid/main.py
srun python -u $HOME/GA-PSO_Hybrid/main_array.py

EXIT_CODE=$?
echo "Python script exited with code: $EXIT_CODE"

END_TIME=$(date +%s)
echo "Job finished at: $(date -d @$END_TIME)"
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "Total execution time: $(date -u -d @$EXECUTION_TIME +%H:%M:%S)"

exit $EXIT_CODE