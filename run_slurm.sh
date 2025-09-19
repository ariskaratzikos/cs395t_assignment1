#!/bin/bash
#SBATCH -J transformer_profiling  # Job name
#SBATCH -o logs/slurm_out_%j.log  # Standard output and error log (%j expands to jobID)
#SBATCH -p gpu-a100-dev               # Partition to submit to
#SBATCH -N 1                      # Request 1 node
#SBATCH -n 1                      # Run a single task
#SBATCH --mem=80G                 # Request 32 GB of memory
#SBATCH -t 2:00:00               # Job run time (hh:mm:ss)

# --- Environment Setup ---
# Load necessary modules
module load cuda/12.2

source /work/10906/arisk/conda/etc/profile.d/conda.sh
conda activate /work/10906/arisk/conda/envs/transformer_ml

cd /work/10906/arisk/ls6/FND_ML/cs395t_assignment1

# Ensure the log directory exists
mkdir -p logs

# Run the main experiment script
# This will execute all experiments defined in config.yaml
echo "Starting all experiments..."
python run_experiments.py --tags vanilla_opts

# To run only a specific set of experiments, you can use the --tags argument.
# For example, to run only the DDP and model size experiments:
# python run_experiments.py --tags ddp model_size

echo "All experiments complete."
