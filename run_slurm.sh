#!/bin/bash

# --- Environment Setup ---
# Load necessary modules
module load cuda/12.2

source to/your/conda
conda activate venv

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
