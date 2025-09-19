import argparse
import subprocess
import os
import json
import sys
from pathlib import Path
from src.experiment_generator import generate_experiments

def run_all_experiments(tags=None):
    """
    Generates and runs all experiments, logging output only to files.
    """
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    experiments = generate_experiments()
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    if tags:
        print(f"Filtering experiments with tags: {tags}")
        experiments = [exp for exp in experiments if exp.get('tag') in tags]

    print(f"Found {len(experiments)} experiments to run.")

    for i, exp_config in enumerate(experiments):
        run_name = exp_config['run_name']
        print(f"\n--- Running Experiment {i+1}/{len(experiments)}: {run_name} ---")

        num_processes = exp_config.get('num_gpus', 1)
        
        cmd = [
            sys.executable,
            "-m", "torch.distributed.run",
            "--standalone",
            "--nproc_per_node", str(num_processes),
            "-m", "src.train"
        ]

        cmd.append("--config")
        cmd.append(json.dumps(exp_config))
        
        log_path = os.path.join(log_dir, f"{run_name}.log")
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Logging to: {log_path}") # This will still print to the console

        try:
            exp_env = os.environ.copy()
            exp_env["HF_HOME"] = "/work/10906/arisk/ls6/.cache"
            
            # --- MODIFICATION START ---
            # Open the log file first
            with open(log_path, 'w') as log_file:
                # Tell Popen to write stdout and stderr directly to the log file
                process = subprocess.Popen(
                    cmd, 
                    stdout=log_file, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    env=exp_env
                )
            
            # Wait for the process to complete
            process.wait()
            # --- MODIFICATION END ---

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            print(f"--- Experiment {run_name} completed successfully. ---")
        except subprocess.CalledProcessError as e:
            print(f"!!! Experiment {run_name} failed with exit code {e.returncode}. Check log for details: {log_path} !!!")
        except KeyboardInterrupt:
            print("\nAborting experiment, terminating subprocess...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("\nAborting all experiments.")
            return

    print("\nAll experiments complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Transformer profiling experiments.")
    parser.add_argument(
        "--tags",
        nargs='+',
        help="Filter experiments to run only those matching the given tags (e.g., seq_len, bs_fp16)."
    )
    args = parser.parse_args()
    run_all_experiments(tags=args.tags)