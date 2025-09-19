# CS395T Assignment 1: Scalable Transformer Training & Profiling

Minimal, reproducible setup to profile a causal Transformer under different settings (sequence length, batch size, precision, gradient checkpointing, DDP, Flash/SDPA), and to analyze throughput, memory, and quality.

## Setup

1) Create an environment and install deps
```bash
conda create -n transformer_ass1 python=3.11 -y
conda activate transformer_ass1
pip install -r requirements.txt
```

## How it works
- Training uses pure PyTorch DDP (`src/train.py`), mixed precision optional.
- Model is a custom vanilla Transformer (`src/vanilla_transformer.py`) with causal attention and PAD masking.
- Datasets are tokenized/cached per sequence length (`processed_datasets/…`).
- Per-step training metrics are written to `results/details/<run_name>_metrics.csv` with columns: `step,loss,perplexity,accuracy,lr`.
- Each run appends a summary row to `results.csv` including tokens/sec throughput, memory, and eval metrics.

## Run experiments
Use tags to pick subsets (see `src/experiment_generator.py` for definitions).

Run everything:
```bash
python run_experiments.py
```

Run by tags (examples):
```bash
# Sequence length sweep
python run_experiments.py --tags seq_len

# Batch size sweeps (FP32 and FP16)
python run_experiments.py --tags bs_fp32 bs_fp16

# Optimization toggles (baseline, grad checkpointing, flash/SDPA)
python run_experiments.py --tags opts vanilla_opts
```

Outputs:
- Per-run console/logs: `logs/<run_name>.log`
- Detailed training curves: `results/details/<run_name>_metrics.csv`
- Summary rows: `results.csv`

## Analyze results
Generate plots:
```bash
python analysis.py
```
Artifacts go to `results/`:
- `memory_vs_sequence_length.png`
- `throughput_vs_batch_size.png` (tokens/sec)
- `optimization_*_comparison.png`
- `throughput_vs_model_size.png`
- `perplexity_over_time.png`
- Tables in `results/tables/`

## Tips & troubleshooting
- If you change tokenization/sequence length logic, delete `processed_datasets/` for a clean re-tokenization.
- Eval metrics are averaged per token across the full validation set; prefer these over noisy train-step snapshots to judge convergence.
- Throughput in `results.csv` is tokens/sec aggregated over all DDP ranks.
- If a detailed metrics CSV is empty for a very short run, re-run with a lower `logging_steps` so at least one point is logged.

