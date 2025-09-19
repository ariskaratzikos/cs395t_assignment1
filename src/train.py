import argparse
import json
import csv
import os
import math
import time
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

from src.profiler import Profiler, DetailedLogger
from .dataset import get_tokenized_dataset
from .model import get_model_and_tokenizer

class CausalLMDataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        import torch
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


from torch.backends.cuda import sdp_kernel, SDPBackend

def _time_one_step(mdl, batch, device, backend: SDPBackend):
    # force specific backend for one forward+backward timing
    mdl.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with sdp_kernel(backend):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            out = mdl(**{k: v.to(device) for k, v in batch.items()})
            loss = out.loss
        loss.backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0  # ms

def sanity_sdpa_backends(model, batch, device):
    mdl = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # warmup
    for _ in range(2):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            mdl(**{k: v.to(device) for k, v in batch.items()}).loss.backward()
        mdl.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    for name, bk in [("FLASH", SDPBackend.FLASH_ATTENTION),
                     ("MEM_EFF", SDPBackend.EFFICIENT_ATTENTION),
                     ("MATH", SDPBackend.MATH)]:
        try:
            ms = _time_one_step(mdl, batch, device, bk)
            print(f"[SDPA={name}] one step: {ms:.2f} ms")
        except Exception as e:
            print(f"[SDPA={name}] not available: {repr(e)}")


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def set_seed(seed: int, local_rank: int):
    random.seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    torch.manual_seed(seed + local_rank)
    torch.cuda.manual_seed_all(seed + local_rank)

def run_training_loop(config, model, train_dataset, tokenizer, device, local_rank):
    if local_rank == 0:
        detailed_logger = DetailedLogger(config['run_name'])
    
    collator = CausalLMDataCollator(tokenizer)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=config['batch_size'],
        collate_fn=collator, sampler=train_sampler,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 3e-4))
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.get('gradient_accumulation_steps', 1))
    if 'max_steps' in config and config['max_steps'] > 0:
        max_steps = config['max_steps']
        num_epochs = math.ceil(max_steps / num_update_steps_per_epoch)
    else:
        num_epochs = config.get('num_epochs', 1)
        max_steps = num_epochs * num_update_steps_per_epoch

    if local_rank == 0:
        print(f"  - Batches per epoch: {len(train_dataloader)}")
        print(f"  - Gradient Accumulation Steps: {config.get('gradient_accumulation_steps', 1)}")
        print(f"  - Num Epochs: {num_epochs}")
        print(f"  - Total optimization steps: {max_steps}")

    lr_scheduler = get_scheduler(
        name=config.get('lr_scheduler', 'cosine'),
        optimizer=optimizer,
        num_warmup_steps=config.get('num_warmup_steps', 400),
        num_training_steps=max_steps
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=config.get('fp16', False))
    progress_bar = tqdm(range(max_steps), disable=(local_rank != 0), desc="Training")
    completed_steps, grad_accum_steps = 0, config.get('gradient_accumulation_steps', 1)
    logged_perplexities, logged_accuracies = [], []
    tokens_processed_local = 0

    model.train()
    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            if completed_steps >= max_steps: break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config.get('fp16', False)):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / grad_accum_steps

            labels = batch["labels"]
            tokens_this_batch = (labels[..., 1:] != -100).sum().item()
            tokens_processed_local += tokens_this_batch

            if not torch.isfinite(loss):
                if local_rank == 0: print(f"Skipping step {completed_steps} due to non-finite loss.")
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or step == len(train_dataloader) - 1:
                if config.get("max_grad_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % config.get('logging_steps', 250) == 0:
                    logits = outputs.logits
                    labels = batch["labels"]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_active_mask = shift_labels != -100
                    predictions = torch.argmax(shift_logits, dim=-1)
                    num_correct = (predictions.eq(shift_labels) & shift_active_mask).sum()
                    num_active_tokens = shift_active_mask.sum()

                    loss_tensor = torch.tensor([loss.item() * grad_accum_steps], device=device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss = loss_tensor.item() / dist.get_world_size()

                    correct_tensor = torch.tensor([num_correct.item()], device=device)
                    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
                    
                    active_tensor = torch.tensor([num_active_tokens.item()], device=device)
                    dist.all_reduce(active_tensor, op=dist.ReduceOp.SUM)

                    try:
                        perplexity = math.exp(avg_loss)
                    except OverflowError:
                        perplexity = float("inf")
                    
                    accuracy = correct_tensor.item() / active_tensor.item() if active_tensor.item() > 0 else 0.0
                    
                    logged_perplexities.append(perplexity)
                    logged_accuracies.append(accuracy)

                    if local_rank == 0:
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "ppl": f"{perplexity:.2f}", "acc": f"{accuracy:.3f}"})
                        detailed_logger.log({ "step": completed_steps, "loss": avg_loss, "perplexity": perplexity, "accuracy": accuracy, "lr": lr_scheduler.get_last_lr()[0] })
    dist.barrier()
    tokens_tensor = torch.tensor([tokens_processed_local], device=device, dtype=torch.float64)
    dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
    tokens_processed_total = int(tokens_tensor.item())
    if local_rank == 0:
        detailed_logger.close()
        
    avg_perplexity = sum(logged_perplexities) / len(logged_perplexities) if logged_perplexities else 'N/A'
    avg_accuracy = sum(logged_accuracies) / len(logged_accuracies) if logged_accuracies else 'N/A'
    
    return {
        "avg_perplexity": avg_perplexity,
        "avg_accuracy": avg_accuracy,
        "completed_steps": completed_steps,
        "world_size": dist.get_world_size(),
        "tokens_processed": tokens_processed_total
    }

def run_evaluation(config, model, eval_dataset, tokenizer, device, local_rank):
    """Runs evaluation using DDP with corrected metric calculation."""
    collator = CausalLMDataCollator(tokenizer)
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config['batch_size'], collate_fn=collator, 
        sampler=eval_sampler, num_workers=8, pin_memory=True, persistent_workers=True
    )

    model.eval()
    total_loss, total_correct, total_active_tokens = 0.0, 0, 0
    progress_bar = tqdm(range(len(eval_dataloader)), disable=(local_rank != 0), desc="Evaluating")

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config.get('fp16', False)):
            outputs = model(**batch)

        logits = outputs.logits
        labels = batch["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_active_mask = shift_labels != -100

        predictions = torch.argmax(shift_logits, dim=-1)
        total_correct += (predictions.eq(shift_labels) & shift_active_mask).sum().item()
        total_active_tokens += shift_active_mask.sum().item()

        batch_token_count = shift_active_mask.sum().item()
        total_loss += outputs.loss.item() * batch_token_count
        progress_bar.update(1)

    metrics_tensor = torch.tensor([total_loss, float(total_correct), float(total_active_tokens)], dtype=torch.float64, device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    
    total_loss_sum, total_correct_sum, total_active_tokens_sum = metrics_tensor.tolist()

    avg_loss = (total_loss_sum / total_active_tokens_sum) if total_active_tokens_sum > 0 else 0.0
    accuracy = (total_correct_sum / total_active_tokens_sum) if total_active_tokens_sum > 0 else 0.0
    try:
        perplexity = math.exp(avg_loss)
    except (OverflowError, ValueError):
        perplexity = float("inf")
    
    return {"eval_loss": avg_loss, "eval_perplexity": perplexity, "eval_accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser(description="Run a single training experiment.")
    parser.add_argument("--config", type=str, required=True, help="JSON string of the experiment config.")
    args = parser.parse_args()
    config = json.loads(args.config)

    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    
    try:
        set_seed(config.get("seed", 42), local_rank)

        if local_rank == 0:
            import torch.backends.cuda as bc
            print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
            print(f"flash_sdp_enabled={bc.flash_sdp_enabled()} mem_eff_sdp_enabled={bc.mem_efficient_sdp_enabled()} math_sdp_enabled={bc.math_sdp_enabled()}")

        model, tokenizer = get_model_and_tokenizer(config)
        model.to(device)
        model = DDP(model, device_ids=[local_rank])
        
        tokenized_datasets = get_tokenized_dataset(
            config['dataset_name'], config['dataset_config'], tokenizer, 
            config['sequence_length'], max_samples=config.get('max_train_samples')
        )

        if local_rank == 0 and config.get("use_flash_attention", False) and config.get("debug_flash", True):
            collator = CausalLMDataCollator(tokenizer)
            bsz = min(config.get("batch_size", 8), len(tokenized_datasets['train']))
            debug_features = [tokenized_datasets['train'][i] for i in range(bsz)]
            debug_batch = collator(debug_features)
            print("\n[Flash Debug] Probing SDPA backends on a single mini-batch...")
            sanity_sdpa_backends(model, debug_batch, device)
            print("[Flash Debug] Done.\n")
        dist.barrier()

        if local_rank == 0:
            print("Starting training loop...")
        
        with Profiler() as profiler:
            train_results = run_training_loop(config, model, tokenized_datasets['train'], tokenizer, device, local_rank)
        
        eval_results = {}
        if 'validation' in tokenized_datasets and len(tokenized_datasets['validation']) > 0:
            if local_rank == 0: print("\nStarting evaluation...")
            eval_results = run_evaluation(config, model.module, tokenized_datasets['validation'], tokenizer, device, local_rank)
            
        if local_rank == 0:
            print("\n--- TRAINING & EVALUATION COMPLETE ---")
            total_params_m = sum(p.numel() for p in model.module.parameters()) / 1_000_000
            # Token-based throughput: use total active tokens processed across all ranks
            tokens_total = train_results.get('tokens_processed', 0)
            throughput_tokens = tokens_total / profiler.total_time if profiler.total_time > 0 else 0

            def format_metric(metric):
                return f"{metric:.3f}" if isinstance(metric, float) else metric

            print(f"Training Time: {profiler.total_time:.2f} seconds")
            print(f"Throughput: {throughput_tokens:.2f} tokens/sec")
            print(f"Peak GPU Memory: {profiler.peak_memory_gb:.3f} GB")
            print(f"Average Train Perplexity: {format_metric(train_results.get('avg_perplexity'))}")
            print(f"Average Train Accuracy: {format_metric(train_results.get('avg_accuracy'))}")
            print(f"Eval Perplexity: {format_metric(eval_results.get('eval_perplexity'))}")
            print(f"Eval Accuracy: {format_metric(eval_results.get('eval_accuracy'))}")
            
            results_file = "results.csv"
            file_exists = os.path.isfile(results_file)
            
            # NOTE: we store tokens/sec into throughput_samples_per_sec for compatibility
            csv_data = {
                "run_name": config.get('run_name'),
                "total_params_m": total_params_m,
                "sequence_length": config.get('sequence_length'),
                "batch_size": config.get('batch_size'),
                "fp16": config.get('fp16', False),
                "grad_checkpoint": config.get('gradient_checkpointing', False),
                "flash_attention": config.get('use_flash_attention', False),
                "ddp_gpus": train_results.get('world_size'),
                "training_time_sec": profiler.total_time,
                "throughput_samples_per_sec": throughput_tokens,
                "peak_gpu_mem_gb": profiler.peak_memory_gb,
                "avg_perplexity": train_results.get('avg_perplexity'),
                "avg_accuracy": train_results.get('avg_accuracy'),
                "eval_loss": eval_results.get('eval_loss'),
                "eval_perplexity": eval_results.get('eval_perplexity'),
                "eval_accuracy": eval_results.get('eval_accuracy')
            }

            with open(results_file, 'a', newline='') as csvfile:
                fieldnames = list(csv_data.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in csv_data.items()})
            print(f"\nResults appended to {results_file}")
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()