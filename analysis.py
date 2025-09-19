import argparse
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _ensure_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_bool(df, cols):
    for c in cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
    return df

def _save_table(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[table] saved -> {out_path}")

def _save_plot(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")

def _intersect(a, b):
    return sorted(list(set(a).intersection(set(b))))

def plot_memory_vs_seq_len(df, outdir):
    sel = df[df["run_name"].str.startswith("seq_len_")].copy()
    if sel.empty:
        print("No seq_len_* rows found; skipping memory_vs_sequence_length.")
        return None
    sel = sel.sort_values("sequence_length")
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(sel["sequence_length"], sel["peak_gpu_mem_gb"], marker="o")
    ax.set_title("GPU Memory vs. Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.grid(True)
    out = os.path.join(outdir, "memory_vs_sequence_length.png")
    _save_plot(fig, out)
    tbl = sel[["sequence_length", "batch_size", "fp16", "peak_gpu_mem_gb"]]
    _save_table(tbl, os.path.join(outdir, "tables", "memory_vs_sequence_length.csv"))
    return out

def plot_memory_vs_batch_size(df, outdir):
    sel = df[df["run_name"].str.startswith(("bs_fp32_", "bs_fp16_"))].copy()
    if sel.empty:
        print("No bs_fp32_/bs_fp16_ rows found; skipping memory_vs_batch_size.")
        return None
    fig = plt.figure()
    ax = fig.gca()
    fp32 = sel[sel["fp16"] == False].sort_values("batch_size")
    fp16 = sel[sel["fp16"] == True].sort_values("batch_size")
    if not fp32.empty:
        ax.plot(fp32["batch_size"], fp32["peak_gpu_mem_gb"], marker="o", label="FP32")
    if not fp16.empty:
        ax.plot(fp16["batch_size"], fp16["peak_gpu_mem_gb"], marker="s", linestyle="--", label="FP16")
    ax.set_title("GPU Memory vs. Batch Size")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.grid(True)
    ax.legend()
    out = os.path.join(outdir, "memory_vs_batch_size.png")
    _save_plot(fig, out)
    rows = []
    bss = sorted(sel["batch_size"].unique())
    for bs in bss:
        r = {"batch_size": bs}
        sub32 = sel[(sel["batch_size"]==bs) & (sel["fp16"]==False)]
        sub16 = sel[(sel["batch_size"]==bs) & (sel["fp16"]==True)]
        r["fp32_peak_mem_gb"] = sub32["peak_gpu_mem_gb"].mean() if not sub32.empty else np.nan
        r["fp16_peak_mem_gb"] = sub16["peak_gpu_mem_gb"].mean() if not sub16.empty else np.nan
        rows.append(r)
    tbl = pd.DataFrame(rows)
    _save_table(tbl, os.path.join(outdir, "tables", "memory_vs_batch_size.csv"))
    return out

def plot_throughput_vs_batch_size(df, outdir):
    sel = df[df["run_name"].str.startswith(("bs_fp32_", "bs_fp16_"))].copy()
    if sel.empty:
        print("No bs_fp32_/bs_fp16_ rows found; skipping throughput_vs_batch_size.")
        return None
    fig = plt.figure()
    ax = fig.gca()
    fp32 = sel[sel["fp16"] == False].sort_values("batch_size")
    fp16 = sel[sel["fp16"] == True].sort_values("batch_size")
    if not fp32.empty:
        ax.plot(fp32["batch_size"], fp32["throughput_samples_per_sec"], marker="o", label="FP32")
    if not fp16.empty:
        ax.plot(fp16["batch_size"], fp16["throughput_samples_per_sec"], marker="s", linestyle="--", label="FP16")
    ax.set_title("Throughput vs. Batch Size (FP32 vs FP16)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.grid(True)
    ax.legend()
    out = os.path.join(outdir, "throughput_vs_batch_size.png")
    _save_plot(fig, out)
    rows = []
    for bs in sorted(sel["batch_size"].unique()):
        row = {"batch_size": bs}
        s32 = sel[(sel["batch_size"]==bs) & (sel["fp16"]==False)]
        s16 = sel[(sel["batch_size"]==bs) & (sel["fp16"]==True)]
        row["fp32_throughput"] = s32["throughput_samples_per_sec"].mean() if not s32.empty else np.nan
        row["fp16_throughput"] = s16["throughput_samples_per_sec"].mean() if not s16.empty else np.nan
        rows.append(row)
    tbl = pd.DataFrame(rows)
    _save_table(tbl, os.path.join(outdir, "tables", "throughput_vs_batch_size.csv"))
    return out

def plot_mixed_precision_bars(df, outdir):
    sel = df[df["run_name"].str.startswith(("bs_fp32_", "bs_fp16_"))].copy()
    if sel.empty:
        print("No bs_fp32_/bs_fp16_ rows found; skipping mixed_precision_* plots.")
        return (None, None)
    bs32 = sel[sel["fp16"]==False]["batch_size"].unique()
    bs16 = sel[sel["fp16"]==True]["batch_size"].unique()
    common_bs = sorted(list(set(bs32).intersection(set(bs16))))
    if not common_bs:
        print("No common batch sizes across precisions; skipping mixed precision bars.")
        return (None, None)
    # Throughput bars
    fig_t = plt.figure()
    ax_t = fig_t.gca()
    x = np.arange(len(common_bs))
    width = 0.35
    thr32 = [sel[(sel["batch_size"]==b)&(sel["fp16"]==False)]["throughput_samples_per_sec"].mean() for b in common_bs]
    thr16 = [sel[(sel["batch_size"]==b)&(sel["fp16"]==True)]["throughput_samples_per_sec"].mean() for b in common_bs]
    ax_t.bar(x - width/2, thr32, width, label="FP32")
    ax_t.bar(x + width/2, thr16, width, label="FP16")
    ax_t.set_title("Mixed Precision: Throughput by Batch Size")
    ax_t.set_xlabel("Batch Size")
    ax_t.set_ylabel("Throughput (tokens/sec)")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels([str(b) for b in common_bs])
    ax_t.legend()
    ax_t.grid(True, axis="y")
    out_thr = os.path.join(outdir, "mixed_precision_throughput_by_batch.png")
    _save_plot(fig_t, out_thr)
    # Memory bars
    fig_m = plt.figure()
    ax_m = fig_m.gca()
    mem32 = [sel[(sel["batch_size"]==b)&(sel["fp16"]==False)]["peak_gpu_mem_gb"].mean() for b in common_bs]
    mem16 = [sel[(sel["batch_size"]==b)&(sel["fp16"]==True)]["peak_gpu_mem_gb"].mean() for b in common_bs]
    ax_m.bar(x - width/2, mem32, width, label="FP32")
    ax_m.bar(x + width/2, mem16, width, label="FP16")
    ax_m.set_title("Mixed Precision: Memory by Batch Size")
    ax_m.set_xlabel("Batch Size")
    ax_m.set_ylabel("Peak GPU Memory (GB)")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels([str(b) for b in common_bs])
    ax_m.legend()
    ax_m.grid(True, axis="y")
    out_mem = os.path.join(outdir, "mixed_precision_memory_by_batch.png")
    _save_plot(fig_m, out_mem)
    rows = []
    for b, t32, t16, m32, m16 in zip(common_bs, thr32, thr16, mem32, mem16):
        rows.append({
            "batch_size": b,
            "throughput_fp16_vs_fp32_x": (t16 / t32) if (t32 and not math.isclose(t32,0.0)) else np.nan,
            "memory_fp16_change_pct": 100.0 * (m16 - m32) / m32 if (m32 and not math.isclose(m32,0.0)) else np.nan
        })
    tbl = pd.DataFrame(rows)
    _save_table(tbl, os.path.join(outdir, "tables", "mixed_precision_speedups.csv"))
    return (out_thr, out_mem)

def plot_optimization_comparisons(df, outdir):
    """
    Compare Baseline vs Gradient Checkpointing vs FlashAttention.
    - Vertical bars (no side labels)
    - Short names in legend: 'Baseline', 'Grad Ckpt', 'FlashAttn'
    - Consistent colors across throughput & memory plots
    - Value labels on bars
    """
    order = ["vanilla_baseline", "vanilla_gradient_checkpointing", "opt_flash_attention"]
    short = {
        "vanilla_baseline": "Baseline",
        "vanilla_gradient_checkpointing": "Grad Ckpt",
        "opt_flash_attention": "FlashAttn",
    }
    colors = {
        "Baseline":   "#4C78A8",
        "Grad Ckpt":  "#F58518",
        "FlashAttn":  "#54A24B",
    }

    sel = df[df["run_name"].isin(order)].copy()
    if sel.empty:
        print("No baseline/gc/flash rows found; skipping optimization comparisons.")
        return (None, None)

    # Aggregate (mean in case there are multiple rows per run)
    agg = (sel.groupby("run_name", as_index=False)
              .agg(throughput=("throughput_samples_per_sec", "mean"),
                   peak_mem_gb=("peak_gpu_mem_gb", "mean")))

    # Reorder rows to our fixed display order
    agg["__ord"] = agg["run_name"].map({k:i for i,k in enumerate(order)})
    agg = agg.sort_values("__ord")

    # Helper: draw vertical bars with legend + value labels
    def _bar_plot(values, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(8, 5))
        handles = []
        labels = []
        xs = np.arange(len(agg))
        for i, row in enumerate(agg.itertuples(index=False)):
            name = getattr(row, "run_name")
            disp = short[name]
            color = colors[disp]
            val = values[i]
            bar = ax.bar(i, val, color=color, width=0.6)
            # value label
            ax.bar_label(bar, fmt="%.0f", padding=3, fontsize=9)
            # legend bookkeeping
            handles.append(bar[0])
            labels.append(disp)

        # axes formatting
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])  # no names on the side
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # unique legend in original order
        uniq = []
        uniq_labels = []
        for h, lab in zip(handles, labels):
            if lab not in uniq_labels:
                uniq.append(h)
                uniq_labels.append(lab)
        ax.legend(uniq, uniq_labels, title="Optimization", loc="upper right")

        out_path = os.path.join(outdir, fname)
        _save_plot(fig, out_path)
        return out_path

    # Build vectors in display order
    thr_vals = [agg.loc[agg["run_name"] == r, "throughput"].values[0] for r in order if r in agg["run_name"].values]
    mem_vals = [agg.loc[agg["run_name"] == r, "peak_mem_gb"].values[0] for r in order if r in agg["run_name"].values]

    out_thr = _bar_plot(thr_vals, "Throughput (tokens/sec)",
                        "Optimization Comparison — Throughput",
                        "optimization_throughput_comparison.png")

    out_mem = _bar_plot(mem_vals, "Peak GPU Memory (GB)",
                        "Optimization Comparison — Peak GPU Memory",
                        "optimization_memory_comparison.png")

    # Save a tidy table with short names too
    tidy = agg[["run_name", "throughput", "peak_mem_gb"]].copy()
    tidy.insert(0, "name", tidy["run_name"].map(short))
    tidy = tidy.drop(columns=["run_name"]).rename(columns={"peak_mem_gb": "peak_gpu_mem_gb"})
    _save_table(tidy, os.path.join(outdir, "tables", "optimization_comparison.csv"))

    return (out_thr, out_mem)

def plot_throughput_vs_model_size(df, outdir):
    sel = df[df["run_name"].str.startswith("model_")].copy()
    if sel.empty:
        print("No model_* rows found; skipping throughput_vs_model_size.")
        return None
    sel = sel.sort_values("total_params_m")
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(sel["total_params_m"], sel["throughput_samples_per_sec"], marker="o")
    for _, r in sel.iterrows():
        ax.text(r["total_params_m"], r["throughput_samples_per_sec"], f'{int(round(r["total_params_m"]))}M', ha="center", va="bottom")
    ax.set_title("Throughput vs. Model Size")
    ax.set_xlabel("Total Parameters (Millions)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.grid(True)
    out = os.path.join(outdir, "throughput_vs_model_size.png")
    _save_plot(fig, out)
    return out

def plot_time_series(details_dir, outdir):
    if not os.path.isdir(details_dir):
        print(f"No details dir found @ {details_dir}; skipping time series.")
        return None
    files = [f for f in os.listdir(details_dir) if f.endswith("_metrics.csv")]
    if not files:
        print(f"No *_metrics.csv in {details_dir}; skipping time series.")
        return None
    fig = plt.figure()
    ax = fig.gca()
    for f in files:
        path = os.path.join(details_dir, f)
        try:
            d = pd.read_csv(path)
            if "perplexity" in d.columns and "step" in d.columns:
                ax.plot(d["step"], d["perplexity"], label=f.replace("_metrics.csv",""))
        except Exception as e:
            print(f"Could not process {f}: {e}")
    ax.set_title("Perplexity During Training")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    out = os.path.join(outdir, "perplexity_over_time.png")
    _save_plot(fig, out)
    return out

def plot_eval_over_steps_by_model_size(details_dir, outdir):
    """
    Reads results/details/model_*_metrics.csv files and plots:
      - model_size_perplexity_over_steps.png
      - model_size_accuracy_over_steps.png
    X-axis = training steps; one line per model size with legend like '67M', '95M', ...
    """
    if not os.path.isdir(details_dir):
        print(f"No details dir @ {details_dir}; skipping model-size over-steps plots.")
        return (None, None)

    files = [f for f in os.listdir(details_dir)
             if f.startswith("model_") and f.endswith("_metrics.csv")]
    if not files:
        print("No model_*_metrics.csv files found; skipping.")
        return (None, None)

    # Perplexity
    fig_p = plt.figure()
    axp = fig_p.gca()
    for fname in sorted(files):
        path = os.path.join(details_dir, fname)
        try:
            d = pd.read_csv(path)
            if {"step", "perplexity"} <= set(d.columns):
                base = fname.replace("_metrics.csv", "")
                size_tag = base.split("_")[-1]  # e.g. '67M'
                label = size_tag if size_tag.upper().endswith("M") else base
                axp.plot(d["step"], d["perplexity"], label=label)
        except Exception as e:
            print(f"Could not process {fname}: {e}")
    axp.set_title("Perplexity vs Training Steps (by Model Size)")
    axp.set_xlabel("Training Steps")
    axp.set_ylabel("Perplexity")
    axp.grid(True)
    axp.legend()
    out_p = os.path.join(outdir, "model_size_perplexity_over_steps.png")
    _save_plot(fig_p, out_p)

    # Accuracy
    fig_a = plt.figure()
    axa = fig_a.gca()
    for fname in sorted(files):
        path = os.path.join(details_dir, fname)
        try:
            d = pd.read_csv(path)
            if {"step", "accuracy"} <= set(d.columns):
                base = fname.replace("_metrics.csv", "")
                size_tag = base.split("_")[-1]
                label = size_tag if size_tag.upper().endswith("M") else base
                axa.plot(d["step"], d["accuracy"], label=label)
        except Exception as e:
            print(f"Could not process {fname}: {e}")
    axa.set_title("Accuracy vs Training Steps (by Model Size)")
    axa.set_xlabel("Training Steps")
    axa.set_ylabel("Accuracy")
    axa.grid(True)
    axa.legend()
    out_a = os.path.join(outdir, "model_size_accuracy_over_steps.png")
    _save_plot(fig_a, out_a)

    return (out_p, out_a)

def plot_eval_over_steps_by_seq_len(details_dir, outdir):
    """
    Reads results/details/seq_len_*_metrics.csv files and plots:
      - seq_len_perplexity_over_steps.png
      - seq_len_accuracy_over_steps.png
    X-axis = training steps; legend labels EXACTLY 'Sequence length 128', 'Sequence length 512', ...
    """
    if not os.path.isdir(details_dir):
        print(f"No details dir @ {details_dir}; skipping seq-len over-steps plots.")
        return (None, None)

    files = [f for f in os.listdir(details_dir)
             if f.startswith("seq_len_") and f.endswith("_metrics.csv")]
    if not files:
        print("No seq_len_*_metrics.csv files found; skipping.")
        return (None, None)

    def _label_from_fname(fname):
        base = fname.replace("_metrics.csv", "")
        # Attempt to extract trailing integer (e.g., '...128')
        digits = "".join(ch for ch in base if ch.isdigit())
        if digits:
            return f"Sequence length {int(digits)}"
        return base

    # Perplexity
    fig_p = plt.figure()
    axp = fig_p.gca()
    for fname in sorted(files):
        path = os.path.join(details_dir, fname)
        try:
            d = pd.read_csv(path)
            if {"step", "perplexity"} <= set(d.columns):
                axp.plot(d["step"], d["perplexity"], label=_label_from_fname(fname))
        except Exception as e:
            print(f"Could not process {fname}: {e}")
    axp.set_title("Perplexity vs Training Steps (by Sequence Length)")
    axp.set_xlabel("Training Steps")
    axp.set_ylabel("Perplexity")
    axp.grid(True)
    axp.legend()
    out_p = os.path.join(outdir, "seq_len_perplexity_over_steps.png")
    _save_plot(fig_p, out_p)

    # Accuracy
    fig_a = plt.figure()
    axa = fig_a.gca()
    for fname in sorted(files):
        path = os.path.join(details_dir, fname)
        try:
            d = pd.read_csv(path)
            if {"step", "accuracy"} <= set(d.columns):
                axa.plot(d["step"], d["accuracy"], label=_label_from_fname(fname))
        except Exception as e:
            print(f"Could not process {fname}: {e}")
    axa.set_title("Accuracy vs Training Steps (by Sequence Length)")
    axa.set_xlabel("Training Steps")
    axa.set_ylabel("Accuracy")
    axa.grid(True)
    axa.legend()
    out_a = os.path.join(outdir, "seq_len_accuracy_over_steps.png")
    _save_plot(fig_a, out_a)

    return (out_p, out_a)

def write_report(df, outdir):
    base = df[df["run_name"]=="vanilla_baseline"]
    gc   = df[df["run_name"]=="vanilla_gradient_checkpointing"]
    fa   = df[df["run_name"]=="opt_flash_attention"]
    lines = []
    lines.append("# Profiling & Optimization Report\n")
    lines.append("**Source CSV:** results2.csv\n")
    seq = df[df["run_name"].str.startswith("seq_len_")].sort_values("sequence_length")
    if not seq.empty:
        first = seq.iloc[0]
        last  = seq.iloc[-1]
        lines.append("## Memory vs Sequence Length\n")
        lines.append(f"- Peak memory from **{first['peak_gpu_mem_gb']:.2f} GB @ {int(first['sequence_length'])}** to **{last['peak_gpu_mem_gb']:.2f} GB @ {int(last['sequence_length'])}**.\n")
    mm = df[df["run_name"].str.startswith(("bs_fp32_","bs_fp16_"))]
    if not mm.empty:
        bs32 = sorted(mm[mm["fp16"]==False]["batch_size"].unique())
        bs16 = sorted(mm[mm["fp16"]==True]["batch_size"].unique())
        common = sorted(list(set(bs32).intersection(set(bs16))))
        if common:
            b = max(common)
            t32 = mm[(mm["batch_size"]==b)&(mm["fp16"]==False)]["throughput_samples_per_sec"].mean()
            t16 = mm[(mm["batch_size"]==b)&(mm["fp16"]==True)]["throughput_samples_per_sec"].mean()
            m32 = mm[(mm["batch_size"]==b)&(mm["fp16"]==False)]["peak_gpu_mem_gb"].mean()
            m16 = mm[(mm["batch_size"]==b)&(mm["fp16"]==True)]["peak_gpu_mem_gb"].mean()
            if t32 and m32:
                lines.append("## Mixed Precision (example at largest common batch size)\n")
                lines.append(f"- BS={b}: throughput **FP32 {t32:.0f} → FP16 {t16:.0f}** (×{t16/t32:.2f}), memory **{m32:.2f} → {m16:.2f} GB** ({(m16-m32)/m32*100:.1f}%).\n")
    if not base.empty and not gc.empty:
        lines.append("## Gradient Checkpointing\n")
        mem_delta = (gc["peak_gpu_mem_gb"].iloc[0] - base["peak_gpu_mem_gb"].iloc[0]) / base["peak_gpu_mem_gb"].iloc[0] * 100.0
        thr_delta = (gc["throughput_samples_per_sec"].iloc[0] - base["throughput_samples_per_sec"].iloc[0]) / base["throughput_samples_per_sec"].iloc[0] * 100.0
        lines.append(f"- Peak memory change: **{mem_delta:.1f}%**; Throughput change: **{thr_delta:.1f}%** (vs baseline).\n")
    if not base.empty and not fa.empty:
        lines.append("## Flash Attention\n")
        mem_delta = (fa["peak_gpu_mem_gb"].mean() - base["peak_gpu_mem_gb"].iloc[0]) / base["peak_gpu_mem_gb"].iloc[0] * 100.0
        thr_delta = (fa["throughput_samples_per_sec"].mean() - base["throughput_samples_per_sec"].iloc[0]) / base["throughput_samples_per_sec"].iloc[0] * 100.0
        lines.append(f"- Peak memory change: **{mem_delta:.1f}%**; Throughput change: **{thr_delta:.1f}%** (vs baseline).\n")
    md = "\n".join(lines).strip() + "\n"
    path = os.path.join(outdir, "profiling_optimization_report.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"[report] saved -> {path}")
    return path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="results.csv")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--details_dir", type=str, default="results/details")
    args = p.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: results file not found: {args.results}")
        return

    df = pd.read_csv(args.results)
    num_cols = ["total_params_m","sequence_length","batch_size","training_time_sec",
                "throughput_samples_per_sec","peak_gpu_mem_gb","eval_loss","eval_perplexity",
                "avg_perplexity","avg_accuracy","eval_accuracy"]
    df = _ensure_numeric(df, num_cols)
    bool_cols = ["fp16","grad_checkpoint","flash_attention"]
    df = _ensure_bool(df, bool_cols)
    os.makedirs(args.outdir, exist_ok=True)

    # Core plots/tables
    plot_memory_vs_seq_len(df, args.outdir)
    plot_memory_vs_batch_size(df, args.outdir)
    plot_throughput_vs_batch_size(df, args.outdir)
    plot_mixed_precision_bars(df, args.outdir)
    plot_optimization_comparisons(df, args.outdir)
    plot_throughput_vs_model_size(df, args.outdir)
    plot_time_series(args.details_dir, args.outdir)

    # New: eval over steps by model size & by sequence length
    plot_eval_over_steps_by_model_size(args.details_dir, args.outdir)
    plot_eval_over_steps_by_seq_len(args.details_dir, args.outdir)

    # Short report
    write_report(df, args.outdir)

if __name__ == "__main__":
    main()