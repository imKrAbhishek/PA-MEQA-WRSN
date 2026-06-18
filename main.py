# =============================================================================
# main.py — Final execution file (QL vs MEQA vs PA-MEQA)
# =============================================================================

import os
import time
import numpy as np

from config import CFG
from train import train
from visualization import (
    fig1_training,
    fig2_comparison,
    fig3_priority,
    fig4_cycle,
    fig5_dashboard,
    fig6_scalability,
    fig7_hyperparams,
)

# Ensure results folder exists
os.makedirs("results", exist_ok=True)


# =============================================================================
# COMPARISON TABLE
# =============================================================================
def print_comparison(pa_logs, mq_logs, ql_logs, w=80):

    print("\n" + "=" * 72)
    print(" FINAL COMPARISON — Q-Learning | MEQA | PA-MEQA")
    print("=" * 72)

    print(f"{'Metric':<15}{'QL':>10}{'MEQA':>10}{'PA-MEQA':>12}"
          f"{'PA vs MQ Δ%':>15}{'PA Best?':>10}")
    print("-" * 72)

    metrics = [
        ("reward", "Total Reward", False),
        ("qed",    "QED",          False),
        ("scr",    "SCR",          False),
        ("cdp",    "CDP",          False),
        ("noff",   "NONs",         True),
    ]

    for key, label, lower_better in metrics:

        qv = np.mean([np.mean(h[key][-w:]) for h in ql_logs])
        mv = np.mean([np.mean(h[key][-w:]) for h in mq_logs])
        pv = np.mean([np.mean(h[key][-w:]) for h in pa_logs])

        delta = 100 * (pv - mv) / (abs(mv) + 1e-9)

        best = (pv < mv) if lower_better else (pv > mv)
        mark = "✓" if best else "✗"

        print(f"{label:<15}{qv:>10.4f}{mv:>10.4f}{pv:>12.4f}"
              f"{delta:>15.2f}%{mark:>10}")

    print("=" * 72 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    t0 = time.time()

    # Seeds for reproducibility
    seeds = range(CFG.SEED, CFG.SEED + CFG.N_RUNS)

    # ─────────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────────
    print(f"\nTraining Q-Learning  ({CFG.N_RUNS} runs × {CFG.G_MAX} iters)...")
    ql_logs = [train(s, mode="ql") for s in seeds]

    print(f"Training MEQA        ({CFG.N_RUNS} runs × {CFG.G_MAX} iters)...")
    mq_logs = [train(s, mode="meqa") for s in seeds]

    print(f"Training PA-MEQA     ({CFG.N_RUNS} runs × {CFG.G_MAX} iters)...")
    pa_logs = [train(s, mode="pa") for s in seeds]

    # ─────────────────────────────────────────────────────────────
    # RESULTS TABLE
    # ─────────────────────────────────────────────────────────────
    print_comparison(pa_logs, mq_logs, ql_logs)

    # ─────────────────────────────────────────────────────────────
    # FIGURES
    # ─────────────────────────────────────────────────────────────
    print("Generating figures...")

    fig1_training(pa_logs, mq_logs, ql_logs)
    fig2_comparison(pa_logs, mq_logs, ql_logs)
    fig3_priority(pa_logs)
    fig4_cycle(pa_logs, mq_logs, ql_logs)
    fig5_dashboard(pa_logs, mq_logs, ql_logs)
    fig6_scalability(pa_logs, mq_logs, ql_logs)
    fig7_hyperparams(pa_logs, mq_logs, ql_logs)

    # ─────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────
    print(f"\nAll done in {time.time() - t0:.1f}s")
    print("Results saved to: results/")