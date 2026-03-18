"""Run differential evolution for the AI-READI dataset at a single K value.

Usage (typically called by the companion Slurm script):

    python tools/screeplot_aireadi.py --K 3

The script saves a pickle file to  results/screeplot/aireadi_Loss2_K{K}.pkl
containing a dict with keys: K, best_cutoffs, min_loss, seed.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method import Distribution, run_de
from tools.ai_readi_tools import load_ai_readi_cohort

SEED = 20241225
LOSS = "Loss2"
RAN = (40, 400)
M = 200
OUTPUT_DIR = REPO_ROOT / "results" / "screeplot"


def main() -> None:
    parser = argparse.ArgumentParser(description="Screeplot DE run for AI-READI")
    parser.add_argument("--K", type=int, required=True, help="Number of thresholds")
    args = parser.parse_args()

    K = args.K
    print(f"[screeplot_aireadi] K={K}, loss={LOSS}, seed={SEED}")

    # ---- load cohort --------------------------------------------------------
    cohort = load_ai_readi_cohort(REPO_ROOT)
    data_class = Distribution(cohort["gl"], ran=RAN, M=M)

    # ---- run DE -------------------------------------------------------------
    best_cutoffs, min_loss = run_de(data_class, K=K, loss=LOSS, seed=SEED)
    print(f"  cutoffs = {best_cutoffs[1:-1]}")
    print(f"  loss    = {min_loss}")

    # ---- save ---------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"aireadi_{LOSS}_K{K}.pkl"
    result = {
        "K": K,
        "best_cutoffs": best_cutoffs,
        "min_loss": min_loss,
        "seed": SEED,
    }
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
