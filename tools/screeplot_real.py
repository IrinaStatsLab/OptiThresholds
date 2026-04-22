"""Run real-data screeplot jobs, either once or in paired local batches.

Usage:

    python tools/screeplot_real.py --dataset healthy --K 3
    python tools/screeplot_real.py --datasets healthy t1d combined --max-workers 5

Each run saves a pickle file to `results/screeplot/{dataset}_{loss}_K{K}.pkl`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method import run_de
from tools.real_loss_tools import DATASET_SPECS, MAIN_OBJECTIVE_BY_DATASET, load_real_loss_dataset

SEED = 20241225
VALID_LOSSES = ("Loss1", "Loss2")
DATASET_LABELS = dict(DATASET_SPECS)
OUTPUT_DIR = REPO_ROOT / "results" / "screeplot"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real-data screeplot jobs. "
            "Provide --K for one job, or omit --K to run paired K batches in local parallel."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_LABELS),
        help="Single dataset key. With --K, this is required.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_LABELS),
        help="Dataset keys for batch mode. Defaults to healthy t1d combined when --K is omitted.",
    )
    parser.add_argument(
        "--loss",
        choices=VALID_LOSSES,
        help="Loss function. Defaults to the manuscript's main loss for each dataset.",
    )
    parser.add_argument("--K", type=int, help="Number of thresholds for single-job mode.")
    parser.add_argument(
        "--K-values",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Threshold counts for batch mode. Defaults to 1 2 ... 10.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of local worker processes in batch mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing pickle files instead of skipping them.",
    )
    args = parser.parse_args()

    if args.dataset and args.datasets:
        parser.error("Use either --dataset or --datasets, not both.")
    if args.K is not None and args.dataset is None:
        parser.error("--dataset is required when --K is provided.")
    return args


def _result_path(dataset_key: str, loss: str, threshold_count: int) -> Path:
    return OUTPUT_DIR / f"{dataset_key}_{loss}_K{threshold_count}.pkl"


def run_screeplot_job(
    dataset_key: str,
    threshold_count: int,
    loss: str | None = None,
    *,
    verbose: bool = True,
) -> dict[str, object]:
    loss = loss or MAIN_OBJECTIVE_BY_DATASET[dataset_key]
    dataset = load_real_loss_dataset(REPO_ROOT, dataset_key)
    if verbose:
        print(
            f"[screeplot_real] dataset={dataset_key} ({dataset.label}), "
            f"loss={loss}, K={threshold_count}, seed={SEED}",
            flush=True,
        )

    best_cutoffs, min_loss = run_de(dataset.data_class, K=threshold_count, loss=loss, seed=SEED)
    if verbose:
        print(f"  cutoffs = {best_cutoffs[1:-1]}", flush=True)
        print(f"  loss    = {min_loss}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _result_path(dataset_key, loss, threshold_count)
    result = {
        "dataset_key": dataset_key,
        "dataset_label": dataset.label,
        "loss": loss,
        "K": threshold_count,
        "best_cutoffs": best_cutoffs,
        "min_loss": min_loss,
        "seed": SEED,
    }
    with out_path.open("wb") as file_obj:
        pickle.dump(result, file_obj)
    if verbose:
        print(f"  saved -> {out_path}", flush=True)
    return result


def _build_pairs(K_values: list[int]) -> list[tuple[int, ...]]:
    ordered_values = sorted(set(K_values))
    pairs: list[tuple[int, ...]] = []
    left = 0
    right = len(ordered_values) - 1
    while left <= right:
        if left == right:
            pairs.append((ordered_values[left],))
        else:
            pairs.append((ordered_values[left], ordered_values[right]))
        left += 1
        right -= 1
    return pairs


def _run_pair(dataset_key: str, K_pair: tuple[int, ...], loss: str | None, overwrite: bool) -> list[str]:
    resolved_loss = loss or MAIN_OBJECTIVE_BY_DATASET[dataset_key]
    logs: list[str] = []
    for threshold_count in K_pair:
        out_path = _result_path(dataset_key, resolved_loss, threshold_count)
        if out_path.exists() and not overwrite:
            logs.append(f"[skip] dataset={dataset_key} loss={resolved_loss} K={threshold_count} -> {out_path}")
            continue
        result = run_screeplot_job(dataset_key, threshold_count, loss=resolved_loss, verbose=False)
        logs.append(
            f"[done] dataset={dataset_key} loss={resolved_loss} K={threshold_count} "
            f"loss_value={float(result['min_loss']):.6f} -> {out_path}"
        )
    return logs


def run_screeplot_batch(
    dataset_keys: list[str],
    *,
    loss: str | None = None,
    K_values: list[int] | None = None,
    max_workers: int = 5,
    overwrite: bool = False,
) -> None:
    K_values = sorted(set(K_values or list(range(1, 11))))
    K_pairs = _build_pairs(K_values)
    worker_count = min(max(1, max_workers), len(K_pairs), os.cpu_count() or max_workers)

    for dataset_key in dataset_keys:
        resolved_loss = loss or MAIN_OBJECTIVE_BY_DATASET[dataset_key]
        print(
            f"[screeplot_real:batch] dataset={dataset_key} ({DATASET_LABELS[dataset_key]}), "
            f"loss={resolved_loss}, workers={worker_count}",
            flush=True,
        )
        print(f"  K pairs: {K_pairs}", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_run_pair, dataset_key, K_pair, loss, overwrite)
                for K_pair in K_pairs
            ]
            for future in futures:
                for line in future.result():
                    print(f"  {line}", flush=True)


def main() -> None:
    args = _parse_args()
    if args.K is not None:
        run_screeplot_job(dataset_key=args.dataset, threshold_count=args.K, loss=args.loss)
        return

    dataset_keys = args.datasets or ([args.dataset] if args.dataset else ["healthy", "t1d", "combined"])
    run_screeplot_batch(
        dataset_keys=dataset_keys,
        loss=args.loss,
        K_values=args.K_values,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
