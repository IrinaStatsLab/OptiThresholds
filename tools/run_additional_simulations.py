from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.additional_simulations import describe_tasks, run_task


def parse_args():
    parser = argparse.ArgumentParser(description="Run additional simulation tasks for n=50 and n=100.")
    parser.add_argument("--group", choices=["sa", "non_sa"], required=True)
    parser.add_argument("--n", type=int, choices=[50, 100], required=True)
    parser.add_argument("--task-id", type=int, default=None, help="Defaults to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--describe", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.describe:
        for idx, task in enumerate(describe_tasks(args.group)):
            print(idx, task)
        return

    task_id = args.task_id
    if task_id is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            raise SystemExit("--task-id is required outside Slurm.")
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    outputs = run_task(
        group=args.group,
        task_id=task_id,
        n=args.n,
        reps=args.reps,
        n_jobs=args.n_jobs,
        n_obs=args.n_obs,
    )

    print(f"Completed task {task_id} for group={args.group}, n={args.n}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
