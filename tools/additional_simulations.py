from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

from tools.simul import REPO_ROOT, oracle_result_path, run_oracle_parallel, run_parallel_method

NOISE_LEVELS = [0, 5, 10]
LOSS_TYPES = ["Loss1", "Loss2"]
SUPPORTED_N = [50, 100]

SETTING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Setting1": {
        "method_dir": REPO_ROOT / "results" / "simulation",
        "k": {"de": 3, "sa": 2, "ss": 3, "paa": 3},
    },
    "Setting2": {
        "method_dir": REPO_ROOT / "results" / "simulation2",
        "k": {"de": 3, "de_k2": 2, "sa": 2, "ss": 3, "paa": 2},
    },
}

SA_TASKS: List[Dict[str, Any]] = [
    {"setting": setting, "noise": noise, "loss": loss}
    for setting in ["Setting1", "Setting2"]
    for loss in LOSS_TYPES
    for noise in NOISE_LEVELS
]

NON_SA_TASKS: List[Dict[str, Any]] = [
    {"setting": setting, "noise": noise}
    for setting in ["Setting1", "Setting2"]
    for noise in NOISE_LEVELS
]


def result_dir(setting: str) -> Path:
    return SETTING_CONFIGS[setting]["method_dir"]


def load_oracle_result(setting: str, n: int, noise: int, loss: str):
    with open(oracle_result_path(setting, n, noise, loss), "rb") as f:
        return pickle.load(f)


def describe_tasks(group: str) -> List[Dict[str, Any]]:
    if group == "sa":
        return SA_TASKS
    if group == "non_sa":
        return NON_SA_TASKS
    raise ValueError(f"Unsupported group: {group}")


def _setting1_non_sa_outputs(n: int, noise: int) -> List[str]:
    cfg = SETTING_CONFIGS["Setting1"]["k"]
    outputs = [
        str(result_dir("Setting1") / f"de-n{n}K{cfg['de']}_noise{noise}_{loss}.pkl")
        for loss in LOSS_TYPES
    ]
    outputs += [
        str(result_dir("Setting1") / f"ss-n{n}_noise{noise}_{loss}.pkl")
        for loss in LOSS_TYPES
    ]
    outputs += [
        str(oracle_result_path("Setting1", n, noise, loss))
        for loss in LOSS_TYPES
    ]
    outputs.append(str(result_dir("Setting1") / f"paa-n{n}_noise{noise}.pkl"))
    return outputs


def _setting2_non_sa_outputs(n: int, noise: int) -> List[str]:
    cfg = SETTING_CONFIGS["Setting2"]["k"]
    outputs = [
        str(result_dir("Setting2") / f"de-n{n}K{cfg['de']}_noise{noise}_{loss}.pkl")
        for loss in LOSS_TYPES
    ]
    outputs += [
        str(result_dir("Setting2") / f"de-n{n}K{cfg['de_k2']}_noise{noise}_{loss}.pkl")
        for loss in LOSS_TYPES
    ]
    outputs += [
        str(result_dir("Setting2") / f"ss-n{n}-noise{noise}_{loss}.pkl")
        for loss in LOSS_TYPES
    ]
    outputs += [
        str(oracle_result_path("Setting2", n, noise, loss))
        for loss in LOSS_TYPES
    ]
    outputs.append(str(result_dir("Setting2") / f"paa-n{n}-noise{noise}.pkl"))
    return outputs


def run_task(group: str, task_id: int, n: int, reps: int = 100, n_jobs: int | None = None, n_obs: int = 1000, base_seed: int = 20241225) -> List[str]:
    if n not in SUPPORTED_N:
        raise ValueError(f"Unsupported n: {n}. Expected one of {SUPPORTED_N}.")

    n_jobs = n_jobs or int(os.environ.get("N_JOBS", os.cpu_count() or 1))

    if group == "sa":
        task = SA_TASKS[task_id]
        cfg = SETTING_CONFIGS[task["setting"]]["k"]
        run_parallel_method(
            "sa",
            n=n,
            reps=reps,
            njobs=n_jobs,
            noise=task["noise"],
            K=cfg["sa"],
            loss=task["loss"],
            setting=task["setting"],
            n_obs=n_obs,
            base_seed=base_seed,
            save=True,
        )
        if task["setting"] == "Setting1":
            return [str(result_dir(task["setting"]) / f"sa-n{n}_noise{task['noise']}_{task['loss']}.pkl")]
        return [str(result_dir(task["setting"]) / f"sa-n{n}K{cfg['sa']}-noise{task['noise']}_{task['loss']}.pkl")]

    if group == "non_sa":
        task = NON_SA_TASKS[task_id]
        setting = task["setting"]
        noise = task["noise"]
        cfg = SETTING_CONFIGS[setting]["k"]

        for loss in LOSS_TYPES:
            run_parallel_method("de", n=n, reps=reps, njobs=n_jobs, noise=noise, K=cfg["de"], loss=loss, setting=setting, n_obs=n_obs, base_seed=base_seed, save=True)
            if setting == "Setting2":
                run_parallel_method("de", n=n, reps=reps, njobs=n_jobs, noise=noise, K=cfg["de_k2"], loss=loss, setting=setting, n_obs=n_obs, base_seed=base_seed, save=True)
            run_parallel_method("ss", n=n, reps=reps, njobs=n_jobs, noise=noise, K=cfg["ss"], loss=loss, setting=setting, n_obs=n_obs, base_seed=base_seed, save=True)
            run_oracle_parallel(n=n, reps=reps, njobs=n_jobs, noise=noise, loss=loss, setting=setting, n_obs=n_obs, base_seed=base_seed, save=True)

        run_parallel_method("paa", n=n, reps=reps, njobs=n_jobs, noise=noise, K=cfg["paa"], setting=setting, n_obs=n_obs, base_seed=base_seed, save=True)

        if setting == "Setting1":
            return _setting1_non_sa_outputs(n, noise)
        return _setting2_non_sa_outputs(n, noise)

    raise ValueError(f"Unsupported group: {group}")
