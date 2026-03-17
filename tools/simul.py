# A Python file of functions generating various simulation data and running the methods on them.

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method import Distribution, run_de, agglomerative_discrete, divisive_discrete, agglomerative_BC, fitness
from scipy.stats import truncnorm

SEED = 20241225


def simul_uniform_mixture(n_samples, setting="Setting1", eps = 0, seed=None, **params):
    """
    Simulate mixture of uniform distributions data with given weights

    Parameters
    ----------
    n_samples : int
        number of distribution samples
    setting: str
        "Setting1" or "Setting2" for the different settings of the weights (see the paper)
    eps : float
        Standard deviation of the Gaussian noise added to the thresholds
    seed : int, default=None
        Random seed for the random number generator
    **params : additional parameters to pass to the function `generate_mixture`
        In case one wants to use different n_obs or thresholds in the simulation

    Returns
    -------
    samples : np.ndarray of shape (n_samples, n_obs)
        Simulated data for each distribution sample
    """
    params.setdefault('n_obs', 1000)

    # Different seeds for different noise levels/settings
    if seed is not None:
        seed += eps * 1000
        if setting == "Setting2":
            seed += 50000

    RS = np.random.RandomState(seed)

    # Weights generation based on the Setting 1/2
    if setting == "Setting1":
        alpha = (.3, .4, .2, .1)
        weight_matrix = RS.dirichlet(np.array(alpha) * 20, n_samples)

    elif setting == "Setting2":
        fixed_weights=(.2, .1)
        w1, w2 = RS.dirichlet(np.array([.5, .5]) * 5, n_samples).T * (1 - np.sum(fixed_weights))
        
        # stack the copys of fixed weights through the rows
        fixed_weights_tile = np.tile(fixed_weights, (n_samples, 1))   
        weight_matrix =  np.column_stack((w1, w2, fixed_weights_tile))

    else:
        raise ValueError("Invalid setting: " + setting)

    # Generate empirical mixture distributions
    samples = np.zeros((n_samples, params['n_obs']))
    for i in range(n_samples):
        samples[i] = generate_mixture(weight_matrix[i], eps=eps, random_state=RS, **params)
    
    return samples


def generate_mixture(weights, thresholds=(40, 70, 180, 250, 400), n_obs=1000, eps=0, random_state=None):
    """
    Generate observations from a mixture of uniform distributions given weights vector

    Parameters
    ----------
    weights : np.ndarray 
        Nonnegative weights of the mixture components, summing to 1
    thresholds : array-like indexable vector. 
        Uniform distributions are drawn from the intervals (thresholds[i], thresholds[i+1])
        Must satisfy thresholds[0] < thresholds[1] < ... < thresholds[-1], and len(weights) == len(thresholds) - 1
    n_obs : int, default 1000
        Number of observations to generate
    eps : float
        Standard deviation of the Gaussian noise added to the thresholds
    random_state : np.random.RandomState, default=None 
        Random number generator, often inherited from the parent function
    """
    n_intervals = len(weights)
    assert n_intervals == len(thresholds) - 1
    assert np.all(np.diff(thresholds) > 0)

    RS = random_state if random_state is not None else np.random.RandomState(None)
    
    thresholds = np.array(thresholds, dtype=float)
    
    # Add noise to the intermediate thresholds
    if eps > 0:
        # Truncated normal distribution to [-30, 30]
        a, b = -30 / eps, 30 / eps
        noise = truncnorm.rvs(a, b, scale=eps, size=len(thresholds) - 2, random_state=RS)
        thresholds[1:-1] += noise

    # Generate mixture uniform samples
    samples = np.zeros(n_obs)
    for i in range(n_obs):
        # Choose an index from the np.arange(n_intervals) with mixture weights
        idx = RS.choice(n_intervals, p=weights)

        # Draw a uniform sample
        samples[i] = RS.uniform(thresholds[idx], thresholds[idx + 1])

    return samples



#### Method implementations ####

_THRESHOLD_GRID = list(np.arange(40, 401, 2))


def run_parallel_method(method, n, reps, njobs, noise=0, K=3, loss="Loss1",
                        setting="Setting1", n_obs=1000, base_seed=SEED, save=True):
    """Run one of the notebook simulation methods: ``de``, ``sa``, ``ss``, or ``paa``."""
    _validate_setting(setting)

    # For each monte Carlo repetition, generate the same data for all methods, and run the methods on it.
    def run_1rep(i):
        print(f"Running rep {i + 1}")
        seed = base_seed + i
        data = simul_uniform_mixture(n, setting=setting, eps=noise, seed=seed, n_obs=n_obs)
        data_class = Distribution(data, ran=(40., 400.), M=200)

        if method == "de":
            return run_de(data_class, K=K, loss=loss, seed=_de_seed_offset(setting, K, loss) + i)
        if method == "sa":
            return agglomerative_discrete(data_class, K=K, loss=loss, thresholds=_THRESHOLD_GRID)
        if method == "ss":
            return divisive_discrete(data_class, K=K, loss=loss, thre_list=np.array(_THRESHOLD_GRID))
        if method == "paa":
            return agglomerative_BC(data_class, K=K, thresholds=_THRESHOLD_GRID)
        raise ValueError(f"Unsupported method: {method}")

    result_list = Parallel(n_jobs=njobs)(delayed(run_1rep)(i) for i in range(reps))

    if save:
        result_path = _result_path(method, setting, n, noise, loss=loss, K=K)
        with open(result_path, 'wb') as f:
            pickle.dump(result_list, f)

    return result_list


def run_oracle_parallel(n, reps, njobs, noise=0, loss="Loss1",
                        setting="Setting1", n_obs=1000, base_seed=SEED, save=True):
    """Run the oracle-threshold loss evaluation used in the simulation notebook."""
    _validate_setting(setting)

    if setting == "Setting1":
        oracle_cutoffs = np.array([70, 180, 250], dtype=float)
    elif loss == "Loss1":
        oracle_cutoffs = np.array([70, 180, 250], dtype=float)
    else:
        oracle_cutoffs = np.array([70, 180], dtype=float)

    def run_1rep(i):
        print(f"Running oracle rep {i + 1}")
        seed = base_seed + i
        data = simul_uniform_mixture(n, setting=setting, eps=noise, seed=seed, n_obs=n_obs)
        data_class = Distribution(data, ran=(40., 400.), M=200)
        data_class.Wdist_matrix()
        return fitness(oracle_cutoffs, data_class, loss=loss)

    result_list = Parallel(n_jobs=njobs)(delayed(run_1rep)(i) for i in range(reps))

    if save:
        result_path = oracle_result_path(setting, n, noise, loss)
        with open(result_path, 'wb') as f:
            pickle.dump(result_list, f)

    return result_list


## validations of inputs and output paths for compatibility ##
# Designed only for the specific settings used in the paper, not for general use.

def _validate_setting(setting):
    if setting not in {"Setting1", "Setting2"}:
        raise ValueError(f"Invalid setting: {setting}")


def oracle_result_path(setting, n, noise, loss):
    _validate_setting(setting)
    result_dir = "results/simulation" if setting == "Setting1" else "results/simulation2"
    return REPO_ROOT / result_dir / f"oracle-n{n}_noise{noise}_{loss}.pkl"


def _result_path(method, setting, n, noise, loss=None, K=None):
    _validate_setting(setting)
    result_dir = "results/simulation" if setting == "Setting1" else "results/simulation2"

    if method == "de":
        filename = f"de-n{n}K{K}_noise{noise}_{loss}.pkl"
    elif method == "sa":
        if setting == "Setting1":
            filename = f"sa-n{n}_noise{noise}_{loss}.pkl"
        else:
            filename = f"sa-n{n}K{K}-noise{noise}_{loss}.pkl"
    elif method == "ss":
        if setting == "Setting1":
            filename = f"ss-n{n}_noise{noise}_{loss}.pkl"
        else:
            filename = f"ss-n{n}-noise{noise}_{loss}.pkl"
    elif method == "paa":
        if setting == "Setting1":
            filename = f"paa-n{n}_noise{noise}.pkl"
        else:
            filename = f"paa-n{n}-noise{noise}.pkl"
    else:
        raise ValueError(f"Unsupported method: {method}")

    return REPO_ROOT / result_dir / filename


_DE_SEED_OFFSETS = {
    ("Setting1", 3, "Loss1"): 0,
    ("Setting1", 3, "Loss2"): 200,
    ("Setting2", 3, "Loss1"): 1000,
    ("Setting2", 3, "Loss2"): 2000,
    ("Setting2", 2, "Loss1"): 10000,
    ("Setting2", 2, "Loss2"): 20000,
}

def _de_seed_offset(setting, K, loss):
    try:
        return _DE_SEED_OFFSETS[(setting, K, loss)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DE configuration: setting={setting}, K={K}, loss={loss}"
        ) from exc
