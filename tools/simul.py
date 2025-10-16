# A Python file of functions generating various simulation data

import numpy as np
from scipy.stats import truncnorm


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
