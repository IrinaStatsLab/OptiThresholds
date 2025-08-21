import numpy as np

from scipy.stats import ecdf
from scipy.interpolate import interp1d
from scipy.optimize import LinearConstraint, differential_evolution


class Distribution:
    """
    Class representing the distributional representations wearable device data (e.g., CGM data) and their amalgamated histograms. 
    The class stores the distributional data in the form of histograms and quantiles.
    """

    def __init__(self, data, ran=None, M=200):
        """
        Initialize with data and range.

        Parameters:
            data (list): List of wearable device measurements for each subject.
            ran (tuple): Range of data. Default is None, which set minimum and maximum values from the data.
            M (int): Number of quantile grid points for Wasserstein distance computation.
        """
        # Check if data is an iterable of lists or arrays
        try:
            iter(data)
            if not all(hasattr(datum, '__iter__') and not isinstance(datum, str) for datum in data):
                raise ValueError("Data must be an iterable of lists or arrays of glucose values.")
        except TypeError:
            raise ValueError("Data must be an iterable of lists or arrays of glucose values.")

        if ran is not None:
            assert np.min(np.min(data)) >= ran[0] and np.max(np.max(data)) <= ran[1], (
                "Data out of range: specify the correct min/max range of measurement levels"
            )
        else:
            ran = [np.min(np.min(data)) - 1e-8, np.max(np.max(data)) + 1e-8]
        
        self.ran = ran
        self.data = [np.array(datum) for datum in data]
        self.M = M
        self.gr = np.linspace(0, 1, M+2)    # Grid for quantiles
        
        # Get quantile functions and empirical CDFs for each subject
        self.qtiles = self.compute_quantiles()
        self.F_list = self.get_ecdfs()

    def compute_quantiles(self):
        """Compute quantiles for each subject's data."""
        return [np.quantile(data, self.gr) for data in self.data]
    
    def get_ecdfs(self):
        """ Get the empirical CDF callables of the CGM measurements"""
        return [ecdf(data).cdf.evaluate for data in self.data]

    def cutoff_amalgamation(self, cutoffs, fixed=None):
        """
        Compute piecewise-linearized quantiles given the cutoffs, 
        which correspond to the amalgamated histograms based on the cutoffs.

        Parameters:
            cutoffs (list): List of cutoffs (excluding endpoints).
            fixed (list): Optional fixed thresholds for semi-supervised approach.

        Returns:
            list: Amalgamated quantiles.
        """
        if fixed is not None:
            assert np.min(fixed) >= self.ran[0] and np.max(fixed) <= self.ran[1], "Fixed thresholds out of range"
            cutoffs = np.sort(np.r_[fixed, cutoffs])    # Sort is required for the correct interpolation

        q_a_list = list()
        for i in range(len(self.data)):
            knots = np.unique(np.concatenate([[0], self.F_list[i](cutoffs), [1]]))  # endpoints for ecdf 0, 1 values
            q_knots = np.quantile(self.data[i], knots)
            interp_func = interp1d(knots, q_knots, kind='linear', fill_value=(np.min(self.data[i]), np.max(self.data[i])))
            
            # Amalgamated quantiles
            q_a_list.append(interp_func(self.gr)) 
        
        self.q_a = q_a_list
        return q_a_list
    
    def dists_for_Loss1(self, cutoffs, fixed=None, Wdist="W2"):
        """
        Compute q-Wasserstein distances d_W^q between original and amalgamated distributions.
        q = 1, 2 for W1 and W2 distances, respectively.

        Parameters:
            cutoffs (list): Cutoffs for amalgamation.
            fixed (list): Optional fixed thresholds.
            Wdist (str): Distance metric, "W1" or "W2".

        Returns:
            np.ndarray: List of distances.
        """
        # Compute the quantiles of the amalgamation given cutoffs
        self.cutoff_amalgamation(cutoffs, fixed)

        n = len(self.data)
        distances = np.zeros(n)
        for i in range(n):
            if Wdist == "W2":
                distances[i] = np.sum((self.qtiles[i] - self.q_a[i]) ** 2) / (self.M + 1)
            elif Wdist == "W1":
                distances[i] = np.sum(np.abs(self.qtiles[i] - self.q_a[i])) / (self.M + 1)
        return distances
    
    def Wdist_matrix(self, cutoffs=None, fixed=None, change=True, Wdist="W2"):
        """
        Compute (upper triangular) Wasserstein distance matrix for Loss2 computation.

        Parameters:
            cutoffs (list): Cutoffs for amalgamation.
            fixed (list): Optional fixed thresholds.
            change (bool): Whether to update the amalgamated matrix.
            Wdist (str): Distance metric, "W1" or "W2".

        Returns:
            np.ndarray: Distance matrix.
        """
        n = len(self.data)
        dist_matrix = np.zeros((n, n))
        qtiles = self.qtiles

        if cutoffs is not None:
            self.cutoff_amalgamation(cutoffs, fixed)
            qtiles = self.q_a
        
        for i in range(n-1):
            for j in range(i+1, n):
                if Wdist == "W2":
                    # Square-root for the actual W2 distance ########### <- this will change many of my results. Test this.
                    dist = np.sqrt(np.sum((qtiles[i] - qtiles[j])**2) / (self.M + 1))
                elif Wdist == "W1":
                    dist = np.sum(np.abs(qtiles[i] - qtiles[j])) / (self.M + 1)
                dist_matrix[i, j] = dist
        
        if cutoffs is None:
            self.dist_matrix = dist_matrix
        elif change:
            self.dist_matrix_amalg = dist_matrix

        return dist_matrix

    def __repr__(self):
        return f"Distributions of {len(self.data)} subjects"


##########################################################
######### Optimizations for data-driven cutoffs ##########
##########################################################

def fitness(cutoffs, data_class, loss, fixed=None, Wdist="W2"):
    """
    Fitness function for the differential evolution algorithm
    """
    if loss == "Loss1":
        return np.mean(data_class.dists_for_Loss1(cutoffs, fixed, Wdist=Wdist))
    elif loss == "Loss2":
        mat_orig = data_class.dist_matrix    # computed only once
        mat_amalg = data_class.Wdist_matrix(cutoffs, fixed=fixed, Wdist=Wdist)
        n = mat_orig.shape[0] 
        return np.sum((mat_orig - mat_amalg) ** 2) * 2 / n / (n - 1)
    

def run_de(data_class, loss, K=4, fixed=None, Wdist="W2",
            maxiter=1000, popsize=15, tol=1e-5, init=None, polish=False,
            disp=False, seed=None, **params):
    """
    Run the differential evolution (DE) algorithm for optimizing cutoffs.

    Parameters:
        data_class (Distribution): data converted to the class Distribution
        loss (string): "Loss1" or "Loss2"
        K (int): Number of cutoffs to optimize.
        fixed (list): Optional fixed thresholds.
        Wdist (str): Distance metric, "W1" or "W2".
        maxiter (int): Maximum number of iterations.
        popsize (int): Population size for the DE algorithm.
        tol (float): Convergence tolerance for DE.
        init (str or None): Initialization method, "custom" or None.
        polish (bool): Whether to refine the result with local optimization.
        disp (bool): Display optimization process.
        seed (int or None): Random seed for reproducibility.
        params (dict): Additional parameters for differential_evolution.

    Returns:
        tuple: Optimized cutoffs and the minimum loss value.
    """
    assert loss in ["Loss1", "Loss2", "BC"], "Invalid loss function"

    # Precompute distance matrix for Loss2
    if loss == "Loss2":
        data_class.Wdist_matrix(Wdist=Wdist)

    ran = data_class.ran
    bounds = [ran] * K  # Bounds for cutoffs

    # Linear constraint for monotonicity
    constr_matrix = (np.eye(K, k=1) - np.eye(K))[:-1]
    linear_constraint = LinearConstraint(constr_matrix, 
                                         lb=0., ub=np.inf) if K > 1 else ()
    
    result = differential_evolution(
        func=fitness,
        bounds=bounds,
        args=(data_class, loss, fixed, Wdist),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,       # Relative tolerance for convergence,
        seed=seed,
        callback=None,
        disp=disp,
        polish=polish,
        constraints=linear_constraint,
        **params
    )
    
    res = np.sort(np.r_[fixed, result.x]) if fixed is not None else result.x
    best_cutoffs = np.concatenate(([ran[0]], res, [ran[1]]))
    min_loss = result.fun
    
    return best_cutoffs, min_loss



### Greedy approaches for threshold optimization

def agglomerative_discrete(data_class, K, loss, thresholds=None, Wdist="W2", verbose=False):
    """
    Iteratively merge thresholds to achieve the target number of bins using a greedy approach.

    Parameters:
        data_class (Distribution): Instance of Distribution class.
        K (int): Target number of thresholds.
        loss (str): Loss function type, "Loss1" or "Loss2".
        thresholds (list): Initial thresholds; if None, defaults to range-based thresholds.
        Wdist (str): Distance metric, "W1" or "W2".
        verbose (bool): Whether to print progress.

    Returns:
        dict: Final thresholds and history of changes.
    """
    assert loss in ["Loss1", "Loss2"], "Invalid loss function specified"

    if thresholds is None:
        thresholds = list(np.arange(data_class.ran[0] + 1, data_class.ran[1] + 1, 1))

    if loss == "Loss2":
        data_class.Wdist_matrix(Wdist=Wdist)  # precompute the distance matrix for the original data

    loss_history = {}
    thres_history = [thresholds]

    while len(thresholds) > K:
        best_loss = np.inf
        best_threshold = None

        # Evaluate optimal threshold to delete
        for potential_threshold in thresholds:
            # Remove the potential threshold and calculate loss
            temp_thresholds = thresholds.copy()
            temp_thresholds.remove(potential_threshold)
            loss_val = fitness(temp_thresholds, data_class, loss=loss, Wdist=Wdist)

            if loss_val < best_loss:
                best_loss = loss_val
                best_threshold = potential_threshold

        # Remove the threshold minimizing the loss
        thresholds.remove(best_threshold)
        if verbose:
            print("Removed threshold:", best_threshold)

        loss_history[len(thresholds)] = best_loss
        thres_history.append([thresholds.copy()])

    print("Final loss:", best_loss)
    
    return {
        'thresholds': thresholds,
        'loss_history': loss_history,
        'thresholds_history': thres_history
    }


def divisive_discrete(data_class, K, loss, thresholds=None, thre_list=None, Wdist="W2"):
    """
    Stepwise divisive thresholding of histograms based on given loss function.

    Parameters:
        data_class (Distribution): Instance of Distribution class.
        K (int): Target number of thresholds.
        loss (str): Loss function type, "Loss1" or "Loss2".
        thresholds (list): Initial fixed thresholds.
        thre_list (list): List of candidate thresholds; defaults to integer values in the data range.
        Wdist (str): Distance metric, "W1" or "W2".

    Returns:
        dict: Final thresholds and their history.
    """
    assert loss in ["Loss1", "Loss2"], "Invalid loss function string"

    loss_history = {}
    thres_history = []

    if thresholds is None:
        thresholds = []
    else:
        assert isinstance(thresholds, list), "Fixed thresholds must be a list"

    if thre_list is None:
        thre_list = np.arange(data_class.ran[0] + 1, data_class.ran[1] + 1, 1)
    
    if not all(threshold in thre_list for threshold in thresholds):
        raise ValueError("Invalid fixed thresholds. Some values are not in thre_list.")

    if loss == "Loss2":
        # precompute the distance matrix for the original data
        data_class.Wdist_matrix(Wdist=Wdist)   

    # Iterative search for the optimal thresholds
    while len(thresholds) < K:
        best_loss = np.inf
        best_threshold = None

        # Evaluate potential new thresholds
        for potential_threshold in thre_list:
            if potential_threshold in thresholds:
                continue

            # Insert the potential threshold and calculate loss
            temp_thresholds = sorted(thresholds + [potential_threshold])
            loss_val = fitness(temp_thresholds, data_class, loss=loss, Wdist=Wdist)

            if loss_val < best_loss:
                best_loss = loss_val
                best_threshold = potential_threshold

        # Update thresholds with the best found threshold
        thresholds.append(best_threshold)
        thresholds.sort()
        print("Picked thresholds:", thresholds)

        loss_history[len(thresholds)] = best_loss
        thres_history.append(best_threshold)

    print("Final loss:", best_loss)

    return {
        'thresholds': thresholds,
        'loss_history': loss_history,
        'thresholds_history': thres_history
    }


###########################################################
# Bray-Curtis criterion for agglomerative amalgamation (the method PAA)
# Adapted from the paper: Principal amalgamation analysis for microbiome data (2022), by Li et al.

def agglomerative_BC(data_class, K, thresholds=None, verbose=False):
    """
    Agglomerative method for reducing thresholds based on Bray-Curtis criterion.

    Parameters:
        data_class (Distribution): Instance of Distribution class.
        K (int): Target number of thresholds.
        thresholds (list): Initial thresholds; if None, defaults integer values in the data range
        verbose (bool): Whether to print progress.

    Returns:
        dict: Final thresholds and their history.
    """
    if thresholds is None:
        thresholds = list(np.arange(data_class.ran[0] - 0.5, data_class.ran[1] + 0.5, 1))
    a, b = data_class.ran[0] - 0.5, data_class.ran[1] + 0.5
    cutoffs = np.r_[a, thresholds, b]

    n_obs = np.sum(np.histogram(data_class.data[0], bins=cutoffs)[0])
    compo_list = [np.histogram(data_class.data[i], bins=cutoffs)[0] / n_obs for i in range(len(data_class.data))]
    compo_matrix = np.array(compo_list)

    thres_history = []

    while len(thresholds) > K:
        criterion = np.zeros(len(thresholds))
        for l in range(len(thresholds)):
            criterion[l] = BC_criterion(compo_matrix, l)

        l = np.argmin(criterion)
        removed = thresholds.pop(l)
        if verbose:
            print("Removed threshold:", removed)
        compo_matrix[:, l] += compo_matrix[:, l+1]
        compo_matrix = np.delete(compo_matrix, l+1, axis=1)
        thres_history.append(thresholds.copy())

    return {
        "thresholds": thresholds,
        "thres_history": thres_history
    }


def BC_criterion(compo_matrix, l):
    """
    Compute Bray-Curtis criterion for merging thresholds, derived by Li et al. (2022).

    Parameters:
        compo_matrix (np.ndarray): Composition matrix.
        l (int): Index of threshold to evaluate.

    Returns:
        float: Computed Bray-Curtis criterion value.
    """
    n = compo_matrix.shape[0]
    val = 0
    for i in range(n):
        for j in range(i + 1, n):
            val += (min(compo_matrix[i, l], compo_matrix[j, l]) + min(compo_matrix[i, l+1], compo_matrix[j, l+1]) - 
                    min(compo_matrix[i, l] + compo_matrix[i, l+1], compo_matrix[j, l] + compo_matrix[j, l+1])) ** 2
    return val
