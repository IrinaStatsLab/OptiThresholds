from math import ceil
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def calculate_tir_metrics(gl_values, thresholds=[70, 181], minmax=[40, 401]):
    """Calculate Time in Range metrics for given glucose values and thresholds"""
    gl_array = np.array(gl_values)
    total_readings = len(gl_array)

    extended_thresholds = [minmax[0]] + list(thresholds) + [minmax[1]]

    tir_list = []
    for i in range(len(extended_thresholds) - 1):
        if extended_thresholds[i] >= extended_thresholds[i + 1]:
            raise ValueError("Thresholds must be in ascending order.")
        tir = np.sum((gl_array >= extended_thresholds[i]) & (gl_array < extended_thresholds[i + 1])) / total_readings
        tir_list.append(tir) 
    
    return tir_list


def add_tir_metrics(df, thresholds=[70, 181], minmax=[40, 401]):
    """Add TIR metrics to the DataFrame"""
    tir_metrics = df['gl'].apply(lambda x: calculate_tir_metrics(x, thresholds, minmax))
    extended_thresholds = [minmax[0]] + list(thresholds) + [minmax[1]]

    column_names = [f'TIR_{ceil(extended_thresholds[i])}_{ceil(extended_thresholds[i + 1] - 1)}'
                     for i in range(len(extended_thresholds) - 1)]

    tir_metrics_df = tir_metrics.apply(lambda x: pd.Series(x))
    tir_metrics_df.columns = column_names
    df = pd.concat([df, tir_metrics_df], axis=1)
    
    return df


def setup_r_environment():
    """
    Setup R environment with automatic path detection.
    
    This function attempts to locate R installation automatically and
    sets up the necessary environment variables for rpy2 to work.
    
    Raises:
        RuntimeError: If R installation cannot be found
    """
    if 'R_HOME' not in os.environ:
        # Try common R installation paths on Windows
        possible_paths = [
            r'C:\Program Files\R\R-4.4.1',
            r'C:\Program Files\R\R-4.4.0',
            r'C:\Program Files\R\R-4.3.3',
            r'C:\Program Files\R\R-4.3.2', 
            r'C:\Program Files\R\R-4.3.1',
            r'C:\Program Files\R\R-4.3.0',
            r'C:\Program Files\R\R-4.2.3',
            r'C:\Program Files\R\R-4.2.2',
            r'C:\Program Files\R\R-4.2.1',
            r'C:\Program Files\R\R-4.2.0',
            r'C:\Program Files\R\R-4.1.3',
            # Alternative installation paths
            r'C:\Program Files (x86)\R\R-4.4.1',
            r'C:\Program Files (x86)\R\R-4.3.3',
            r'C:\Users\%USERNAME%\Documents\R\R-4.4.1',
            # Unix/Linux paths (in case this runs on other systems)
            '/usr/lib/R',
            '/usr/local/lib/R',
            '/opt/R',
        ]
        
        r_home = None
        for path in possible_paths:
            # Expand environment variables like %USERNAME%
            expanded_path = os.path.expandvars(path)
            if Path(expanded_path).exists():
                r_home = expanded_path
                break
        
        if r_home is None:
            raise RuntimeError(
                "R installation not found. Please either:\n"
                "1. Install R from https://cran.r-project.org/\n" 
                "2. Set R_HOME environment variable manually\n"
                "3. Add R to your system PATH\n\n"
                f"Searched in the following locations:\n" + 
                "\n".join([f"  - {os.path.expandvars(p)}" for p in possible_paths])
            )
        
        os.environ['R_HOME'] = r_home
        
        # Set up PATH for R binaries
        if sys.platform.startswith('win'):
            bin_path = Path(r_home) / 'bin' / 'x64'
        else:
            bin_path = Path(r_home) / 'bin'
            
        if bin_path.exists():
            os.environ['PATH'] += os.pathsep + str(bin_path)
        
        print(f"Using R installation at: {r_home}")
    else:
        print(f"Using existing R_HOME: {os.environ['R_HOME']}")


def clarke_test(df, response, model_a_vars, model_b_vars, digits=3):
    """
    Performs Clarke test to compare two non-nested models.
    """
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import Formula

    clarke = importr('clarkeTest')
    stats = importr('stats')

    # Activate automatic conversion between pandas and R data frames
    pandas2ri.activate()

    # Convert DataFrame to R data frame
    r_df = pandas2ri.py2rpy(df)

    # Create formulas for the two models
    formula_a = Formula(f"{response} ~ " + " + ".join(model_a_vars))
    formula_b = Formula(f"{response} ~ " + " + ".join(model_b_vars))

    # Fit the two models using R's lm function
    model_a = stats.lm(formula_a, data=r_df)
    model_b = stats.lm(formula_b, data=r_df)

    # Perform Clarke test using clarke::clarke_test
    clarke_result = clarke.clarke_test(model_a, model_b, digits=digits)

    from scipy.stats import binomtest
    statistic = clarke_result.rx2('stat')[0]
    n = clarke_result.rx2('nobs')[0]
    p_value = binomtest(statistic, n=n, p=0.5, alternative='two-sided').pvalue

    return statistic, p_value 