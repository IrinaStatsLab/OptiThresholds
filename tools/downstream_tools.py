from __future__ import annotations

"""Generic downstream helpers for threshold-based real-data workflows.

This module contains reusable feature-construction helpers, model-comparison
utilities, and Wasserstein-regression bridges used across notebooks and small
scripts. Caller-specific threshold sets should stay at the notebook or script
layer and be passed into the helpers here.
"""

from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from tools.r_tools import ensure_r_packages, load_rpy2, r_literal

__all__ = [
    'DEFAULT_FPCA_OPTIONS',
    'ensure_r_packages',
    'calculate_tir_metrics',
    'tir_column_names',
    'add_tir_metrics',
    'build_tir_modeling_data',
    'pooled_quantile_thresholds',
    'compute_quantile_matrix',
    'r_squared',
    'clarke_test',
    'fit_wr_scalar_model',
    'fit_wr_binomial_model',
]

# Wasserstein regression hyperparameters
DEFAULT_FPCA_OPTIONS: dict[str, object] = {
    'methodSelectK': 'FVE',
    'FVEthreshold': 0.95,
    'useBinnedData': 'OFF',
}

_R_HELPERS_INITIALIZED = False

_R_HELPERS = r"""
fit_wr_binomial_internal <- function(qf_matrix, y, qSup, fpca_options, link_name) {
  y <- as.integer(y)
  qSup <- as.numeric(qSup)
  fit <- WR::WR(
    X = list(glucose = qf_matrix),
    Y = as.numeric(y),
    qSup = qSup,
    FPCAoptnsX = fpca_options
  )

  xi <- as.matrix(fit$fpcaLogX[[1]]$xiEst)
  if (is.null(dim(xi))) {
    xi <- matrix(xi, ncol = 1)
  }

  dfReg <- as.data.frame(xi)
  names(dfReg) <- paste0('xi_', seq_len(ncol(dfReg)))
  dfReg$y <- y

  second_stage <- glm(y ~ ., data = dfReg, family = binomial(link = link_name))
  fitted_prob <- as.numeric(fitted(second_stage))
  linear_predictor <- as.numeric(predict(second_stage, type = 'link'))

  list(
    fit = fit,
    yfit = as.numeric(fit$Yfit),
    beta = as.numeric(fit$beta[[1]]),
    work_grid = as.numeric(fit$workGridX[[1]]),
    xi = xi,
    second_stage_fitted = fitted_prob,
    second_stage_linear_predictor = linear_predictor,
    second_stage_residuals = as.numeric(residuals(second_stage, type = 'response')),
    second_stage_aic = as.numeric(AIC(second_stage)),
    second_stage_ncoef = length(coef(second_stage)),
    second_stage_deviance = as.numeric(second_stage$deviance),
    second_stage_null_deviance = as.numeric(second_stage$null.deviance)
  )
}
"""


def calculate_tir_metrics(gl_values, thresholds=[70, 181], minmax=[40, 401]):
    """Return compositional TIR proportions for one glucose trajectory."""
    gl_array = np.asarray(gl_values, dtype=float)
    total_readings = len(gl_array)
    extended_thresholds = [minmax[0]] + list(thresholds) + [minmax[1]]

    tir_list = []
    for i in range(len(extended_thresholds) - 1):
        lower = extended_thresholds[i]
        upper = extended_thresholds[i + 1]
        if lower >= upper:
            raise ValueError('Thresholds must be in ascending order.')
        tir = np.sum((gl_array >= lower) & (gl_array < upper)) / total_readings
        tir_list.append(tir)

    return tir_list


def tir_column_names(thresholds, minmax=(40, 401)) -> list[str]:
    """Return the TIR column names induced by a threshold vector."""
    extended_thresholds = [minmax[0]] + list(thresholds) + [minmax[1]]
    return [
        f'TIR_{ceil(extended_thresholds[i])}_{ceil(extended_thresholds[i + 1] - 1)}'
        for i in range(len(extended_thresholds) - 1)
    ]


def add_tir_metrics(df, thresholds=[70, 181], minmax=[40, 401]):
    """Append TIR composition columns to a dataframe containing `gl` lists."""
    tir_metrics = df['gl'].apply(lambda x: calculate_tir_metrics(x, thresholds, minmax))
    tir_metrics_df = tir_metrics.apply(lambda x: pd.Series(x))
    tir_metrics_df.columns = tir_column_names(thresholds, minmax=minmax)
    return pd.concat([df, tir_metrics_df], axis=1)


def build_tir_modeling_data(
    df: pd.DataFrame,
    threshold_sets: Mapping[str, Sequence[float]],
    *,
    drop_gl: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add TIR features for caller-supplied threshold sets.

    The caller owns the threshold-set names and cutoff values. This helper only
    appends the matching TIR columns and returns the derived feature map.
    """
    modeling_data = df.copy()
    feature_map: dict[str, list[str]] = {}

    for label, thresholds in threshold_sets.items():
        threshold_list = list(thresholds)
        modeling_data = add_tir_metrics(modeling_data, thresholds=threshold_list)
        feature_map[label] = tir_column_names(threshold_list)

    modeling_data = modeling_data.loc[:, ~modeling_data.columns.duplicated()].copy()
    if drop_gl and 'gl' in modeling_data.columns:
        modeling_data = modeling_data.drop(columns=['gl'])

    return modeling_data, feature_map


def pooled_quantile_thresholds(
    glucose_lists: Sequence[Sequence[float] | np.ndarray],
    threshold_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled probability levels and empirical quantile cutoffs."""
    pooled = np.concatenate([np.asarray(values, dtype=float) for values in glucose_lists])
    probs = np.arange(1, threshold_count + 1, dtype=float) / (threshold_count + 1)
    cutoffs = np.quantile(pooled, probs)
    if np.any(np.diff(cutoffs) <= 0):
        raise ValueError(f'Naive thresholds for K={threshold_count} are not strictly increasing: {cutoffs}')
    return probs, cutoffs


def compute_quantile_matrix(glucose_lists, probs: np.ndarray) -> np.ndarray:
    """Construct the subject-by-grid quantile matrix used by WR-style models."""
    probs = np.asarray(probs, dtype=float)
    return np.vstack([np.quantile(np.asarray(values, dtype=float), probs) for values in glucose_lists])


def r_squared(y: Sequence[float], fitted_values: Sequence[float]) -> float:
    """Compute the standard coefficient of determination from fitted values."""
    y = np.asarray(y, dtype=float)
    fitted_values = np.asarray(fitted_values, dtype=float)
    sse = np.sum((y - fitted_values) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - sse / sst)


def _clarke_preference_details(
    statistic: int,
    nobs: int,
    p_value: float,
    model_a_label: str,
    model_b_label: str,
) -> dict[str, Any]:
    """Summarize the preferred model implied by a Clarke test result."""
    wins_model_a = int(statistic)
    wins_model_b = int(nobs - statistic)
    half = nobs / 2.0

    if statistic > half:
        preferred_model = model_a_label
        preferred_index = 1
        preferred_wins = wins_model_a
    elif statistic < half:
        preferred_model = model_b_label
        preferred_index = 2
        preferred_wins = wins_model_b
    else:
        preferred_model = None
        preferred_index = None
        preferred_wins = wins_model_a

    return {
        'stat': wins_model_a,
        'nobs': int(nobs),
        'p_value': float(p_value),
        'wins_model_a': wins_model_a,
        'wins_model_b': wins_model_b,
        'model_a_label': model_a_label,
        'model_b_label': model_b_label,
        'preferred_model': preferred_model,
        'preferred_index': preferred_index,
        'preferred_wins': int(preferred_wins),
    }


def clarke_test(
    df,
    response,
    model_a_vars,
    model_b_vars,
    digits=3,
    return_details: bool = False,
    model_a_label: str = 'Model A',
    model_b_label: str = 'Model B',
):
    """Run the R `clarkeTest` comparison for two non-nested linear models."""
    ro, default_converter, _, localconverter = load_rpy2()
    from rpy2.robjects import Formula, pandas2ri
    from rpy2.robjects.packages import importr

    clarke = importr('clarkeTest')
    stats = importr('stats')

    with localconverter(default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)

    formula_a = Formula(f"{response} ~ " + ' + '.join(model_a_vars))
    formula_b = Formula(f"{response} ~ " + ' + '.join(model_b_vars))

    model_a = stats.lm(formula_a, data=r_df)
    model_b = stats.lm(formula_b, data=r_df)
    clarke_result = clarke.clarke_test(model_a, model_b, digits=digits)

    statistic = clarke_result.rx2('stat')[0]
    n = clarke_result.rx2('nobs')[0]
    p_value = binomtest(statistic, n=n, p=0.5, alternative='two-sided').pvalue
    if return_details:
        return _clarke_preference_details(
            statistic=int(statistic),
            nobs=int(n),
            p_value=float(p_value),
            model_a_label=model_a_label,
            model_b_label=model_b_label,
        )
    return statistic, p_value


def _initialize_r_helpers(ro) -> None:
    """Register the WR-specific R helper block once per Python session."""
    global _R_HELPERS_INITIALIZED
    if _R_HELPERS_INITIALIZED:
        return

    ro.r(_R_HELPERS)
    _R_HELPERS_INITIALIZED = True


def fit_wr_scalar_model(
    y: Sequence[float],
    qf_matrix_nq: np.ndarray,
    probs: Sequence[float],
    fpca_options: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Fit scalar-response Wasserstein regression through the WR R package."""
    ro, default_converter, numpy2ri, localconverter = load_rpy2()

    y = np.asarray(y, dtype=float)
    probs = np.asarray(probs, dtype=float)
    qf_matrix_nq = np.asarray(qf_matrix_nq, dtype=float)
    qf_matrix_qn = qf_matrix_nq.T
    resolved_fpca_options = {**DEFAULT_FPCA_OPTIONS, **(fpca_options or {})}
    fpca_expr = ', '.join(f"{key} = {r_literal(value)}" for key, value in resolved_fpca_options.items())

    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['qf_matrix_py'] = numpy2ri.py2rpy(qf_matrix_qn)
        ro.globalenv['y_py'] = numpy2ri.py2rpy(y)
        ro.globalenv['qsup_py'] = numpy2ri.py2rpy(probs)

    ro.r(f"fpca_options_py <- list({fpca_expr})")
    fit_obj = ro.r(
        'WR::WR(X = list(glucose = qf_matrix_py), Y = as.numeric(y_py), qSup = as.numeric(qsup_py), FPCAoptnsX = fpca_options_py)'
    )
    fitted_wr = np.asarray(fit_obj.rx2('Yfit'), dtype=float)

    return {
        'fit_obj': fit_obj,
        'fitted': fitted_wr,
    }


def fit_wr_binomial_model(
    y: Sequence[int],
    qf_matrix_nq: np.ndarray,
    probs: Sequence[float],
    fpca_options: dict[str, object] | None = None,
    link: str = 'logit',
) -> dict[str, Any]:
    """Fit a WR-based predictor representation with a binomial GLM second stage."""
    ro, default_converter, numpy2ri, localconverter = load_rpy2()
    _initialize_r_helpers(ro)

    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    qf_matrix_nq = np.asarray(qf_matrix_nq, dtype=float)
    qf_matrix_qn = qf_matrix_nq.T
    resolved_fpca_options = {**DEFAULT_FPCA_OPTIONS, **(fpca_options or {})}
    fpca_expr = ', '.join(f"{key} = {r_literal(value)}" for key, value in resolved_fpca_options.items())

    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['qf_matrix_py'] = numpy2ri.py2rpy(qf_matrix_qn)
        ro.globalenv['y_py'] = numpy2ri.py2rpy(y)
        ro.globalenv['qsup_py'] = numpy2ri.py2rpy(probs)

    ro.r(f"fpca_options_py <- list({fpca_expr})")
    ro.globalenv['link_name_py'] = link
    result = ro.r('fit_wr_binomial_internal(qf_matrix_py, y_py, qsup_py, fpca_options_py, link_name_py)')

    fit_obj = result.rx2('fit')
    fitted_wr = np.asarray(result.rx2('yfit'), dtype=float)
    fitted_prob = np.asarray(result.rx2('second_stage_fitted'), dtype=float)
    linear_predictor = np.asarray(result.rx2('second_stage_linear_predictor'), dtype=float)
    residuals_second_stage = np.asarray(result.rx2('second_stage_residuals'), dtype=float)
    beta = np.asarray(result.rx2('beta'), dtype=float)
    work_grid = np.asarray(result.rx2('work_grid'), dtype=float)
    xi = np.asarray(result.rx2('xi'), dtype=float)

    conditional_model = {
        'fitted': fitted_prob,
        'residuals': residuals_second_stage,
        'aic': float(np.asarray(result.rx2('second_stage_aic'), dtype=float)[0]),
        'npar': int(np.asarray(result.rx2('second_stage_ncoef'), dtype=int)[0]),
        'deviance': float(np.asarray(result.rx2('second_stage_deviance'), dtype=float)[0]),
        'null_deviance': float(np.asarray(result.rx2('second_stage_null_deviance'), dtype=float)[0]),
        'link': link,
    }

    return {
        'fit_obj': fit_obj,
        'wr_scalar_fitted': fitted_wr,
        'fitted_prob': fitted_prob,
        'linear_predictor': linear_predictor,
        'beta': beta,
        'work_grid': work_grid,
        'xi': xi,
        'conditional_model': conditional_model,
    }
