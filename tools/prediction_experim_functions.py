from __future__ import annotations

"""Prediction and model-comparison helpers for real-data notebook workflows.

This module contains prediction-side utilities: model packing, Gaussian
comparison summaries, Clarke's test, and the Wasserstein-regression bridge to
R. Data-loading and feature-construction helpers live in `tools.ai_readi_tools`.
"""

import json
from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest, norm

from tools.ai_readi_tools import add_tir_metrics
from tools.r_tools import ensure_r_packages, load_rpy2, r_literal

__all__ = [
    'DEFAULT_FPCA_OPTIONS',
    'ensure_r_packages',
    'build_wr_modeling_data',
    'compute_quantile_matrix',
    'pack_lm_model',
    'r_squared',
    'clarke_test',
    'clarke_gaussian',
    'fit_wr_scalar_model',
]

# Wasserstein regression hyperparameters
DEFAULT_FPCA_OPTIONS: dict[str, object] = {
    'methodSelectK': 'FVE',
    'FVEthreshold': 0.95,
    'useBinnedData': 'OFF',
}

DE_THRESHOLD_SETS: dict[str, list[int]] = {
    'de_k2': [96, 170],
    'de_k4': [90, 128, 172, 232],
}

DE_FEATURE_MAP: dict[str, list[str]] = {
    'de_k2': ['TIR_40_95', 'TIR_96_169', 'TIR_170_400'],
    'de_k4': ['TIR_40_89', 'TIR_90_127', 'TIR_128_171', 'TIR_172_231', 'TIR_232_400'],
}

_R_HELPERS_INITIALIZED = False

_R_HELPERS = r"""
fit_wr_scalar_internal <- function(qf_matrix, y, qSup, fpca_options) {
  y <- as.numeric(y)
  qSup <- as.numeric(qSup)
  fit <- WR::WR(
    X = list(glucose = qf_matrix),
    Y = y,
    qSup = qSup,
    FPCAoptnsX = fpca_options
  )

  xi <- as.matrix(fit$fpcaLogX[[1]]$xiEst)
  if (is.null(dim(xi))) {
    xi <- matrix(xi, ncol = 1)
  }

  dfReg <- as.data.frame(xi)
  names(dfReg) <- paste0('xi_', seq_len(ncol(dfReg)))
  dfReg$y_centered <- y - mean(y)

  second_stage <- lm(y_centered ~ 0 + ., data = dfReg)
  centered_fitted <- as.numeric(fitted(second_stage))
  fitted_values <- centered_fitted + mean(y)

  list(
    fit = fit,
    yfit = as.numeric(fit$Yfit),
    beta = as.numeric(fit$beta[[1]]),
    work_grid = as.numeric(fit$workGridX[[1]]),
    xi = xi,
    second_stage_fitted = fitted_values,
    second_stage_residuals = as.numeric(resid(second_stage)),
    second_stage_aic = as.numeric(AIC(second_stage)),
    second_stage_ncoef = length(coef(second_stage))
  )
}
"""


def build_wr_modeling_data(filtered_data) -> tuple:
    """Add the settled DE compositional baselines used in prediction notebooks.

    The returned feature map identifies which TIR columns belong to the K=2 and
    K=4 DE threshold sets.
    """
    modeling_data = filtered_data.copy()
    for thresholds in DE_THRESHOLD_SETS.values():
        modeling_data = add_tir_metrics(modeling_data, thresholds=thresholds)

    modeling_data = modeling_data.loc[:, ~modeling_data.columns.duplicated()].copy()
    feature_map = {name: columns.copy() for name, columns in DE_FEATURE_MAP.items()}
    return modeling_data, feature_map


def compute_quantile_matrix(glucose_lists, probs: np.ndarray) -> np.ndarray:
    """Construct the subject-by-grid quantile matrix used by WR-style models.

    Each row corresponds to one subject and each column corresponds to one
    probability level in the supplied grid.
    """
    probs = np.asarray(probs, dtype=float)
    return np.vstack([np.quantile(np.asarray(values, dtype=float), probs) for values in glucose_lists])


def pack_lm_model(model) -> dict[str, Any]:
    """Pack a fitted OLS model into the structure used by comparison helpers.

    This keeps baseline linear fits and WR conditional fits on the same
    dictionary interface for AIC, R^2, and Clarke-style summaries.
    """
    return {
        'fitted': np.asarray(model.fittedvalues, dtype=float),
        'residuals': np.asarray(model.resid, dtype=float),
        'aic': float(model.aic),
        'npar': int(len(model.params)),
        'r2': float(model.rsquared),
        'model': model,
    }


def r_squared(y: Sequence[float], fitted_values: Sequence[float]) -> float:
    """Compute the standard coefficient of determination from fitted values."""
    y = np.asarray(y, dtype=float)
    fitted_values = np.asarray(fitted_values, dtype=float)
    sse = np.sum((y - fitted_values) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - sse / sst)


def _gaussian_individual_loglik(
    y: Sequence[float],
    fitted_values: Sequence[float],
    residuals: Sequence[float],
    npar: int,
) -> np.ndarray:
    """Return per-observation Gaussian log-likelihood contributions."""
    y = np.asarray(y, dtype=float)
    fitted_values = np.asarray(fitted_values, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n = len(y)
    sigma = np.sqrt(np.var(residuals, ddof=1) * ((n - 1) / (n - npar)))
    return norm.logpdf(y, loc=fitted_values, scale=sigma)


def clarke_test(df, response, model_a_vars, model_b_vars, digits=3):
    """Run the R `clarkeTest` comparison for two non-nested linear models.

    This helper is used by threshold and L-moment notebooks where the models are
    specified by response and predictor column names in a pandas dataframe.
    """
    ro, _, _, _ = load_rpy2()
    from rpy2.robjects import Formula, pandas2ri
    from rpy2.robjects.packages import importr

    clarke = importr('clarkeTest')
    stats = importr('stats')

    # Activate conversion so the pandas frame can be passed directly to R's lm.
    pandas2ri.activate()
    r_df = pandas2ri.py2rpy(df)

    formula_a = Formula(f"{response} ~ " + ' + '.join(model_a_vars))
    formula_b = Formula(f"{response} ~ " + ' + '.join(model_b_vars))

    model_a = stats.lm(formula_a, data=r_df)
    model_b = stats.lm(formula_b, data=r_df)
    clarke_result = clarke.clarke_test(model_a, model_b, digits=digits)

    statistic = clarke_result.rx2('stat')[0]
    n = clarke_result.rx2('nobs')[0]
    p_value = binomtest(statistic, n=n, p=0.5, alternative='two-sided').pvalue
    return statistic, p_value


def clarke_gaussian(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    y: Sequence[float],
) -> dict[str, float]:
    """Clarke test for WR method under the Gaussian assumption.

    Both model inputs must follow the structure produced by `pack_lm_model` or
    by the `conditional_model` entry returned from `fit_wr_scalar_model`.
    """
    y = np.asarray(y, dtype=float)
    loglik_a = _gaussian_individual_loglik(y, model_a['fitted'], model_a['residuals'], model_a['npar'])
    loglik_b = _gaussian_individual_loglik(y, model_b['fitted'], model_b['residuals'], model_b['npar'])
    n = len(y)
    correction = (model_a['npar'] - model_b['npar']) * (np.log(n) / (2.0 * n))
    stat = int(np.sum(loglik_a - loglik_b > correction))
    p_value = float(binomtest(stat, n=n, p=0.5, alternative='two-sided').pvalue)
    return {'stat': stat, 'p_value': p_value}


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
    """Fit scalar-response Wasserstein regression through the WR R package.

    The return value contains the raw R fit, predictor-side summaries, and a
    reconstructed second-stage Gaussian model used for conditional AIC and
    Clarke-style comparison inside the notebook.
    """
    ro, default_converter, numpy2ri, localconverter = load_rpy2()
    _initialize_r_helpers(ro)

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
    result = ro.r('fit_wr_scalar_internal(qf_matrix_py, y_py, qsup_py, fpca_options_py)')

    fit_obj = result.rx2('fit')
    fitted_wr = np.asarray(result.rx2('yfit'), dtype=float)
    fitted_second_stage = np.asarray(result.rx2('second_stage_fitted'), dtype=float)
    residuals_second_stage = np.asarray(result.rx2('second_stage_residuals'), dtype=float)
    beta = np.asarray(result.rx2('beta'), dtype=float)
    work_grid = np.asarray(result.rx2('work_grid'), dtype=float)
    xi = np.asarray(result.rx2('xi'), dtype=float)

    conditional_model = {
        'fitted': fitted_second_stage,
        'residuals': residuals_second_stage,
        'aic': float(np.asarray(result.rx2('second_stage_aic'), dtype=float)[0]),
        'npar': int(np.asarray(result.rx2('second_stage_ncoef'), dtype=int)[0]),
    }

    return {
        'fit_obj': fit_obj,
        'fitted': fitted_wr,
        'beta': beta,
        'work_grid': work_grid,
        'xi': xi,
        'conditional_model': conditional_model,
        'fit_diff_max': float(np.max(np.abs(fitted_wr - fitted_second_stage))),
    }
