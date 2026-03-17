from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from statsmodels.formula.api import ols

from tools.ai_readi_tools import add_tir_metrics, load_ai_readi_cohort
from tools.prediction_experim_functions import clarke_gaussian, clarke_test, pack_lm_model
from tools.r_tools import setup_r_environment


RESPONSES = ["hdl_c", "log_triglycerides", "AIP"]
TRADITIONAL_THRESHOLDS = {
    "K=2": [70, 181],
    "K=4": [54, 70, 181, 251],
}
DE_THRESHOLDS = {
    "K=2": [96, 170],
    "K=4": [90, 128, 172, 232],
}


def build_comparison_frame() -> pd.DataFrame:
    """Recreate the AI-READI modeling dataframe needed for lm-vs-lm checks."""
    filtered_data = load_ai_readi_cohort(REPO_ROOT)
    modeling_data = filtered_data.copy()

    for thresholds in [*TRADITIONAL_THRESHOLDS.values(), *DE_THRESHOLDS.values()]:
        modeling_data = add_tir_metrics(modeling_data, thresholds=thresholds)

    return modeling_data.loc[:, ~modeling_data.columns.duplicated()].copy()


def tir_columns(thresholds: list[int]) -> list[str]:
    """Return the TIR column names induced by a threshold set."""
    bounds = [40] + list(thresholds) + [401]
    return [f"TIR_{bounds[i]}_{bounds[i + 1] - 1}" for i in range(len(bounds) - 1)]


def compare_clarke_methods() -> pd.DataFrame:
    """Compare the R Clarke test to the Gaussian plug-in version on identical lm pairs."""
    setup_r_environment()
    modeling_data = build_comparison_frame()
    rows: list[dict[str, object]] = []

    for label in ["K=2", "K=4"]:
        trad_cols = tir_columns(TRADITIONAL_THRESHOLDS[label])
        de_cols = tir_columns(DE_THRESHOLDS[label])

        for response in RESPONSES:
            model_trad = ols(f"{response} ~ " + " + ".join(trad_cols[:-1]), data=modeling_data).fit()
            model_de = ols(f"{response} ~ " + " + ".join(de_cols[:-1]), data=modeling_data).fit()

            stat_r, p_r = clarke_test(modeling_data, response, trad_cols[:-1], de_cols[:-1])
            gaussian = clarke_gaussian(
                pack_lm_model(model_trad),
                pack_lm_model(model_de),
                modeling_data[response].to_numpy(),
            )

            rows.append(
                {
                    "comparison": label,
                    "response": response,
                    "stat_r_clarke": float(stat_r),
                    "p_r_clarke": float(p_r),
                    "stat_gaussian": float(gaussian["stat"]),
                    "p_gaussian": float(gaussian["p_value"]),
                    "stat_diff": float(stat_r - gaussian["stat"]),
                    "p_diff": float(p_r - gaussian["p_value"]),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result_df = compare_clarke_methods()
    print(result_df.to_string(index=False))
