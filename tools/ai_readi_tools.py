"""AI-READI data-processing helpers used by real-data notebooks."""

from __future__ import annotations

from math import ceil
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

AI_READI_STUDY_GROUPS = (
    'healthy',
    'pre_diabetes_lifestyle_controlled',
)

RESPONSE_LABELS: dict[str, str] = {
    'hdl_c': 'HDL-C',
    'log_triglycerides': 'TG',
    'AIP': 'TG/HDL-C',
    'hba1c': 'HbA1c',
}



def calculate_tir_metrics(gl_values, thresholds=[70, 181], minmax=[40, 401]):
    """Return compositional TIR proportions for one glucose trajectory.

    Parameters follow the threshold convention used across the real-data
    notebooks. The returned list sums to one across the induced glucose bins.
    """
    gl_array = np.array(gl_values)
    total_readings = len(gl_array)

    extended_thresholds = [minmax[0]] + list(thresholds) + [minmax[1]]

    tir_list = []
    for i in range(len(extended_thresholds) - 1):
        if extended_thresholds[i] >= extended_thresholds[i + 1]:
            raise ValueError('Thresholds must be in ascending order.')
        tir = np.sum((gl_array >= extended_thresholds[i]) & (gl_array < extended_thresholds[i + 1])) / total_readings
        tir_list.append(tir)

    return tir_list


def add_tir_metrics(df, thresholds=[70, 181], minmax=[40, 401]):
    """Append TIR composition columns to a dataframe containing `gl` lists.

    Typical use is to call this once per threshold set before fitting a linear
    model on the resulting compositional predictors.
    """
    tir_metrics = df['gl'].apply(lambda x: calculate_tir_metrics(x, thresholds, minmax))
    extended_thresholds = [minmax[0]] + list(thresholds) + [minmax[1]]

    column_names = [
        f'TIR_{ceil(extended_thresholds[i])}_{ceil(extended_thresholds[i + 1] - 1)}'
        for i in range(len(extended_thresholds) - 1)
    ]

    tir_metrics_df = tir_metrics.apply(lambda x: pd.Series(x))
    tir_metrics_df.columns = column_names
    return pd.concat([df, tir_metrics_df], axis=1)

def load_ai_readi_cohort(repo_root: Path | str) -> pd.DataFrame:
    """Load the subject-level AI-READI cohort used by comparison notebooks.

    The function aggregates glucose trajectories by subject, merges metadata,
    applies the fixed study-group and HbA1c filters, and creates the transformed
    lipid outcomes used by the prediction notebooks.
    """
    repo_root = Path(repo_root)
    data_dir = repo_root / 'data'

    data = pd.read_csv(data_dir / 'ai-ready.csv')
    grouped_data = data.groupby('id', as_index=False).agg({'gl': list})

    metadata = pd.read_csv(data_dir / '2025-10-06_metadata.csv')
    metadata = metadata.rename(
        columns={
            'participant_id': 'id',
            'HbA1c (%)': 'hba1c',
            'Triglycerides (mg/dL)': 'triglycerides',
            'HDL Cholesterol (mg/dL)': 'hdl_c',
            'Total Cholesterol (mg/dL)': 'total_c',
            'LDL Cholesterol Calculation (mg/dL)': 'ldl_c',
        }
    )
    metadata = metadata[
        ['id', 'age', 'study_group', 'ldl_c', 'hdl_c', 'total_c', 'triglycerides', 'hba1c']
    ]

    cohort = grouped_data.merge(metadata, on='id', how='inner')
    cohort = cohort.loc[cohort['study_group'].isin(AI_READI_STUDY_GROUPS)].copy()

    cohort['log_triglycerides'] = np.nan
    positive_triglycerides = cohort['triglycerides'] > 0
    cohort.loc[positive_triglycerides, 'log_triglycerides'] = np.log(
        cohort.loc[positive_triglycerides, 'triglycerides']
    )
    cohort['AIP'] = cohort['log_triglycerides'] - np.log(cohort['hdl_c'])

    cohort = cohort.dropna().copy()
    cohort = cohort.loc[cohort['hba1c'].between(2.5, 8, inclusive='left')].copy()
    return cohort.reset_index(drop=True)


def format_response_name(response: str) -> str:
    """Map an internal response name to the label used in notebook tables."""
    return RESPONSE_LABELS.get(response, response)
