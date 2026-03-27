from __future__ import annotations

"""AI-READI raw-data processing and cohort-loading helpers."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR = REPO_ROOT.parent / 'dataset'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'data'
AI_READI_CGM_FILENAME = 'ai-ready.csv'
AI_READI_METADATA_FILENAME = '2025-10-06_metadata.csv'

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

METADATA_IDENTIFIERS: tuple[str, ...] = (
    'Triglycerides (mg/dL)',
    'bmi_vsorres, BMI',
    'HbA1c (%)',
    'waist_vsorres, Waist Circumference (cm)',
    'weight_vsorres, Weight (kilograms)',
    'whr_vsorres, Waist to Hip Ratio (WHR)',
    'HDL Cholesterol (mg/dL)',
    'LDL Cholesterol Calculation (mg/dL)',
    'Total Cholesterol (mg/dL)',
)


def flatten_json(payload):
    """Flatten a nested JSON-like object into a single dictionary."""
    out: dict[str, object] = {}

    def flatten(value, name: str = '') -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                flatten(nested_value, name + key + '_')
        elif isinstance(value, list):
            for idx, nested_value in enumerate(value):
                flatten(nested_value, name + str(idx) + '_')
        else:
            out[name[:-1]] = value

    flatten(payload)
    return out


def convert_time_string_to_datetime(t_str: str) -> datetime:
    """Convert a UTC time string like `2023-08-01T20:39:33Z` to datetime."""
    return datetime.strptime(t_str, '%Y-%m-%dT%H:%M:%SZ')


def _replace_alt_glucose_value(val, low_value: float, high_value: float):
    if val == 'Low':
        return low_value
    if val == 'High':
        return high_value
    return val


def process_participant_data(
    participant_id,
    glucose_filepath,
    raw_data_dir: Path | str,
    low_value: float = 40,
    high_value: float = 400,
) -> pd.DataFrame | None:
    """Process CGM data for a single AI-READI participant."""
    raw_data_dir = Path(raw_data_dir)
    cgm_path = raw_data_dir / glucose_filepath

    try:
        with cgm_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        print(f'Error processing participant {participant_id}: {exc}')
        return None

    list_of_body_dicts = [flatten_json(observation) for observation in data['body']['cgm']]
    df_participant = pd.DataFrame.from_records(list_of_body_dicts)
    df_participant = df_participant.rename(
        columns={
            'effective_time_frame_time_interval_start_date_time': 'start_time',
            'effective_time_frame_time_interval_end_date_time': 'end_time',
        }
    )

    if df_participant.shape[0] < 260:
        print(
            '    '
            f'Participant {participant_id} excluded due to insufficient record counts (<1 day): '
            f'{df_participant.shape[0]}'
        )
        return None

    df_participant['start_dtime'] = df_participant['start_time'].map(convert_time_string_to_datetime)
    total_dur_min = (
        df_participant['start_dtime'].iloc[-1] - df_participant['start_dtime'].iloc[0]
    ).total_seconds() / 60.0
    non_missing_rate = df_participant.shape[0] / (total_dur_min / 5.0)
    if non_missing_rate < 0.7:
        print(f'    Participant {participant_id} excluded due to low non-missing rate: {non_missing_rate:.2f}')
        print(f'    Duration (day): {total_dur_min / 1440:.2f}, Records: {df_participant.shape[0]}')
        return None

    df_participant['blood_glucose_value'] = df_participant['blood_glucose_value'].map(
        lambda value: _replace_alt_glucose_value(value, low_value, high_value)
    )
    df_participant['participant_id'] = participant_id
    return df_participant


def parse_measurement(df: pd.DataFrame, identifier: str) -> pd.DataFrame:
    """Select one clinical measurement source value."""
    return df.loc[df['measurement_source_value'] == identifier].copy()


def build_ai_readi_glucose_csv(
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    output_filename: str = AI_READI_CGM_FILENAME,
) -> pd.DataFrame:
    """Build the processed AI-READI glucose CSV stored in repo `data/`."""
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    manifest_path = raw_data_dir / 'wearable_blood_glucose' / 'manifest.tsv'
    df_manifest = pd.read_csv(manifest_path, sep='\t')

    print('Starting to load data for all participants...')
    print(f'Total participants to process: {len(df_manifest)}')

    all_participants_data: list[pd.DataFrame] = []
    failed_participants: list[str] = []

    for idx, row in df_manifest.iterrows():
        participant_id = row['participant_id']
        glucose_filepath = row['glucose_filepath']
        if idx % 100 == 0:
            print(f'Processing participant {idx + 1}/{len(df_manifest)}: ID {participant_id}')

        df_participant = process_participant_data(
            participant_id=participant_id,
            glucose_filepath=glucose_filepath,
            raw_data_dir=raw_data_dir,
            low_value=40,
            high_value=400,
        )
        if df_participant is not None:
            all_participants_data.append(df_participant)
        else:
            failed_participants.append(participant_id)

    print('\nData loading completed!')
    print(f'Successfully processed: {len(all_participants_data)} participants')
    print(f'Failed to process: {len(failed_participants)} participants')
    if failed_participants:
        print(f'Failed participant IDs: {failed_participants[:10]}...')

    if not all_participants_data:
        raise RuntimeError('No participant data was successfully processed.')

    combined_df = pd.concat(all_participants_data, ignore_index=True)
    combined_df = combined_df[['blood_glucose_value', 'start_dtime', 'participant_id']].rename(
        columns={
            'blood_glucose_value': 'gl',
            'start_dtime': 'time',
            'participant_id': 'id',
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    combined_df.to_csv(output_path, index=False)
    print(f'Saved processed glucose CSV: {output_path}')
    return combined_df


def build_ai_readi_metadata_csv(
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    output_filename: str = AI_READI_METADATA_FILENAME,
) -> pd.DataFrame:
    """Build the pinned AI-READI metadata CSV stored in repo `data/`."""
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    participants_df = pd.read_csv(raw_data_dir / 'participants.tsv', sep='\t')
    measurement_df = pd.read_csv(raw_data_dir / 'clinical_data' / 'measurement.csv')

    final_df = pd.DataFrame(
        {
            'participant_id': participants_df['participant_id'],
            'age': participants_df['age'],
            'study_group': participants_df['study_group'],
            'clinical_site': participants_df['clinical_site'],
        }
    )

    for identifier in METADATA_IDENTIFIERS:
        temp_df = parse_measurement(measurement_df, identifier).rename(
            columns={'value_as_number': identifier, 'person_id': 'participant_id'}
        )
        temp_df = temp_df[['participant_id', identifier]]
        final_df = final_df.merge(temp_df, on='participant_id', how='left')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    final_df.to_csv(output_path, index=False)
    print(f'Saved processed metadata CSV: {output_path}')
    return final_df


def process_ai_readi_dataset(
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the end-to-end AI-READI preprocessing pipeline."""
    glucose_df = build_ai_readi_glucose_csv(raw_data_dir=raw_data_dir, output_dir=output_dir)
    metadata_df = build_ai_readi_metadata_csv(raw_data_dir=raw_data_dir, output_dir=output_dir)
    return glucose_df, metadata_df


def load_ai_readi_cohort(
    repo_root: Path | str = REPO_ROOT,
    *,
    data_filename: str = AI_READI_CGM_FILENAME,
    metadata_filename: str = AI_READI_METADATA_FILENAME,
) -> pd.DataFrame:
    """Load the subject-level AI-READI cohort used by the notebooks."""
    repo_root = Path(repo_root)
    data_dir = repo_root / 'data'

    data = pd.read_csv(data_dir / data_filename)
    grouped_data = data.groupby('id', as_index=False).agg({'gl': list})

    metadata = pd.read_csv(data_dir / metadata_filename)
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
    """Map an internal response name to the notebook-facing label."""
    return RESPONSE_LABELS.get(response, response)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process raw AI-READI exports into repo data files.')
    parser.add_argument(
        '--raw-data-dir',
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help='Directory containing AI-READI raw exports such as participants.tsv and wearable_blood_glucose/.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory where processed ai-ready.csv and pinned metadata CSV will be written.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f'Raw data directory: {args.raw_data_dir}')
    print(f'Output directory: {args.output_dir}')
    process_ai_readi_dataset(raw_data_dir=args.raw_data_dir, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
