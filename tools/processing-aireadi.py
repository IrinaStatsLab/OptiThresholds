import numpy as np
import pandas as pd
import json
from datetime import datetime

def flatten_json(y):
    out = {}

    def flatten(x, name=""):
        # print(f'type(x) is {type(x)}')
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + "_")
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def convert_time_string_to_datetime(t_str):
    """Converts time string to datetime format. Does not convert to local time.
    Args:
        t_str (str): UTC time string such as 2023-08-01T20:39:33Z
    Returns: datetime object
    """
    datetime_object = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%SZ")  # 4 digit Year
    return datetime_object


def process_participant_data(participant_id, glucose_filepath, data_dir, low_value=40, high_value=400):
    """
    Process CGM data for a single participant.
    
    Args:
        participant_id: The participant ID
        glucose_filepath: Path to the participant's glucose data file (relative to data_dir)
        data_dir: Base data directory
        low_value: Value to replace "Low" readings with
        high_value: Value to replace "High" readings with
    
    Returns:
        pandas.DataFrame: Processed CGM data with participant_id column added
    """
    try:
        # Construct full path to CGM data file
        cgm_path = os.path.join(data_dir, glucose_filepath)
        
        # Read the mHealth formatted data as json
        with open(cgm_path, "r") as f:
            data = json.load(f)
        
        # CGM observations are in a list of nested dicts; flatten these
        list_of_body_dicts = list()
        for observation in data["body"]["cgm"]:
            flat_obs = flatten_json(observation)
            list_of_body_dicts.append(flat_obs)
        
        # Convert to pandas dataframe
        df_participant = pd.DataFrame.from_records(list_of_body_dicts)
        
        # Rename columns to more readable names
        df_participant.rename(
            columns={
                "effective_time_frame_time_interval_start_date_time": "start_time",
                "effective_time_frame_time_interval_end_date_time": "end_time",
            },
            inplace=True,
        )
        
        # Exclusion criterion: must have at least 260 records
        if df_participant.shape[0] < 260:
            print(f"    Participant {participant_id} excluded due to insufficient record counts (<1 day): {df_participant.shape[0]}")
            return None


        # Convert start_time to datetime
        df_participant["start_dtime"] = df_participant.apply(
            lambda row: convert_time_string_to_datetime(row["start_time"]), axis=1
        )

        # Missing rate check using start_dtime
        total_dur_min = (df_participant["start_dtime"].iloc[-1] - df_participant["start_dtime"].iloc[0]).total_seconds() / 60
        non_missing_rate = df_participant.shape[0] / (total_dur_min / 5)
        if non_missing_rate < 0.7:
            print(f"    Participant {participant_id} excluded due to low non-missing rate: {non_missing_rate:.2f}")
            print(f"    Duration (day): {total_dur_min/1440:.2f}, Records: {df_participant.shape[0]}")
            return None
        
        # Handle non-numeric blood glucose values (Low/High)
        def replace_alt(val, low_value, high_value):
            if val == "Low":
                return low_value
            elif val == "High":
                return high_value
            else:
                return val

        df_participant["blood_glucose_value"] = df_participant.apply(
            lambda x: replace_alt(x["blood_glucose_value"], low_value, high_value), axis=1
        )
        
        # Add participant ID column
        df_participant['participant_id'] = participant_id
        
        return df_participant
        
    except Exception as e:
        print(f"Error processing participant {participant_id}: {str(e)}")
        return None



###
## CGM Data Processing


import os
data_dir = os.getcwd()

# Manifest file for the cgm data
cgm_manifest_path = os.path.join(data_dir, 'wearable_blood_glucose', 'manifest.tsv')
dfm = pd.read_csv(cgm_manifest_path, sep='\t')

# Load all participants' data
print("Starting to load data for all participants...")
print(f"Total participants to process: {len(dfm)}")

# Initialize list to store all participant dataframes
all_participants_data = []
failed_participants = []

# Process each participant
for idx, row in dfm.iterrows():
    participant_id = row['participant_id']
    glucose_filepath = row['glucose_filepath']
    
    # Show progress every 100 participants
    if idx % 100 == 0:
        print(f"Processing participant {idx + 1}/{len(dfm)}: ID {participant_id}")
    
    # Process this participant's data
    df_participant = process_participant_data(
        participant_id=participant_id,
        glucose_filepath=glucose_filepath,
        data_dir=data_dir,
        low_value=40,
        high_value=400,
    )
    
    if df_participant is not None:
        all_participants_data.append(df_participant)
    else:
        failed_participants.append(participant_id)

print(f"\nData loading completed!")
print(f"Successfully processed: {len(all_participants_data)} participants")
print(f"Failed to process: {len(failed_participants)} participants")

if failed_participants:
    print(f"Failed participant IDs: {failed_participants[:10]}...")  # Show first 10


# Combine all participant data into a single DataFrame
if all_participants_data:
    print("Combining all participant data into a single DataFrame...")
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_participants_data, ignore_index=True)
    
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"Unique participants in combined data: {combined_df['participant_id'].nunique()}")
    print(f"Date range: {combined_df['start_dtime'].min()} to {combined_df['start_dtime'].max()}")
    
    # Display basic statistics
    print("\nBasic statistics for blood glucose values:")
    print(combined_df['blood_glucose_value'].describe())
    
else:
    print("No participant data was successfully loaded!")


# Select only columns 'blood_glucose_value', 'start_dtime', and 'participant_id'
selected_columns = ['blood_glucose_value', 'start_dtime', 'participant_id']

combined_df = combined_df[selected_columns]
print(f"Final DataFrame shape after selection: {combined_df.shape}")

# Rename columns for clarity
combined_df.rename(columns={
    'blood_glucose_value': 'gl',
    'start_dtime': 'time',
    'participant_id': 'id'
}, inplace=True)


# Save the final dataframe to CSV
filename = "ai-ready.csv"

combined_df.to_csv(os.path.join(data_dir, filename), index=False)

###
# Metadata processing

# Load clinical data from TSV and CSV files
participants_df = pd.read_csv(os.path.join(data_dir, 'participants.tsv'), sep='\t')
measurement_df = pd.read_csv(os.path.join(data_dir, "clinical_data", "measurement.csv"))

# Put variables you want to add as dictionary mapping concept names to their respective concept IDs
identifiers = {
'Triglycerides (mg/dL)', 
'bmi_vsorres, BMI', 'HbA1c (%)', 'waist_vsorres, Waist Circumference (cm)', 'weight_vsorres, Weight (kilograms)',
'HbA1c (%)',
'whr_vsorres, Waist to Hip Ratio (WHR)', 
'HDL Cholesterol (mg/dL)',
'LDL Cholesterol Calculation (mg/dL)',
'Total Cholesterol (mg/dL)',
}

def parse_measurement(df, identifier):
    temp_df = df[df['measurement_source_value'] == identifier]
    return temp_df


final_df = pd.DataFrame(columns=['participant_id', 'age', 'study_group', 'clinical_site'])

final_df['participant_id'] = participants_df['participant_id']
final_df['age'] = participants_df['age']
final_df['study_group'] = participants_df['study_group']
final_df['clinical_site'] = participants_df['clinical_site']

for key in identifiers:
    temp_df = parse_measurement(measurement_df, key)
    temp_df = temp_df.rename(columns={'value_as_number': key})
    temp_df = temp_df.rename(columns={'person_id': 'participant_id'})
    temp_df = temp_df[['participant_id', key]]

    final_df = pd.merge(final_df, temp_df, on='participant_id', how='left')


from datetime import date

# Add today's date to the final_df
final_df.to_csv(str(date.today()) + "_metadata.csv", index=False)
