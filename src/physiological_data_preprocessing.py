import pandas as pd
import numpy as np

def process_intraday_heart_rate(participant_id: str, key_name: str = "bpm",file_name: str = "heart_rate") -> pd.DataFrame:
    """
    Processes the intraday (minute-by-minute) heart rate data to calculate the
    average heart rate for each day.

    Args:
        participant_id: The identifier for the participant.
        file_name: The name of the JSON file containing intraday heart rate.

    Returns:
        A DataFrame with the average daily heart rate.
    """
    file_path = f"./PMDATA/{participant_id}/fitbit/{file_name}.json"
    df = pd.read_json(file_path)
    df["heart_rate_value"] = df["value"].apply(lambda x: x[key_name])
    df = df.drop("value", axis = 1)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    daily_avg_hr_df = df.set_index("dateTime").resample("D")['heart_rate_value'].mean().reset_index()
    daily_avg_hr_df.rename(columns={"heart_rate_bpm": "avg_daily_heart_rate"}, inplace=True)
    daily_avg_hr_df["dateTime"] = daily_avg_hr_df["dateTime"].dt.date
    return daily_avg_hr_df

def load_resting_heart_rate(participant_id: str) -> pd.DataFrame:
    """
    Loads and processes the daily resting_heart_rate.json file.

    Args:
        participant_id: The identifier for the participant (e.g., 'p07').

    Returns:
        A DataFrame with 'date' and 'resting_heart_rate' columns.
    """
    file_path = f"./PMDATA/{participant_id}/fitbit/resting_heart_rate.json"
    df = pd.read_json(file_path)

    # Standardize the date column name and type
    df['dateTime'] = pd.to_datetime(df['dateTime'])

    # The 'value' column is a dictionary {'value': 58, 'error': 0}. We extract the number.
    df['resting_heart_rate'] = df['value'].apply(lambda x: x.get('value'))

    # Select only the columns needed for the final merged dataset

    df['resting_heart_rate'] = df['resting_heart_rate'].replace(0.0, np.nan)
    final_df = df[['dateTime', 'resting_heart_rate']].copy()
    
    # Handle days where data might be missing

    return final_df

def process_calories(participant_id: str, key_name: str = "value",file_name: str = "calories") -> pd.DataFrame:
    file_path = f"./PMDATA/{participant_id}/fitbit/{file_name}.json"
    df = pd.read_json(file_path)
    df.rename(columns = {"value":"calories"}, inplace = True)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    daily_avg_calories_df = df.set_index("dateTime").resample("D")['calories'].sum().reset_index()
    daily_avg_calories_df["dateTime"] = daily_avg_calories_df["dateTime"].dt.date
    return daily_avg_calories_df

def get_daily_aggregate(participant_id,file_name, aggregate_type):
    """
    Loads a time-series JSON (e.g., very_active_minutes), aggregates it to a 
    daily level, and prepares it for merging.

    Args:
        participant_id: The ID of the participant (e.g., 'p07').
        file_name: The base name of the JSON file (e.g., 'sedentary_minutes').
        agg_method: The aggregation method as a string, typically 'sum' or 'mean'.

    Returns:
        A DataFrame with 'date' and the daily aggregated value column.
    """

    file_path = f"./PMDATA/{participant_id}/fitbit/{file_name}.json"
    df = pd.read_json(file_path)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    new_column_name = f"{file_name}_{aggregate_type}"
    df.rename(columns={'value': new_column_name}, inplace=True)
    daily_df = df.set_index('dateTime')[new_column_name].resample('D').agg(aggregate_type).reset_index()    
    return daily_df

def process_resting_heart_rate(participant_id: str, json_file_name: str = "resting_heart_rate") -> pd.DataFrame:
    """
    Processes the daily resting heart rate data. This data is already aggregated
    by day and does not require resampling.

    Args:
        participant_id: The identifier for the participant.
        json_file_name: The name of the JSON file containing daily resting heart rate.

    Returns:
        A DataFrame with the daily resting heart rate.
    """
    # Load the JSON file containing the daily resting heart rate values
    file_path = f"./PMDATA/{participant_id}/fitbit/{json_file_name}.json"
    rhr_df = pd.read_json(file_path)

    # The actual RHR value is nested within the 'value' column dictionary
    # The structure is [{"dateTime": "...", "value": {"value": 58, "error": 4}}, ...]
    rhr_df['resting_heart_rate'] = rhr_df['value'].apply(lambda x: x.get('value'))
    
    # Drop the original complex column
    rhr_df = rhr_df.drop('value', axis=1)
    
    # Convert dateTime to date object for clean merging
    rhr_df['dateTime'] = pd.to_datetime(rhr_df['dateTime']).dt.date

    return rhr_df

def load_sleep_score(participant_id: str) -> pd.DataFrame:
    """
    Loads and processes the sleep_score.csv file for the MVP.

    Args:
        participant_id: The ID of the participant

    Returns:
        A DataFrame with 'date' and 'avg_overall_sleep_score' columns.
    """
    # Define the file path
    file_path = f"./PMDATA/{participant_id}/fitbit/sleep_score.csv"
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # normalize to get just the date part of the timestamp.
    df['dateTime'] = df['timestamp'].dt.date
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    
    # Select and rename the columns required for the MVP.
    # We only need the overall_score.
    mvp_df = df[['dateTime', 'overall_score']].copy()
    mvp_df.rename(columns={'overall_score': 'avg_overall_sleep_score'}, inplace=True)
    final_df = mvp_df.groupby('dateTime').mean().reset_index()
    return final_df


def deal_with_sleep_data_json(participant_id):
    sleep_df = pd.read_json(f"./PMDATA/{participant_id}/fitbit/sleep.json")
    sleep_df['dateOfSleep'] = pd.to_datetime(sleep_df['dateOfSleep'])
    sleep_df['startTime'] = pd.to_datetime(sleep_df['startTime'])
    sleep_df['endTime'] = pd.to_datetime(sleep_df['endTime'].str.replace('Z', '')) # Handle 'Z' if present
    def extract_sleep_levels(levels_data):
        if isinstance(levels_data, dict) and 'summary' in levels_data:
            summary = levels_data['summary']
            return {
                'deep_sleep_minutes': summary.get('deep', {}).get('minutes'),
                'light_sleep_minutes': summary.get('light', {}).get('minutes'),
                'rem_sleep_minutes': summary.get('rem', {}).get('minutes'),
                'awake_minutes_in_sleep': summary.get('wake', {}).get('minutes')
            }
        return {
            'deep_sleep_minutes': None,
            'light_sleep_minutes': None,
            'rem_sleep_minutes': None,
            'awake_minutes_in_sleep': None
        }
    sleep_levels_extracted = sleep_df['levels'].apply(extract_sleep_levels).apply(pd.Series)
    sleep_df = pd.concat([sleep_df, sleep_levels_extracted], axis=1)
    # Select and rename columns for clarity
    sleep_df_processed = sleep_df[[
        'logId', 'dateOfSleep', 'duration', 'minutesToFallAsleep', 'minutesAsleep',
        'minutesAwake', 'minutesAfterWakeup', 'timeInBed', 'efficiency',
        'deep_sleep_minutes', 'light_sleep_minutes', 'rem_sleep_minutes', 'awake_minutes_in_sleep'
    ]].copy()
    sleep_df_processed.rename(columns={'logId': 'sleep_log_entry_id'}, inplace=True)

    sleep_score_df = pd.read_csv(f"./PMDATA/{participant_id}/fitbit/sleep_score.csv")
    sleep_score_df['timestamp'] = pd.to_datetime(sleep_score_df['timestamp'].str.replace('Z', ''))
    sleep_score_df_processed = sleep_score_df[[
        'sleep_log_entry_id', 'timestamp', 'overall_score', 'composition_score',
        'revitalization_score', 'duration_score', 'restlessness'
    ]].copy()

    sleep_df_processed.drop_duplicates(inplace = True)
    sleep_score_df_processed.drop_duplicates(inplace = True)
    
    combined_sleep_df = pd.merge(
        sleep_df_processed,
        sleep_score_df_processed,
        on='sleep_log_entry_id',
        how='outer'
    )

    combined_sleep_df.rename(columns = {"dateOfSleep":"dateTime"}, inplace = True)

    return combined_sleep_df


def aggregate_daily_sleep_data(combined_sleep_df):
    """
    Aggregates the combined sleep DataFrame to a daily level.

    Args:
        combined_sleep_df (pd.DataFrame): DataFrame containing merged sleep.json and sleep_score.csv data.

    Returns:
        pd.DataFrame: Daily aggregated sleep metrics.
    """
    # Ensure 'dateTime' is of datetime.date type for grouping
    combined_sleep_df['dateTime'] = pd.to_datetime(combined_sleep_df['dateTime']).dt.date

    # Define aggregation dictionary for daily summary
    daily_sleep_metrics = combined_sleep_df.groupby('dateTime').agg(
        # Sums for durations/minutes (assuming multiple sleep sessions like naps per day)
        total_sleep_duration_minutes=('duration', 'sum'),
        total_minutes_asleep=('minutesAsleep', 'sum'),
        total_minutes_awake_during_sleep=('minutesAwake', 'sum'),
        total_time_in_bed_minutes=('timeInBed', 'sum'),
        total_deep_sleep_minutes=('deep_sleep_minutes', 'sum'),
        total_light_sleep_minutes=('light_sleep_minutes', 'sum'),
        total_rem_sleep_minutes=('rem_sleep_minutes', 'sum'),
        total_awake_minutes_in_sleep_stages=('awake_minutes_in_sleep', 'sum'),

        # Averages for scores and single-session metrics
        avg_minutes_to_fall_asleep=('minutesToFallAsleep', 'mean'), # Avg across sessions for the day
        avg_minutes_after_wakeup=('minutesAfterWakeup', 'mean'),   # Avg across sessions for the day
        avg_sleep_efficiency=('efficiency', 'mean'),
        avg_overall_sleep_score=('overall_score', 'mean'),
        avg_composition_score=('composition_score', 'mean'),
        avg_revitalization_score=('revitalization_score', 'mean'),
        avg_duration_score=('duration_score', 'mean'),
        avg_restlessness=('restlessness', 'mean')
    ).reset_index()

    # Fill NaNs for sum-based metrics with 0 (if no sleep data for a given day)
    # The default behavior of sum() is to ignore NaNs, so if there's no data for a day,
    # the sum will naturally be 0 after grouping.
    # We can ensure this for clarity.
    daily_sleep_metrics.fillna({
        'total_sleep_duration_minutes': 0,
        'total_minutes_asleep': 0,
        'total_minutes_awake_during_sleep': 0,
        'total_time_in_bed_minutes': 0,
        'total_deep_sleep_minutes': 0,
        'total_light_sleep_minutes': 0,
        'total_rem_sleep_minutes': 0,
        'total_awake_minutes_in_sleep_stages': 0,
    }, inplace=True)

    # Rename the date column to a common name for final merging with other dataframes
    daily_sleep_metrics.rename(columns={'dateTime': 'dateTime'}, inplace=True)

    return daily_sleep_metrics

def preprocess_hr_zones(participant_id):
    hr_zones_df = pd.read_json(f"./PMDATA/{participant_id}/fitbit/time_in_heart_rate_zones.json")
    hr_zones_df['dateTime'] = pd.to_datetime(hr_zones_df['dateTime'])

    def extract_hr_zones(value_data):
        if isinstance(value_data, dict) and 'valuesInZones' in value_data:
            zones = value_data['valuesInZones']
            return {
                'time_in_below_default_zone1_minutes': zones.get('BELOW_DEFAULT_ZONE_1', 0.0),
                'time_in_fat_burn_zone_minutes': zones.get('IN_DEFAULT_ZONE_1', 0.0), # Fat Burn Zone 
                'time_in_cardio_zone_minutes': zones.get('IN_DEFAULT_ZONE_2', 0.0),   # Cardio Zone 
                'time_in_peak_zone_minutes': zones.get('IN_DEFAULT_ZONE_3', 0.0)     # Peak Zone 
            }
        # Return None or appropriate default if 'valuesInZones' is missing or not a dict
        return {
            'time_in_below_default_zone1_minutes': None,
            'time_in_fat_burn_zone_minutes': None,
            'time_in_cardio_zone_minutes': None,
            'time_in_peak_zone_minutes': None
        }

    hr_zones_extracted = hr_zones_df['value'].apply(extract_hr_zones).apply(pd.Series)

    hr_zones_df_processed = pd.concat([hr_zones_df['dateTime'], hr_zones_extracted], axis=1)
        
    hr_zones_df_processed['dateTime'] = hr_zones_df_processed['dateTime'].dt.date

    return hr_zones_df_processed

    
def preprocess_heart_rate(participant_id,json_file_name,key_name = "bpm"): 
    heart_rate_df = pd.read_json(f"./PMDATA/{participant_id}/fitbit/{json_file_name}.json")
    heart_rate_df["heart_rate_value"] = heart_rate_df["value"].apply(lambda x: x[key_name])
    heart_rate_df = heart_rate_df.drop("value", axis = 1)
    daily_heart_rate = heart_rate_df.set_index("dateTime").resample("D").mean().reset_index()
    daily_heart_rate.rename(columns = {"heart_rate_value":"avg_daily_heart_rate"}, inplace = True)
    daily_heart_rate["dateTime"] = daily_heart_rate["dateTime"].dt.date
    return daily_heart_rate

def preprocess_exercise_data(participant_id):
    """
    Processes and aggregates daily exercise data from exercise.json.
    """
    try:
        exercise_df = pd.read_json(f"./PMDATA/{participant_id}/fitbit/exercise.json")
    except ValueError:
        return pd.DataFrame()

    if exercise_df.empty:
        return pd.DataFrame()

    exercise_df['startTime'] = pd.to_datetime(exercise_df['startTime'])
    exercise_df['exercise_date'] = exercise_df['startTime'].dt.date

    selected_exercise_df = exercise_df[[
        "exercise_date",
        "activityName",
        "calories",
        "duration",  # This is in milliseconds
        "steps",
        "distance",
        "averageHeartRate"
    ]].copy()

    daily_exercise_metrics = selected_exercise_df.groupby('exercise_date').agg(
        total_exercise_calories=('calories', 'sum'),
        total_exercise_duration_ms=('duration', 'sum'), # Aggregate in MS first
        total_exercise_steps=('steps', 'sum'),
        total_exercise_distance_km=('distance', 'sum'),
        avg_exercise_heart_rate=('averageHeartRate', 'mean'),
        num_exercise_sessions=('activityName', 'count')
    ).reset_index()

    daily_exercise_metrics['total_exercise_duration_minutes'] = daily_exercise_metrics['total_exercise_duration_ms'] / 60000
    daily_exercise_metrics.drop('total_exercise_duration_ms', axis=1, inplace=True)

    fill_values = {
        'total_exercise_calories': 0,
        'total_exercise_duration_minutes': 0,
        'total_exercise_steps': 0,
        'total_exercise_distance_km': 0,
        'avg_exercise_heart_rate': 0, 
        'num_exercise_sessions': 0
    }
    daily_exercise_metrics.fillna(fill_values, inplace=True)

    daily_exercise_metrics.rename(columns={"exercise_date": "dateTime"}, inplace=True)

    return daily_exercise_metrics