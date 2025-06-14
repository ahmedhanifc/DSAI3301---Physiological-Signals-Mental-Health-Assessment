import pandas as pd


def preprocess_srpe_data(participant_id):
    srpe_df = pd.read_csv(f"./PMDATA/{participant_id}/pmsys/srpe.csv")

    # 1. Convert 'end_date_time' to datetime and extract date
    srpe_df['end_date_time'] = pd.to_datetime(srpe_df['end_date_time'].str.replace('Z', ''))
    srpe_df['srpe_date'] = srpe_df['end_date_time'].dt.date

    # 2. Calculate Session Training Load (sRPE)
    srpe_df['perceived_exertion'] = pd.to_numeric(srpe_df['perceived_exertion'], errors='coerce')
    srpe_df['duration_min'] = pd.to_numeric(srpe_df['duration_min'], errors='coerce')
    srpe_df['session_load_srpe'] = srpe_df['perceived_exertion'] * srpe_df['duration_min']

    # 3. Handle 'activity_names' (it's a string representation of a list, so convert it)
    import ast
    srpe_df['activity_names_list'] = srpe_df['activity_names'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # 4. Aggregate to Daily Level
    daily_srpe_metrics = srpe_df.groupby('srpe_date').agg(
        total_daily_training_load_srpe=('session_load_srpe', 'sum'),
        total_daily_duration_min=('duration_min', 'sum'),
        avg_daily_perceived_exertion=('perceived_exertion', 'mean'),
        num_daily_srpe_sessions=('end_date_time', 'count'), 
        unique_daily_activity_names=('activity_names_list', lambda x: list(set([item for sublist in x for item in sublist])))
    ).reset_index()

    # Rename date column for merging consistency
    daily_srpe_metrics.rename(columns={'srpe_date': 'dateTime'}, inplace=True)

    # Fill NaNs for sum-based metrics with 0 where appropriate (if no training on a day)
    daily_srpe_metrics.fillna({
        'total_daily_training_load_srpe': 0,
        'total_daily_duration_min': 0,
        'avg_daily_perceived_exertion': 0 # Or NaN if you prefer to distinguish no training vs 0 exertion
    }, inplace=True)

    return daily_srpe_metrics


def preprocess_reporting_data(participant_id):
    reporting_df = pd.read_csv(f"./PMDATA/{participant_id}/googledocs/reporting.csv")

    # 1. Convert 'date' and 'timestamp' columns
    reporting_df['date'] = pd.to_datetime(reporting_df['date'], format='%d/%m/%Y')
    reporting_df['dateTime'] = reporting_df['date'].dt.date

    # 2. Process 'meals' column to create boolean flags for each meal type
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Evening']
    for meal in meal_types:
        reporting_df[f'had_{meal.lower()}'] = reporting_df['meals'].str.contains(meal, case=False, na=False)

    # 3. Process 'alcohol_consumed' column
    reporting_df['alcohol_consumed_flag'] = reporting_df['alcohol_consumed'].apply(lambda x: 1 if x == 'Yes' else 0)

    agg_dict = {
        'weight': ('weight', 'last'), # Take the last weight reported for the day
        'total_glasses_of_fluid': ('glasses_of_fluid', 'sum'), # Sum all fluid intake for the day
        'alcohol_consumed_daily_flag': ('alcohol_consumed_flag', 'max'), # 1 if alcohol was consumed at least once, 0 otherwise
    }

    # Add meal flags to aggregation dict using 'max'
    for meal in meal_types:
        agg_dict[f'had_{meal.lower()}'] = (f'had_{meal.lower()}', 'max')

    daily_reporting_metrics = reporting_df.groupby('dateTime').agg(
        **agg_dict
    ).reset_index()

    # 5. Handle NaNs in 'weight' after daily aggregation
    daily_reporting_metrics['weight'] = daily_reporting_metrics['weight'].interpolate(method='linear', limit_direction='both', limit_area='inside')
    daily_reporting_metrics['weight'] = daily_reporting_metrics['weight'].fillna(method='ffill').fillna(method='bfill') # Fill remaining NaNs at edges

    # Ensure boolean/int columns are correct type after aggregation
    for meal in meal_types:
        daily_reporting_metrics[f'had_{meal.lower()}'] = daily_reporting_metrics[f'had_{meal.lower()}'].astype(int)
    daily_reporting_metrics['alcohol_consumed_daily_flag'] = daily_reporting_metrics['alcohol_consumed_daily_flag'].astype(int)

    return daily_reporting_metrics

def preprocess_injury_data(participant_id):
    injury_df = pd.read_csv(f"./PMDATA/{participant_id}/pmsys/injury.csv")

    # Convert 'effective_time_frame' to datetime and extract date
    injury_df['effective_time_frame'] = pd.to_datetime(injury_df['effective_time_frame'])
    injury_df['injury_date'] = injury_df['effective_time_frame'].dt.date

    # Process 'injuries' column (which contains dictionaries)
    # Convert string representation of dict to actual dict for empty check
    injury_df['injuries_dict'] = injury_df['injuries'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)

    # Flag if any injury was reported for the entry
    injury_df['has_injury_reported_session'] = injury_df['injuries_dict'].apply(lambda x: 1 if x else 0)

    # Count the number of distinct injury areas reported in that session
    injury_df['num_injury_areas_session'] = injury_df['injuries_dict'].apply(lambda x: len(x) if x else 0)

    # Optional: Extract severity and specific locations if needed for detailed analysis
    # For simplicity, we'll stick to a count and a general flag for the dashboard.
    # If a specific injury (e.g., 'right_foot') is important, you can add flags:
    # injury_df['has_right_foot_injury'] = injury_df['injuries_dict'].apply(lambda x: 1 if 'right_foot' in x else 0)

    # Aggregate to daily level
    # Since it's weekly, and we just want to know if there was an injury report on a day
    daily_injury_metrics = injury_df.groupby('injury_date').agg(
        has_injury_reported=('has_injury_reported_session', 'max'), # Use max to get 1 if any injury was reported that day
        num_unique_injury_areas_daily=('num_injury_areas_session', 'max') # Max number of areas reported on that day
    ).reset_index()

    # Rename the date column for merging consistency
    daily_injury_metrics.rename(columns={'injury_date': 'dateTime'}, inplace=True)

    # Ensure has_injury_reported is int (0 or 1)
    daily_injury_metrics['has_injury_reported'] = daily_injury_metrics['has_injury_reported'].astype(int)

    return daily_injury_metrics

def get_daily_aggregate(participant_id,json_file_name, aggregate_type):
    specific = pd.read_json(f"./PMDATA/{participant_id}/fitbit/{json_file_name}.json")
    if(aggregate_type == "sum"):
        specific.rename(columns = {"value":f"{json_file_name}_sum"}, inplace = True)
        specific = specific.set_index("dateTime").resample("D").sum().reset_index()
    if(aggregate_type == "mean"):
        specific.rename(columns = {"value":f"{json_file_name}_mean"}, inplace = True)
        specific = specific.set_index("dateTime").resample("D").mean().reset_index()
    return specific
    


def merge_dfs(all_dfs):
    master_df = all_dfs[0].copy()
    for i in range(1,len(all_dfs)):
        master_df = pd.merge(master_df, all_dfs[i], on = "dateTime",how = "outer")

    master_df.drop_duplicates(subset=['dateTime'],inplace = True)
    return master_df

def convert_column_to_datetime(all_dfs):
    for i in range(len(all_dfs)):
        all_dfs[i]["dateTime"] = pd.to_datetime(all_dfs[i]["dateTime"])

    return all_dfs

def analyze_gaps(df, column_name):
    """
    Analyzes the structure of contiguous missing value gaps in a DataFrame column.
    """
    is_missing = df[column_name].isnull()
    if not is_missing.any():
        return {
            "num_gaps":0,
            "max_gap_length":0,
            "avg_gap_length":0
        }
        
    gap_starts = is_missing.ne(is_missing.shift()).cumsum()
    gaps = is_missing.groupby(gap_starts).sum()
    actual_gaps = gaps[gaps > 0]
    if actual_gaps.empty:
        return {
            'num_gaps': 0,
            'max_gap_length': 0,
            'avg_gap_length': 0.0
        }
        
    return {
        'num_gaps': len(actual_gaps),
        'max_gap_length': actual_gaps.max(),
        'avg_gap_length': actual_gaps.mean()
    }


def get_missing_proportion(df):
    return pd.DataFrame({
    "missing_count":df.isnull().sum(),
    "missing_percent": (df.isnull().sum() / len(df)) * 100
}).sort_values(by = "missing_count", ascending= False)



