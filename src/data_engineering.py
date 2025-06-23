import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

def calculate_moving_average(df, columns,frequency : int = 7):
    '''
        Parameters:
            df, columns -> dataframe and columns you want to take moving average of
            frequency --> e.g.  7 day moving average or 28 day moving average
        
        returns
            df with the newly established Moving Average Columns
    '''
    df_v2 = df.copy()
    for column in columns:
        moving_average_column_name = f'{column}_MA_{frequency}'
        df_v2[moving_average_column_name] = df_v2[column].rolling(window = frequency).mean()

    return df_v2


def calculate_delta_features(df, columns):
    df = df.copy()
    
    for column in columns:
        if column in df.columns:
            column_name = f"{column}_delta_day_over_day"
            df[column_name] = df[column].diff(1)
    return df

def calculate_deviation_features(df, columns):
    df = df.copy()
    moving_average_windows = ['MA_7', 'MA_30']
    for column in columns:
        for window in moving_average_windows:
            ma_column_name = f"{column}_{window}"
            if ma_column_name in df.columns and column in df.columns:
                df[f'{column}_deviation_{window}'] = df[column] - df[ma_column_name]


    return df

def calculate_lagged_features(df,columns,N):
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[f"{column}_lag{N}"] = df[column].shift(N)

    return df


def calculate_rolling_baselines(df, columns, dateTime_column,window_size, calculate_mean, calculate_std):
    df = df.copy()
    df[dateTime_column] = pd.to_datetime(df[dateTime_column])
    df.index = df[dateTime_column]
    df.sort_index(inplace=True)

    for column in columns:
        if column in df.columns:
            if calculate_mean:
                column_name = f"{column}_{window_size}_mean"
                df[column_name] = df[column].rolling(window_size).mean()
                # df[column_name] = df[column_name].bfill()
                
            if calculate_std:
                column_name = f"{column}_{window_size}_std"
                df[column_name] = df[column].rolling(window_size).std()
                # df[column_name] = df[column_name].bfill()
            
    df = df.dropna()
    return df


def calculate_subjective_score(df: pd.DataFrame, subjective_cols: list) -> pd.Series:
    """
    Calculates the subjective distress score based on a list of columns.

    Args:
        df: The input DataFrame.
        subjective_cols: A list of column names to evaluate. The rule is
                         that a score <= 2 for any of these columns adds
                         1 point to the daily score.

    Returns:
        A pandas Series containing the calculated Subjective_Distress_Score
        for each day.
    """
    # Initialize a Series of zeros to hold the score
    total_score = pd.Series(0, index=df.index)
    
    for col in subjective_cols:
        # For each column, create the boolean mask, convert to int (0 or 1),
        # and add it to the total score.
        total_score += (df[col] <= 2).astype(int)
        
    return total_score


def calculate_physiological_score(df: pd.DataFrame, deviation_configs: list) -> pd.Series:
    """
    Calculates the physiological deviation score based on a list of
    configuration dictionaries.

    Args:
        df: The input DataFrame.
        deviation_configs: A list of dictionaries. Each dictionary must contain:
                           'metric': the column to check (e.g., 'resting_heart_rate')
                           'mean': the baseline mean column
                           'std': the baseline std dev column
                           'direction': 'above' or 'below' for the deviation

    Returns:
        A pandas Series containing the calculated Physiological_Deviation_Score
        for each day.
    """
    total_score = pd.Series(0, index=df.index)

    # Iterate through the configuration list
    for config in deviation_configs:
        metric_col = config['metric']
        mean_col = config['mean']
        std_col = config['std']
        direction = config['direction']
        
        # Dynamically apply the logic based on the 'direction'
        if direction == 'above':
            points = (df[metric_col] > (df[mean_col] + df[std_col])).astype(int)
        elif direction == 'below':
            points = (df[metric_col] < (df[mean_col] - df[std_col])).astype(int)
        else:
            # If config is invalid, return no points for that rule
            points = 0
            
        total_score += points
        
    return total_score

def add_risk_scores(df_with_baselines: pd.DataFrame, subjective_cols_config: list, physiological_cols_config: list) -> pd.DataFrame:
    """
    Main wrapper function to calculate and append all daily risk sub-scores.

    Args:
        df_with_baselines: DataFrame with all features and baseline columns.
        subjective_cols_config: List of columns for the subjective score.
        physiological_cols_config: List of config dicts for the physiological score.

    Returns:
        A new DataFrame with the two score columns appended.
    """
    df_out = df_with_baselines.copy()

    df_out['Subjective_Distress_Score'] = calculate_subjective_score(
        df=df_out,
        subjective_cols=subjective_cols_config
    )
    
    df_out['Physiological_Deviation_Score'] = calculate_physiological_score(
        df=df_out,
        deviation_configs=physiological_cols_config
    )
    
    print("Successfully calculated and appended daily risk scores using refactored functions.")
    return df_out



def engineer_target_variable(df: pd.DataFrame, risk_threshold: int, lookahead_window: int) -> pd.DataFrame:
    """
    Engineers the final predictive target variable and intermediate risk columns.

    This function performs the final three steps of feature engineering:
    1.  Calculates the Composite_Risk_Score by summing the sub-scores.
    2.  Identifies the daily High_Risk_State based on a threshold.
    3.  Creates the predictive target Is_High_Risk_Next_7_Days by looking
        ahead in the High_Risk_State column.

    Args:
        df: DataFrame containing the subjective and physiological
                        risk scores.
        risk_threshold: The integer value at which the composite score is
                        considered a high-risk state (e.g., 3).
        lookahead_window: The number of subsequent days to check for a
                          high-risk state (e.g., 7).

    Returns:
        A new pandas DataFrame with the composite score, daily risk state,
        and the final predictive target column appended.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    #Calculating Composite Risk Score
    df['Composite_Risk_Score'] = df['Subjective_Distress_Score'] + df['Physiological_Deviation_Score']

    # Identify Daily High Risk State
    df['High_Risk_State'] = (df['Composite_Risk_Score'] >= risk_threshold).astype(int)

    # Engineer Predictive Target by Looking Ahead
        # To look forward, shift the 'High_Risk_State' column backwards,
        # then apply a rolling max over the lookahead window.
        # This finds if a '1' exists in any of the future N days.
    df['Is_High_Risk_Next_7_Days'] = df['High_Risk_State'].shift(-lookahead_window).rolling(window=lookahead_window, min_periods=1).max()

    # The above operation leaves NaNs at the end. We will handle this
    # by dropping them in the final modeling preparation step.
    # For now, we can fill with an invalid value like -1 to show they are not usable.
    df['Is_High_Risk_Next_7_Days'] = df['Is_High_Risk_Next_7_Days'].fillna(-1).astype(int)
    return df


def generate_mock_appointments(participant_id, start_date_str, end_date_str, filename="./cleaned_data/Appointments.csv"):
    """
    Generates a mock CSV file of appointments for a list of participants.

    Args:
        participant_ids (list): A list of participant ID strings.
        start_date_str (str): The start date for the appointment range (e.g., "2019-11-01").
        end_date_str (str): The end date for the appointment range (e.g., "2019-12-31").
        filename (str): The name of the output CSV file.
    """
    
    # Define the possible outcomes for an appointment
    statuses = ["Attended", "No-Show", "Cancelled", "Attended", "Attended"] # Weighted towards 'Attended'
    
    # Convert string dates to date objects
    start_date = date.fromisoformat(start_date_str)
    end_date = date.fromisoformat(end_date_str)
    delta_days = (end_date - start_date).days

    appointments_data = []

    print(f"Generating mock data for {participant_id} participant")

    # Each participant will have a random number of appointments (e.g., 2 to 6)
    num_appointments = random.randint(4, 12)
    
    for _ in range(num_appointments):
        # Generate a random date within the specified range
        random_day_offset = random.randint(0, delta_days)
        appointment_date = start_date + timedelta(days=random_day_offset)
        
        # Assign a random status
        appointment_status = random.choice(statuses)
        
        # Add the record to our list
        appointments_data.append({
            "participant_id": participant_id,
            "AppointmentDate": appointment_date,
            "Status": appointment_status
        })

    return appointments_data