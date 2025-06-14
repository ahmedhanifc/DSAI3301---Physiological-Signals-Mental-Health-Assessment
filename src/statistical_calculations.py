import pandas as pd
import numpy as np

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
