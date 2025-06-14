import pandas as pd


def preprocess_wellness_csv(participant_id):
    file_path = f"./PMDATA/{participant_id}/pmsys/wellness.csv"
    df = pd.read_csv(file_path)
    df['effective_time_frame'] = pd.to_datetime(df['effective_time_frame'])
    df['dateTime'] = df['effective_time_frame'].dt.date
    df["dateTime"] = pd.to_datetime(df['dateTime'])
    df = df.sort_values('effective_time_frame').drop_duplicates(subset=['dateTime'], keep='last')

    mvp_features = [
        'dateTime',
        'fatigue',
        'mood',
        'stress',
        'sleep_quality'
    ]
    df = df[mvp_features].copy()


    return df