SUBJECTIVE_COLS = ['fatigue', 'mood', 'stress', 'sleep_quality']

participant_information = {"60301085":["Ahmed Hanif"],
                           "60107567":["Ahmad-Abdel-Hafez"],
                           "60300310":["Djihane Mahrez"],
                           "60301986":["Umm Kulsoom"],
                           "60106652":["Hafsa Farhan"],
                           "60105379":["Adil Waheed"],
                           "60106321":["Mohammad Ashjar"],
                           "60107629":["Reda Bendraou"]}

PHYSIOLOGICAL_CONFIG = [
    {
        'metric': 'resting_heart_rate',
        'mean': 'resting_heart_rate_14_mean',
        'std': 'resting_heart_rate_14_std',
        'direction': 'above'
    },
    {
        'metric': 'avg_overall_sleep_score',
        'mean': 'avg_overall_sleep_score_14_mean',
        'std': 'avg_overall_sleep_score_14_std',
        'direction': 'below'
    }
]

RISK_THRESHOLD = 3
LOOKAHEAD_DAYS = 7

X_columns = [
    'resting_heart_rate',
    'avg_overall_sleep_score',
    'fatigue',
    'mood',
    'stress',
    'sleep_quality',
    'very_active_minutes_sum',
    'sedentary_minutes_sum',
    'resting_heart_rate_14_mean',
    'resting_heart_rate_14_std',
    'avg_overall_sleep_score_14_mean',
    'avg_overall_sleep_score_14_std',
]

y_column = 'Is_High_Risk_Next_7_Days'