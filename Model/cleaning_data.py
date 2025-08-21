import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#results.csv cleaning
#
#
#q
df_results = pd.read_csv('./results.csv')
df_results = df_results.drop(columns = ['city', 'country', 'neutral'], errors = 'ignore')
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'HomeWin'
    elif row['home_score'] < row['away_score']:
        return 'AwayWin'
    else:
        return 'Draw'
def goal_diff(row):
    return abs(row['home_score'] - row['away_score'])

df_results['result'] = df_results.apply(get_result, axis=1)
df_results['goal_diff'] = df_results.apply(goal_diff, axis = 1)

#encoding (easier access for machine learning model)
all_team = pd.concat([df_results['home_team'], df_results['away_team']]).unique()
le.fit(all_team)
df_results['home_team_encoded'] = le.transform(df_results['home_team'])
df_results['away_team_encoded'] = le.transform(df_results['away_team'])
result_encoder = LabelEncoder()
df_results['result_encoded'] = result_encoder.fit_transform(df_results['result'])


