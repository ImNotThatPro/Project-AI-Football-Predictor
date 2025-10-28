import matplotlib as plt
import pandas as pd 

df_results = pd.read_csv('Model/cleaned_with_elo_results.csv')

print(df_results['home_team'].value_counts().head(50))