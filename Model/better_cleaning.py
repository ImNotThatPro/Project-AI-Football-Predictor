import matplotlib.pyplot as plt

import pandas as pd 

bdf_results = pd.read_csv('Model/cleaned_with_elo_results.csv')

print(bdf_results['home_team'].value_counts().head(50))
print(bdf_results['result_encoded'].value_counts(normalize=True).head(10))
print(bdf_results['date'].min())


print(bdf_results['elo_diff'].describe())
plt.hist(bdf_results['elo_diff'], bins=50)
plt.title('Distribution of Elo Differences')
plt.show()