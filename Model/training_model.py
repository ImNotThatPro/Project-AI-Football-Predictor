from cleaning_data import df_results, result_encoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import joblib

le = LabelEncoder()

##Elo rating system building
#
#
#making dict to store each team elo in and make a seen history(K is the default factor to calculate elo)(ReadMe)
team_elos = {}
elo_history = {}
default_elo = 1500
K = 20

#from the elo of each team returning whether we expected the higher elo team to win, if not read update_elo to understand(ReadMe)
def expected_result(elo1, elo2):
    return 1 / (1 + 10 **((elo2 - elo1) / 400))

#if higher elo team win, gain some elo base on the other team strength, if lower elo team win gain quite a bit of elo based on the higher elo team lost(ReadMe)
def update_elo(elo, expected, actual):
    return elo + K * (actual - expected)

#creating a list of away and home elo for better readability
home_elo_list = []
away_elo_list = []
elo_diff = []
for index, row in df_results.iterrows():
    #getting the team name from each match
    team1 = row['home_team']
    team2 = row['away_team']
    score1 = row['home_score']
    score2 = row['away_score']
    #set to 1500 if not seen in history
    if team1 not in team_elos:
        team_elos[team1] = default_elo
        elo_history[team1] = [default_elo]
    if team2 not in team_elos:
        team_elos[team2] = default_elo
        elo_history[team2] = [default_elo]
    #getting elo after each match or default elo at the start
    elo1 = team_elos[team1]
    elo2 = team_elos[team2]
    #adding the current elo to the home and away list
    home_elo_list.append(elo1)
    away_elo_list.append(elo2)
    #it is what it is
    elo_diff.append(elo1-elo2)
    #getting expected result of each time (read function)
    expected1 = expected_result(elo1, elo2)
    expected2 = expected_result(elo2, elo1)
    #simple if win= 1, lose =0, draw =0.5 for model to read better
    if score1 > score2:
        actual1, actual2 = 1, 0
    elif score1 < score2:
        actual1, actual2 = 0, 1
    else:
        actual1, actual2 = 0.5, 0.5
    #after a loop of expected and prediction, finally get all value to update_elo(read function )
    new_elo1 = update_elo(elo1, expected1, actual1)
    new_elo2 = update_elo(elo2, expected2, actual2)
    #changing elo(this = live rating for the current to date match)
    team_elos[team1]= new_elo1
    team_elos[team2] = new_elo2
    #This = rating from the current match, but also can see history of the change, rise and fall of the team according to the index
    elo_history[team1].append(new_elo1)
    elo_history[team2].append(new_elo2)

df_results['home_team_elo'] = home_elo_list
df_results['away_team_elo'] = away_elo_list
df_results['elo_diff'] = elo_diff

df_results.to_csv('Model/cleaned_with_elo_results.csv', index=False)
#splitting data(X = dataframe without answer, y = answer)
X = df_results[['home_team_encoded', 'away_team_encoded', 'elo_diff']]
y = df_results['result_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 42)
#Processing data for machine learning to read(i guess?)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Training model(old model, does not have a tuning round)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluate how well the (first) model performed
y_predicted = model.predict(X_test)
print(f'New RandomForest model accuracy:{accuracy_score(y_test ,model.predict(X_test))}')


##Optional shits for show
#
#
#
# Taking team name and predict
all_team = pd.concat([df_results['home_team'], df_results['away_team']]).unique()
team_mapping = {team : i for i, team in enumerate(sorted(all_team))}

def pred_match(teamA, teamB):
    teamA_encoded = int(team_mapping.get(teamA))
    teamB_encoded = int(team_mapping.get(teamB))
    if teamA_encoded is None or teamB_encoded is None:
        return 'Unknown team'

    eloA = float(team_elos.get(teamA, 1500))
    eloB = float(team_elos.get(teamB, 1500))
    elo_diff = eloA - eloB

    # Build input as a 2D list of floats
    input_data = [[teamA_encoded, teamB_encoded, elo_diff]]

    # Convert to DataFrame with column names (to silence that warning)
    import pandas as pd
    input_df = pd.DataFrame(input_data, columns=['home_team_encoded', 'away_team_encoded', 'elo_diff'])

    input_scaled = scaler.transform(input_df)

    result_code = model.predict(input_scaled)[0]

    print('Class order:', result_encoder.classes_)
    print(model.predict_proba(input_scaled))

    result_label = result_encoder.inverse_transform([result_code])[0]
    return result_label


print(pred_match('Argentina','Portugal'))



##Making simple visualization(does not look good due to the dataset size)
#
#
step = max(1, len(y_test)//2000)  # plot max 2000 points for readability
plt.figure(figsize=(12,6))
plt.scatter(range(0, len(y_test), step), y_test[::step], color='blue', label='Actual', alpha=0.3)
plt.scatter(range(0, len(y_predicted), step), y_predicted[::step], color='red', label='Predicted', marker='x', alpha=0.3)
plt.title('Random Forest: Actual vs Predicted Football Match Outcomes (Downsampled)')
plt.xlabel('Match Index')
plt.ylabel('Encoded Result')
plt.legend()
plt.show()

joblib.dump(model, 'models/rf_model.joblib', compress=3)
joblib.dump(scaler, 'models/rf_scaler.joblib')
joblib.dump(result_encoder, 'models/result_encoder.joblib')
joblib.dump(team_mapping, 'models/team_mapping.joblib')
joblib.dump(accuracy_score(y_test ,model.predict(X_test)), 'models/rf_accuracy.joblib')