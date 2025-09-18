from cleaning_data import df_results, result_encoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

le = LabelEncoder()

#spliting data(X = dataframe without answer, y = answer)
X = df_results[['home_team_encoded', 'away_team_encoded']]
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
print(f'First RandomForest model accuracy:{accuracy_score(y_test ,model.predict(X_test))}')

##Tuning the model using GridSearchCV(barely got any better than the first model)
#
#
#
def RandomForest_GridSearchCV(X_train, y_train):
    param_grid_RandomForests = {
        'n_estimators': [10, 50, 100,200],#numbers of trees
        'max_depth': [10, 20],#how deep can the tree can go
        'min_samples_split' : [2, 5],#how many split do i want the sample to get(do not forget adding the S in samples!)
        'min_samples_leaf': [1, 2]#like branches of an trees?(also do not forget the S for god sake)
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state= 42),
                               param_grid_RandomForests,
                               cv=5,#nums of fold cross validation(watch youtube if dumb ah)
                               scoring='accuracy',#how to display scoring
                               n_jobs =-1,#use all cpu power to go faster?
                               verbose=1#see the progress so far
                               )
    grid_search.fit(X_train,y_train)
    best_model = grid_search.best_estimator_
    return best_model

best_model_RandomForest = RandomForest_GridSearchCV(X_train, y_train)
accuracy_RandomForest = best_model_RandomForest.score(X_test, y_test)
print('Second RandomForest model accuracy:', accuracy_RandomForest)

##Tuning model using KNN
#
#
#
X_train_KNN, X_test_KNN, y_train_KNN, y_test_KNN = train_test_split(X, y, test_size= 0.25, random_state= 42)

KNN_scaler = StandardScaler()
X_train_scaled = KNN_scaler.fit_transform(X_train_KNN)
X_test_scaled = KNN_scaler.transform(X_test_KNN)
def KNN_tune_model(X_train_scaled, y_train_KNN):
    param_grid_KNN = {
        'n_neighbors': range(1,21),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid_KNN, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_KNN)
    return grid_search.best_estimator_

best_model_KNN = KNN_tune_model(X_train_scaled, y_train_KNN)
#Evaluate how well the KNN model performed
def evaluate_model_KNN(model, X_test_scaled, y_test_KNN):#The same as the second model evaluate
    prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_KNN, prediction)
    return accuracy

accuracy_KNN = evaluate_model_KNN(best_model_KNN, X_test_scaled, y_test_KNN)
print(f'KNN model accuracy= {accuracy_KNN*100:.2f}%')


##Optional shits for show
#
#
#
# Taking team name and predict
all_team = pd.concat([df_results['home_team'], df_results['away_team']]).unique()
team_mapping = {team : i for i, team in enumerate(sorted(all_team))}
def pred_match(teamA, teamB):
    teamA_encoded = team_mapping.get(teamA)
    teamB_encoded = team_mapping.get(teamB)
    if teamA_encoded is None or teamB_encoded is None:
        return 'Unknown team'
    input_data = [[teamA_encoded, teamB_encoded]]
    input_scaled = scaler.transform(input_data)
    result_code = model.predict(input_scaled)[0]
    #Model confidence
    print('Class order:', result_encoder.classes_)
    print(model.predict_proba(input_scaled))
    result_label = result_encoder.inverse_transform([result_code])[0]
    return result_label
print(pred_match('Spain','Germany'))

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