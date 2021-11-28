import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# TODO: Use pre-processed data from pca.py

# # Using NB to predict winner of a single game; does better than logistic regression in terms of accuracy
# # Get numerical data from game_stats.csv. Skip team_id.
# # goalie_stats.csv and skater_stats.csv data not used yet since need to join with game_stats to have a 'won' label
# game_data_columns = ['game_id', 'won', 'goals', 'shots', 'pim', 'powerPlayOpportunities', 'injured_player_count']
# game_data = pd.read_csv("cleaned_data/game_stats.csv", usecols=game_data_columns)
# # Change 'won' column from boolean to int type
# game_data['won'] = game_data['won'].astype(int)
# # Change NaN in 'injured_player_count' column to 0
# game_data['injured_player_count'] = game_data['injured_player_count'].fillna(0)
#
# # TODO: Split data into training and testing data appropriately for the project.
# X = game_data.loc[:, game_data.columns != 'won']  # Data
# y = game_data.loc[:, game_data.columns == 'won']  # Data's labels
# y = np.squeeze(y)  # Prevent shape warning
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

# ----------------------------------------------------------------------------------------------------------------------
# mnb = MultinomialNB()
#
# mnb.fit(X_train, y_train)
#
# y_pred = mnb.predict(X_test)
# probabilities = mnb.predict_proba(X_test)
# np.set_printoptions(suppress=True)
# # print(probabilities)
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# ----------------------------------------------------------------------------------------------------------------------

# Using NB for multi-class classification
# game_data_columns = ['team_id', 'total_first_half_season_wins', 'total_first_half_season_shots',
#                      'total_first_half_season_goals', 'total_first_half_season_pim',
#                      'total_first_half_season_powerPlayOpportunities', 'end_season_playoff_standing']
# game_data = pd.read_csv("cleaned_data_v3/first_half_season_summary.csv", usecols=game_data_columns)
#
# # Change NaN in 'injured_player_count' column to 0
# game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].fillna(0)
#
# # TODO: Split data into training and testing data appropriately for the project.
# X = game_data.loc[:, game_data.columns != 'end_season_playoff_standing']  # Data
# y = game_data.loc[:, game_data.columns == 'end_season_playoff_standing']  # Data's labels
# y = np.squeeze(y)  # Prevent shape warning
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
#
# mnb = MultinomialNB()
#
# mnb.fit(X_train, y_train)
#
# y_pred = mnb.predict(X_test)
# probabilities = mnb.predict_proba(X_test)
# np.set_printoptions(suppress=True)
# # print(probabilities)
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# ----------------------------------------------------------------------------------------------------------------------

# Using NB for binary blassification of playoff standings
game_data_columns = ['team_id', 'total_first_half_season_wins', 'total_first_half_season_shots',
                     'total_first_half_season_goals', 'total_first_half_season_pim',
                     'total_first_half_season_powerPlayOpportunities', 'end_season_playoff_standing']
game_data = pd.read_csv("cleaned_data_v3/first_half_season_summary.csv", usecols=game_data_columns)

# Change NaN in 'injured_player_count' column to 0
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].fillna(0)

# TODO: Split data into training and testing data appropriately for the project.
X = game_data.loc[:, game_data.columns != 'end_season_playoff_standing']  # Data
y = game_data.loc[:, game_data.columns == 'end_season_playoff_standing']  # Data's labels
y = np.squeeze(y)  # Prevent shape warning
y = np.where(y > 0, 1, 0)  # For binary classification, change any playoff standing that is not 0, to 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

mnb = MultinomialNB()

mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)
probabilities = mnb.predict_proba(X_test)
np.set_printoptions(suppress=True)
# print(probabilities)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
