import pandas as pd
import numpy as np
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# TODO: Use pre-processed data from pca.py

# Get numerical data from game_stats.csv. Skip team_id.
# goalie_stats.csv and skater_stats.csv data not used yet since need to join with game_stats to have a 'won' label
game_data_columns = ['game_id', 'won', 'goals', 'shots', 'pim', 'powerPlayOpportunities', 'injured_player_count']
game_data = pd.read_csv("cleaned_data/game_stats.csv", usecols=game_data_columns)
# Change 'won' column from boolean to int type
game_data['won'] = game_data['won'].astype(int)
# Change NaN in 'injured_player_count' column to 0
game_data['injured_player_count'] = game_data['injured_player_count'].fillna(0)

# TODO: Split data into training and testing data by first and second half of the season.
X = game_data.loc[:, game_data.columns != 'won'] # Data
y = game_data.loc[:, game_data.columns == 'won'] # Data's labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --------------------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression implementation follows https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

# Info on current balance of win and loss rows
rows = X.shape[0]
win_rows = y[y['won'] == 1].shape[0]
loss_rows = y[y['won'] == 0].shape[0]
print("Rows: {}. Win rows: {}. Loss rows: {}".format(rows, win_rows, loss_rows))

# Balance data to have equal win and loss rows
oversampled = SMOTE(random_state=0)
oversampled_X, oversampled_y = oversampled.fit_resample(X_train, y_train)
X = pd.DataFrame(data=oversampled_X, columns=X_train.columns)
y = pd.DataFrame(data=oversampled_y, columns=['won'])
# Info on currentbalance of win and loss rows
rows = X.shape[0]
win_rows = y[y['won'] == 1].shape[0]
loss_rows = y[y['won'] == 0].shape[0]
print("Rows: {}. Win rows: {}. Loss rows: {}".format(rows, win_rows, loss_rows))

# Above link used Recursive Feature Elimination for data pre-processing at this point.

# Model implementation
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())

# Link removed any insignificant feature having p-value >= .05 at this point (P>|z| column)
# Note: game_data features are all significant (game_stats.csv version on 10/30/21)

# Logistic Regression Classifier Model Fitting
logreg = LogisticRegression()
logreg.fit(X_train, np.ravel(y_train))

# Predict test data labels
y_pred = logreg.predict(X_test)
score = logreg.score(X_test, y_test)
print('Accuracy on test data: {:.2f}'.format(score))

print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

# Link calculates precision, recall, F-measure, and support at this point

# Link plots receiver operating characteristic (ROC) curve at this point

# TODO: Utilize y_pred to determine predicted playoff bracket rankings