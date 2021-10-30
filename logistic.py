import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Get Numerical Data 
# Skip team_id
game_data_columns = ['game_id', 'won', 'goals', 'shots', 'pim', 'powerPlayOpportunities', 'injured_player_count']

game_data = pd.read_csv("cleaned_data/game_stats.csv", usecols=game_data_columns)
# Change 'won' column from boolean to int type
game_data['won'] = game_data['won'].astype(int)
# Blanks in 'injured_player_count' changed from NaN type to 0
game_data['injured_player_count'] = game_data['injured_player_count'].fillna(0)

goalie_data_columns = ['game_id', 'team_id', 'player_id', 'timeOnIce', 'shots', 'saves', 'savePercentage']
goalie_data = pd.read_csv("cleaned_data/goalie_stats.csv", usecols=goalie_data_columns)
goalie_data['savePercentage'] = goalie_data['savePercentage'].fillna(0)

skater_data_columns = ['game_id', 'team_id', 'player_id', 'timeOnIce', 'assists', 'goals', 'shots','penaltyMinutes']
skater_data = pd.read_csv("cleaned_data/skater_stats.csv", usecols=skater_data_columns)

# -------------------------------------------------------------------------------------------------------------- #
# Following https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

plt.rc("font", size=14)

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Variable to Predict: 'won' from game_data
# Print info on training data 'won' labels
print(game_data['won'].value_counts())
sns.countplot(x='won',data=game_data, palette='hls')
# plt.show()

losses = len(game_data[game_data['won']==0])
wins = len(game_data[game_data['won']==1])
pct_loss = losses/(losses+wins)
print("Percentage of losses", pct_loss*100)
pct_wins = wins/(losses+wins)
print("Percentage of wins", pct_wins*100)

X = game_data.loc[:, game_data.columns != 'won']
y = game_data.loc[:, game_data.columns == 'won']

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
os_data_y= pd.DataFrame(data=os_data_y,columns=['won'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of losses in oversampled data",len(os_data_y[os_data_y['won']==0]))
print("Number of wins",len(os_data_y[os_data_y['won']==1]))
print("Proportion of loss data in oversampled data is ",len(os_data_y[os_data_y['won']==0])/len(os_data_X))
print("Proportion of win data in oversampled data is ",len(os_data_y[os_data_y['won']==1])/len(os_data_X))

# Could perform Recursive Feature Elimination here

# Logistic Regression Model
# Notes: Running on just on game_data and all features are significant.
X = os_data_X
y= os_data_y
logit_model=sm.Logit(y,X)
result=logit_model.fit()
# print(result.summary2())

# Remove any features that have p-values greater or equal to .05 here (look at P>|z| column)
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
