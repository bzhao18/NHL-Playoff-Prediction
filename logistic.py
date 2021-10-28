import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

plt.rc("font", size=14)

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Get Numerical Data
game_data_columns = ['team_id', 'game_id', 'won', 'goals', 'shots', 'pim', 'powerPlayOpportunities', 'injured_player_count']

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

# Variable to Predict: 'won' from game_data
# Print info on training data 'won' labels
print(game_data['won'].value_counts())
sns.countplot(x='won',data=game_data, palette='hls')
plt.show()
