# PCA Implementation on Datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import visualize as v

from sklearn.preprocessing import MinMaxScaler

# only numerical columns in data set
game_data_columns = ['team_id', 'game_id', 'won', 'goals', 'shots', 'pim', 'powerPlayOpportunities',
                     'injured_player_count']
goalie_data_columns = ['game_id', 'team_id', 'player_id', 'timeOnIce', 'shots', 'saves', 'savePercentage']
skater_data_columns = ['game_id', 'team_id', 'player_id', 'timeOnIce', 'assists', 'goals', 'shots', 'penaltyMinutes']

first_half_columns = ['team_id', 'season', 'total_first_half_season_wins', 'total_first_half_season_shots',
                      'total_first_half_season_goals',
                      'total_first_half_season_pim', 'total_first_half_season_powerPlayOpportunities']
# For visualization only
# This is shortening the names for data visualization -> better legibility
first_half_columns_modified = ['team_id', 'season_in_2000s', 'first_half_season_wins', 'first_half_season_shots',
                               'first_half_season_goals',
                               'first_half_season_pim', 'first_half_season_ppo']

v4_avg_columns = ['team_id', 'Total Wins', 'Shots', 'Blocked Shots', 'Goals', 'Power Play Goals',
                'Power Play Opportunities', 'PIM', 'Player Hits', 'Giveaways', 'Takeaways', 'Injured Players']

v4_total_columns = ['team_id', 'Total Wins', 'Total Shots', 'Blocked Shots', 'Goals', 'Power Play Goals',
                'Power Play Opportunities', 'PIM', 'Player Hits', 'Giveaways', 'Takeaways', 'Injured Players']

v5_gbg_first_half_columns = ['Home Shots', 'Home Shots Blocked', 'Home Power Play Opportunities', 'Home PIM', 
                            'Home Player Hits', 'Home Giveaways', 'Home Takeaways', 'Home Injured Players', 
                            'Away Shots', 'Away Shots Blocked', 'Away Power Play Opportunities', 'Away PIM', 
                            'Away Player Hits', 'Away Giveaways', 'Away Takeaways', 'Away Injured Players']

# reading in csv files
game_data = pd.read_csv("cleaned_data/game_stats.csv", usecols=game_data_columns)
# changing 'won' column from boolean to int type
game_data['won'] = game_data['won'].astype(int)
# blank spots in 'injured_player_count' changed from NaN type to 0
game_data['injured_player_count'] = game_data['injured_player_count'].fillna(0)

goalie_data = pd.read_csv("cleaned_data/goalie_stats.csv", usecols=goalie_data_columns)
goalie_data['savePercentage'] = goalie_data['savePercentage'].fillna(0)

skater_data = pd.read_csv("cleaned_data/skater_stats.csv", usecols=skater_data_columns)

# first_half_season_summary.csv file
first_half = pd.read_csv("cleaned_data_v3/first_half_season_summary.csv", usecols=first_half_columns)

# first_half_season_avg (cleaned_data_v4)
v4_avg = pd.read_csv("cleaned_data_v4/Logistic Model - Summary/first_half_season_avg.csv", usecols=v4_avg_columns)

# first_half_season_total (cleaned_data_v4)
v4_total = pd.read_csv("cleaned_data_v4/Logistic Model - Summary/first_half_season_total.csv", usecols=v4_total_columns)

# game-by-game v5 cleaned data first half matchups
v5_gbg_first_half = pd.read_csv("cleaned_data_v5/Logistic Regression - GamebyGame/first_half_matchups.csv", usecols=v5_gbg_first_half_columns)

# For visualization only
# The names have been shortened in the file and the seasons have been changed from e.g. 2012 to 12 to reduce cluttering
first_half_modified = pd.read_csv("cleaned_data_v3/first_half_season_summary_modified.csv",
                                  usecols=first_half_columns_modified)

# The visualize function can also be used on the pre-PCA data. However, this will take a bit to run.
# v.visualizeComponents(data=np.array(game_data), labels=game_data_columns, title='Pre-PCA Game Data')
# v.visualizeComponents(data=np.array(goalie_data), labels=goalie_data_columns, title='Pre-PCA Goalie Data')
# v.visualizeComponents(data=np.array(skater_data), labels=skater_data_columns, title='Pre-PCA Skater Data')
# v.visualizeComponents(data=np.array(first_half_modified), labels=first_half_columns_modified,
#                       title='First Half Season Summary Data')
# -------------------------------------------------------------------------------------------------------------- #
## finding the correct number of components to get for each dataset 
## plots graph of variance vs components 
# turn this into function later that reads in data, finds # of columns, sets proper x/y, output component #s

scaler = MinMaxScaler()
# finds correct number of components for dataset read in on line below - replace to find others
data_rescaled = scaler.fit_transform(v5_gbg_first_half)
# 95% of variance
pca = PCA(n_components=0.95)
pca.fit(data_rescaled)
reduced = pca.transform(data_rescaled)

pca = PCA().fit(data_rescaled)

plt.rcParams["figure.figsize"] = (12, 6)

fig, ax = plt.subplots()
# for goalie_stats.csv dataset xi = np.arange(1,8,step=1) because initially 7 components
xi = np.arange(1, 17, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0, 1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 16, step=1))  # change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

ax.grid(axis='x')
plt.show()

# ------------------------------------------------------------------------------------------------------------- #

# PCA for each dataset

## 5 components for game_data.csv
df1 = pd.DataFrame(game_data)
scaling1 = StandardScaler()
scaling1.fit(df1)
Scaled_data1 = scaling1.transform(df1)
# set n_components to number of PCA components that we want
principal1 = PCA(n_components=5)
principal1.fit(Scaled_data1)
x1 = principal1.transform(Scaled_data1)
print(x1.shape)

print(principal1.components_)

# v.visualizeComponents(data=x1, title='Post-PCA Game Data')

## 4 components for goalie_stats.csv
df2 = pd.DataFrame(game_data)
scaling2 = StandardScaler()
scaling2.fit(df2)
Scaled_data2 = scaling2.transform(df2)
# set n_components to number of PCA components that we want
principal2 = PCA(n_components=4)
principal2.fit(Scaled_data2)
x2 = principal2.transform(Scaled_data2)
print(x2.shape)

print(principal2.components_)

# v.visualizeComponents(data=x2, title='Post-PCA Goalie Data')

## 5 components for skater_stats.csv
df3 = pd.DataFrame(skater_data)
scaling3 = StandardScaler()
scaling3.fit(df3)
Scaled_data3 = scaling3.transform(df3)
# set n_components to number of PCA components that we want
principal3 = PCA(n_components=5)
principal3.fit(Scaled_data3)
x3 = principal3.transform(Scaled_data3)
print(x3.shape)

print(principal3.components_)

# v.visualizeComponents(data=x3, title='Post-PCA Skater Data')  # Slow because there are a lot of data points

## 5 components for first_half_season_summary.csv
df4 = pd.DataFrame(first_half)
scaling4 = StandardScaler()
scaling4.fit(df4)
Scaled_data4 = scaling4.transform(df4)
# set n_components to number of PCA components that we want
principal4 = PCA(n_components=5)
principal4.fit(Scaled_data4)
x4 = principal4.transform(Scaled_data4)
print(x4.shape)

print(principal4.components_)

# v.visualizeComponents(data=x4, title='Post-PCA First Half Season Data')  # Slow because there are a lot of data points


## 9 components for first_half_season_avg.csv
df5 = pd.DataFrame(v4_avg)
scaling5 = StandardScaler()
scaling5.fit(df5)
Scaled_data5 = scaling5.transform(df5)
# set n_components to number of PCA components that we want
principal5 = PCA(n_components=9)
principal5.fit(Scaled_data5)
x5 = principal5.transform(Scaled_data5)
print(x5.shape)

print(principal5.components_)

# v.visualizeComponents(data=x5, title='Post-PCA First Half Season Average cleaned_data_v4')  # Slow because there are a lot of data points


## 9 components for first_half_season_total.csv
df6 = pd.DataFrame(v4_total)
scaling6 = StandardScaler()
scaling6.fit(df6)
Scaled_data6 = scaling6.transform(df6)
# set n_components to number of PCA components that we want
principal6 = PCA(n_components=9)
principal6.fit(Scaled_data6)
x6 = principal6.transform(Scaled_data6)
print(x6.shape)

print(principal6.components_)

# v.visualizeComponents(data=x6, title='Post-PCA First Half Season Total cleaned_data_v4')  # Slow because there are a lot of data points