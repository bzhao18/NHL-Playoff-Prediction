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

# reading in csv files
game_data = pd.read_csv("cleaned_data/game_stats.csv", usecols=game_data_columns)
# changing 'won' column from boolean to int type
game_data['won'] = game_data['won'].astype(int)
# blank spots in 'injured_player_count' changed from NaN type to 0
game_data['injured_player_count'] = game_data['injured_player_count'].fillna(0)

goalie_data = pd.read_csv("cleaned_data/goalie_stats.csv", usecols=goalie_data_columns)
goalie_data['savePercentage'] = goalie_data['savePercentage'].fillna(0)

skater_data = pd.read_csv("cleaned_data/skater_stats.csv", usecols=skater_data_columns)

# The visualize function can also be used on the pre-PCA data. However, this will take a bit to run.
# v.visualizeComponents(data=np.array(game_data), labels=game_data_columns)
# v.visualizeComponents(data=np.array(goalie_data), labels=goalie_data_columns)
# v.visualizeComponents(data=np.array(skater_data), labels=skater_data_columns)

# -------------------------------------------------------------------------------------------------------------- #
## finding the correct number of components to get for each dataset 
## plots graph of variance vs components 
# turn this into function later that reads in data, finds # of columns, sets proper x/y, output component #s

scaler = MinMaxScaler()
# finds correct number of components for dataset read in on line below - replace to find others
data_rescaled = scaler.fit_transform(game_data)
# 95% of variance
pca = PCA(n_components=0.95)
pca.fit(data_rescaled)
reduced = pca.transform(data_rescaled)

pca = PCA().fit(data_rescaled)

plt.rcParams["figure.figsize"] = (12, 6)

fig, ax = plt.subplots()
# for goalie_stats.csv dataset xi = np.arange(1,8,step=1) because initially 7 components
xi = np.arange(1, 9, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0, 1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 11, step=1))  # change from 0-based array index to 1-based human-readable label
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

# v.visualizeComponents(data=x1)

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

# v.visualizeComponents(data=x2)

## 5 components for skater_stats.csv
df3 = pd.DataFrame(skater_data)
scaling3 = StandardScaler()
scaling3.fit(df3)
Scaled_data3 = scaling3.transform(df3)
# set n_components to number of PCA components that we want
principal3 = PCA(n_components=5)
principal3.fit(Scaled_data1)
x3 = principal3.transform(Scaled_data3)
print(x3.shape)

print(principal3.components_)

# v.visualizeComponents(data=x3)  # Slow because there are a lot of data points
