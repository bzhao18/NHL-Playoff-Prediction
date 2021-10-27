# PCA Implementation on Datasets
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



# only numerical columns in data set
game_data_columns = ['team_id', 'game_id', 'won', 'goals', 'shots', 'pim', 'powerPlayOpportunities', 'injured_player_count']
goalie_data_columns = ['game_id', 'team_id', 'player_id', 'timeOnIce', 'shots', 'saves', 'savePercentage']
skater_data_columns = ['game_id', 'team_id', 'player_id', 'timeOnIce', 'assists', 'goals', 'shots','penaltyMinutes']

# reading in csv files
game_data = pd.read_csv("cleaned_data/game_stats.csv", usecols=game_data_columns)
# changing 'won' column from boolean to int type
game_data['won'] = game_data['won'].astype(int)
# blank spots in 'injured_player_count' changed from NaN type to 0
game_data['injured_player_count'] = game_data['injured_player_count'].fillna(0)

goalie_data = pd.read_csv("cleaned_data/goalie_stats.csv", usecols=goalie_data_columns)

skater_data = pd.read_csv("cleaned_data/skater_stats.csv", usecols=skater_data_columns)

## game_stats.csv PCA

scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(game_data)
#95% of variance
pca = PCA(n_components = 0.95)
pca.fit(data_rescaled)
reduced = pca.transform(data_rescaled)

pca = PCA().fit(data_rescaled)


plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 9, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

# plot shows that we need 5 components

df1=pd.DataFrame(game_data)
scaling=StandardScaler()
scaling.fit(df1)
Scaled_data=scaling.transform(df1)
# set n_components to number of PCA components that we want
principal=PCA(n_components=5)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)
print(x.shape)

print(principal.components_)
