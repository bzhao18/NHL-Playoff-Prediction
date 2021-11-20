import pandas as pd
import numpy as np

# TODO: Read in data from x_test.csv
def read_data(file_name):
    data = pd.read_csv(file_name)
    return data

playoff_standings = read_data('x_test.csv')

# TODO: Create visualizations: x axis should be "true_standings" and y axis should be "predicted_standings"
def visualize_predicted_true(data, season):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    # change null data to be 0
    data = data.loc[data['season'] == season, :]
    data['predicted_standings'] = data['predicted_standings'].fillna(0)
    plt.scatter(data['true_standings'], data['predicted_standings'])
    plt.xlabel('True Standings')
    plt.ylabel('Predicted Standings')
    plt.title('True vs. Predicted Standings')
    plt.show()

#visualize_predicted_true(playoff_standings, 2000)

# TODO: Create visualizations: x axis should be "true_standings" and y axis should be "playoffs" with range 0-1 inclusive
# the key is that playoffs is a probability from 0-1
def visualize_playoff_probability(data, season):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    # convert playoffs to doubles
    data['playoffs'] = data['playoffs'].astype(float)
    # change null data to be 0
    data = data.loc[data['season'] == season]
    plt.scatter(data['true_standings'], data['playoffs'])
    plt.xlabel('True Standings')
    plt.ylabel('Probability of Playoffs')
    plt.xticks(np.arange(0, data['true_standings'].max() + 1, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(-0.05, 1.05)
    plt.title('True Standing vs Probability of Playoffs: ' + str(season))
    plt.show()

seasons = [2008, 2010, 2000, 2013].sort()
for season in seasons:
    visualize_playoff_probability(playoff_standings, season)