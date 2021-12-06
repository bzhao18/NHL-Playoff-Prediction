import pandas as pd
import numpy as np

# TODO: Read in data from x_test.csv
def read_data(file_name):
    data = pd.read_csv(file_name)
    return data

playoff_standings = read_data('season_avg.csv')

# TODO: Create visualizations: x axis should be "true_standings" and y axis should be "predicted_standings"
def visualize_predicted_true(data, season):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    # change null data to be 0
    data['predicted_standings'] = data['predicted_standings'].fillna(0)
    # change 0 to be 17
    data['Standing'] = data['Standing'].replace(0, 17)
    data['predicted_standings'] = data['predicted_standings'].replace(0, 17)
    plt.scatter(data['Standing'], data['predicted_standings'])
    plt.xlabel('True Standings')
    plt.yticks(np.arange(1, data['predicted_standings'].max() + 1, 1))
    plt.ylabel('Predicted Standings')
    plt.title('True vs. Predicted Standings: ' + str(season))
    plt.show()

#visualize_predicted_true(playoff_standings, 2000)

# TODO: Create visualizations: x axis should be "true_standings" and y axis should be "playoffs" with range 0-1 inclusive
# the key is that playoffs is a probability from 0-1
def visualize_playoff_probability(data, season):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    # convert playoffs to doubles
    data['prob_playoffs'] = data['prob_playoffs'].astype(float)
    # change null data to be 0
    data['predicted_standings'] = data['predicted_standings'].fillna(0)
    # change 0 to be 17
    data['Standing'] = data['Standing'].replace(0, 17)
    plt.scatter(data['Standing'], data['prob_playoffs'])
    plt.xlabel('True Standings')
    plt.ylabel('Probability of Playoffs')
    plt.xticks(np.arange(1, data['Standing'].max() + 1, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(-0.05, 1.05)
    plt.title('True Standing vs Probability of Playoffs: ' + str(season))
    plt.show()

visualize_playoff_probability(playoff_standings, 2018)
#visualize_predicted_true(playoff_standings, 2018)