# TODO: Create a neural network to perform classification on ranking of NHL teams in the playoffs.
import pandas as pd
def get_data(columns, filePath):
    data = pd.read_csv(filePath, names=columns)
    return data

# creat neural network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
game_data_columns = ['team_id','season','total_first_half_season_wins','total_first_half_season_shots','total_first_half_season_goals','total_first_half_season_pim','total_first_half_season_powerPlayOpportunities','end_season_playoff_standing']
game_data = get_data(game_data_columns, "cleaned_data_v3/first_half_season_summary.csv")
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].fillna(0) # Change NaNs to 0
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].astype(int)

# use the data to train the model
X = game_data.drop(['end_season_playoff_standing'], axis=1)
y = game_data['end_season_playoff_standing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
