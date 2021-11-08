import pandas as pd
import numpy as np
import math
import random
import statsmodels as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

# Transform data using PCA and use transformed data for logistic regression
def logistic_pca(data, n_components):
    # Sample .2 (round up) number of seasons
    seasons = set(data['season'])
    num_seasons = len(seasons)
    num_test_seasons = math.ceil(num_seasons * .2)
    test_seasons = random.sample(seasons, num_test_seasons) # Need seed approach if want same sample when run again
    print("Test seasons are:", test_seasons)

    # Split data into training and test sets
    X_train = data.loc[~data['season'].isin(test_seasons)]
    X_test = data.loc[data['season'].isin(test_seasons)]
    y_train = X_train.loc[:, 'end_season_playoff_standing']
    y_test = X_test.loc[:, 'end_season_playoff_standing']
    X_train = X_train.drop('end_season_playoff_standing', 1)
    X_test = X_test.drop('end_season_playoff_standing', 1)
    print("Proportion training: {0:.4f} | Proportion testing: {1:.4f}".format(X_train.shape[0] / data.shape[0], X_test.shape[0] / data.shape[0]))

    # standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    # logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train, np.ravel(y_train))

    # Predict test data labels
    y_pred = logreg.predict(X_test)
    score = logreg.score(X_test, y_test)
    print('Accuracy on test data: {:.2f}'.format(score))

    # Compute confusion matrix
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=16)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC: %0.2f" % roc_auc)
    # plot ROC curve
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

game_data_columns = ['team_id','season','total_first_half_season_wins','total_first_half_season_shots','total_first_half_season_goals','total_first_half_season_pim','total_first_half_season_powerPlayOpportunities','end_season_playoff_standing']
game_data = pd.read_csv("cleaned_data_v3/first_half_season_summary.csv", usecols=game_data_columns)
# Change 'won' column from boolean to int type
# game_data['won'] = game_data['won'].astype(int)
# Change NaN in 'injured_player_count' column to 0
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].fillna(0)
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].astype(int)

logistic_pca(game_data, 5)




