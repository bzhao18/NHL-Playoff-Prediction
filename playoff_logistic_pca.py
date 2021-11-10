import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Transform data using PCA to be used for Logistic Regression to predict making it to the playoffs
def logistic_pca(data, n_components):
    # Sample .2 (round up) number of seasons
    seasons = set(data['season'])
    num_seasons = len(seasons)
    num_test_seasons = math.ceil(num_seasons * .2)
    # random.seed(150) # Uncomment to get same sample. Change seed value to get different sample
    test_seasons = random.sample(seasons, num_test_seasons)
    print("Test seasons are:", test_seasons)

    # Split data into training and test sets
    X_train = data.loc[~data['season'].isin(test_seasons)]
    X_test = data.loc[data['season'].isin(test_seasons)]
    y_train = X_train.loc[:, 'end_season_playoff_standing']
    y_test = X_test.loc[:, 'end_season_playoff_standing']
    X_train = X_train.drop('end_season_playoff_standing', 1)
    X_test = X_test.drop('end_season_playoff_standing', 1)
    print("Proportion training: {0:.4f} | Proportion testing: {1:.4f}".format(X_train.shape[0] / data.shape[0], X_test.shape[0] / data.shape[0]))

    # Info on current balance of standings
    balance_info(X_train, y_train)
    # Balance data to have equal win and loss rows
    oversample = SMOTE(random_state=0)
    oversample_X_train, oversample_y_train = oversample.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(data=oversample_X_train, columns=X_train.columns)
    y_train = pd.DataFrame(data=oversample_y_train, columns=['end_season_playoff_standing'])
    balance_info(X_train, y_train)

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Would typically run model implementation that's commented out below to remove insignificant
    # features of p-value >= .05 (P>|z| column), and then rerun model implementation.
    # However, this requires the labels to be either 0s or 1s as Logistic Regression requires
    # import statsmodels.api as sm
    # logit_model = sm.Logit(y_train, X_train)
    # result = logit_model.fit()
    # print(result.summary2())

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, np.ravel(y_train))

    # Predict test data labels
    y_pred = logreg.predict(X_test)
    print(y_pred)
    score = logreg.score(X_test, y_test)
    print('Accuracy on test data: {:.2f}'.format(score))

    # Confusion matrix
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=16)
    roc_auc = auc(fpr, tpr)
    print("ROC Area Under the Curve (AUC): %0.2f" % roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # See link in logistic.py to implement other measurements like precision, recall, F-measure, and support

def balance_info(X_train, y_train):
    standing_1_rows = y_train[y_train == 1].shape[0]
    standing_2_rows = y_train[y_train == 2].shape[0]
    standing_3_rows = y_train[y_train == 3].shape[0]
    standing_4_rows = y_train[y_train == 4].shape[0]
    standing_5_rows = y_train[y_train == 5].shape[0]
    standing_6_rows = y_train[y_train == 6].shape[0]
    standing_7_rows = y_train[y_train == 7].shape[0]
    standing_8_rows = y_train[y_train == 8].shape[0]
    standing_9_rows = y_train[y_train == 9].shape[0]
    standing_10_rows = y_train[y_train == 10].shape[0]
    standing_11_rows = y_train[y_train == 11].shape[0]
    standing_12_rows = y_train[y_train == 12].shape[0]
    standing_13_rows = y_train[y_train == 13].shape[0]
    standing_14_rows = y_train[y_train == 14].shape[0]
    standing_15_rows = y_train[y_train == 15].shape[0]
    standing_16_rows = y_train[y_train == 16].shape[0]
    no_playoff_rows = y_train[y_train == 0].shape[0]
    print("Rows: {}. Rank 1: {}. Rank 2: {}. Rank 3: {}. Rank 4: {}. Rank 5: {}. Rank 6: {}. Rank 7: {}. Rank 8: {}. Rank 9: {}. Rank 10: {}. Rank 11: {}. Rank 12: {}. Rank 13: {}. Rank 14: {}. Rank 15: {}. Rank 16: {}. No playoffs: {}. ".format(X_train.shape[0], standing_1_rows, standing_2_rows, standing_3_rows, standing_4_rows, standing_5_rows, standing_6_rows, standing_7_rows, standing_8_rows, standing_9_rows, standing_10_rows, standing_11_rows, standing_12_rows, standing_13_rows, standing_14_rows, standing_15_rows, standing_16_rows, no_playoff_rows))
    
# Read in data
game_data_columns = ['team_id','season','total_first_half_season_wins','total_first_half_season_shots','total_first_half_season_goals','total_first_half_season_pim','total_first_half_season_powerPlayOpportunities','end_season_playoff_standing']
game_data = pd.read_csv("cleaned_data_v3/first_half_season_summary.csv", usecols=game_data_columns)
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].fillna(0) # Change NaNs to 0
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].astype(int)

logistic_pca(game_data, 5)