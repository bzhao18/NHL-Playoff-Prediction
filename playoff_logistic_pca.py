import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv

# Transform data using PCA to be used for Logistic Regression to predict making it to the playoffs
# TODO: Dynamic retrieval of n_componenets from PCA
def logistic_pca(data, n_components):
    # Sample 20% number (round up) of seasons
    seasons = set(data['season'])
    num_seasons = len(seasons)
    num_test_seasons = math.ceil(num_seasons * .2)
    random.seed(150) # Uncomment to get same sample. Change seed value to get different sample.
    test_seasons = random.sample(seasons, num_test_seasons)
    print("Test seasons are:", test_seasons)
    

    # Split data into training and test sets
    X_train = data.loc[~data['season'].isin(test_seasons)]
    X_test = data.loc[data['season'].isin(test_seasons)]
    y_train = X_train.loc[:, 'end_season_playoff_standing']
    y_test = X_test.loc[:, 'end_season_playoff_standing']
    X_train = X_train.drop('end_season_playoff_standing', 1)
    X_test = X_test.drop('end_season_playoff_standing', 1)
    # write x_test to csv file
    # X_test.to_csv('x_test.csv')
    print("Proportion training: {0:.4f} | Proportion testing: {1:.4f}".format(X_train.shape[0] / data.shape[0], X_test.shape[0] / data.shape[0]))

    # Print info on current balance of standings
    balance_info(X_train, y_train)
    # Balance data to have equal number of rows for making it the playoffs and not
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

    # Remove insignificant PCA dimensions of p-value >= .05.
    summary = logit_model_summary(X_train, y_train) # statsmodels.api's Logistic Regression
    # print(summary)
    p_values = summary.tables[1]['P>|z|']
    remove_dim_bool, dims_to_remove = check_insignificant(p_values)
    while remove_dim_bool:
        print("Removing PCA dimensions at indices:", dims_to_remove)
        X_train = np.delete(X_train, dims_to_remove, axis=1)
        X_test = np.delete(X_test, dims_to_remove, axis=1)
        summary = logit_model_summary(X_train, y_train)
        p_values = summary.tables[1]['P>|z|']
        remove_dim_bool, dims_to_remove = check_insignificant(p_values)

    # sklearn.linear_model's Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, np.ravel(y_train))

    # Predict test data labels
    y_pred = logreg.predict(X_test)
    score = logreg.score(X_test, y_test)
    print('Accuracy on test data: {:.2f}'.format(score))

    # Confusion matrix
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    # # append probabilites to x_test.csv for each column
    # probabilities = logreg.predict_proba(X_test)
    # with open('x_test.csv', 'a') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # add 2 columns for probabilities
    #     writer.writerow(['prob_no_playoffs', 'prob_playoffs'])
    #     for row in probabilities:
    #         writer.writerow(row)

    

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
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

# Print the current balance of data rows that do an do not make it to the playoffs
def balance_info(X_train, y_train):
    playoff_rows = y_train[y_train == 1].shape[0]
    no_playoff_rows = y_train[y_train == 0].shape[0]
    print("Rows: {}. Playoffs: {}. No playoffs: {}. ".format(X_train.shape[0], playoff_rows, no_playoff_rows))

# For any p-value p of a PCA dimension, if p >= .05, then add the dimension index to a list of indices of
# dimensions to be removed
def check_insignificant(p_values):
    removed_dim_bool = False
    dims_to_remove = []
    for index, p in enumerate(p_values):
        if p >= .05:
            dims_to_remove.append(index)
            removed_dim_bool = True
    return removed_dim_bool, dims_to_remove

# Get summary statistics of statsmodels.api's Logistic Regression
def logit_model_summary(X_train, y_train):
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit()
    return result.summary2()

# Read in data
game_data_columns = ['team_id','season','total_first_half_season_wins','total_first_half_season_shots','total_first_half_season_goals','total_first_half_season_pim','total_first_half_season_powerPlayOpportunities','end_season_playoff_standing']
game_data = pd.read_csv("cleaned_data_v3/first_half_season_summary.csv", usecols=game_data_columns)
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].fillna(0) # Change NaNs to 0
game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].astype(int)

# test_seasons = [2008, 2010, 2000, 2013]
# test_data_cols = ['team_id','team','season','total_first_half_season_wins','total_first_half_season_shots','total_first_half_season_goals','total_first_half_season_pim','total_first_half_season_powerPlayOpportunities','end_season_playoff_standing']
# test_data = pd.read_csv("cleaned_data_v3/first_half_season_summary.csv", usecols=test_data_cols)
# test_data['end_season_playoff_standing'] = test_data['end_season_playoff_standing'].fillna(0)
# test_data = test_data[test_data['season'].isin(test_seasons)]
# with open('x_test.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(test_data_cols)
#     for index, row in test_data.iterrows():
#         writer.writerow(row)

game_data['end_season_playoff_standing'] = game_data['end_season_playoff_standing'].mask(game_data['end_season_playoff_standing'] > 0, 1)

logistic_pca(game_data, 5)