import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from predict_second_half_helpers import *

# Transform data using PCA to be used for Logistic Regression to predict second half game outcomes
def logistic_pca(training, testing, n_components, label_col):
    y_train = training.loc[:,label_col]
    X_train = training.drop(label_col, 1)
    y_test = testing.loc[:,label_col]
    X_test = testing.drop(label_col, 1)
    print(X_test)

    print("Training rows: {} | Testing rows: {}".format(X_train.shape[0], X_test.shape[0]))
    total_rows = X_train.shape[0] + X_test.shape[0]
    print("Proportion training: {0:.4f} | Proportion testing: {1:.4f}".format(X_train.shape[0] / total_rows, X_test.shape[0] / total_rows))

    # Print current balance of rows of home team wins and losses
    balance_info(y_train, label_col)
    # Balance data
    oversample = SMOTE(random_state=0)
    oversample_X_train, oversample_y_train = oversample.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(data=oversample_X_train, columns=X_train.columns)
    y_train = pd.DataFrame(data=oversample_y_train, columns=[label_col])
    balance_info(y_train, label_col)

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

    # statsmodels.api's Logistic Regression
    summary = logit_model_summary(X_train, y_train)
    # print(summary)
    p_values = summary.tables[1]['P>|z|']
    remove_dim_bool, dims_to_remove = check_insignificant(p_values)
    # Remove insignificant PCA dimensions of p-value >= .05.
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

    # Update teams' seasons' win totals by producing cleaned_data_v6/Logistic Regression - GamebyGame/win_totals.csv
    wins = get_season_win_totals(y_pred)

    # # Predict 16 teams that make it to each seasons' playoffs based on updated win totals
    # get_playoff_teams(wins)

# Print the current balance of data rows of home team wins and losses
def balance_info(y_train, label_col):
    np_y_train = y_train.to_numpy()
    wins = np.where(np_y_train == 1)[0]
    losses = np.where(np_y_train == 0)[0]
    print("Current Training Data Balance. Total Rows: {}. Home wins: {}. Home losses: {}. ".format(y_train.shape[0], wins.shape[0], losses.shape[0]))

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

def get_data(columns, filePath):
    data = pd.read_csv(filePath, usecols=columns)
    return data

# Read in data
columns = ['Home Shots', 'Home Shots Blocked', 'Home Power Play Opportunities', 'Home PIM', 'Home Player Hits', 'Home Giveaways', 'Home Takeaways', 'Home Injured Players', 'Away Shots', 'Away Shots Blocked', 'Away Power Play Opportunities', 'Away PIM', 'Away Player Hits', 'Away Giveaways', 'Away Takeaways', 'Away Injured Players', 'Home Win']
first_half = get_data(columns, "cleaned_data_v6/Logistic Regression - GamebyGame/first_half_matchups.csv")

# Run if stats_second_half_matchups.csv not generated yet in cleaned_data_v6/Logistic Regression - GamebyGame
predict_second_half()

second_half = get_data(columns, "cleaned_data_v6/Logistic Regression - GamebyGame/stats_second_half_matchups.csv")

logistic_pca(first_half, second_half, 13, 'Home Win')