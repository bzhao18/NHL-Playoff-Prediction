from operator import mul
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
from sklearn.model_selection import train_test_split
import csv
from sklearn import metrics

# Transform data using PCA to be used for Logistic Regression to predict making it to the playoffs
# TODO: Dynamic retrieval of n_componenets from PCA
def logistic_pca(data, n_components, season_col, standing_col):
    X = data.loc[:, data.columns != 'end_season_playoff_standing']  # Data
    y = data.loc[:, data.columns == 'end_season_playoff_standing']  # Data's labels
    y = np.squeeze(y)  # Prevent shape warning
    y = np.where(y > 0, 1, 0)  # For binary classification, change any playoff standing that is not 0, to 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)


    # Split data into training and test sets - test only on season 2018
    X_train = data[data[season_col] != 2018]
    X_test = data[data[season_col] == 2018]
    y_train = X_train.loc[:, standing_col]
    y_test = X_test.loc[:, standing_col]
    X_train = X_train.drop(standing_col, 1)
    X_test = X_test.drop(standing_col, 1)
    print("Proportion training: {0:.4f} | Proportion testing: {1:.4f}".format(X_train.shape[0] / data.shape[0], X_test.shape[0] / data.shape[0]))

    # Print info on current balance of standings
    balance_info(X_train, y_train)
    # Balance data to have equal number of rows for making it the playoffs and not
    oversample = SMOTE(random_state=0)
    oversample_X_train, oversample_y_train = oversample.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(data=oversample_X_train, columns=X_train.columns)
    y_train = pd.DataFrame(data=oversample_y_train, columns=[standing_col])
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

    # logistic regression on labels 0-16
    logit_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    logit_model.fit(X_train, y_train)
    # no two data points should have the same label
    y_pred = logit_model.predict(X_test)
    probabilities = logit_model.predict_proba(X_test)

    indices = []
    for i in range(16):
        probSeg = np.expand_dims(probabilities[:, i], axis=1)
        index = np.argmax(probSeg)
        indices.append(index)
        probabilities[index, i] = -1
        y_pred[index] = i + 1

    for i in range(31):
        if i not in indices:
            y_pred[i] = 0



    #print 
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, y_pred)))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


    # append probabilites to x_test.csv for each column
    # probabilities = logreg.predict_proba(X_test)
    # with open('season_avg.csv', 'a') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # add 2 columns for probabilities
    #     writer.writerow(['prob_no_playoffs', 'prob_playoffs'])
    #     for row in probabilities:
    #         writer.writerow(row)

    

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

# Print the current balance of data rows that do an do not make it to the playoffs
def balance_info(X_train, y_train):
    playoff_rows = y_train[y_train == 1].shape[0]
    no_playoff_rows = y_train[y_train == 0].shape[0]
    print("Rows: {}. Playoffs: {}. No playoffs: {}. ".format(X_train.shape[0], playoff_rows, no_playoff_rows))


def get_data(columns, filePath):
    data = pd.read_csv(filePath, usecols=columns)
    return data

def v4_data(type):
    if type == 'avg':
        shots = "Shots"
    else:
        shots = "Total Shots"
    data_col = ['Season' , 'team_id', 'Total Wins',shots,'Blocked Shots','Goals','Power Play Goals','Power Play Opportunities','PIM','Player Hits','Giveaways','Takeaways','Injured Players','Standing']
    data = get_data(data_col, "cleaned_data_v4/Logistic Model - Summary/first_half_season_" + type + ".csv")
    data['Standing'] = data['Standing'].fillna(0) # Change NaNs to 0
    data['Standing'] = data['Standing'].astype(int)
    # test_seasons = [2008, 2006, 2003]
    # test_data_cols = ['Season' , 'team_id','Team', 'Total Wins',shots,'Blocked Shots','Goals','Power Play Goals','Power Play Opportunities','PIM','Player Hits','Giveaways','Takeaways','Injured Players','Standing']
    # test_data = get_data(test_data_cols, "cleaned_data_v4/Logistic Model - Summary/first_half_season_" + type + ".csv")
    # test_data['Standing'] = test_data['Standing'].fillna(0)
    # test_data = test_data[test_data['Season'] == 2018]
    # with open('season_avg.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(test_data_cols)
    #     for index, row in test_data.iterrows():
    #         writer.writerow(row)
    #data['Standing'] = data['Standing'].mask(data['Standing'] > 0, 1)
    logistic_pca(data, 9, 'Season', 'Standing')

#v3_data()
# v4_data('total')
v4_data('avg')

