import pandas as pd
import numpy as np
import statsmodels as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

# transform data using PCA and use transformed data for logistic regression
def logistic_pca(data, target, n_components):
    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
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

    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
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

game_data_columns = ['game_id', 'won', 'goals', 'shots', 'pim', 'powerPlayOpportunities', 'injured_player_count']
game_data = pd.read_csv("cleaned_data/game_stats.csv", usecols=game_data_columns)
# Change 'won' column from boolean to int type
game_data['won'] = game_data['won'].astype(int)
# Change NaN in 'injured_player_count' column to 0
game_data['injured_player_count'] = game_data['injured_player_count'].fillna(0)

X = game_data.loc[:, game_data.columns != 'won'] # Data
y = game_data.loc[:, game_data.columns == 'won'] # Data's labels

logistic_pca(X, y, 5)




