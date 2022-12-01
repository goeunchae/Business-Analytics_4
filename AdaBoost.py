import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Decision stump used as weak classifier
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X.iloc[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions
    

class Adaboost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")

            # greedy search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = X.iloc[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


from sklearn.model_selection import train_test_split
import datetime
import math

# AdaBoost -1 
def run_adaboost_1():

    names = ['data_1','data_2','data_3','data_4']
    weaks = [i for i in range(5,40,5)]
    
    name_lst = []
    weak_lst = []
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []
    time_lst = []
    
    for name in names:
        for weak in weaks:
            data = pd.read_csv(f'./data/{name}.csv'.format(name))
            
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            y[y == 0] = -1

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
            import time
            start = time.time()

            # Adaboost classification with 5 weak classifiers
            clf = Adaboost(n_clf=weak)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            acc = accuracy_score(y_pred, y_test)
            p = precision_score(y_pred, y_test)
            r = recall_score(y_pred, y_test)
            f = f1_score(y_pred, y_test)
            
            sec = (time.time()- start)
            
            time = str(datetime.timedelta(seconds=sec)).split(".")[0]

            acc_lst.append(acc)
            p_lst.append(p)
            r_lst.append(r)
            f_lst.append(f)
            name_lst.append(name)
            weak_lst.append(weak)
            time_lst.append(time)
            
    result = pd.DataFrame({'data':name_lst, '# of weak models':weak_lst, 'accuracy': acc_lst,
                                'precision_score': p_lst, 'recall_score': r_lst, 'f1_score': f_lst, 'time': time_lst})
    result.to_csv('./results/adaboost_1.csv',index=False)
    
#run_adaboost_1()

# AdaBoost -2 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
def run_adaboost_2():
    names = ['data_1','data_2','data_3','data_4']
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []
    time_lst = []
    
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
            
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        model = AdaBoostClassifier()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        param_grid = {'n_estimators': [i for i in range(5,40,5)], 
                    'learning_rate': [0.1, 0.3, 0.5, 0.7, 1]}


        grid_search = GridSearchCV(model,param_grid,scoring="accuracy", refit=True,cv=10, return_train_score=True)

        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"Best params: {best_params}")
        
        import time 
        start = time.time()
        math.factorial(1234567)
        if_clf = AdaBoostClassifier(**best_params)
        if_clf.fit(X_train, y_train)

        y_pred = if_clf.predict(X_test)

        acc = accuracy_score(y_pred, y_test)
        p = precision_score(y_pred, y_test)
        r = recall_score(y_pred, y_test)
        f = f1_score(y_pred, y_test)
        
        end = time.time()
        sec = (end - start)
        time = str(datetime.timedelta(seconds=sec)).split(".")[0]
        
        acc_lst.append(acc)
        p_lst.append(p)
        r_lst.append(r)
        f_lst.append(f)
        time_lst.append(time)
            
    result = pd.DataFrame({'data':names, 'accuracy': acc_lst,
                                'precision_score': p_lst, 'recall_score': r_lst, 'f1_score': f_lst, 'time':time_lst})
    result.to_csv('./results/adaboost_2.csv',index=False)
    
#run_adaboost_2()
    