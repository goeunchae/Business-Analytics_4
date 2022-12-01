import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


class GradientBoostingFromScratch():
    
    def __init__(self, n_trees, learning_rate, max_depth=1):
        self.n_trees=n_trees; self.learning_rate=learning_rate; self.max_depth=max_depth;
        
    def fit(self, x, y):
        self.trees = []
        self.F0 = y.mean()
        Fm = self.F0 
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(x, y)
            Fm += self.learning_rate * tree.predict(x)
            self.trees.append(tree)
            
    def predict(self, x):
        return self.F0 + self.learning_rate * np.sum([tree.predict(x) for tree in self.trees], axis=0)
    
from sklearn.model_selection import train_test_split
import datetime
import math

# GBM -1 
def run_gbm_1():

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
            clf = GradientBoostingFromScratch(n_trees=weak, learning_rate=0.3, max_depth=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            y_pred[y_pred < 0] = -1
            y_pred[y_pred >=0] = 1
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
    result.to_csv('./results/gbm_1.csv',index=False)   
    
#run_gbm_1() 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# GBM -2 
def run_gbm_2():
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
        
        model = GradientBoostingClassifier()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        param_grid = {'n_estimators': [i for i in range(100,300,50)], 
                    'learning_rate': [0.1, 0.3, 0.5, 0.7, 1],
                    'min_samples_split': [i for i in range(2,10,2)],
                    'max_depth': [i for i in range(3,30,5)],
                    'loss': ['log_loss','deviance','exponential']}


        grid_search = GridSearchCV(model,param_grid,scoring="accuracy", refit=True,cv=10, return_train_score=True)

        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"Best params: {best_params}")
        
        import time 
        start = time.time()
        math.factorial(1234567)
        if_clf = GradientBoostingClassifier(**best_params)
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
    result.to_csv('./results/gbm_2.csv',index=False)
    
run_gbm_2()