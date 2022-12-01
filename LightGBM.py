import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


import datetime
import math

from sklearn.model_selection import train_test_split
import lightgbm as lgb 
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# lightGBM -2 
def run_lightgbm():
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
        
        model = LGBMClassifier()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        param_grid = {'n_estimators': [i for i in range(100,300,50)], 
                    'learning_rate': [0.1, 0.3, 0.5, 0.7, 1],
                    'max_depth': [i for i in range(3,30,5)],
                    'num_leaves': [i for i in range(20,50,3)]}


        grid_search = GridSearchCV(model,param_grid,scoring="accuracy", refit=True,cv=10, return_train_score=True)

        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"Best params: {best_params}")
        
        import time 
        start = time.time()
        math.factorial(1234567)
        if_clf = LGBMClassifier(**best_params)
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
    result.to_csv('./results/lightgbm_2.csv',index=False)
    
run_lightgbm()