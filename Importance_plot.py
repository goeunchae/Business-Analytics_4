from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import graphviz
from sklearn.tree import export_graphviz

# Decision Tree 
def dt_importance():
    names = ['data_1','data_2','data_3','data_4']
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
            
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        if name == 'data_1':
            params = {'max_depth':1, 'min_samples_split':2}
            
        if name == 'data_2':
            params = {'max_depth':4, 'min_samples_split':2}
            
        if name == 'data_3':
            params = {'max_depth':16, 'min_samples_split':8}
            
        else:
            params = {'max_depth':19, 'min_samples_split':2}
            
        tree = DecisionTreeClassifier(**params)
        tree.fit(X_train, y_train)
        
        export_graphviz(tree,feature_names=X.columns,impurity=False, filled=True) 

        with open('tree.dot') as file_reader:

            dot_graph = file_reader.read()
        
        dot = graphviz.Source(dot_graph) # dot_graph의 source 저장
        dot.render(filename=f'./pics/tree_{name}.png'.format(name)) # png로 저장
        
        feature_imp = tree.feature_importances_
        n_feature = X.shape[1]
        idx = np.arange(n_feature)
        plt.figure(figsize=(15,15))
        plt.barh(idx, feature_imp, align='center')
        plt.yticks(idx, X.columns)
        plt.xlabel('feature importance', size=15)
        plt.ylabel('feature', size=15)
        plt.savefig(f'./pics/tree_feature_{name}.png'.format(name))
        
#dt_importance()


# AdaBoost
def ada_importance():
    names = ['data_1','data_2','data_3','data_4']
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
            
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        if name == 'data_1':
            params = {'learning_rate':0.1, 'n_estimators':30}
            
        if name == 'data_2':
            params = {'learning_rate':1, 'n_estimators':10}
            
        if name == 'data_3':
            params = {'learning_rate':0.7, 'n_estimators':35}
            
        else:
            params = {'learning_rate':0.7, 'n_estimators':20}
            
        tree = AdaBoostClassifier(**params)
        tree.fit(X_train, y_train)

        feature_imp = tree.feature_importances_
        n_feature = X.shape[1]
        idx = np.arange(n_feature)
        plt.figure(figsize=(15,15))
        plt.barh(idx, feature_imp, align='center', color='lightpink')
        plt.yticks(idx, X.columns)
        plt.xlabel('feature importance', size=15)
        plt.ylabel('feature', size=15)
        plt.savefig(f'./pics/adaboost_feature_{name}.png'.format(name))
        
#ada_importance()

# GBM
def gbm_importance():
    names = ['data_1','data_2','data_3','data_4']
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
            
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        if name == 'data_1':
            params = {'learning_rate':0.7, 'loss':'log_loss', 'max_depth':18, 'min_samples_split':6, 'n_estimators':250}
            
        if name == 'data_2':
            params = {'learning_rate':0.1, 'loss':'exponential', 'max_depth':3, 'min_samples_split':6, 'n_estimators':100}
            
        if name == 'data_3':
            params = {'learning_rate':0.3, 'loss':'exponential', 'max_depth':8, 'min_samples_split':6, 'n_estimators':100}
            
        else:
            params = {'learning_rate':0.5, 'loss':'exponential', 'max_depth':3, 'min_samples_split':2, 'n_estimators':150}
            
        tree = GradientBoostingClassifier(**params)
        tree.fit(X_train, y_train)

        feature_imp = tree.feature_importances_
        n_feature = X.shape[1]
        idx = np.arange(n_feature)
        plt.figure(figsize=(15,15))
        plt.barh(idx, feature_imp, align='center', color='darkseagreen')
        plt.yticks(idx, X.columns)
        plt.xlabel('feature importance', size=15)
        plt.ylabel('feature', size=15)
        plt.savefig(f'./pics/gbm_feature_{name}.png'.format(name))
        
#gbm_importance()

# XGBoost
def xgb_importance():
    names = ['data_1','data_2','data_3','data_4']
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
            
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        tree = XGBClassifier(n_estimators=5, max_depth=5, learning_rate=1, objective='binary:logistic')
        tree.fit(X_train, y_train)

        feature_imp = tree.feature_importances_
        n_feature = X.shape[1]
        idx = np.arange(n_feature)
        plt.figure(figsize=(15,15))
        plt.barh(idx, feature_imp, align='center', color='lightblue')
        plt.yticks(idx, X.columns)
        plt.xlabel('feature importance', size=15)
        plt.ylabel('feature', size=15)
        plt.savefig(f'./pics/xgboost_feature_{name}.png'.format(name))
        
#xgb_importance()

# LightGBM
def ligbm_importance():
    names = ['data_1','data_2','data_3','data_4']
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
            
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
                
        if name == 'data_1':
            params = {'learning_rate':0.1, 'max_depth':18, 'n_estimators':100, 'num_leaves':23}
        
        if name == 'data_2':
            params = {'learning_rate':0.1, 'max_depth':3, 'n_estimators':100, 'num_leaves':20}
        
        if name == 'data_3':
            params = {'learning_rate':0.1, 'max_depth':13, 'n_estimators':150, 'num_leaves':38}
        
        if name == 'data_4':
            params = {'learning_rate':0.3, 'max_depth':13, 'n_estimators':150, 'num_leaves':20}
        
        tree = LGBMClassifier(**params)
        tree.fit(X_train, y_train)

        feature_imp = tree.feature_importances_
        n_feature = X.shape[1]
        idx = np.arange(n_feature)
        plt.figure(figsize=(15,15))
        plt.barh(idx, feature_imp, align='center', color='lightsalmon')
        plt.yticks(idx, X.columns)
        plt.xlabel('feature importance', size=15)
        plt.ylabel('feature', size=15)
        plt.savefig(f'./pics/lightgbm_feature_{name}.png'.format(name))
        
#ligbm_importance()