import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from sklearn.model_selection import train_test_split
import datetime
import math
import warnings
warnings.filterwarnings('ignore')

class TreeBooster():
 
    def __init__(self, X, g, h, params, max_depth, idxs=None):
        self.params = params
        self.max_depth = max_depth
        assert self.max_depth >= 0, 'max_depth must be nonnegative'
        self.min_child_weight = params['min_child_weight'] \
            if params['min_child_weight'] else 1.0
        self.reg_lambda = params['reg_lambda'] if params['reg_lambda'] else 1.0
        self.gamma = params['gamma'] if params['gamma'] else 0.0
        self.colsample_bynode = params['colsample_bynode'] \
            if params['colsample_bynode'] else 1.0
        if isinstance(g, pd.Series): g = g.values
        if isinstance(h, pd.Series): h = h.values
        if idxs is None: idxs = np.arange(len(g))
        self.X, self.g, self.h, self.idxs = X, g, h, idxs
        self.n, self.c = len(idxs), X.shape[1]
        self.value = -g[idxs].sum() / (h[idxs].sum() + self.reg_lambda) # Eq (5)
        self.best_score_so_far = 0.
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()

    def _maybe_insert_child_nodes(self):
        for i in range(self.c): self._find_better_split(i)
        if self.is_leaf: return
        x = self.X.values[self.idxs,self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]
        self.left = TreeBooster(self.X, self.g, self.h, self.params, 
                                self.max_depth - 1, self.idxs[left_idx])
        self.right = TreeBooster(self.X, self.g, self.h, self.params, 
                                 self.max_depth - 1, self.idxs[right_idx])

    @property
    def is_leaf(self): return self.best_score_so_far == 0.
    
    def _find_better_split(self, feature_idx):
        x = self.X.values[self.idxs, feature_idx]
        g, h = self.g[self.idxs], self.h[self.idxs]
        sort_idx = np.argsort(x)
        sort_g, sort_h, sort_x = g[sort_idx], h[sort_idx], x[sort_idx]
        sum_g, sum_h = g.sum(), h.sum()
        sum_g_right, sum_h_right = sum_g, sum_h
        sum_g_left, sum_h_left = 0., 0.

        for i in range(0, self.n - 1):
            g_i, h_i, x_i, x_i_next = sort_g[i], sort_h[i], sort_x[i], sort_x[i + 1]
            sum_g_left += g_i; sum_g_right -= g_i
            sum_h_left += h_i; sum_h_right -= h_i
            if sum_h_left < self.min_child_weight or x_i == x_i_next:continue
            if sum_h_right < self.min_child_weight: break

            gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.reg_lambda))
                            + (sum_g_right**2 / (sum_h_right + self.reg_lambda))
                            - (sum_g**2 / (sum_h + self.reg_lambda))
                            ) - self.gamma/2 # Eq(7) in the xgboost paper
            if gain > self.best_score_so_far: 
                self.split_feature_idx = feature_idx
                self.best_score_so_far = gain
                self.threshold = (x_i + x_i_next) / 2
                
    def predict(self, X):
        return np.array([self._predict_row(row) for i, row in X.iterrows()])

    def _predict_row(self, row):
        if self.is_leaf: 
            return self.value
        child = self.left if row[self.split_feature_idx] <= self.threshold \
            else self.right
        return child._predict_row(row)


class XGBoostModel():
    def __init__(self, params, random_seed=None):
        self.params = defaultdict(lambda: None, params)
        self.subsample = self.params['subsample'] \
            if self.params['subsample'] else 1.0
        self.learning_rate = self.params['learning_rate'] \
            if self.params['learning_rate'] else 0.3
        self.base_prediction = self.params['base_score'] \
            if self.params['base_score'] else 0.5
        self.max_depth = self.params['max_depth'] \
            if self.params['max_depth'] else 5
        self.rng = np.random.default_rng(seed=random_seed)
                
    def fit(self, X, y, objective, num_boost_round, verbose=False):
        current_predictions = self.base_prediction * np.ones(shape=y.shape)
        self.boosters = []
        for i in range(num_boost_round):
            gradients = objective.gradient(y, current_predictions)
            hessians = objective.hessian(y, current_predictions)
            sample_idxs = None if self.subsample == 1.0 \
                else self.rng.choice(len(y), 
                                     size=math.floor(self.subsample*len(y)), 
                                     replace=False)
            booster = TreeBooster(X, gradients, hessians, 
                                  self.params, self.max_depth, sample_idxs)
            current_predictions += self.learning_rate * booster.predict(X)
            self.boosters.append(booster)
            if verbose: 
                print(f'[{i}] train loss = {objective.loss(y, current_predictions)}')
            
    def predict(self, X):
        return (self.base_prediction + self.learning_rate 
                * np.sum([booster.predict(X) for booster in self.boosters], axis=0))
        
class SquaredErrorObjective():
    def loss(self, y, pred): return np.mean((y - pred)**2)
    def gradient(self, y, pred): return pred - y
    def hessian(self, y, pred): return np.ones(len(y))
    
# XGBoost -1 
def run_xgboost_1():

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

            # XGBoost classification
            params = {
            'learning_rate': 1,
            'max_depth': weak,
            'subsample': 0.8,
            'reg_lambda': 1.5,
            'gamma': 0.0,
            'min_child_weight': 25,
            'base_score': 0.0,
            'tree_method': 'exact',
        }
            
            num_boost_round = 50
            
            clf = XGBoostModel(params, random_seed=42)
            clf.fit(X_train, y_train, SquaredErrorObjective(), num_boost_round)
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
            
    result = pd.DataFrame({'data':name_lst, 'max_depth':weak_lst, 'accuracy': acc_lst,
                                'precision_score': p_lst, 'recall_score': r_lst, 'f1_score': f_lst, 'time': time_lst})
    result.to_csv('./results/xgboost_1.csv',index=False)
    
run_xgboost_1()        
        
# XGBoost -2
from xgboost import XGBClassifier
from xgboost import plot_importance

def run_xgboost_2():
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        import time 
        start = time.time()
        math.factorial(1234567)
        
        model_xgb = XGBClassifier(n_estimators=5, max_depth=5, learning_rate=1, objective='binary:logistic')
        
        model_xgb.fit(X_train, y_train)

        y_pred = model_xgb.predict(X_test)

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
    result.to_csv('./results/xgboost_2.csv',index=False)
    
#run_xgboost_2()    