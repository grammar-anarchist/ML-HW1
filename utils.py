import numpy as np
import pandas as pd
import re
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.base import BaseEstimator, TransformerMixin

def print_metrics(y_train, y_train_pred, y_test, y_test_pred):
    print('R^2 train: ', r2_score(y_train, y_train_pred))
    print('MSE train: ', MSE(y_train, y_train_pred))
    print('R^2 test: ', r2_score(y_test, y_test_pred))
    print('MSE test: ', MSE(y_test, y_test_pred))

class CustomImputer(TransformerMixin):
    def __init__(self, numeric_imputer, categorical_imputer, 
                 numeric_columns, categorical_columns):
        self.numeric_imputer = numeric_imputer
        self.categorical_imputer = categorical_imputer
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.numeric_columns] = self.numeric_imputer.transform(X[self.numeric_columns])
        X[self.categorical_columns] = self.categorical_imputer.transform(X[self.categorical_columns])
        return X
    
def torque_split(x):
    'tr'
    if pd.isna(x):
        return [np.nan, np.nan]

    x = x.replace(',', '')
    nums = re.compile(r'[\d\.]+').findall(x)
    nm = re.compile(r'[Nn]?[Mm]?').match(x)
    kgm = re.compile(r'kgm').match(x)

    if 0 == len(nums) or len(nums) > 3:
        return [np.nan, np.nan]
    if not nm and not kgm:
        return [np.nan, np.nan]
        
    if len(nums) == 3:
        torque, max_rpm = float(nums[0]), float(nums[2])
    elif len(nums) == 2:
        torque, max_rpm = float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        if nm or kgm:
            torque, max_rpm = float(nums[0]), np.nan
        else:
            torque, max_rpm = np.nan, float(nums[0])
    if kgm:
        torque *= 9.8
    
    return [torque, max_rpm]
    
class StringToNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()   

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['engine'] = X['engine'].str.split().str[0].astype('float')
        X['max_power'] = X['max_power'].str.split().apply(lambda x: x[0] if isinstance(x, list) and len(x) == 2 else np.nan).astype('float')
        X['mileage'] = X['mileage'].str.split().str[0].astype('float')
        X['torque'], X['max_torque_rpm'] = zip(*X['torque'].apply(torque_split))
        return X
    
class IntAssigner(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()   

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[['engine', 'seats']] = X[['engine', 'seats']].astype('int')
        return X

class NameExtractor(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['name'] = X['name'].str.split().str[0]
        return X
