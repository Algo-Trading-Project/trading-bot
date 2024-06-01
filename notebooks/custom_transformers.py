from sklearn.base import BaseEstimator, TransformerMixin
    
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

class RollingMinMaxScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_cols = []
        for col in X.columns:
            if '_rz_' in col or col in ('symbol_id'):
                continue

            for window_size in self.window_sizes:
                col_name = col + '_rmm_' + str(window_size)
                rolling_min = X[col].rolling(window = window_size).min()
                rolling_max = X[col].rolling(window = window_size).max()
                new_col = pd.Series((X[col] - rolling_min) / (rolling_max - rolling_min), name = col_name)

                new_cols.append(new_col)
        
        X = pd.concat([X] + new_cols, axis = 1)
        return X

class RollingZScoreScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_cols = []

        for col in X.columns:
            if '_rmm_' in col or col in ('symbol_id'):
                continue

            for window_size in self.window_sizes:
                col_name = col + '_rz_' + str(window_size)
                rolling_mean = X[col].rolling(window = window_size).mean()
                rolling_std = X[col].rolling(window = window_size).std()
                new_col = pd.Series((X[col] - rolling_mean) / rolling_std, name = col_name)

                new_cols.append(new_col)   

        X = pd.concat([X] + new_cols, axis = 1)
        
        return X

class LagFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, lags):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = []
        for col in X.columns:
            if col in ('symbol_id'):
                continue

            for lag in self.lags:
                col_name = col + '_lag_' + str(lag)
                cols.append(pd.Series(X[col].shift(lag), name = col_name))

        X = pd.concat([X] + cols, axis = 1)

        return X
               
class FillNaN(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fill nan or infinity values for each column with the rolling mean of that column
        for col in X.columns:
            if col in ('symbol_id'):
                continue
            
            try:
                # Replace inf values with nan
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                X[col] = X[col].fillna(X[col].rolling(window = 2 * 24).mean().fillna(0))
            except:
                continue

        return X
    
class ReturnsFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for window_size in self.window_sizes:
            X[f'returns_{window_size}'] = X['price_close'].pct_change(window_size)

        return X
