from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import warnings
import ta
import ta.trend

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from numba import jit
try:
    from .helper import *
except:
    from helper import *

class RollingMinMaxScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_cols = []
        for col in X.columns:

            if '_rz_' in col \
            or 'num_consecutive' in col \
            or '_ma_1' in col \
            or col in ('symbol_id', 'time_period_end') \
            or 'triple_barrier_label_w' in col:
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

        windows_triple_barrier_label = [2 * 12, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]
        max_holding_times_triple_barrier_label = [2 * 12, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]

        triple_barrier_label_cols = [
            f'triple_barrier_label_w{w}_h{h}' 
            for w in windows_triple_barrier_label 
            for h in max_holding_times_triple_barrier_label
        ]

        for col in X.columns:
            if '_rmm_' in col \
            or 'num_consecutive' in col \
            or col in ('symbol_id', 'time_period_end') \
            or col in triple_barrier_label_cols:
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
            if col in ('symbol_id', 'time_period_end') or \
               'triple_barrier_label_w' in col:
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
            if col in ('symbol_id', 'time_period_end'):
                continue
            
            try:
                # Replace inf values with nan
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                # Replace nan values with the rolling mean for that column
                X[col] = X[col].fillna(X[col].rolling(window = 2 * 24 * 7).mean().fillna(0))
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

            # Rolling mean of returns
            X[f'returns_{window_size}_ma_1d'] = X[f'returns_{window_size}'].rolling(window = 2 * 24).mean()
            X[f'returns_{window_size}_ma_1w'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 7).mean()
            X[f'returns_{window_size}_ma_1m'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 30).mean()

            # Rolling variance of returns
            X[f'returns_{window_size}_var_1d'] = X[f'returns_{window_size}'].rolling(window = 2 * 24).var()
            X[f'returns_{window_size}_var_1w'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 7).var()
            X[f'returns_{window_size}_var_1m'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 30).var()

            # Rolling skewness of returns
            X[f'returns_{window_size}_skew_1d'] = X[f'returns_{window_size}'].rolling(window = 2 * 24).skew()
            X[f'returns_{window_size}_skew_1w'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 7).skew()
            X[f'returns_{window_size}_skew_1m'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 30).skew()

            # Rolling kurtosis of returns
            X[f'returns_{window_size}_kurt_1d'] = X[f'returns_{window_size}'].rolling(window = 2 * 24).kurt()
            X[f'returns_{window_size}_kurt_1w'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 7).kurt()
            X[f'returns_{window_size}_kurt_1m'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 30).kurt()

            # # Rolling count of consecutive positive returns up to the current time, reset after a negative return
            # num_consecutive_pos_returns = 0
            # num_consecutive_pos_returns_list = []
            # for i in range(len(X)):
            #     if X[f'returns_{window_size}'].iloc[i] > 0:
            #         num_consecutive_pos_returns += 1
            #     else:
            #         num_consecutive_pos_returns = 0
            #     num_consecutive_pos_returns_list.append(num_consecutive_pos_returns)

            # X[f'num_consecutive_pos_returns_{window_size}'] = num_consecutive_pos_returns_list

            # # Rolling count of consecutive negative returns up to the current time, reset after a positive return
            # num_consecutive_neg_returns = 0
            # num_consecutive_neg_returns_list = []
            # for i in range(len(X)):
            #     if X[f'returns_{window_size}'].iloc[i] < 0:
            #         num_consecutive_neg_returns += 1
            #     else:
            #         num_consecutive_neg_returns = 0
            #     num_consecutive_neg_returns_list.append(num_consecutive_neg_returns)

            # X[f'num_consecutive_neg_returns_{window_size}'] = num_consecutive_neg_returns_list

        return X

@jit(nopython=True)
def compute_t_value_for_window(window_data):
    y = window_data
    X = np.arange(len(y)).reshape(-1, 1).astype(np.float64)
    ones = np.ones_like(X)
    X = np.hstack((ones, X))  # Add intercept term

    # Perform linear regression manually
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    beta = np.linalg.solve(XTX, XTy)
    coef = beta[1]

    # Calculate standard error of the coefficient
    y_pred = np.dot(X, beta)
    residuals = y - y_pred
    residual_sum_of_squares = np.sum(residuals ** 2)
    denominator = np.sqrt(np.sum((X[:, 1] - np.mean(X[:, 1])) ** 2))
    if denominator == 0:
        return np.nan  # Return NaN if denominator is zero to avoid division by zero
    se_coef = np.sqrt(residual_sum_of_squares / (len(y) - 2)) / denominator

    # Calculate t-value
    if se_coef == 0:
        return np.nan  # Return NaN if standard error is zero to avoid division by zero
    t_value = coef / se_coef
    return t_value

@jit(nopython=True)
def calculate_t_value(price_close, window_size):
    n = len(price_close)
    t_values = np.full(n, np.nan)
    for start in range(n - window_size + 1):
        end = start + window_size
        window_data = price_close[start:end].astype(np.float64)
        t_values[end - 1] = compute_t_value_for_window(window_data)
    return t_values

class RegressionFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for window_size in self.window_sizes:
            t_values = calculate_t_value(X['price_close'].values.astype(np.float64), window_size)
            X[f't_value_{window_size}'] = t_values
        return X

class PriceFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # RSI
        X['rsi_1d'] = ta.momentum.RSIIndicator(close = X['price_close'], window = 2 * 24).rsi()
        X['rsi_1w'] = ta.momentum.RSIIndicator(close = X['price_close'], window = 2 * 24 * 7).rsi()
        X['rsi_1m'] = ta.momentum.RSIIndicator(close = X['price_close'], window = 2 * 24 * 30).rsi()

        # MFI
        X['mfi_1d'] = ta.volume.MFIIndicator(high = X['price_high'], low = X['price_low'], close = X['price_close'], volume = X['volume_traded'], window = 2 * 24).money_flow_index()
        X['mfi_1w'] = ta.volume.MFIIndicator(high = X['price_high'], low = X['price_low'], close = X['price_close'], volume = X['volume_traded'], window = 2 * 24 * 7).money_flow_index()
        X['mfi_1m'] = ta.volume.MFIIndicator(high = X['price_high'], low = X['price_low'], close = X['price_close'], volume = X['volume_traded'], window = 2 * 24 * 30).money_flow_index()

        # PPO
        X['ppo_1d'] = ta.momentum.PercentagePriceOscillator(close = X['price_close'], window_slow = 2 * 24, window_fast = 2 * 24 // 2, window_sign = 2 * 24 // 3).ppo()
        X['ppo_1w'] = ta.momentum.PercentagePriceOscillator(close = X['price_close'], window_slow = 2 * 24 * 7, window_fast = 2 * 24 * 7 // 2, window_sign = 2 * 24 * 7 // 3).ppo()
        X['ppo_1m'] = ta.momentum.PercentagePriceOscillator(close = X['price_close'], window_slow = 2 * 24 * 30, window_fast = 2 * 24 * 30 // 2, window_sign = 2 * 24 * 30 // 3).ppo()

        # Aroon
        X['aroon_up_1d'], X['aroon_down_1d'] = ta.trend.AroonIndicator(high = X['price_high'], low = X['price_low'], window = 2 * 24).aroon_up(), ta.trend.AroonIndicator(high = X['price_high'], low = X['price_low'], window = 2 * 24).aroon_down()
        X['aroon_up_1w'], X['aroon_down_1w'] = ta.trend.AroonIndicator(high = X['price_high'], low = X['price_low'], window = 2 * 24 * 7).aroon_up(), ta.trend.AroonIndicator(high = X['price_high'], low = X['price_low'], window = 2 * 24 * 7).aroon_down()
        X['aroon_up_1m'], X['aroon_down_1m'] = ta.trend.AroonIndicator(high = X['price_high'], low = X['price_low'], window = 2 * 24 * 30).aroon_up(), ta.trend.AroonIndicator(high = X['price_high'], low = X['price_low'], window = 2 * 24 * 30).aroon_down()

        # CCI
        X['cci_1d'] = ta.trend.CCIIndicator(high = X['price_high'], low = X['price_low'], close = X['price_close'], window = 2 * 24).cci()
        X['cci_1w'] = ta.trend.CCIIndicator(high = X['price_high'], low = X['price_low'], close = X['price_close'], window = 2 * 24 * 7).cci()
        X['cci_1m'] = ta.trend.CCIIndicator(high = X['price_high'], low = X['price_low'], close = X['price_close'], window = 2 * 24 * 30).cci()

        return X
    
class TripleBarrierLabelFeatures(BaseEstimator, TransformerMixin):
        
        def __init__(self, windows, max_holding_times):
            self.windows = windows
            self.max_holding_times = max_holding_times
        
        def fit(self, X, y=None):
            return self
    
        def transform(self, X):
            X = X.copy()
            for window in self.windows:
                for max_holding_time in self.max_holding_times:
                    labels, trade_returns = calculate_triple_barrier_labels(X, window = window, max_holding_time = max_holding_time)
                    X[f'triple_barrier_label_w{window}_h{max_holding_time}'] = labels
                    X[f'trade_returns_w{window}_h{max_holding_time}'] = trade_returns

                    # Lag the labels by the max_holding_time to avoid lookahead bias
                    X[f'triple_barrier_label_w{window}_h{max_holding_time}_lag{max_holding_time}'] = X[f'triple_barrier_label_w{window}_h{max_holding_time}'].shift(max_holding_time)

                    # Lag the trade returns by the max_holding_time to avoid lookahead bias
                    X[f'trade_returns_w{window}_h{max_holding_time}_lag{max_holding_time}'] = X[f'trade_returns_w{window}_h{max_holding_time}'].shift(max_holding_time)

                    # Moving average of the lagged labels
                    X[f'triple_barrier_label_w{window}_h{max_holding_time}_lag{max_holding_time}_ma_1d'] = X[f'triple_barrier_label_w{window}_h{max_holding_time}_lag{max_holding_time}'].rolling(window = 2 * 24).mean()
                    X[f'triple_barrier_label_w{window}_h{max_holding_time}_lag{max_holding_time}_ma_1w'] = X[f'triple_barrier_label_w{window}_h{max_holding_time}_lag{max_holding_time}'].rolling(window = 2 * 24 * 7).mean()
                    X[f'triple_barrier_label_w{window}_h{max_holding_time}_lag{max_holding_time}_ma_1m'] = X[f'triple_barrier_label_w{window}_h{max_holding_time}_lag{max_holding_time}'].rolling(window = 2 * 24 * 30).mean()

                    # Moving average of the lagged trade returns
                    X[f'trade_returns_w{window}_h{max_holding_time}_lag{max_holding_time}_ma_1d'] = X[f'trade_returns_w{window}_h{max_holding_time}_lag{max_holding_time}'].rolling(window = 2 * 24).mean()
                    X[f'trade_returns_w{window}_h{max_holding_time}_lag{max_holding_time}_ma_1w'] = X[f'trade_returns_w{window}_h{max_holding_time}_lag{max_holding_time}'].rolling(window = 2 * 24 * 7).mean()
                    X[f'trade_returns_w{window}_h{max_holding_time}_lag{max_holding_time}_ma_1m'] = X[f'trade_returns_w{window}_h{max_holding_time}_lag{max_holding_time}'].rolling(window = 2 * 24 * 30).mean()

                    # Drop the original trade returns, lagged labels, and lagged trade returns
                    X.drop(columns = [
                        f'trade_returns_w{window}_h{max_holding_time}', 
                        f'triple_barrier_label_w{window}_h{max_holding_time}_lag{max_holding_time}', 
                        f'trade_returns_w{window}_h{max_holding_time}_lag{max_holding_time}'
                    ], inplace = True)

            return X
        
class CorrelationFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # eth_usd_coinbase = QUERY(
        # """
        # SELECT
        #     time_period_end, 
        #     price_close
        # FROM market_data.ml_dataset
        # WHERE
        #     symbol_id = 'ETH_USD_COINBASE'
        # ORDER BY time_period_end
        # """
        # )
        # eth_usd_coinbase['time_period_end'] = pd.to_datetime(eth_usd_coinbase['time_period_end'], utc=True)

        # btc_usd_coinbase = QUERY(
        # """
        # SELECT
        #     time_period_end, 
        #     price_close
        # FROM market_data.ml_dataset
        # WHERE
        #     symbol_id = 'BTC_USD_COINBASE'
        # ORDER BY time_period_end
        # """
        # )
        # btc_usd_coinbase['time_period_end'] = pd.to_datetime(btc_usd_coinbase['time_period_end'], utc=True)
        
        # for window_size in self.window_sizes:
        #     eth_usd_coinbase[f'returns_{window_size}'] = eth_usd_coinbase['price_close'].pct_change(window_size)
        #     btc_usd_coinbase[f'returns_{window_size}'] = btc_usd_coinbase['price_close'].pct_change(window_size)

        X.time_period_end = pd.to_datetime(X.time_period_end, utc = True)

        # merged = pd.merge(X, eth_usd_coinbase, on = 'time_period_end', suffixes = ('', '_ETH'), how = 'left',)
        # merged = pd.merge(merged, btc_usd_coinbase, on = 'time_period_end', suffixes = ('', '_BTC'), how = 'left')

        X = X.set_index('time_period_end')

        for window_size in self.window_sizes:
            # Calculate autocorrelation of returns
            X[f'autocorrelation_returns_{window_size}_1d'] = X[f'returns_{window_size}'].rolling(window = 2 * 24).apply(lambda x: pd.Series(x).autocorr(), raw = True)
            X[f'autocorrelation_returns_{window_size}_1w'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 7).apply(lambda x: pd.Series(x).autocorr(), raw = True)
            X[f'autocorrelation_returns_{window_size}_1m'] = X[f'returns_{window_size}'].rolling(window = 2 * 24 * 30).apply(lambda x: pd.Series(x).autocorr(), raw = True)

            # # Calculate cross-correlation of returns with ETH
            # X[f'cross_correlation_returns_{window_size}_1d'] = merged[[f'returns_{window_size}', f'returns_{window_size}_ETH']].rolling(window = 2 * 24).corr(pairwise=True).unstack().iloc[:, 1]
            # X[f'cross_correlation_returns_{window_size}_1w'] = merged[[f'returns_{window_size}', f'returns_{window_size}_ETH']].rolling(window = 2 * 24 * 7).corr(pairwise=True).unstack().iloc[:, 1]
            # X[f'cross_correlation_returns_{window_size}_1m'] = merged[[f'returns_{window_size}', f'returns_{window_size}_ETH']].rolling(window = 2 * 24 * 30).corr(pairwise=True).unstack().iloc[:, 1]

            # # Calculate cross-correlation of returns with BTC
            # X[f'cross_correlation_returns_{window_size}_1d_BTC'] = merged[[f'returns_{window_size}', f'returns_{window_size}_BTC']].rolling(window = 2 * 24).corr(pairwise=True).unstack().iloc[:, 1]
            # X[f'cross_correlation_returns_{window_size}_1w_BTC'] = merged[[f'returns_{window_size}', f'returns_{window_size}_BTC']].rolling(window = 2 * 24 * 7).corr(pairwise=True).unstack().iloc[:, 1]
            # X[f'cross_correlation_returns_{window_size}_1m_BTC'] = merged[[f'returns_{window_size}', f'returns_{window_size}_BTC']].rolling(window = 2 * 24 * 30).corr(pairwise=True).unstack().iloc[:, 1]

            # # Calculate difference of returns to ETH
            # X[f'returns_diff_{window_size}'] = X[f'returns_{window_size}'] - merged[f'returns_{window_size}_ETH']

            # # Calculate difference of returns to BTC
            # X[f'returns_diff_{window_size}_BTC'] = X[f'returns_{window_size}'] - merged[f'returns_{window_size}_BTC']

            # # ETH returns
            # X[f'returns_{window_size}_ETH'] = merged[f'returns_{window_size}_ETH']

            # # BTC returns
            # X[f'returns_{window_size}_BTC'] = merged[f'returns_{window_size}_BTC']

        return X