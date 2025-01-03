from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm
from utils.db_utils import QUERY
from analysis.ml.labeling import calculate_triple_barrier_labels

import pandas as pd
import numpy as np
import warnings
import ta
import duckdb
import holidays

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class TAFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, windows):
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['log_open'] = np.log(X['open'])
        X['log_high'] = np.log(X['high'])
        X['log_low'] = np.log(X['low'])
        X['log_close'] = np.log(X['close'])
        X['log_volume'] = np.log(X['volume'])

        for window in self.windows:
            # EMA
            X[f'ema_{window}'] = ta.trend.ema_indicator(close = X['close'], window = window)
            X[f'close_above_ema_{window}'] = X['close'] > X[f'ema_{window}']
            X[f'close_below_ema_{window}'] = X['close'] < X[f'ema_{window}']

            # Log EMA
            X[f'log_ema_{window}'] = ta.trend.ema_indicator(close = X['log_close'], window = window)
            X[f'log_close_above_ema_{window}'] = X['log_close'] > X[f'log_ema_{window}']
            X[f'log_close_below_ema_{window}'] = X['log_close'] < X[f'log_ema_{window}']
            
            # Close cross over/under EMA
            X[f'close_cross_over_ema_{window}'] = (X['close'] > X[f'ema_{window}']) & (X['close'].shift(1) < X[f'ema_{window}'])
            X[f'close_cross_under_ema_{window}'] = (X['close'] < X[f'ema_{window}']) & (X['close'].shift(1) > X[f'ema_{window}'])

            # Log Close cross over/under EMA
            X[f'log_close_cross_over_ema_{window}'] = (X['log_close'] > X[f'log_ema_{window}']) & (X['log_close'].shift(1) < X[f'log_ema_{window}'])
            X[f'log_close_cross_under_ema_{window}'] = (X['log_close'] < X[f'log_ema_{window}']) & (X['log_close'].shift(1) > X[f'log_ema_{window}'])

            # Bollinger Bands
            X[f'bb_hband_{window}'], X[f'bb_lband_{window}'] = ta.volatility.bollinger_hband(close = X['close'], window = window), ta.volatility.bollinger_lband(close = X['close'], window = window)

            # Log Bollinger Bands
            X[f'log_bb_hband_{window}'], X[f'log_bb_lband_{window}'] = ta.volatility.bollinger_hband(close = X['log_close'], window = window), ta.volatility.bollinger_lband(close = X['log_close'], window = window)
            
            # Close above/below Bollinger Bands
            X[f'close_above_bb_hband_{window}'] = (X['close'] > X[f'bb_hband_{window}']).astype(int)
            X[f'close_below_bb_lband_{window}'] = (X['close'] < X[f'bb_lband_{window}']).astype(int)

            # Log Close above/below Bollinger Bands
            X[f'log_close_above_bb_hband_{window}'] = (X['log_close'] > X[f'log_bb_hband_{window}']).astype(int)
            X[f'log_close_below_bb_lband_{window}'] = (X['log_close'] < X[f'log_bb_lband_{window}']).astype(int)

            # Close cross over/under Bollinger Bands
            X[f'close_cross_over_bb_hband_{window}'] = (X['close'] > X[f'bb_hband_{window}']) & (X['close'].shift(1) < X[f'bb_hband_{window}'])
            X[f'close_cross_under_bb_lband_{window}'] = (X['close'] < X[f'bb_lband_{window}']) & (X['close'].shift(1) > X[f'bb_lband_{window}'])

            # Log Close cross over/under Bollinger Bands
            X[f'log_close_cross_over_bb_hband_{window}'] = (X['log_close'] > X[f'log_bb_hband_{window}']) & (X['log_close'].shift(1) < X[f'log_bb_hband_{window}'])
            X[f'log_close_cross_under_bb_lband_{window}'] = (X['log_close'] < X[f'log_bb_lband_{window}']) & (X['log_close'].shift(1) > X[f'log_bb_lband_{window}'])
            
            # RSI
            X[f'rsi_{window}'] = ta.momentum.rsi(close = X['close'], window = window)

            # Log RSI
            X[f'log_rsi_{window}'] = ta.momentum.rsi(close = X['log_close'], window = window)

        return X
        
class TimeFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        X['day_of_week'] = X['time_period_end'].dt.dayofweek
        X['day_of_month'] = X['time_period_end'].dt.day
        X['month'] = X['time_period_end'].dt.month
        X['year'] = X['time_period_end'].dt.year

        # Holidays
        us_holidays = holidays.US()
        X['is_holiday'] = X['time_period_end'].dt.date.astype(str).map(lambda x: x in us_holidays).astype(int)

        return X
    
class RollingZScoreScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes
        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_cols = []

        triple_barrier_label_cols = [col for col in X if 'triple_barrier_label_h' in col]
        trade_returns_cols = [col for col in X if 'trade_returns_h' in col]
        cols_1d = [col for col in X if '_1d' in col]

        for col in X.columns:
            if '_rmm_' in col \
            or 'num_consecutive' in col \
            or '_ind' in col \
            or col in ('symbol_id', 'asset_id_base', 'asset_id_quote', 'exchange_id', 'time_period_end') \
            or col in triple_barrier_label_cols \
            or col in ('hour', 'day_of_week', 'day_of_month', 'month', 'year', 'is_holiday') \
            or col in cols_1d \
            or col in trade_returns_cols:
                continue

            for window_size in self.window_sizes:
                col_name = col + '_rz_' + str(window_size)
                rolling_mean = X[col].rolling(window = window_size, min_periods = 1).mean()
                rolling_std = X[col].rolling(window = window_size, min_periods = 1).std()
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
            if col in ('symbol_id', 'asset_id_base', 'asset_id_quote', 'exchange_id', 'time_period_end') or \
            col in ('hour', 'day_of_week', 'day_of_month', 'month') or \
            'triple_barrier_label' in col or \
            'trade_returns' in col:
                continue

            for lag in self.lags:
                col_name = col + '_lag_' + str(lag)
                cols.append(pd.Series(X[col].shift(lag), name = col_name))

        X = pd.concat([X] + cols, axis = 1)

        return X
               
class ReturnsFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes, lookback_windows):
        self.window_sizes = window_sizes
        self.lookback_windows = lookback_windows

        self.ml_dataset = QUERY(
            """
            SELECT *
            FROM market_data.ml_dataset
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        self.ml_dataset['symbol_id'] = self.ml_dataset['asset_id_base'] + '_' + self.ml_dataset['asset_id_quote'] + '_' + self.ml_dataset['exchange_id']
        tokens = self.ml_dataset['symbol_id'].unique().tolist()

        for window in self.window_sizes:
            self.ml_dataset[f'returns_{window}'] = 0
            self.ml_dataset[f'log_returns_{window}'] = 0

        for token in tokens:
            filter = self.ml_dataset['symbol_id'] == token
            for window in self.window_sizes:
                self.ml_dataset.loc[filter, f'returns_{window}'] = self.ml_dataset.loc[filter, 'close'].pct_change(window).fillna(0)
                self.ml_dataset.loc[filter, f'log_returns_{window}'] = np.log(self.ml_dataset.loc[filter, 'close'] / self.ml_dataset.loc[filter, 'close'].shift(window)).fillna(0)

        for window in self.window_sizes:
            # Cross-sectional returns for each symbol for each time period
            pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'returns_{window}')
            pivot = pivot.fillna(0)
            pivot.columns = [f'{col}_returns_{window}' for col in pivot.columns]

            # Cross-sectional log returns for each symbol for each time period
            pivot_log = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'log_returns_{window}')
            pivot_log = pivot_log.fillna(0)
            pivot_log.columns = [f'{col}_log_returns_{window}' for col in pivot_log.columns]

            # Cross-sectional dollar volume for each symbol for each time period
            self.ml_dataset['dollar_volume'] = self.ml_dataset['close'] * self.ml_dataset['volume']
            pivot_volume = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = 'dollar_volume')
            pivot_volume = pivot_volume.fillna(0)
            pivot_volume.columns = [f'{col}_volume' for col in pivot_volume.columns]

            # Cross-sectional number of trades for each symbol for each time period
            pivot_num_trades = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = 'trades')
            pivot_num_trades = pivot_num_trades.fillna(0)
            pivot_num_trades.columns = [f'{col}_num_trades' for col in pivot_num_trades.columns]

            # Cross-sectional returns rank for each symbol for each time period
            pivot_rank = pivot.rank(axis = 1, method = 'first', ascending = False)
            pivot_rank = pivot_rank.fillna(0)
            pivot_rank.columns = [f'{col}_returns_{window}_rank' for col in pivot_rank.columns]

            # Average cross-sectional returns for each time period
            pivot_rank[f'avg_cross_sectional_returns_{window}'] = pivot.mean(axis = 1)

            # Standard deviation of cross-sectional returns for each time period
            pivot_rank[f'std_cross_sectional_returns_{window}'] = pivot.std(axis = 1)

            # Skewness of cross-sectional returns for each time period
            pivot_rank[f'skewness_cross_sectional_returns_{window}'] = pivot.skew(axis = 1)

            # Kurtosis of cross-sectional returns for each time period
            pivot_rank[f'kurtosis_cross_sectional_returns_{window}'] = pivot.kurt(axis = 1)

            # Median of cross-sectional returns for each time period
            pivot_rank[f'median_cross_sectional_returns_{window}'] = pivot.median(axis = 1)

            # 10th percentile of cross-sectional returns for each time period
            pivot_rank[f'10th_percentile_cross_sectional_returns_{window}'] = pivot.quantile(0.1, axis = 1)

            # 90th percentile of cross-sectional returns for each time period
            pivot_rank[f'90th_percentile_cross_sectional_returns_{window}'] = pivot.quantile(0.9, axis = 1)

            # Cross-sectional log returns rank for each symbol for each time period
            pivot_log_rank = pivot_log.rank(axis = 1, method = 'first', ascending = False)
            pivot_log_rank = pivot_log_rank.fillna(0)
            pivot_log_rank.columns = [f'{col}_log_returns_{window}_rank' for col in pivot_log_rank.columns]

            # Average cross-sectional log returns for each time period
            pivot_log_rank[f'avg_cross_sectional_log_returns_{window}'] = pivot_log.mean(axis = 1)

            # Standard deviation of cross-sectional log returns for each time period
            pivot_log_rank[f'std_cross_sectional_log_returns_{window}'] = pivot_log.std(axis = 1)

            # Skewness of cross-sectional log returns for each time period
            pivot_log_rank[f'skewness_cross_sectional_log_returns_{window}'] = pivot_log.skew(axis = 1)

            # Kurtosis of cross-sectional log returns for each time period
            pivot_log_rank[f'kurtosis_cross_sectional_log_returns_{window}'] = pivot_log.kurt(axis = 1)

            # Median of cross-sectional log returns for each time period
            pivot_log_rank[f'median_cross_sectional_log_returns_{window}'] = pivot_log.median(axis = 1)

            # 10th percentile of cross-sectional log returns for each time period
            pivot_log_rank[f'10th_percentile_cross_sectional_log_returns_{window}'] = pivot_log.quantile(0.1, axis = 1)

            # 90th percentile of cross-sectional log returns for each time period
            pivot_log_rank[f'90th_percentile_cross_sectional_log_returns_{window}'] = pivot_log.quantile(0.9, axis = 1)

            # Cross-sectional volume rank for each symbol for each time period
            pivot_volume_rank = pivot_volume.rank(axis = 1, method = 'first', ascending = False)
            pivot_volume_rank = pivot_volume_rank.fillna(0)
            pivot_volume_rank.columns = [f'{col}_volume_rank' for col in pivot_volume_rank.columns]

            # Average cross-sectional volume for each time period
            pivot_volume_rank[f'avg_cross_sectional_volume_{window}'] = pivot_volume.mean(axis = 1)

            # Standard deviation of cross-sectional volume for each time period
            pivot_volume_rank[f'std_cross_sectional_volume_{window}'] = pivot_volume.std(axis = 1)

            # Skewness of cross-sectional volume for each time period
            pivot_volume_rank[f'skewness_cross_sectional_volume_{window}'] = pivot_volume.skew(axis = 1)

            # Kurtosis of cross-sectional volume for each time period
            pivot_volume_rank[f'kurtosis_cross_sectional_volume_{window}'] = pivot_volume.kurt(axis = 1)

            # Median of cross-sectional volume for each time period
            pivot_volume_rank[f'median_cross_sectional_volume_{window}'] = pivot_volume.median(axis = 1)

            # 10th percentile of cross-sectional volume for each time period
            pivot_volume_rank[f'10th_percentile_cross_sectional_volume_{window}'] = pivot_volume.quantile(0.1, axis = 1)

            # 90th percentile of cross-sectional volume for each time period
            pivot_volume_rank[f'90th_percentile_cross_sectional_volume_{window}'] = pivot_volume.quantile(0.9, axis = 1)

            # Cross-sectional number of trades rank for each symbol for each time period
            pivot_num_trades_rank = pivot_num_trades.rank(axis = 1, method = 'first', ascending = False)
            pivot_num_trades_rank = pivot_num_trades_rank.fillna(0)
            pivot_num_trades_rank.columns = [f'{col}_num_trades_rank' for col in pivot_num_trades_rank.columns]

            # Average cross-sectional number of trades for each time period
            pivot_num_trades_rank[f'avg_cross_sectional_num_trades_{window}'] = pivot_num_trades.mean(axis = 1)

            # Standard deviation of cross-sectional number of trades for each time period
            pivot_num_trades_rank[f'std_cross_sectional_num_trades_{window}'] = pivot_num_trades.std(axis = 1)

            # Skewness of cross-sectional number of trades for each time period
            pivot_num_trades_rank[f'skewness_cross_sectional_num_trades_{window}'] = pivot_num_trades.skew(axis = 1)

            # Kurtosis of cross-sectional number of trades for each time period
            pivot_num_trades_rank[f'kurtosis_cross_sectional_num_trades_{window}'] = pivot_num_trades.kurt(axis = 1)

            # Median of cross-sectional number of trades for each time period
            pivot_num_trades_rank[f'median_cross_sectional_num_trades_{window}'] = pivot_num_trades.median(axis = 1)

            # 10th percentile of cross-sectional number of trades for each time period
            pivot_num_trades_rank[f'10th_percentile_cross_sectional_num_trades_{window}'] = pivot_num_trades.quantile(0.1, axis = 1)

            # 90th percentile of cross-sectional number of trades for each time period
            pivot_num_trades_rank[f'90th_percentile_cross_sectional_num_trades_{window}'] = pivot_num_trades.quantile(0.9, axis = 1)
            
            # Merge the cross-sectional features back into the original data
            self.ml_dataset = pd.merge(self.ml_dataset, pivot_rank, left_index = True, right_index = True, how = 'left', suffixes = ('', '__remove'))
            self.ml_dataset = pd.merge(self.ml_dataset, pivot_log_rank, left_index = True, right_index = True, how = 'left', suffixes = ('', '__remove'))
            self.ml_dataset = pd.merge(self.ml_dataset, pivot_volume_rank, left_index = True, right_index = True, how = 'left', suffixes = ('', '__remove'))
            self.ml_dataset = pd.merge(self.ml_dataset, pivot_num_trades_rank, left_index = True, right_index = True, how = 'left', suffixes = ('', '__remove'))

            # Drop the columns with '__remove' suffix
            self.ml_dataset = self.ml_dataset.drop(columns = [col for col in self.ml_dataset.columns if '__remove' in col], axis = 1)

        for window in self.window_sizes:
            self.ml_dataset = self.ml_dataset.drop(columns = [f'returns_{window}', f'log_returns_{window}'], axis = 1)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def pos_volatility(returns):
            pos = returns[returns > 0]
            return pos.mean() / pos.std()

        def neg_volatility(returns):
            neg = returns[returns < 0]
            return neg.mean() / neg.std()

        for window_size in self.window_sizes:
            for lookback_window in self.lookback_windows:
                # Calculate returns
                X[f'returns_{window_size}'] = X['close'].pct_change(window_size).fillna(0)

                # Calculate rolling mean of returns
                X[f'avg_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).mean()

                # Calculate 10th percentile of returns
                X[f'10th_percentile_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).quantile(0.1)

                # Calculate rolling median of returns
                X[f'median_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).median()

                # Calculate 90th percentile of returns
                X[f'90th_percentile_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).quantile(0.9)

                # Calculate rolling min of returns
                X[f'min_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).min()

                # Calculate rolling max of returns
                X[f'max_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).max()
                
                # Calculate rolling standard deviation of returns
                X[f'volatility_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).std()

                # Calculate rolling skewness of returns
                X[f'skewness_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).skew()

                # Calculate rolling kurtosis of returns
                X[f'kurtosis_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).kurt()
                
                # Calculate log returns
                X[f'log_returns_{window_size}'] = np.log(X['close'] / X['close'].shift(window_size))

                # Calculate rolling mean of log returns
                X[f'avg_log_returns_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).mean()

                # Calculate 10th percentile of log returns
                X[f'10th_percentile_log_returns_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).quantile(0.1)

                # Calculate rolling median of log returns
                X[f'median_log_returns_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).median()

                # Calculate 90th percentile of log returns
                X[f'90th_percentile_log_returns_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).quantile(0.9)

                # Calculate rolling min of log returns
                X[f'min_log_returns_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).min()

                # Calculate rolling max of log returns
                X[f'max_log_returns_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).max()

                # Calculate rolling standard deviation of log returns
                X[f'volatility_log_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).std()

                # Calculate rolling skewness of log returns
                X[f'skewness_log_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).skew()

                # Calculate rolling kurtosis of log returns
                X[f'kurtosis_log_{window_size}_{lookback_window}'] = X[f'log_returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).kurt()

                # Positive / Negative Volatility ratio (returns)
                X[f'pos_volatility_ratio_{lookback_window}'] = X[f'returns_1'].rolling(window = lookback_window, min_periods = 1).apply(pos_volatility, raw = True)
                X[f'neg_volatility_ratio_{lookback_window}'] = X[f'returns_1'].rolling(window = lookback_window, min_periods = 1).apply(neg_volatility, raw = True)
                X[f'pos_neg_volatility_ratio_{lookback_window}'] = (X[f'returns_1'].rolling(window = lookback_window, min_periods = 1).apply(pos_volatility, raw = True) / X[f'returns_1'].rolling(window = lookback_window, min_periods = 1).apply(neg_volatility, raw = True)).replace([np.inf, -np.inf], np.nan)

                # Positive / Negative Volatility ratio (log returns)
                X[f'pos_volatility_ratio_log_{lookback_window}'] = X[f'log_returns_1'].rolling(window = lookback_window, min_periods = 1).apply(pos_volatility, raw = True)
                X[f'neg_volatility_ratio_log_{lookback_window}'] = X[f'log_returns_1'].rolling(window = lookback_window, min_periods = 1).apply(neg_volatility, raw = True)
                X[f'pos_neg_volatility_ratio_log_{lookback_window}'] = (X[f'log_returns_1'].rolling(window = lookback_window, min_periods = 1).apply(pos_volatility, raw = True) / X[f'log_returns_1'].rolling(window = lookback_window, min_periods = 1).apply(neg_volatility, raw = True)).replace([np.inf, -np.inf], np.nan)

        # Cross-sectional rank features for current token and time period
        curr_token = X['symbol_id'].iloc[0]
        # Columns specific to the current token and columns for cross-sectional metrics
        cols = [col for col in self.ml_dataset.columns if (any([x in col for x in ('avg', 'std', 'skewness', 'kurtosis', 'median', '10th_percentile', '90th_percentile')])) or (col[:len(curr_token)] == curr_token and 'rank' in col)]

        X = pd.merge(X, self.ml_dataset[cols + ['time_period_end', 'symbol_id']], on = ['time_period_end', 'symbol_id'], how = 'left', suffixes = ('', '__remove'))
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)

        # Rename cross-sectional features to a standard format
        X = X.rename(columns = {col: col.replace(curr_token + '_', '') for col in X.columns if curr_token in col})

        return X
    
class TripleBarrierLabelFeatures(BaseEstimator, TransformerMixin):
        
        def __init__(self, max_holding_times, std_lookback_windows):
            self.max_holding_times = max_holding_times
            self.std_lookback_windows = std_lookback_windows
        
        def fit(self, X, y=None):
            return self
    
        def transform(self, X): 
            X = X.copy()
            for max_holding_time in self.max_holding_times:
                for std_lookback_window in self.std_lookback_windows:
                    labels, trade_returns, start_date_indices, end_date_indices = calculate_triple_barrier_labels(X, max_holding_time = max_holding_time, std_lookback_window = std_lookback_window)
                    
                    X[f'triple_barrier_label_h{max_holding_time}'] = labels
                    X[f'trade_returns_h{max_holding_time}'] = trade_returns
                    X[f'start_date_triple_barrier_label_h{max_holding_time}'] = pd.NaT
                    X[f'end_date_triple_barrier_label_h{max_holding_time}'] = pd.NaT

                    for i in range(len(start_date_indices)):
                        start_date_index = start_date_indices[i]
                        end_date_index = end_date_indices[i]         

                        if pd.isna(start_date_index):
                            X.loc[i, f'start_date_triple_barrier_label_h{max_holding_time}'] = pd.NaT
                        else:               
                            X.loc[i, f'start_date_triple_barrier_label_h{max_holding_time}'] = X.loc[start_date_index, 'time_period_end']

                        if pd.isna(end_date_index):
                            X.loc[i, f'end_date_triple_barrier_label_h{max_holding_time}'] = pd.NaT
                        else:
                            X.loc[i, f'end_date_triple_barrier_label_h{max_holding_time}'] = X.loc[end_date_index, 'time_period_end']
                
            return X

class CorrelationFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes, lookback_windows, period = None):
        self.window_sizes = window_sizes
        self.lookback_windows = lookback_windows
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        eth_usd_coinbase = QUERY(
            """
            SELECT
                time_period_end, 
                asset_id_base,
                asset_id_quote,
                exchange_id,
                open,
                high,
                low,
                close,
                volume
            FROM market_data.ml_dataset
            WHERE
                asset_id_base = 'ETH' AND
                asset_id_quote = 'USDT' AND
                exchange_id = 'BINANCE'
            ORDER BY time_period_end
            """
        )
        eth_usd_coinbase['time_period_end'] = pd.to_datetime(eth_usd_coinbase['time_period_end'])

        btc_usd_coinbase = QUERY(
            """
            SELECT
                time_period_end,
                asset_id_base,
                asset_id_quote,
                exchange_id,
                open,
                high,
                low,
                close,
                volume
            FROM market_data.ml_dataset
            WHERE
                asset_id_base = 'BTC' AND
                asset_id_quote = 'USDT' AND
                exchange_id = 'BINANCE'
            ORDER BY time_period_end
            """
        )
        btc_usd_coinbase['time_period_end'] = pd.to_datetime(btc_usd_coinbase['time_period_end'])

        for window_size in self.window_sizes:
            eth_usd_coinbase[f'returns_{window_size}'] = eth_usd_coinbase['close'].pct_change(window_size).fillna(0)
            btc_usd_coinbase[f'returns_{window_size}'] = btc_usd_coinbase['close'].pct_change(window_size).fillna(0)

        X.time_period_end = pd.to_datetime(X.time_period_end)

        merged = pd.merge(X, eth_usd_coinbase, on = 'time_period_end', suffixes = ('', '_ETH'), how = 'left',)
        merged = pd.merge(merged, btc_usd_coinbase, on = 'time_period_end', suffixes = ('', '_BTC'), how = 'left')

        for window_size in self.window_sizes:
            for lookback_window in self.lookback_windows:
                # Calculate autocorrelation of returns
                X[f'autocorrelation_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1).corr(X[f'returns_{window_size}'])

                merged_rolling = merged[f'returns_{window_size}'].rolling(window = lookback_window, min_periods = 1)

                # Calculate cross-correlation of returns with ETH
                X[f'cross_correlation_returns_{window_size}_{lookback_window}_ETH'] = merged_rolling.corr(merged[f'returns_{window_size}_ETH'])

                # Calculate cross-correlation of returns with BTC
                X[f'cross_correlation_returns_{window_size}_{lookback_window}_BTC'] = merged_rolling.corr(merged[f'returns_{window_size}_BTC'])

        return X
        
class OrderBookFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        # Set up the connection to the DuckDB database
        self.conn = duckdb.connect(
            database = '/Users/louisspencer/Desktop/Trading-Bot-Data-Pipelines/data/database.db',
            read_only = False
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Join the order book data with the features
        asset_id_base = X['asset_id_base'].iloc[0]
        asset_id_quote = X['asset_id_quote'].iloc[0]
        exchange_id = X['exchange_id'].iloc[0]

        query = f"""
            SELECT *
            FROM market_data.order_book_snapshot_1m
            WHERE
                asset_id_base = '{asset_id_base}' AND
                asset_id_quote = '{asset_id_quote}' AND
                exchange_id = '{exchange_id}'
            ORDER BY received_time
        """
        ob = QUERY(query)
        
        # Remove microseconds from the timestamp and change name of the column to match the original data
        ob['received_time'] = pd.to_datetime(ob['received_time']).dt.floor('s')
        ob = ob.rename(columns = {'received_time': 'time_period_end'})
        # Remove duplicates
        ob = ob.drop_duplicates(subset = ['time_period_end'], keep = 'first')
        # Set the timestamp as the index
        ob = ob.set_index('time_period_end')
        # Fill missing values
        ob = ob.asfreq('1min').ffill()
        ob = ob.drop(columns = ['sequence_number'], axis = 1)

        # Downsample the order book data to 1d
        bid_cols = [f'bid_{i}_price' for i in range(20)]
        ask_cols = [f'ask_{i}_price' for i in range(20)]
        bid_size_cols = [f'bid_{i}_size' for i in range(20)]
        ask_size_cols = [f'ask_{i}_size' for i in range(20)]
        combined_cols = bid_cols + ask_cols + bid_size_cols + ask_size_cols

        agg_dict = {col: 'last' for col in combined_cols}
        ob_1d = ob.resample('1d', label='right', closed='left').agg({
            'asset_id_base': 'last',
            'asset_id_quote': 'last',
            'exchange_id': 'last',
            **agg_dict
        }).reset_index()

        # Calculate total volumes and imbalances
        total_bid_dollar_volume = pd.Series(index = ob_1d.index, data = 0)
        total_bid_volume = pd.Series(index = ob_1d.index, data = 0)

        total_ask_dollar_volume = pd.Series(index = ob_1d.index, data = 0)
        total_ask_volume = pd.Series(index = ob_1d.index, data = 0)

        for i in range(20):
            total_bid_dollar_volume += ob_1d[bid_cols[i]] * ob_1d[bid_size_cols[i]]
            total_ask_dollar_volume += ob_1d[ask_cols[i]] * ob_1d[ask_size_cols[i]]

            total_bid_volume += ob_1d[bid_size_cols[i]]
            total_ask_volume += ob_1d[ask_size_cols[i]]
            
            ob_1d[f'bid_ask_spread_{i}'] = ob_1d[ask_cols[i]] - ob_1d[bid_cols[i]]
            ob_1d[f'imbalance_{i}'] = (ob_1d[bid_size_cols[i]] - ob_1d[ask_size_cols[i]]) / (ob_1d[ask_size_cols[i]] + ob_1d[bid_size_cols[i]])

            ob_1d[f'bid_dollar_volume_{i}'] = ob_1d[bid_cols[i]] * ob_1d[bid_size_cols[i]]
            ob_1d[f'ask_dollar_volume_{i}'] = ob_1d[ask_cols[i]] * ob_1d[ask_size_cols[i]]

        total_dollar_imbalance = (total_bid_dollar_volume - total_ask_dollar_volume) / (total_bid_dollar_volume + total_ask_dollar_volume)
        total_volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        
        ob_1d['total_bid_dollar_volume'] = total_bid_dollar_volume
        ob_1d['total_ask_dollar_volume'] = total_ask_dollar_volume
        ob_1d['total_ob_dollar_volume'] = total_bid_dollar_volume + total_ask_dollar_volume

        ob_1d['total_bid_volume'] = total_bid_volume
        ob_1d['total_ask_volume'] = total_ask_volume
        ob_1d['total_ob_volume'] = total_bid_volume + total_ask_volume

        ob_1d['total_dollar_imbalance'] = total_dollar_imbalance
        ob_1d['total_volume_imbalance'] = total_volume_imbalance

        # Drop unnecessary columns
        ob_1d = ob_1d.drop(columns = bid_cols + ask_cols + bid_size_cols + ask_size_cols, axis = 1)

        # Merge the order book features with the original data
        X = pd.merge(X, ob_1d, on = 'time_period_end', suffixes = ('', '__remove'), how = 'left')
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)
        
        return X

class MicroStructureFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, windows):
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for window in self.windows:
            # Roll measure
            diff_close = X['close'].diff()
            diff_close_lag_1 = diff_close.shift(1)
            X[f'roll_measure_{window}'] = 2 * np.sqrt(diff_close.rolling(window = window, min_periods = 1).cov(diff_close_lag_1).abs())

            # Roll impact measure (Roll measure divided by dollar volume)
            X[f'roll_impact_measure_{window}'] = X[f'roll_measure_{window}'] / (X['volume'] * X['close'])

            # Amihud measure
            X[f'amihud_measure_{window}'] = (X['close'].pct_change().abs() / (X['volume'] * X['close'])).rolling(window = window, min_periods = 1).mean()

            # Kyle lambda
            X[f'kyle_lambda_{window}'] = (X['close'] - X['close'].shift(window)) / (np.sign(diff_close) * X['volume']).rolling(window = window, min_periods = 1).sum()

            # VPIN
            diff_close_std = diff_close.rolling(window = window, min_periods = 1).std()
            estimated_buy_volume = X['volume'] * norm.cdf(diff_close / diff_close_std)
            estimated_sell_volume = X['volume'] - estimated_buy_volume
            vpin_component = (estimated_buy_volume - estimated_sell_volume).abs() / X['volume']
            X[f'vpin_{window}'] = vpin_component.rolling(window = window, min_periods = 1).mean()

        return X
            
class TokenCategoryFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass

class MacroeconomicFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass

class CrossTokenFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, period, window_sizes):
        self.period = period
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        eth_usd_coinbase = QUERY(
            """
            SELECT
                time_period_end, 
                asset_id_base,
                asset_id_quote,
                exchange_id,
                open,
                high,
                low,
                close,
                volume
            FROM market_data.ml_dataset
            WHERE
                asset_id_base = 'ETH' AND
                asset_id_quote = 'USDT' AND
                exchange_id = 'BINANCE'
            ORDER BY time_period_end
            """
        )
        eth_usd_coinbase['time_period_end'] = pd.to_datetime(eth_usd_coinbase['time_period_end'])

        btc_usd_coinbase = QUERY(
            """
            SELECT
                time_period_end,
                asset_id_base,
                asset_id_quote,
                exchange_id,
                open,
                high,
                low,
                close,
                volume
            FROM market_data.ml_dataset
            WHERE
                asset_id_base = 'BTC' AND
                asset_id_quote = 'USDT' AND
                exchange_id = 'BINANCE'
            ORDER BY time_period_end
            """
        )
        btc_usd_coinbase['time_period_end'] = pd.to_datetime(btc_usd_coinbase['time_period_end'])

        if self.period == '1d':

            eth_usd_coinbase = eth_usd_coinbase.set_index('time_period_end').sort_index().resample('1D', label = 'right', closed = 'left').agg({
                'asset_id_base': 'first',
                'asset_id_quote': 'first',
                'exchange_id': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

            btc_usd_coinbase = btc_usd_coinbase.set_index('time_period_end').sort_index().resample('1D', label = 'right', closed = 'left').agg({
                'asset_id_base': 'first',
                'asset_id_quote': 'first',
                'exchange_id': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

        elif self.period == '15min':
                
            eth_usd_coinbase = eth_usd_coinbase.set_index('time_period_end').sort_index().resample('15min', label = 'right', closed = 'left').agg({
                'asset_id_base': 'first',
                'asset_id_quote': 'first',
                'exchange_id': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

            btc_usd_coinbase = btc_usd_coinbase.set_index('time_period_end').sort_index().resample('15min', label = 'right', closed = 'left').agg({
                'asset_id_base': 'first',
                'asset_id_quote': 'first',
                'exchange_id': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

        elif self.period == '1h':
                    
            eth_usd_coinbase = eth_usd_coinbase.set_index('time_period_end').sort_index().resample('1H', label = 'right', closed = 'left').agg({
                'asset_id_base': 'first',
                'asset_id_quote': 'first',
                'exchange_id': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

            btc_usd_coinbase = btc_usd_coinbase.set_index('time_period_end').sort_index().resample('1H', label = 'right', closed = 'left').agg({
                'asset_id_base': 'first',
                'asset_id_quote': 'first',
                'exchange_id': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
        # Specify window sizes for TA features
        ta_features = TAFeatures(self.window_sizes)
        
        eth_usd_coinbase = ta_features.fit_transform(eth_usd_coinbase)
        btc_usd_coinbase = ta_features.fit_transform(btc_usd_coinbase)

        eth_usd_coinbase.drop(columns = ['asset_id_base', 'asset_id_quote', 'exchange_id'], axis = 1, inplace = True)
        btc_usd_coinbase.drop(columns = ['asset_id_base', 'asset_id_quote', 'exchange_id'], axis = 1, inplace = True)

        merged = pd.merge(X, eth_usd_coinbase, on = 'time_period_end', suffixes = ('', '_ETH'), how = 'left',)
        merged = pd.merge(merged, btc_usd_coinbase, on = 'time_period_end', suffixes = ('', '_BTC'), how = 'left')

        return merged

class FillNaTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col == 'time_period_end':
                continue
            elif X[col].dtype in ('O', 'object'):
                if X[col].isnull().sum() > 0:
                    mode = X[col].mode().loc[0]
                    X[col] = X[col].fillna(mode)
            else:
                if X[col].isnull().sum() > 0:
                    # Fill missing values with the rolling mean
                    X[col] = X[col].fillna(X[col].rolling(window = 3, min_periods = 1).mean()).bfill().ffill()

        return X
