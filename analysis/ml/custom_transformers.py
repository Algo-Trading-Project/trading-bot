from sklearn.base import BaseEstimator, TransformerMixin
from utils.db_utils import QUERY

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class TimeFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):       
        X['day_of_month'] = X['time_period_end'].dt.day.astype('category')
        X['day_of_week'] = X['time_period_end'].dt.dayofweek.astype('category')
        X['month'] = X['time_period_end'].dt.month.astype('category')
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
        z_cols = [col for col in X if 'zscore' in col]

        # Remove duplicate columns
        X = X.loc[:, ~X.columns.duplicated()]

        for col in X.columns:
            if col in ('symbol_id', 'asset_id_base', 'asset_id_quote', 'exchange_id', 'time_period_end') \
            or col in triple_barrier_label_cols \
            or col in z_cols \
            or col in ('hour', 'day_of_week', 'day_of_month', 'month', 'year', 'is_holiday') \
            or X[col].dtype in ['object', 'category'] \
            or col in trade_returns_cols:
                continue

            # If col has all NaNs, fill with NaN
            if X[col].isna().all():
                col_name = col + '_rz_'

            for window_size in self.window_sizes:
                if window_size == 'expanding':
                    col_name = col + '_rz_expanding'
                    # If col has all NaNs, fill with NaN
                    if X[col].isna().all():
                        new_col = pd.Series([np.nan] * len(X), name = col_name)
                    else:
                        rolling_mean = X[col].shift(1).expanding().mean()
                        rolling_std = X[col].shift(1).expanding().std()
                        new_col = pd.Series((X[col] - rolling_mean) / rolling_std, name = col_name).clip(-10, 10)
                else:
                    col_name = col + '_rz_' + str(window_size)
                    # If col has all NaNs, fill with NaN
                    if X[col].isna().all():
                        new_col = pd.Series([np.nan] * len(X), name = col_name)
                    else:
                        rolling_mean = X[col].shift(1).rolling(window = window_size, min_periods = 7).mean()
                        rolling_std = X[col].shift(1).rolling(window = window_size, min_periods = 7).std()
                        new_col = pd.Series((X[col] - rolling_mean) / rolling_std, name = col_name).clip(-10, 10)

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
            col in ('open', 'high', 'low', 'close', 'volume', 'trades') or \
            'triple_barrier_label' in col or \
            'trade_returns' in col:
                continue

            for lag in self.lags:
                col_name = str(col) + '_lag_' + str(lag)
                new_col = X[col].shift(lag)
                new_col.name = col_name
                cols.append(new_col)

        X = pd.concat([X] + cols, axis = 1)

        return X
               
class ReturnsFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes, lookback_windows):
        self.window_sizes = window_sizes
        self.lookback_windows = lookback_windows

        self.spot_data = QUERY(
            """
            SELECT *
            FROM market_data.ml_dataset_1d
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        self.spot_data['symbol_id'] = self.spot_data['asset_id_base'] + '_' + self.spot_data['asset_id_quote'] + '_' + self.spot_data['exchange_id']
        self.spot_data['time_period_end'] = pd.to_datetime(self.spot_data['time_period_end'])
        
        self.futures_data = QUERY(
            """
            SELECT *
            FROM market_data.ml_dataset_futures_1d
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        self.futures_data['symbol_id'] = self.futures_data['asset_id_base'] + '_' + self.futures_data['asset_id_quote'] + '_' + self.futures_data['exchange_id']
        self.futures_data['time_period_end'] = pd.to_datetime(self.futures_data['time_period_end'])
        
        # Left join becasue I only want rows w/ existing spot data, futures data are optional
        self.ml_dataset = pd.merge(
            self.spot_data, 
            self.futures_data, 
            on = ['time_period_end', 'symbol_id'], 
            how = 'outer', 
            suffixes = ('_spot', '_futures')
        )
        self.ml_dataset.rename({
            'time_period_end_spot': 'time_period_end',
            'asset_id_base_spot': 'asset_id_base',
            'asset_id_quote_spot': 'asset_id_quote',
            'exchange_id_spot': 'exchange_id',
            'symbol_id_spot': 'symbol_id'
        }, inplace = True, axis = 1)
        self.ml_dataset.drop(columns = ['asset_id_base_futures', 'asset_id_quote_futures', 'exchange_id_futures', 'symbol_id_futures', 'time_period_end_futures'], inplace = True, errors = 'ignore', axis = 1)
        tokens = sorted(self.ml_dataset['symbol_id'].unique().tolist())

        # Calculate cross-sectional 1d returns features
        final_features = []

        # N-day returns
        for window in self.window_sizes:
            self.ml_dataset[f'spot_returns_{window}'] = np.nan
            self.ml_dataset[f'futures_returns_{window}'] = np.nan

        for token in tokens:
            filter = self.ml_dataset['symbol_id'] == token
            for window in self.window_sizes:
                print(f'Calculating returns for {token} with window size {window}...')
                # Spot clipping of returns
                if window == 1:
                    clip_upper_bound = 0.57
                elif window == 7:
                    clip_upper_bound = 3.55
                elif window == 30:
                    clip_upper_bound = 9.44
                elif window == 180:
                    clip_upper_bound = 59
                self.ml_dataset.loc[filter, f'spot_returns_{window}'] = self.ml_dataset.loc[filter, 'close_spot'].pct_change(window)
                # Futures clipping of returns
                if window == 1:
                    clip_upper_bound = 0.47
                elif window == 7:
                    clip_upper_bound = 2.05
                elif window == 30:
                    clip_upper_bound = 7.09
                elif window == 180:
                    clip_upper_bound = 22.7
                self.ml_dataset.loc[filter, f'futures_returns_{window}'] = self.ml_dataset.loc[filter, 'close_futures'].pct_change(window)

                for lookback in self.lookback_windows:
                    # Spot metrics
                    # Avg returns
                    self.ml_dataset.loc[filter, f'avg_spot_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'spot_returns_{window}'].rolling(lookback, min_periods = 7).mean()
                    # Std returns
                    self.ml_dataset.loc[filter, f'std_spot_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'spot_returns_{window}'].rolling(lookback, min_periods = 7).std()
                    # Sharpe ratio
                    self.ml_dataset.loc[filter, f'spot_sharpe_ratio_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'avg_spot_returns_{window}_{lookback}'] / self.ml_dataset.loc[filter, f'std_spot_returns_{window}_{lookback}']
                    # Std negative returns
                    self.ml_dataset.loc[filter, f'std_negative_spot_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'spot_returns_{window}'].rolling(lookback, min_periods = 7).apply(lambda x: np.std(x[x < 0]), raw = False)
                    # Sortino ratio
                    self.ml_dataset.loc[filter, f'spot_sortino_ratio_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'avg_spot_returns_{window}_{lookback}'] / self.ml_dataset.loc[filter, f'std_negative_spot_returns_{window}_{lookback}']

                    # Futures metrics
                    # Avg returns
                    self.ml_dataset.loc[filter, f'avg_futures_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'futures_returns_{window}'].rolling(lookback, min_periods = 7).mean()
                    # Std returns
                    self.ml_dataset.loc[filter, f'std_futures_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'futures_returns_{window}'].rolling(lookback, min_periods = 7).std()
                    # Sharpe ratio
                    self.ml_dataset.loc[filter, f'futures_sharpe_ratio_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'avg_futures_returns_{window}_{lookback}'] / self.ml_dataset.loc[filter, f'std_futures_returns_{window}_{lookback}']
                    # Std negative returns
                    self.ml_dataset.loc[filter, f'std_negative_futures_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'futures_returns_{window}'].rolling(lookback, min_periods = 7).apply(lambda x: np.std(x[x < 0]), raw = False)
                    # Sortino ratio
                    self.ml_dataset.loc[filter, f'futures_sortino_ratio_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'avg_futures_returns_{window}_{lookback}'] / self.ml_dataset.loc[filter, f'std_negative_futures_returns_{window}_{lookback}']

        for window in self.window_sizes:
            # Perform cross-sectional rank features for each time period
            spot_returns_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'spot_returns_{window}', dropna = False)
            spot_returns_pivot_percentile = spot_returns_pivot.rank(axis = 1, pct = True)
            spot_returns_pivot_percentile.columns = [col + f'returns_percentile_{window}_css' for col in spot_returns_pivot.columns]

            # Cross-sectional z-scores for each symbol for each time period
            spot_returns_cs_mean = spot_returns_pivot.mean(axis = 1)
            spot_returns_cs_std = spot_returns_pivot.std(axis = 1)            
            spot_returns_pivot_zscore = (spot_returns_pivot.sub(spot_returns_cs_mean, axis = 0)).div(spot_returns_cs_std, axis = 0)
            spot_returns_pivot_zscore.columns = [col + f'returns_zscore_{window}_css' for col in spot_returns_pivot_zscore.columns]

            # 4 moments of cross-sectional returns for each symbol for each time period
            spot_returns_pivot[f'cs_spot_returns_mean_{window}'] = spot_returns_pivot.mean(axis = 1)
            spot_returns_pivot[f'cs_spot_returns_std_{window}'] = spot_returns_pivot.std(axis = 1)
            spot_returns_pivot[f'cs_spot_returns_skew_{window}'] = spot_returns_pivot.skew(axis = 1)
            spot_returns_pivot[f'cs_spot_returns_kurtosis_{window}'] = spot_returns_pivot.kurtosis(axis = 1)
            spot_returns_pivot[f'cs_spot_returns_median_{window}'] = spot_returns_pivot.median(axis = 1)
            
            futures_returns_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'futures_returns_{window}', dropna = False)
            futures_returns_pivot_percentile = futures_returns_pivot.rank(axis = 1, pct = True)
            futures_returns_pivot_percentile.columns = [col + f'returns_percentile_{window}_csf' for col in futures_returns_pivot.columns]

            futures_returns_cs_mean = futures_returns_pivot.mean(axis = 1)
            futures_returns_cs_std = futures_returns_pivot.std(axis = 1)
            futures_returns_pivot_zscore = (futures_returns_pivot.sub(futures_returns_cs_mean, axis = 0)).div(futures_returns_cs_std, axis = 0)
            futures_returns_pivot_zscore.columns = [col + f'returns_zscore_{window}_csf' for col in futures_returns_pivot_zscore.columns]

            # 4 moments of cross-sectional returns for each symbol for each time period
            futures_returns_pivot[f'cs_futures_returns_mean_{window}'] = futures_returns_pivot.mean(axis = 1)
            futures_returns_pivot[f'cs_futures_returns_std_{window}'] = futures_returns_pivot.std(axis = 1)
            futures_returns_pivot[f'cs_futures_returns_skew_{window}'] = futures_returns_pivot.skew(axis = 1)
            futures_returns_pivot[f'cs_futures_returns_kurtosis_{window}'] = futures_returns_pivot.kurtosis(axis = 1)
            futures_returns_pivot[f'cs_futures_returns_median_{window}'] = futures_returns_pivot.median(axis = 1)

            # Append the percentile and z-score features to the final features list
            final_features.append(spot_returns_pivot_percentile)
            final_features.append(futures_returns_pivot_percentile)
            final_features.append(spot_returns_pivot_zscore)
            final_features.append(futures_returns_pivot_zscore)

            # Append the 4 moments of cross-sectional returns to the final features list
            spot_cs_returns_moments_cols = [col for col in spot_returns_pivot.columns if any([x in col for x in ['mean', 'std', 'skew', 'kurtosis', 'median']])]
            futures_cs_returns_moments_cols = [col for col in futures_returns_pivot.columns if any([x in col for x in ['mean', 'std', 'skew', 'kurtosis', 'median']])]

            final_features.append(spot_returns_pivot[spot_cs_returns_moments_cols])
            final_features.append(futures_returns_pivot[futures_cs_returns_moments_cols])

            for lookback in self.lookback_windows:
                # Rolling ema of cross-sectional returns moments for regime detection
                spot_returns_pivot[f'rolling_ema_{lookback}_cs_spot_returns_mean_{window}'] = spot_returns_pivot[f'cs_spot_returns_mean_{window}'].ewm(span = lookback, min_periods = 7).mean()
                spot_returns_pivot[f'rolling_ema_{lookback}_cs_spot_returns_std_{window}'] = spot_returns_pivot[f'cs_spot_returns_std_{window}'].ewm(span = lookback, min_periods = 7).mean()
                spot_returns_pivot[f'rolling_ema_{lookback}_cs_spot_returns_skew_{window}'] = spot_returns_pivot[f'cs_spot_returns_skew_{window}'].ewm(span = lookback, min_periods = 7).mean()
                spot_returns_pivot[f'rolling_ema_{lookback}_cs_spot_returns_kurtosis_{window}'] = spot_returns_pivot[f'cs_spot_returns_kurtosis_{window}'].ewm(span = lookback, min_periods = 7).mean()
                spot_returns_pivot[f'rolling_ema_{lookback}_cs_spot_returns_median_{window}'] = spot_returns_pivot[f'cs_spot_returns_median_{window}'].ewm(span = lookback, min_periods = 7).mean()

                futures_returns_pivot[f'rolling_ema_{lookback}_cs_futures_returns_mean_{window}'] = futures_returns_pivot[f'cs_futures_returns_mean_{window}'].ewm(span = lookback, min_periods = 7).mean()
                futures_returns_pivot[f'rolling_ema_{lookback}_cs_futures_returns_std_{window}'] = futures_returns_pivot[f'cs_futures_returns_std_{window}'].ewm(span = lookback, min_periods = 7).mean()
                futures_returns_pivot[f'rolling_ema_{lookback}_cs_futures_returns_skew_{window}'] = futures_returns_pivot[f'cs_futures_returns_skew_{window}'].ewm(span = lookback, min_periods = 7).mean()
                futures_returns_pivot[f'rolling_ema_{lookback}_cs_futures_returns_kurtosis_{window}'] = futures_returns_pivot[f'cs_futures_returns_kurtosis_{window}'].ewm(span = lookback, min_periods = 7).mean()
                futures_returns_pivot[f'rolling_ema_{lookback}_cs_futures_returns_median_{window}'] = futures_returns_pivot[f'cs_futures_returns_median_{window}'].ewm(span = lookback, min_periods = 7).mean()
                
                # Append the rolling mean of cross-sectional features to the final features list
                spot_cs_returns_rolling_moments = spot_returns_pivot.filter(like = f'rolling_ema_{lookback}_cs')
                futures_cs_returns_rolling_moments = futures_returns_pivot.filter(like = f'rolling_ema_{lookback}_cs')

                final_features.append(spot_cs_returns_rolling_moments)
                final_features.append(futures_cs_returns_rolling_moments)

        for lookback in self.lookback_windows:
            # Spot
            # Cross-sectional sharpe ratios for each symbol for each time period
            pivot_spot_sharpe = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'spot_sharpe_ratio_1_{lookback}', dropna = False)
            pivot_spot_sharpe_decile = pivot_spot_sharpe.rank(axis = 1, pct = True)
            pivot_spot_sharpe_decile.columns = [col + f'sharpe_percentile_{lookback}_css' for col in pivot_spot_sharpe_decile.columns]

            # Cross-sectional sharpe z-scores
            pivot_spot_sharpe_cs_mean = pivot_spot_sharpe.mean(axis = 1)
            pivot_spot_sharpe_cs_std = pivot_spot_sharpe.std(axis = 1)
            pivot_spot_sharpe_zscore = pivot_spot_sharpe.sub(pivot_spot_sharpe_cs_mean, axis = 0).div(pivot_spot_sharpe_cs_std, axis = 0)
            pivot_spot_sharpe_zscore.columns = [col + f'sharpe_zscore_{lookback}_css' for col in pivot_spot_sharpe_zscore.columns]

            # 4 moments of cross-sectional sharpe ratios
            pivot_spot_sharpe[f'cs_spot_sharpe_mean_{lookback}'] = pivot_spot_sharpe.mean(axis = 1)
            pivot_spot_sharpe[f'cs_spot_sharpe_std_{lookback}'] = pivot_spot_sharpe.std(axis = 1)
            pivot_spot_sharpe[f'cs_spot_sharpe_skew_{lookback}'] = pivot_spot_sharpe.skew(axis = 1)
            pivot_spot_sharpe[f'cs_spot_sharpe_kurtosis_{lookback}'] = pivot_spot_sharpe.kurtosis(axis = 1)
            pivot_spot_sharpe[f'cs_spot_sharpe_median_{lookback}'] = pivot_spot_sharpe.median(axis = 1)

            # Append 4 moments of cross-sectional sharpe ratios to the final features list
            spot_cs_sharpe_moments_cols = [col for col in pivot_spot_sharpe.columns if any(x in col for x in ['mean', 'std', 'skew', 'kurtosis', 'median'])]
            
            final_features.append(pivot_spot_sharpe[spot_cs_sharpe_moments_cols])
            final_features.append(pivot_spot_sharpe_decile)
            final_features.append(pivot_spot_sharpe_zscore)

            # Cross-sectional sortino ratios for each symbol for each time period
            pivot_spot_sortino = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'spot_sortino_ratio_1_{lookback}', dropna = False)
            pivot_spot_sortino_decile = pivot_spot_sortino.rank(axis = 1, pct = True)
            pivot_spot_sortino_decile.columns = [col + f'sortino_percentile_{lookback}_css' for col in pivot_spot_sortino_decile.columns]

            # Cross-sectional sortino z-scores
            pivot_spot_sortino_cs_mean = pivot_spot_sortino.mean(axis = 1)
            pivot_spot_sortino_cs_std = pivot_spot_sortino.std(axis = 1)
            pivot_spot_sortino_zscore = pivot_spot_sortino.sub(pivot_spot_sortino_cs_mean, axis = 0).div(pivot_spot_sortino_cs_std, axis = 0)
            pivot_spot_sortino_zscore.columns = [col + f'sortino_zscore_{lookback}_css' for col in pivot_spot_sortino_zscore.columns]

            # 4 moments of cross-sectional sortino ratios
            pivot_spot_sortino[f'cs_spot_sortino_mean_{lookback}'] = pivot_spot_sortino.mean(axis = 1)
            pivot_spot_sortino[f'cs_spot_sortino_std_{lookback}'] = pivot_spot_sortino.std(axis = 1)
            pivot_spot_sortino[f'cs_spot_sortino_skew_{lookback}'] = pivot_spot_sortino.skew(axis = 1)
            pivot_spot_sortino[f'cs_spot_sortino_kurtosis_{lookback}'] = pivot_spot_sortino.kurtosis(axis = 1)
            pivot_spot_sortino[f'cs_spot_sortino_median_{lookback}'] = pivot_spot_sortino.median(axis = 1)

            # Append 4 moments of cross-sectional sortino ratios to the final features list
            spot_cs_sortino_moments_cols = [col for col in pivot_spot_sortino.columns if any(x in col for x in ['mean', 'std', 'skew', 'kurtosis', 'median'])]

            final_features.append(pivot_spot_sortino[spot_cs_sortino_moments_cols])
            final_features.append(pivot_spot_sortino_decile)
            final_features.append(pivot_spot_sortino_zscore)

            # Futures
            # Cross-sectional sharpe ratios for each symbol for each time period
            pivot_futures_sharpe = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'futures_sharpe_ratio_1_{lookback}', dropna = False)
            pivot_futures_sharpe_decile = pivot_futures_sharpe.rank(axis = 1, pct = True)
            pivot_futures_sharpe_decile.columns = [col + f'sharpe_percentile_{lookback}_csf' for col in pivot_futures_sharpe_decile.columns]

            # Cross-sectional sharpe z-scores
            pivot_futures_sharpe_cs_mean = pivot_futures_sharpe.mean(axis = 1)
            pivot_futures_sharpe_cs_std = pivot_futures_sharpe.std(axis = 1)
            pivot_futures_sharpe_zscore = pivot_futures_sharpe.sub(pivot_futures_sharpe_cs_mean, axis = 0).div(pivot_futures_sharpe_cs_std, axis = 0)
            pivot_futures_sharpe_zscore.columns = [col + f'sharpe_zscore_{lookback}_csf' for col in pivot_futures_sharpe_zscore.columns]

            # 4 moments of cross-sectional sharpe ratios
            pivot_futures_sharpe[f'cs_futures_sharpe_mean_{lookback}'] = pivot_futures_sharpe.mean(axis = 1)
            pivot_futures_sharpe[f'cs_futures_sharpe_std_{lookback}'] = pivot_futures_sharpe.std(axis = 1)
            pivot_futures_sharpe[f'cs_futures_sharpe_skew_{lookback}'] = pivot_futures_sharpe.skew(axis = 1)
            pivot_futures_sharpe[f'cs_futures_sharpe_kurtosis_{lookback}'] = pivot_futures_sharpe.kurtosis(axis = 1)
            pivot_futures_sharpe[f'cs_futures_sharpe_median_{lookback}'] = pivot_futures_sharpe.median(axis = 1)

            # Append 4 moments of cross-sectional sharpe ratios to the final features list
            futures_cs_sharpe_moments_cols = [col for col in pivot_futures_sharpe.columns if any(x in col for x in ['mean', 'std', 'skew', 'kurtosis', 'median'])]

            final_features.append(pivot_futures_sharpe[futures_cs_sharpe_moments_cols])
            final_features.append(pivot_futures_sharpe_decile)
            final_features.append(pivot_futures_sharpe_zscore)

            # Cross-sectional sortino ratios for each symbol for each time period
            pivot_futures_sortino = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'futures_sortino_ratio_1_{lookback}', dropna = False)
            pivot_futures_sortino_decile = pivot_futures_sortino.rank(axis = 1, pct = True)
            pivot_futures_sortino_decile.columns = [col + f'sortino_percentile_{lookback}_csf' for col in pivot_futures_sortino_decile.columns]

            # Cross-sectional sortino z-scores
            pivot_futures_sortino_cs_mean = pivot_futures_sortino.mean(axis = 1)
            pivot_futures_sortino_cs_std = pivot_futures_sortino.std(axis = 1)
            pivot_futures_sortino_zscore = pivot_futures_sortino.sub(pivot_futures_sortino_cs_mean, axis = 0).div(pivot_futures_sortino_cs_std, axis = 0)
            pivot_futures_sortino_zscore.columns = [col + f'sortino_zscore_{lookback}_csf' for col in pivot_futures_sortino_zscore.columns]

            # 4 moments of cross-sectional sortino ratios
            pivot_futures_sortino[f'cs_futures_sortino_mean_{lookback}'] = pivot_futures_sortino.mean(axis = 1)
            pivot_futures_sortino[f'cs_futures_sortino_std_{lookback}'] = pivot_futures_sortino.std(axis = 1)
            pivot_futures_sortino[f'cs_futures_sortino_skew_{lookback}'] = pivot_futures_sortino.skew(axis = 1)
            pivot_futures_sortino[f'cs_futures_sortino_kurtosis_{lookback}'] = pivot_futures_sortino.kurtosis(axis = 1)
            pivot_futures_sortino[f'cs_futures_sortino_median_{lookback}'] = pivot_futures_sortino.median(axis = 1)

            # Append 4 moments of cross-sectional sortino ratios to the final features list
            futures_cs_sortino_moments_cols = [col for col in pivot_futures_sortino.columns if any(x in col for x in ['mean', 'std', 'skew', 'kurtosis', 'median'])]

            final_features.append(pivot_futures_sortino[futures_cs_sortino_moments_cols])
            final_features.append(pivot_futures_sortino_decile)
            final_features.append(pivot_futures_sortino_zscore)

            for lookback_2 in self.lookback_windows:
                # Rolling ema of cross-sectional features for regime detection
                pivot_spot_sharpe[f'rolling_ema_{lookback_2}_cs_spot_sharpe_mean_{lookback}'] = pivot_spot_sharpe[f'cs_spot_sharpe_mean_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_spot_sharpe[f'rolling_ema_{lookback_2}_cs_spot_sharpe_std_{lookback}'] = pivot_spot_sharpe[f'cs_spot_sharpe_std_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_spot_sharpe[f'rolling_ema_{lookback_2}_cs_spot_sharpe_skew_{lookback}'] = pivot_spot_sharpe[f'cs_spot_sharpe_skew_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_spot_sharpe[f'rolling_ema_{lookback_2}_cs_spot_sharpe_kurtosis_{lookback}'] = pivot_spot_sharpe[f'cs_spot_sharpe_kurtosis_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_spot_sharpe[f'rolling_ema_{lookback_2}_cs_spot_sharpe_median_{lookback}'] = pivot_spot_sharpe[f'cs_spot_sharpe_median_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()

                pivot_futures_sharpe[f'rolling_ema_{lookback_2}_cs_futures_sharpe_mean_{lookback}'] = pivot_futures_sharpe[f'cs_futures_sharpe_mean_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_futures_sharpe[f'rolling_ema_{lookback_2}_cs_futures_sharpe_std_{lookback}'] = pivot_futures_sharpe[f'cs_futures_sharpe_std_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_futures_sharpe[f'rolling_ema_{lookback_2}_cs_futures_sharpe_skew_{lookback}'] = pivot_futures_sharpe[f'cs_futures_sharpe_skew_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_futures_sharpe[f'rolling_ema_{lookback_2}_cs_futures_sharpe_kurtosis_{lookback}'] = pivot_futures_sharpe[f'cs_futures_sharpe_kurtosis_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_futures_sharpe[f'rolling_ema_{lookback_2}_cs_futures_sharpe_median_{lookback}'] = pivot_futures_sharpe[f'cs_futures_sharpe_median_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()

                pivot_spot_sortino[f'rolling_ema_{lookback_2}_cs_spot_sortino_mean_{lookback}'] = pivot_spot_sortino[f'cs_spot_sortino_mean_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_spot_sortino[f'rolling_ema_{lookback_2}_cs_spot_sortino_std_{lookback}'] = pivot_spot_sortino[f'cs_spot_sortino_std_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_spot_sortino[f'rolling_ema_{lookback_2}_cs_spot_sortino_skew_{lookback}'] = pivot_spot_sortino[f'cs_spot_sortino_skew_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_spot_sortino[f'rolling_ema_{lookback_2}_cs_spot_sortino_kurtosis_{lookback}'] = pivot_spot_sortino[f'cs_spot_sortino_kurtosis_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_spot_sortino[f'rolling_ema_{lookback_2}_cs_spot_sortino_median_{lookback}'] = pivot_spot_sortino[f'cs_spot_sortino_median_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                
                pivot_futures_sortino[f'rolling_ema_{lookback_2}_cs_futures_sortino_mean_{lookback}'] = pivot_futures_sortino[f'cs_futures_sortino_mean_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_futures_sortino[f'rolling_ema_{lookback_2}_cs_futures_sortino_std_{lookback}'] = pivot_futures_sortino[f'cs_futures_sortino_std_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_futures_sortino[f'rolling_ema_{lookback_2}_cs_futures_sortino_skew_{lookback}'] = pivot_futures_sortino[f'cs_futures_sortino_skew_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_futures_sortino[f'rolling_ema_{lookback_2}_cs_futures_sortino_kurtosis_{lookback}'] = pivot_futures_sortino[f'cs_futures_sortino_kurtosis_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()
                pivot_futures_sortino[f'rolling_ema_{lookback_2}_cs_futures_sortino_median_{lookback}'] = pivot_futures_sortino[f'cs_futures_sortino_median_{lookback}'].ewm(span = lookback_2, min_periods = 7).mean()

                # Append the rolling mean of cross-sectional features to the final features list  
                spot_cs_sharpe_moments = pivot_spot_sharpe.filter(like = f'rolling_ema_{lookback_2}_cs_spot_sharpe')
                futures_cs_sharpe_moments = pivot_futures_sharpe.filter(like = f'rolling_ema_{lookback_2}_cs_futures_sharpe')
                spot_cs_sortino_moments = pivot_spot_sortino.filter(like = f'rolling_ema_{lookback_2}_cs_spot_sortino')
                futures_cs_sortino_moments = pivot_futures_sortino.filter(like = f'rolling_ema_{lookback_2}_cs_futures_sortino')

                final_features.append(spot_cs_sharpe_moments)
                final_features.append(futures_cs_sharpe_moments)
                final_features.append(spot_cs_sortino_moments)
                final_features.append(futures_cs_sortino_moments)

        self.final_features = pd.concat(final_features, axis = 1)
        self.final_features = self.final_features.reset_index()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['dollar_volume_spot'] = X['close_spot'] * X['volume_spot']
        X['dollar_volume_futures'] = X['close_futures'] * X['volume_futures']
        for window_size in self.window_sizes:
            # Spot
            # Clipping of returns
            if window_size == 1:
                clip_upper_bound = 0.57
            elif window_size == 7:
                clip_upper_bound = 3.55
            elif window_size == 30:
                clip_upper_bound = 9.44
            elif window_size == 180:
                clip_upper_bound = 59
            X[f'spot_returns_{window_size}'] = X['close_spot'].pct_change(window_size)          
            # Futures 
            if window_size == 1:
                clip_upper_bound = 0.47
            elif window_size == 7:
                clip_upper_bound = 2.05
            elif window_size == 30:
                clip_upper_bound = 7.09
            elif window_size == 180:
                clip_upper_bound = 22.7
            X[f'futures_returns_{window_size}'] = X['close_futures'].pct_change(window_size)
            
            # Forward returns for future prediction evaluation
            X[f'forward_returns_{window_size}'] = X[f'spot_returns_{window_size}'].shift(-window_size)
            X[f'futures_forward_returns_{window_size}'] = X[f'futures_returns_{window_size}'].shift(-window_size) 

            for lookback_window in self.lookback_windows:
                # Spot
                # Calculate returns distributional features
                X[f'avg_spot_returns_{window_size}_{lookback_window}'] = X[f'spot_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).mean()
                X[f'median_spot_returns_{window_size}_{lookback_window}'] = X[f'spot_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).median()
                X[f'std_spot_returns_{window_size}_{lookback_window}'] = X[f'spot_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).std()
                X[f'skewness_spot_returns_{window_size}_{lookback_window}'] = X[f'spot_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).skew()
                X[f'kurtosis_spot_returns_{window_size}_{lookback_window}'] = X[f'spot_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).kurt()
                
                # Sharpe ratio
                X[f'spot_sharpe_ratio_{window_size}_{lookback_window}'] = X[f'avg_spot_returns_{window_size}_{lookback_window}'] / X[f'std_spot_returns_{window_size}_{lookback_window}']
                # Sortino ratio
                X[f'std_negative_spot_returns_{window_size}_{lookback_window}'] = X[f'spot_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).apply(lambda x: np.std(x[x < 0]), raw = False)
                X[f'spot_sortino_ratio_{window_size}_{lookback_window}'] = X[f'avg_spot_returns_{window_size}_{lookback_window}'] / X[f'std_negative_spot_returns_{window_size}_{lookback_window}']

                # Futures
                # Calculate returns distributional features
                X[f'avg_futures_returns_{window_size}_{lookback_window}'] = X[f'futures_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).mean()
                X[f'median_futures_returns_{window_size}_{lookback_window}'] = X[f'futures_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).median()
                X[f'std_futures_returns_{window_size}_{lookback_window}'] = X[f'futures_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).std()
                X[f'skewness_futures_returns_{window_size}_{lookback_window}'] = X[f'futures_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).skew()
                X[f'kurtosis_futures_returns_{window_size}_{lookback_window}'] = X[f'futures_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).kurt()
                
                # Sharpe ratio
                X[f'futures_sharpe_ratio_{window_size}_{lookback_window}'] = X[f'avg_futures_returns_{window_size}_{lookback_window}'] / X[f'std_futures_returns_{window_size}_{lookback_window}']
                # Sortino ratio
                X[f'std_negative_futures_returns_{window_size}_{lookback_window}'] = X[f'futures_returns_{window_size}'].rolling(window = lookback_window, min_periods = 7).apply(lambda x: np.std(x[x < 0]), raw = False)
                X[f'futures_sortino_ratio_{window_size}_{lookback_window}'] = X[f'avg_futures_returns_{window_size}_{lookback_window}'] / X[f'std_negative_futures_returns_{window_size}_{lookback_window}']

        # Cross-sectional rank features for current token and time period
        symbol_id = X.iloc[0]['symbol_id']

        # Cross_sectional percentile and z-score columns for the current token
        cross_sectional_ranking_cols = self.final_features.filter(like = symbol_id).columns.to_list()

        # Cross-sectional 4 moments features for the current token
        cross_sectional_moments_cols_spot = self.final_features.filter(like = 'cs_spot').columns.to_list()
        cross_sectional_moments_cols_futures = self.final_features.filter(like = 'cs_futures').columns.to_list()

        # Merge data to final features
        valid_cols = cross_sectional_ranking_cols + cross_sectional_moments_cols_spot + cross_sectional_moments_cols_futures
        X = pd.merge(
            X, 
            self.final_features[['time_period_end'] + valid_cols], 
            on = 'time_period_end', 
            how = 'left', 
            suffixes = ('', '__remove')
        )

        # Drop the columns with '__remove' suffix
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)

        # Rename the cross-sectional percentile columns to remove the symbol_id from it
        for col in cross_sectional_ranking_cols:
            new_col = col.replace(symbol_id, '')
            X.rename({col: new_col}, inplace = True, axis = 1)

        return X
    
class FillNaTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            # If column has all NaN values, skip it
            if X[col].isnull().sum() == len(X):
                continue
            elif col == 'time_period_end':
                continue
            elif X[col].dtype in ('O', 'object', 'category'):
                try:
                    if X[col].isnull().sum() > 0:
                        mode = X[col].mode().loc[0]
                        X[col] = X[col].fillna(mode)
                except Exception as e:
                    continue
            else:
                if X[col].isnull().sum() > 0:
                    # Fill missing values with the rolling mean
                    X[col] = X[col].fillna(X[col].rolling(window = 7).mean())

        return X

class TradeFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, windows, lookback_windows = (30, 90, 180)):
        self.windows = windows
        self.lookback_windows = lookback_windows

        spot_trade_features = QUERY(
            """
            SELECT *
            FROM market_data.spot_trade_features_rolling
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        spot_trade_features['symbol_id'] = spot_trade_features['asset_id_base'] + '_' + spot_trade_features['asset_id_quote'] + '_' + spot_trade_features['exchange_id']
        spot_trade_features['time_period_end'] = pd.to_datetime(spot_trade_features['time_period_end'])

        futures_trade_features = QUERY(
            """
            SELECT *
            FROM market_data.futures_trade_features_rolling
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        futures_trade_features['symbol_id'] = futures_trade_features['asset_id_base'] + '_' + futures_trade_features['asset_id_quote'] + '_' + futures_trade_features['exchange_id']
        futures_trade_features['time_period_end'] = pd.to_datetime(futures_trade_features['time_period_end'])

        trade_features = pd.merge(
            spot_trade_features,
            futures_trade_features,
            on = ['time_period_end', 'symbol_id'],
            how = 'outer',
            suffixes = ('_spot', '_futures')
        )
        trade_features.rename({
            'asset_id_base_spot': 'asset_id_base',
            'asset_id_quote_spot': 'asset_id_quote',
            'exchange_id_spot': 'exchange_id',
            'time_period_end_spot': 'time_period_end'
        }, inplace = True, axis = 1)
        trade_features.drop(columns = ['asset_id_base_futures', 'asset_id_quote_futures', 'exchange_id_futures', 'time_period_end_futures'], inplace = True, axis = 1, errors = 'ignore')        

        final_features = []

        for window in self.windows:
            # Spot
            # Cross-sectional total buy dollar volume percentile for each symbol/day
            spot_total_buy_dollar_volume_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'total_buy_dollar_volume_{window}d_spot', dropna=False).sort_index()
            spot_total_buy_dollar_volume_percentile = spot_total_buy_dollar_volume_pivot.rank(axis=1, pct=True)
            spot_total_buy_dollar_volume_percentile.columns = [f'{col}cs_spot_total_buy_dollar_volume_percentile_{window}d' for col in spot_total_buy_dollar_volume_percentile.columns]

            # Cross-sectional z-score of total buy dollar volume for each symbol/day
            spot_total_buy_dollar_volume_pivot_mean = spot_total_buy_dollar_volume_pivot.mean(axis=1)
            spot_total_buy_dollar_volume_pivot_std = spot_total_buy_dollar_volume_pivot.std(axis=1)
            spot_total_buy_dollar_volume_zscore = spot_total_buy_dollar_volume_pivot.sub(spot_total_buy_dollar_volume_pivot_mean, axis=0).div(spot_total_buy_dollar_volume_pivot_std, axis=0)
            spot_total_buy_dollar_volume_zscore.columns = [f'{col}cs_spot_total_buy_dollar_volume_zscore_{window}d' for col in spot_total_buy_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional total buy dollar volume
            spot_total_buy_dollar_volume_pivot[f'cs_spot_total_buy_dollar_volume_mean_{window}d'] = spot_total_buy_dollar_volume_pivot.mean(axis=1)
            spot_total_buy_dollar_volume_pivot[f'cs_spot_total_buy_dollar_volume_std_{window}d'] = spot_total_buy_dollar_volume_pivot.std(axis=1)
            spot_total_buy_dollar_volume_pivot[f'cs_spot_total_buy_dollar_volume_skew_{window}d'] = spot_total_buy_dollar_volume_pivot.skew(axis=1)
            spot_total_buy_dollar_volume_pivot[f'cs_spot_total_buy_dollar_volume_kurtosis_{window}d'] = spot_total_buy_dollar_volume_pivot.kurtosis(axis=1)
            spot_total_buy_dollar_volume_pivot[f'cs_spot_total_buy_dollar_volume_median_{window}d'] = spot_total_buy_dollar_volume_pivot.median(axis=1)

            # Cross-sectional total sell dollar volume percentile for each symbol/day
            spot_total_sell_dollar_volume_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'total_sell_dollar_volume_{window}d_spot', dropna=False).sort_index()
            spot_total_sell_dollar_volume_percentile = spot_total_sell_dollar_volume_pivot.rank(axis=1, pct=True)
            spot_total_sell_dollar_volume_percentile.columns = [f'{col}cs_spot_total_sell_dollar_volume_percentile_{window}d' for col in spot_total_sell_dollar_volume_percentile.columns]

            # Cross-sectional z-score of total sell dollar volume for each symbol/day
            spot_total_sell_dollar_volume_pivot_mean = spot_total_sell_dollar_volume_pivot.mean(axis=1)
            spot_total_sell_dollar_volume_pivot_std = spot_total_sell_dollar_volume_pivot.std(axis=1)
            spot_total_sell_dollar_volume_zscore = spot_total_sell_dollar_volume_pivot.sub(spot_total_sell_dollar_volume_pivot_mean, axis=0).div(spot_total_sell_dollar_volume_pivot_std, axis=0)
            spot_total_sell_dollar_volume_zscore.columns = [f'{col}cs_spot_total_sell_dollar_volume_zscore_{window}d' for col in spot_total_sell_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional total sell dollar volume
            spot_total_sell_dollar_volume_pivot[f'cs_spot_total_sell_dollar_volume_mean_{window}d'] = spot_total_sell_dollar_volume_pivot.mean(axis=1)
            spot_total_sell_dollar_volume_pivot[f'cs_spot_total_sell_dollar_volume_std_{window}d'] = spot_total_sell_dollar_volume_pivot.std(axis=1)
            spot_total_sell_dollar_volume_pivot[f'cs_spot_total_sell_dollar_volume_skew_{window}d'] = spot_total_sell_dollar_volume_pivot.skew(axis=1)
            spot_total_sell_dollar_volume_pivot[f'cs_spot_total_sell_dollar_volume_kurtosis_{window}d'] = spot_total_sell_dollar_volume_pivot.kurtosis(axis=1)
            spot_total_sell_dollar_volume_pivot[f'cs_spot_total_sell_dollar_volume_median_{window}d'] = spot_total_sell_dollar_volume_pivot.median(axis=1)

            # Cross-sectional total dollar volume percentile for each symbol/day          
            spot_total_dollar_volume_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'total_dollar_volume_{window}d_spot', dropna=False).sort_index()
            spot_total_dollar_volume_percentile = spot_total_dollar_volume_pivot.rank(axis=1, pct=True)
            spot_total_dollar_volume_percentile.columns = [f'{col}cs_spot_total_dollar_volume_percentile_{window}d' for col in spot_total_dollar_volume_percentile.columns]

            # Cross-sectional z-score of total dollar volume for each symbol/day
            spot_total_dollar_volume_pivot_mean = spot_total_dollar_volume_pivot.mean(axis=1)
            spot_total_dollar_volume_pivot_std = spot_total_dollar_volume_pivot.std(axis=1)
            spot_total_dollar_volume_zscore = spot_total_dollar_volume_pivot.sub(spot_total_dollar_volume_pivot_mean, axis=0).div(spot_total_dollar_volume_pivot_std, axis=0)
            spot_total_dollar_volume_zscore.columns = [f'{col}cs_spot_total_dollar_volume_zscore_{window}d' for col in spot_total_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional total dollar volume
            spot_total_dollar_volume_pivot[f'cs_spot_total_dollar_volume_mean_{window}d'] = spot_total_dollar_volume_pivot.mean(axis=1)
            spot_total_dollar_volume_pivot[f'cs_spot_total_dollar_volume_std_{window}d'] = spot_total_dollar_volume_pivot.std(axis=1)
            spot_total_dollar_volume_pivot[f'cs_spot_total_dollar_volume_skew_{window}d'] = spot_total_dollar_volume_pivot.skew(axis=1)
            spot_total_dollar_volume_pivot[f'cs_spot_total_dollar_volume_kurtosis_{window}d'] = spot_total_dollar_volume_pivot.kurtosis(axis=1)
            spot_total_dollar_volume_pivot[f'cs_spot_total_dollar_volume_median_{window}d'] = spot_total_dollar_volume_pivot.median(axis=1)

            # Cross-sectional number of buys percentile for each symbol/day
            spot_num_buys_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'num_buys_{window}d_spot', dropna=False).sort_index()
            spot_num_buys_percentile = spot_num_buys_pivot.rank(axis=1, pct=True)
            spot_num_buys_percentile.columns = [f'{col}cs_spot_num_buys_percentile_{window}d' for col in spot_num_buys_percentile.columns]

            # Cross-sectional z-score of number of buys for each symbol/day
            spot_num_buys_pivot_mean = spot_num_buys_pivot.mean(axis=1)
            spot_num_buys_pivot_std = spot_num_buys_pivot.std(axis=1)
            spot_num_buys_zscore = spot_num_buys_pivot.sub(spot_num_buys_pivot_mean, axis=0).div(spot_num_buys_pivot_std, axis=0)
            spot_num_buys_zscore.columns = [f'{col}cs_spot_num_buys_zscore_{window}d' for col in spot_num_buys_zscore.columns]

            # 4 moments of cross-sectional number of buys
            spot_num_buys_pivot[f'cs_spot_num_buys_mean_{window}d'] = spot_num_buys_pivot.mean(axis=1)
            spot_num_buys_pivot[f'cs_spot_num_buys_std_{window}d'] = spot_num_buys_pivot.std(axis=1)
            spot_num_buys_pivot[f'cs_spot_num_buys_skew_{window}d'] = spot_num_buys_pivot.skew(axis=1)
            spot_num_buys_pivot[f'cs_spot_num_buys_kurtosis_{window}d'] = spot_num_buys_pivot.kurtosis(axis=1)
            spot_num_buys_pivot[f'cs_spot_num_buys_median_{window}d'] = spot_num_buys_pivot.median(axis=1)

            # Cross-sectional number of sells percentile for each symbol/day
            spot_num_sells_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'num_sells_{window}d_spot', dropna=False).sort_index()
            spot_num_sells_percentile = spot_num_sells_pivot.rank(axis=1, pct=True)
            spot_num_sells_percentile.columns = [f'{col}cs_spot_num_sells_percentile_{window}d' for col in spot_num_sells_percentile.columns]

            # Cross-sectional z-score of number of sells for each symbol/day
            spot_num_sells_pivot_mean = spot_num_sells_pivot.mean(axis=1)
            spot_num_sells_pivot_std = spot_num_sells_pivot.std(axis=1)
            spot_num_sells_zscore = spot_num_sells_pivot.sub(spot_num_sells_pivot_mean, axis=0).div(spot_num_sells_pivot_std, axis=0)
            spot_num_sells_zscore.columns = [f'{col}cs_spot_num_sells_zscore_{window}d' for col in spot_num_sells_zscore.columns]

            # 4 moments of cross-sectional number of sells
            spot_num_sells_pivot[f'cs_spot_num_sells_mean_{window}d'] = spot_num_sells_pivot.mean(axis=1)
            spot_num_sells_pivot[f'cs_spot_num_sells_std_{window}d'] = spot_num_sells_pivot.std(axis=1)
            spot_num_sells_pivot[f'cs_spot_num_sells_skew_{window}d'] = spot_num_sells_pivot.skew(axis=1)
            spot_num_sells_pivot[f'cs_spot_num_sells_kurtosis_{window}d'] = spot_num_sells_pivot.kurtosis(axis=1)
            spot_num_sells_pivot[f'cs_spot_num_sells_median_{window}d'] = spot_num_sells_pivot.median(axis=1)

            # Cross-sectional percentage buys percentile for each symbol/day
            spot_pct_buys_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'pct_buys_{window}d_spot', dropna=False).sort_index()
            spot_pct_buys_percentile = spot_pct_buys_pivot.rank(axis=1, pct=True)
            spot_pct_buys_percentile.columns = [f'{col}cs_spot_pct_buys_percentile_{window}d' for col in spot_pct_buys_percentile.columns]

            # Cross-sectional z-score of percentage buys for each symbol/day
            spot_pct_buys_pivot_mean = spot_pct_buys_pivot.mean(axis=1)
            spot_pct_buys_pivot_std = spot_pct_buys_pivot.std(axis=1)
            spot_pct_buys_zscore = spot_pct_buys_pivot.sub(spot_pct_buys_pivot_mean, axis=0).div(spot_pct_buys_pivot_std, axis=0)
            spot_pct_buys_zscore.columns = [f'{col}cs_spot_pct_buys_zscore_{window}d' for col in spot_pct_buys_zscore.columns]

            # 4 moments of cross-sectional percentage buys
            spot_pct_buys_pivot[f'cs_spot_pct_buys_mean_{window}d'] = spot_pct_buys_pivot.mean(axis=1)
            spot_pct_buys_pivot[f'cs_spot_pct_buys_std_{window}d'] = spot_pct_buys_pivot.std(axis=1)
            spot_pct_buys_pivot[f'cs_spot_pct_buys_skew_{window}d'] = spot_pct_buys_pivot.skew(axis=1)
            spot_pct_buys_pivot[f'cs_spot_pct_buys_kurtosis_{window}d'] = spot_pct_buys_pivot.kurtosis(axis=1)
            spot_pct_buys_pivot[f'cs_spot_pct_buys_median_{window}d'] = spot_pct_buys_pivot.median(axis=1)

            # Cross-sectional percentage sells percentile for each symbol/day
            spot_pct_sells_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'pct_sells_{window}d_spot', dropna=False).sort_index()
            spot_pct_sells_percentile = spot_pct_sells_pivot.rank(axis=1, pct=True)
            spot_pct_sells_percentile.columns = [f'{col}cs_spot_pct_sells_percentile_{window}d' for col in spot_pct_sells_percentile.columns]

            # Cross-sectional z-score of percentage sells for each symbol/day
            spot_pct_sells_pivot_mean = spot_pct_sells_pivot.mean(axis=1)
            spot_pct_sells_pivot_std = spot_pct_sells_pivot.std(axis=1)
            spot_pct_sells_zscore = spot_pct_sells_pivot.sub(spot_pct_sells_pivot_mean, axis=0).div(spot_pct_sells_pivot_std, axis=0)
            spot_pct_sells_zscore.columns = [f'{col}cs_spot_pct_sells_zscore_{window}d' for col in spot_pct_sells_zscore.columns]

            # 4 moments of cross-sectional percentage sells
            spot_pct_sells_pivot[f'cs_spot_pct_sells_mean_{window}d'] = spot_pct_sells_pivot.mean(axis=1)
            spot_pct_sells_pivot[f'cs_spot_pct_sells_std_{window}d'] = spot_pct_sells_pivot.std(axis=1)
            spot_pct_sells_pivot[f'cs_spot_pct_sells_skew_{window}d'] = spot_pct_sells_pivot.skew(axis=1)
            spot_pct_sells_pivot[f'cs_spot_pct_sells_kurtosis_{window}d'] = spot_pct_sells_pivot.kurtosis(axis=1)
            spot_pct_sells_pivot[f'cs_spot_pct_sells_median_{window}d'] = spot_pct_sells_pivot.median(axis=1)

            # Cross-sectional trade imbalance percentile for each symbol/day
            spot_trade_dollar_volume_imbalance_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'trade_imbalance_{window}d_spot', dropna=False).sort_index()
            spot_trade_dollar_volume_imbalance_percentile = spot_trade_dollar_volume_imbalance_pivot.rank(axis=1, pct=True)
            spot_trade_dollar_volume_imbalance_percentile.columns = [f'{col}cs_spot_trade_dollar_volume_imbalance_percentile_{window}d' for col in spot_trade_dollar_volume_imbalance_percentile.columns]

            # Cross-sectional z-score of trade imbalance for each symbol/day
            spot_trade_dollar_volume_imbalance_pivot_mean = spot_trade_dollar_volume_imbalance_pivot.mean(axis=1)
            spot_trade_dollar_volume_imbalance_pivot_std = spot_trade_dollar_volume_imbalance_pivot.std(axis=1)
            spot_trade_dollar_volume_imbalance_zscore = spot_trade_dollar_volume_imbalance_pivot.sub(spot_trade_dollar_volume_imbalance_pivot_mean, axis=0).div(spot_trade_dollar_volume_imbalance_pivot_std, axis=0)
            spot_trade_dollar_volume_imbalance_zscore.columns = [f'{col}cs_spot_trade_dollar_volume_imbalance_zscore_{window}d' for col in spot_trade_dollar_volume_imbalance_zscore.columns]

            # 4 moments of cross-sectional trade imbalance
            spot_trade_dollar_volume_imbalance_pivot[f'cs_spot_trade_dollar_volume_imbalance_mean_{window}d'] = spot_trade_dollar_volume_imbalance_pivot.mean(axis=1)
            spot_trade_dollar_volume_imbalance_pivot[f'cs_spot_trade_dollar_volume_imbalance_std_{window}d'] = spot_trade_dollar_volume_imbalance_pivot.std(axis=1)
            spot_trade_dollar_volume_imbalance_pivot[f'cs_spot_trade_dollar_volume_imbalance_skew_{window}d'] = spot_trade_dollar_volume_imbalance_pivot.skew(axis=1)
            spot_trade_dollar_volume_imbalance_pivot[f'cs_spot_trade_dollar_volume_imbalance_kurtosis_{window}d'] = spot_trade_dollar_volume_imbalance_pivot.kurtosis(axis=1)
            spot_trade_dollar_volume_imbalance_pivot[f'cs_spot_trade_dollar_volume_imbalance_median_{window}d'] = spot_trade_dollar_volume_imbalance_pivot.median(axis=1)

            # Cross-sectional percentage buy dollar volume percentile for each symbol/day
            spot_pct_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id',values=f'pct_buy_dollar_volume_{window}d_spot', dropna=False).sort_index()
            spot_pct_buy_dollar_volume_percentile = spot_pct_buy_dollar_volume_pivot.rank(axis=1, pct=True)
            spot_pct_buy_dollar_volume_percentile.columns = [f'{col}cs_spot_pct_buy_dollar_volume_percentile_{window}d' for col in spot_pct_buy_dollar_volume_percentile.columns]

            # Cross-sectional z-score of percentage buy dollar volume for each symbol/day
            spot_pct_buy_dollar_volume_pivot_mean = spot_pct_buy_dollar_volume_pivot.mean(axis=1)
            spot_pct_buy_dollar_volume_pivot_std = spot_pct_buy_dollar_volume_pivot.std(axis=1)
            spot_pct_buy_dollar_volume_zscore = spot_pct_buy_dollar_volume_pivot.sub(spot_pct_buy_dollar_volume_pivot_mean, axis=0).div(spot_pct_buy_dollar_volume_pivot_std, axis=0)
            spot_pct_buy_dollar_volume_zscore.columns = [f'{col}cs_spot_pct_buy_dollar_volume_zscore_{window}d' for col in spot_pct_buy_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional percentage buy dollar volume
            spot_pct_buy_dollar_volume_pivot[f'cs_spot_pct_buy_dollar_volume_mean_{window}d'] = spot_pct_buy_dollar_volume_pivot.mean(axis=1)
            spot_pct_buy_dollar_volume_pivot[f'cs_spot_pct_buy_dollar_volume_std_{window}d'] = spot_pct_buy_dollar_volume_pivot.std(axis=1)
            spot_pct_buy_dollar_volume_pivot[f'cs_spot_pct_buy_dollar_volume_skew_{window}d'] = spot_pct_buy_dollar_volume_pivot.skew(axis=1)
            spot_pct_buy_dollar_volume_pivot[f'cs_spot_pct_buy_dollar_volume_kurtosis_{window}d'] = spot_pct_buy_dollar_volume_pivot.kurtosis(axis=1)
            spot_pct_buy_dollar_volume_pivot[f'cs_spot_pct_buy_dollar_volume_median_{window}d'] = spot_pct_buy_dollar_volume_pivot.median(axis=1)

            # Cross-sectional percentage sell dollar volume percentile for each symbol/day
            spot_pct_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id',values=f'pct_sell_dollar_volume_{window}d_spot', dropna=False).sort_index()
            spot_pct_sell_dollar_volume_percentile = spot_pct_sell_dollar_volume_pivot.rank(axis=1, pct=True)
            spot_pct_sell_dollar_volume_percentile.columns = [f'{col}cs_spot_pct_sell_dollar_volume_percentile_{window}d' for col in spot_pct_sell_dollar_volume_percentile.columns]

            # Cross-sectional z-score of percentage sell dollar volume for each symbol/day
            spot_pct_sell_dollar_volume_pivot_mean = spot_pct_sell_dollar_volume_pivot.mean(axis=1)
            spot_pct_sell_dollar_volume_pivot_std = spot_pct_sell_dollar_volume_pivot.std(axis=1)
            spot_pct_sell_dollar_volume_zscore = spot_pct_sell_dollar_volume_pivot.sub(spot_pct_sell_dollar_volume_pivot_mean, axis=0).div(spot_pct_sell_dollar_volume_pivot_std, axis=0)
            spot_pct_sell_dollar_volume_zscore.columns = [f'{col}cs_spot_pct_sell_dollar_volume_zscore_{window}d' for col in spot_pct_sell_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional percentage sell dollar volume
            spot_pct_sell_dollar_volume_pivot[f'cs_spot_pct_sell_dollar_volume_mean_{window}d'] = spot_pct_sell_dollar_volume_pivot.mean(axis=1)
            spot_pct_sell_dollar_volume_pivot[f'cs_spot_pct_sell_dollar_volume_std_{window}d'] = spot_pct_sell_dollar_volume_pivot.std(axis=1)
            spot_pct_sell_dollar_volume_pivot[f'cs_spot_pct_sell_dollar_volume_skew_{window}d'] = spot_pct_sell_dollar_volume_pivot.skew(axis=1)
            spot_pct_sell_dollar_volume_pivot[f'cs_spot_pct_sell_dollar_volume_kurtosis_{window}d'] = spot_pct_sell_dollar_volume_pivot.kurtosis(axis=1)
            spot_pct_sell_dollar_volume_pivot[f'cs_spot_pct_sell_dollar_volume_median_{window}d'] = spot_pct_sell_dollar_volume_pivot.median(axis=1)

            for lookback_2 in self.lookback_windows:
                # Rolling ema of cross-sectional total buy dollar volume for regime detection
                spot_total_buy_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_spot_total_buy_dollar_volume_mean_{window}d'] = spot_total_buy_dollar_volume_pivot[f'cs_spot_total_buy_dollar_volume_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()
                # Rolling ema of cross-sectional total sell dollar volume for regime detection
                spot_total_sell_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_spot_total_sell_dollar_volume_mean_{window}d'] = spot_total_sell_dollar_volume_pivot[f'cs_spot_total_sell_dollar_volume_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()
                # Rolling ema of cross-sectional total dollar volume for regime detection
                spot_total_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_spot_total_dollar_volume_mean_{window}d'] = spot_total_dollar_volume_pivot[f'cs_spot_total_dollar_volume_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()
                # Rolling ema of cross-sectional number of buys for regime detection
                spot_num_buys_pivot[f'rolling_ema_{lookback_2}_cs_spot_num_buys_mean_{window}d'] = spot_num_buys_pivot[f'cs_spot_num_buys_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()
                # Rolling ema of cross-sectional number of sells for regime detection
                spot_num_sells_pivot[f'rolling_ema_{lookback_2}_cs_spot_num_sells_mean_{window}d'] = spot_num_sells_pivot[f'cs_spot_num_sells_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()
                # Rolling ema of cross-sectional percentage buys for regime detection
                spot_pct_buys_pivot[f'rolling_ema_{lookback_2}_cs_spot_pct_buys_mean_{window}d'] = spot_pct_buys_pivot[f'cs_spot_pct_buys_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()
                # Rolling ema of cross-sectional percentage sells for regime detection
                spot_pct_sells_pivot[f'rolling_ema_{lookback_2}_cs_spot_pct_sells_mean_{window}d'] = spot_pct_sells_pivot[f'cs_spot_pct_sells_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()
                #  Rolling ema of cross-sectional trade imbalance for regime detection
                spot_trade_dollar_volume_imbalance_pivot[f'rolling_ema_{lookback_2}_cs_spot_trade_dollar_volume_imbalance_mean_{window}d'] = spot_trade_dollar_volume_imbalance_pivot[f'cs_spot_trade_dollar_volume_imbalance_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()            
                # Rolling ema of cross-sectional percentage buy dollar volume for regime detection
                spot_pct_buy_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_spot_pct_buy_dollar_volume_mean_{window}d'] = spot_pct_buy_dollar_volume_pivot[f'cs_spot_pct_buy_dollar_volume_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()
                # Rolling ema of cross-sectional percentage sell dollar volume for regime detection
                spot_pct_sell_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_spot_pct_sell_dollar_volume_mean_{window}d'] = spot_pct_sell_dollar_volume_pivot[f'cs_spot_pct_sell_dollar_volume_mean_{window}d'].ewm(span=lookback_2, min_periods=7).mean()

                spot_total_buy_dollar_volume_pivot_ema = spot_total_buy_dollar_volume_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_total_buy_dollar_volume_mean_{window}d')
                spot_total_sell_dollar_volume_pivot_ema = spot_total_sell_dollar_volume_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_total_sell_dollar_volume_mean_{window}d')
                spot_total_dollar_volume_pivot_ema = spot_total_dollar_volume_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_total_dollar_volume_mean_{window}d')
                spot_num_buys_pivot_ema = spot_num_buys_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_num_buys_mean_{window}d')
                spot_num_sells_pivot_ema = spot_num_sells_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_num_sells_mean_{window}d')
                spot_pct_buys_pivot_ema = spot_pct_buys_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_pct_buys_mean_{window}d')
                spot_pct_sells_pivot_ema = spot_pct_sells_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_pct_sells_mean_{window}d')
                spot_trade_dollar_volume_imbalance_pivot_ema = spot_trade_dollar_volume_imbalance_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_trade_dollar_volume_imbalance_mean_{window}d')
                spot_pct_buy_dollar_volume_pivot_ema = spot_pct_buy_dollar_volume_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_pct_buy_dollar_volume_mean_{window}d')
                spot_pct_sell_dollar_volume_pivot_ema = spot_pct_sell_dollar_volume_pivot.filter(like=f'rolling_ema_{lookback_2}_cs_spot_pct_sell_dollar_volume_mean_{window}d')

                # Add rolling ema columns to final features
                final_features.append(spot_total_buy_dollar_volume_pivot_ema)
                final_features.append(spot_total_sell_dollar_volume_pivot_ema)
                final_features.append(spot_total_dollar_volume_pivot_ema)
                final_features.append(spot_num_buys_pivot_ema)
                final_features.append(spot_num_sells_pivot_ema)
                final_features.append(spot_pct_buys_pivot_ema)
                final_features.append(spot_pct_sells_pivot_ema)
                final_features.append(spot_trade_dollar_volume_imbalance_pivot_ema)
                final_features.append(spot_pct_buy_dollar_volume_pivot_ema)
                final_features.append(spot_pct_sell_dollar_volume_pivot_ema)

            # Cross-sectional avg dollar volume percentile for each symbol/day
            spot_avg_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'avg_dollar_volume_{window}d_spot', dropna=False).sort_index()
            spot_avg_dollar_volume_percentile = spot_avg_dollar_volume_pivot.rank(axis=1, pct=True)
            spot_avg_dollar_volume_percentile.columns = [f'{col}cs_spot_avg_dollar_volume_percentile_{window}d' for col in spot_avg_dollar_volume_percentile.columns]

            # Cross-sectional z-score of avg dollar volume for each symbol/day
            spot_avg_dollar_volume_pivot_mean = spot_avg_dollar_volume_pivot.mean(axis=1)
            spot_avg_dollar_volume_pivot_std = spot_avg_dollar_volume_pivot.std(axis=1)
            spot_avg_dollar_volume_zscore = spot_avg_dollar_volume_pivot.sub(spot_avg_dollar_volume_pivot_mean, axis=0).div(spot_avg_dollar_volume_pivot_std, axis=0)
            spot_avg_dollar_volume_zscore.columns = [f'{col}cs_spot_avg_dollar_volume_zscore_{window}d' for col in spot_avg_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional avg dollar volume
            spot_avg_dollar_volume_pivot[f'cs_spot_avg_dollar_volume_mean_{window}d'] = spot_avg_dollar_volume_pivot.mean(axis=1)
            spot_avg_dollar_volume_pivot[f'cs_spot_avg_dollar_volume_std_{window}d'] = spot_avg_dollar_volume_pivot.std(axis=1)
            spot_avg_dollar_volume_pivot[f'cs_spot_avg_dollar_volume_skew_{window}d'] = spot_avg_dollar_volume_pivot.skew(axis=1)
            spot_avg_dollar_volume_pivot[f'cs_spot_avg_dollar_volume_kurtosis_{window}d'] = spot_avg_dollar_volume_pivot.kurtosis(axis=1)
            spot_avg_dollar_volume_pivot[f'cs_spot_avg_dollar_volume_median_{window}d'] = spot_avg_dollar_volume_pivot.median(axis=1)

            # Cross-sectional avg buy dollar volume percentile for each symbol/day
            spot_avg_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'avg_buy_dollar_volume_{window}d_spot', dropna=False).sort_index()
            spot_avg_buy_dollar_volume_percentile = spot_avg_buy_dollar_volume_pivot.rank(axis=1, pct=True)
            spot_avg_buy_dollar_volume_percentile.columns = [f'{col}cs_spot_avg_buy_dollar_volume_percentile_{window}d' for col in spot_avg_buy_dollar_volume_percentile.columns]

            # Cross-sectional z-score of avg buy dollar volume for each symbol/day
            spot_avg_buy_dollar_volume_pivot_mean = spot_avg_buy_dollar_volume_pivot.mean(axis=1)
            spot_avg_buy_dollar_volume_pivot_std = spot_avg_buy_dollar_volume_pivot.std(axis=1)
            spot_avg_buy_dollar_volume_zscore = spot_avg_buy_dollar_volume_pivot.sub(spot_avg_buy_dollar_volume_pivot_mean, axis=0).div(spot_avg_buy_dollar_volume_pivot_std, axis=0)
            spot_avg_buy_dollar_volume_zscore.columns = [f'{col}cs_spot_avg_buy_dollar_volume_zscore_{window}d' for col in spot_avg_buy_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional avg buy dollar volume
            spot_avg_buy_dollar_volume_pivot[f'cs_spot_avg_buy_dollar_volume_mean_{window}d'] = spot_avg_buy_dollar_volume_pivot.mean(axis=1)
            spot_avg_buy_dollar_volume_pivot[f'cs_spot_avg_buy_dollar_volume_std_{window}d'] = spot_avg_buy_dollar_volume_pivot.std(axis=1)
            spot_avg_buy_dollar_volume_pivot[f'cs_spot_avg_buy_dollar_volume_skew_{window}d'] = spot_avg_buy_dollar_volume_pivot.skew(axis=1)
            spot_avg_buy_dollar_volume_pivot[f'cs_spot_avg_buy_dollar_volume_kurtosis_{window}d'] = spot_avg_buy_dollar_volume_pivot.kurtosis(axis=1)
            spot_avg_buy_dollar_volume_pivot[f'cs_spot_avg_buy_dollar_volume_median_{window}d'] = spot_avg_buy_dollar_volume_pivot.median(axis=1)

            # Cross-sectional avg sell dollar volume percentile for each symbol/day
            spot_avg_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'avg_sell_dollar_volume_{window}d_spot', dropna=False).sort_index()
            spot_avg_sell_dollar_volume_percentile = spot_avg_sell_dollar_volume_pivot.rank(axis=1, pct=True)
            spot_avg_sell_dollar_volume_percentile.columns = [f'{col}cs_spot_avg_sell_dollar_volume_percentile_{window}d' for col in spot_avg_sell_dollar_volume_percentile.columns]

            # Cross-sectional z-score of avg sell dollar volume for each symbol/day
            spot_avg_sell_dollar_volume_pivot_mean = spot_avg_sell_dollar_volume_pivot.mean(axis=1)
            spot_avg_sell_dollar_volume_pivot_std = spot_avg_sell_dollar_volume_pivot.std(axis=1)
            spot_avg_sell_dollar_volume_zscore = spot_avg_sell_dollar_volume_pivot.sub(spot_avg_sell_dollar_volume_pivot_mean, axis=0).div(spot_avg_sell_dollar_volume_pivot_std, axis=0)
            spot_avg_sell_dollar_volume_zscore.columns = [f'{col}cs_spot_avg_sell_dollar_volume_zscore_{window}d' for col in spot_avg_sell_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional avg sell dollar volume
            spot_avg_sell_dollar_volume_pivot[f'cs_spot_avg_sell_dollar_volume_mean_{window}d'] = spot_avg_sell_dollar_volume_pivot.mean(axis=1)
            spot_avg_sell_dollar_volume_pivot[f'cs_spot_avg_sell_dollar_volume_std_{window}d'] = spot_avg_sell_dollar_volume_pivot.std(axis=1)
            spot_avg_sell_dollar_volume_pivot[f'cs_spot_avg_sell_dollar_volume_skew_{window}d'] = spot_avg_sell_dollar_volume_pivot.skew(axis=1)
            spot_avg_sell_dollar_volume_pivot[f'cs_spot_avg_sell_dollar_volume_kurtosis_{window}d'] = spot_avg_sell_dollar_volume_pivot.kurtosis(axis=1)
            spot_avg_sell_dollar_volume_pivot[f'cs_spot_avg_sell_dollar_volume_median_{window}d'] = spot_avg_sell_dollar_volume_pivot.median(axis=1)

            if window == 1:
                # Cross-sectional std dollar volume percentile for each symbol/day
                spot_std_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'std_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_std_dollar_volume_percentile = spot_std_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_std_dollar_volume_percentile.columns = [f'{col}cs_spot_std_dollar_volume_percentile_{window}d' for col in spot_std_dollar_volume_percentile.columns]

                # Cross-sectional z-score of std dollar volume for each symbol/day
                spot_std_dollar_volume_pivot_mean = spot_std_dollar_volume_pivot.mean(axis=1)
                spot_std_dollar_volume_pivot_std = spot_std_dollar_volume_pivot.std(axis=1)
                spot_std_dollar_volume_zscore = spot_std_dollar_volume_pivot.sub(spot_std_dollar_volume_pivot_mean, axis=0).div(spot_std_dollar_volume_pivot_std, axis=0)
                spot_std_dollar_volume_zscore.columns = [f'{col}cs_spot_std_dollar_volume_zscore_{window}d' for col in spot_std_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional std dollar volume
                spot_std_dollar_volume_pivot[f'cs_spot_std_dollar_volume_mean_{window}d'] = spot_std_dollar_volume_pivot.mean(axis=1)
                spot_std_dollar_volume_pivot[f'cs_spot_std_dollar_volume_std_{window}d'] = spot_std_dollar_volume_pivot.std(axis=1)                
                spot_std_dollar_volume_pivot[f'cs_spot_std_dollar_volume_skew_{window}d'] = spot_std_dollar_volume_pivot.skew(axis=1)
                spot_std_dollar_volume_pivot[f'cs_spot_std_dollar_volume_kurtosis_{window}d'] = spot_std_dollar_volume_pivot.kurtosis(axis=1)
                spot_std_dollar_volume_pivot[f'cs_spot_std_dollar_volume_median_{window}d'] = spot_std_dollar_volume_pivot.median(axis=1)

                # Cross-sectional std buy dollar volume percentile for each symbol/day
                spot_std_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'std_buy_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_std_buy_dollar_volume_percentile = spot_std_buy_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_std_buy_dollar_volume_percentile.columns = [f'{col}cs_spot_std_buy_dollar_volume_percentile_{window}d' for col in spot_std_buy_dollar_volume_percentile.columns]

                # Cross-sectional z-score of std buy dollar volume for each symbol/day
                spot_std_buy_dollar_volume_pivot_mean = spot_std_buy_dollar_volume_pivot.mean(axis=1)
                spot_std_buy_dollar_volume_pivot_std = spot_std_buy_dollar_volume_pivot.std(axis=1)
                spot_std_buy_dollar_volume_zscore = spot_std_buy_dollar_volume_pivot.sub(spot_std_buy_dollar_volume_pivot_mean, axis=0).div(spot_std_buy_dollar_volume_pivot_std, axis=0)
                spot_std_buy_dollar_volume_zscore.columns = [f'{col}cs_spot_std_buy_dollar_volume_zscore_{window}d' for col in spot_std_buy_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional std buy dollar volume
                spot_std_buy_dollar_volume_pivot[f'cs_spot_std_buy_dollar_volume_mean_{window}d'] = spot_std_buy_dollar_volume_pivot.mean(axis=1)
                spot_std_buy_dollar_volume_pivot[f'cs_spot_std_buy_dollar_volume_std_{window}d'] = spot_std_buy_dollar_volume_pivot.std(axis=1)
                spot_std_buy_dollar_volume_pivot[f'cs_spot_std_buy_dollar_volume_skew_{window}d'] = spot_std_buy_dollar_volume_pivot.skew(axis=1)
                spot_std_buy_dollar_volume_pivot[f'cs_spot_std_buy_dollar_volume_kurtosis_{window}d'] = spot_std_buy_dollar_volume_pivot.kurtosis(axis=1)
                spot_std_buy_dollar_volume_pivot[f'cs_spot_std_buy_dollar_volume_median_{window}d'] = spot_std_buy_dollar_volume_pivot.median(axis=1)


                # Cross-sectional std sell dollar volume percentile for each symbol/day
                spot_std_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'std_sell_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_std_sell_dollar_volume_percentile = spot_std_sell_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_std_sell_dollar_volume_percentile.columns = [f'{col}cs_spot_std_sell_dollar_volume_percentile_{window}d' for col in spot_std_sell_dollar_volume_percentile.columns]

                # Cross-sectional z-score of std sell dollar volume for each symbol/day
                spot_std_sell_dollar_volume_pivot_mean = spot_std_sell_dollar_volume_pivot.mean(axis=1)
                spot_std_sell_dollar_volume_pivot_std = spot_std_sell_dollar_volume_pivot.std(axis=1)
                spot_std_sell_dollar_volume_zscore = spot_std_sell_dollar_volume_pivot.sub(spot_std_sell_dollar_volume_pivot_mean, axis=0).div(spot_std_sell_dollar_volume_pivot_std, axis=0)
                spot_std_sell_dollar_volume_zscore.columns = [f'{col}cs_spot_std_sell_dollar_volume_zscore_{window}d' for col in spot_std_sell_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional std sell dollar volume
                spot_std_sell_dollar_volume_pivot[f'cs_spot_std_sell_dollar_volume_mean_{window}d'] = spot_std_sell_dollar_volume_pivot.mean(axis=1)
                spot_std_sell_dollar_volume_pivot[f'cs_spot_std_sell_dollar_volume_std_{window}d'] = spot_std_sell_dollar_volume_pivot.std(axis=1)
                spot_std_sell_dollar_volume_pivot[f'cs_spot_std_sell_dollar_volume_skew_{window}d'] = spot_std_sell_dollar_volume_pivot.skew(axis=1)
                spot_std_sell_dollar_volume_pivot[f'cs_spot_std_sell_dollar_volume_kurtosis_{window}d'] = spot_std_sell_dollar_volume_pivot.kurtosis(axis=1)
                spot_std_sell_dollar_volume_pivot[f'cs_spot_std_sell_dollar_volume_median_{window}d'] = spot_std_sell_dollar_volume_pivot.median(axis=1)


                # Cross-sectional skewness dollar volume percentile for each symbol/day
                spot_skewness_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'skew_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_skewness_dollar_volume_percentile = spot_skewness_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_skewness_dollar_volume_percentile.columns = [f'{col}cs_spot_skewness_dollar_volume_percentile_{window}d' for col in spot_skewness_dollar_volume_percentile.columns]

                # Cross-sectional z-score of skewness dollar volume for each symbol/day
                spot_skewness_dollar_volume_pivot_mean = spot_skewness_dollar_volume_pivot.mean(axis=1)
                spot_skewness_dollar_volume_pivot_std = spot_skewness_dollar_volume_pivot.std(axis=1)
                spot_skewness_dollar_volume_zscore = spot_skewness_dollar_volume_pivot.sub(spot_skewness_dollar_volume_pivot_mean, axis=0).div(spot_skewness_dollar_volume_pivot_std, axis=0)
                spot_skewness_dollar_volume_zscore.columns = [f'{col}cs_spot_skewness_dollar_volume_zscore_{window}d' for col in spot_skewness_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional skewness dollar volume
                spot_skewness_dollar_volume_pivot[f'cs_spot_skewness_dollar_volume_mean_{window}d'] = spot_skewness_dollar_volume_pivot.mean(axis=1)
                spot_skewness_dollar_volume_pivot[f'cs_spot_skewness_dollar_volume_std_{window}d'] = spot_skewness_dollar_volume_pivot.std(axis=1)
                spot_skewness_dollar_volume_pivot[f'cs_spot_skewness_dollar_volume_skew_{window}d'] = spot_skewness_dollar_volume_pivot.skew(axis=1)
                spot_skewness_dollar_volume_pivot[f'cs_spot_skewness_dollar_volume_kurtosis_{window}d'] = spot_skewness_dollar_volume_pivot.kurtosis(axis=1)
                spot_skewness_dollar_volume_pivot[f'cs_spot_skewness_dollar_volume_median_{window}d'] = spot_skewness_dollar_volume_pivot.median(axis=1)

                # Cross-sectional skewness buy dollar volume percentile for each symbol/day
                spot_skewness_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'skew_buy_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_skewness_buy_dollar_volume_percentile = spot_skewness_buy_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_skewness_buy_dollar_volume_percentile.columns = [f'{col}cs_spot_skewness_buy_dollar_volume_percentile_{window}d' for col in spot_skewness_buy_dollar_volume_percentile.columns]

                # Cross-sectional z-score of skewness buy dollar volume for each symbol/day
                spot_skewness_buy_dollar_volume_pivot_mean = spot_skewness_buy_dollar_volume_pivot.mean(axis=1)
                spot_skewness_buy_dollar_volume_pivot_std = spot_skewness_buy_dollar_volume_pivot.std(axis=1)
                spot_skewness_buy_dollar_volume_zscore = spot_skewness_buy_dollar_volume_pivot.sub(spot_skewness_buy_dollar_volume_pivot_mean, axis=0).div(spot_skewness_buy_dollar_volume_pivot_std, axis=0)
                spot_skewness_buy_dollar_volume_zscore.columns = [f'{col}cs_spot_skewness_buy_dollar_volume_zscore_{window}d' for col in spot_skewness_buy_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional skewness buy dollar volume
                spot_skewness_buy_dollar_volume_pivot[f'cs_spot_skewness_buy_dollar_volume_mean_{window}d'] = spot_skewness_buy_dollar_volume_pivot.mean(axis=1)
                spot_skewness_buy_dollar_volume_pivot[f'cs_spot_skewness_buy_dollar_volume_std_{window}d'] = spot_skewness_buy_dollar_volume_pivot.std(axis=1)
                spot_skewness_buy_dollar_volume_pivot[f'cs_spot_skewness_buy_dollar_volume_skew_{window}d'] = spot_skewness_buy_dollar_volume_pivot.skew(axis=1)
                spot_skewness_buy_dollar_volume_pivot[f'cs_spot_skewness_buy_dollar_volume_kurtosis_{window}d'] = spot_skewness_buy_dollar_volume_pivot.kurtosis(axis=1)
                spot_skewness_buy_dollar_volume_pivot[f'cs_spot_skewness_buy_dollar_volume_median_{window}d'] = spot_skewness_buy_dollar_volume_pivot.median(axis=1)


                # Cross-sectional skewness sell dollar volume percentile for each symbol/day
                spot_skewness_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'skew_sell_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_skewness_sell_dollar_volume_percentile = spot_skewness_sell_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_skewness_sell_dollar_volume_percentile.columns = [f'{col}cs_spot_skewness_sell_dollar_volume_percentile_{window}d' for col in spot_skewness_sell_dollar_volume_percentile.columns]

                # Cross-sectional z-score of skewness sell dollar volume for each symbol/day
                spot_skewness_sell_dollar_volume_pivot_mean = spot_skewness_sell_dollar_volume_pivot.mean(axis=1)
                spot_skewness_sell_dollar_volume_pivot_std = spot_skewness_sell_dollar_volume_pivot.std(axis=1)
                spot_skewness_sell_dollar_volume_zscore = spot_skewness_sell_dollar_volume_pivot.sub(spot_skewness_sell_dollar_volume_pivot_mean, axis=0).div(spot_skewness_sell_dollar_volume_pivot_std, axis=0)
                spot_skewness_sell_dollar_volume_zscore.columns = [f'{col}cs_spot_skewness_sell_dollar_volume_zscore_{window}d' for col in spot_skewness_sell_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional skewness sell dollar volume
                spot_skewness_sell_dollar_volume_pivot[f'cs_spot_skewness_sell_dollar_volume_mean_{window}d'] = spot_skewness_sell_dollar_volume_pivot.mean(axis=1)
                spot_skewness_sell_dollar_volume_pivot[f'cs_spot_skewness_sell_dollar_volume_std_{window}d'] = spot_skewness_sell_dollar_volume_pivot.std(axis=1)
                spot_skewness_sell_dollar_volume_pivot[f'cs_spot_skewness_sell_dollar_volume_skew_{window}d'] = spot_skewness_sell_dollar_volume_pivot.skew(axis=1)
                spot_skewness_sell_dollar_volume_pivot[f'cs_spot_skewness_sell_dollar_volume_kurtosis_{window}d'] = spot_skewness_sell_dollar_volume_pivot.kurtosis(axis=1)
                spot_skewness_sell_dollar_volume_pivot[f'cs_spot_skewness_sell_dollar_volume_median_{window}d'] = spot_skewness_sell_dollar_volume_pivot.median(axis=1)

                # Cross-sectional kurtosis dollar volume percentile for each symbol/day
                spot_kurtosis_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'kurtosis_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_kurtosis_dollar_volume_percentile = spot_kurtosis_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_kurtosis_dollar_volume_percentile.columns = [f'{col}cs_spot_kurtosis_dollar_volume_percentile_{window}d' for col in spot_kurtosis_dollar_volume_percentile.columns]

                # Cross-sectional z-score of kurtosis dollar volume for each symbol/day
                spot_kurtosis_dollar_volume_pivot_mean = spot_kurtosis_dollar_volume_pivot.mean(axis=1)
                spot_kurtosis_dollar_volume_pivot_std = spot_kurtosis_dollar_volume_pivot.std(axis=1)
                spot_kurtosis_dollar_volume_zscore = spot_kurtosis_dollar_volume_pivot.sub(spot_kurtosis_dollar_volume_pivot_mean, axis=0).div(spot_kurtosis_dollar_volume_pivot_std, axis=0)
                spot_kurtosis_dollar_volume_zscore.columns = [f'{col}cs_spot_kurtosis_dollar_volume_zscore_{window}d' for col in spot_kurtosis_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional kurtosis dollar volume
                spot_kurtosis_dollar_volume_pivot[f'cs_spot_kurtosis_dollar_volume_mean_{window}d'] = spot_kurtosis_dollar_volume_pivot.mean(axis=1)
                spot_kurtosis_dollar_volume_pivot[f'cs_spot_kurtosis_dollar_volume_std_{window}d'] = spot_kurtosis_dollar_volume_pivot.std(axis=1)
                spot_kurtosis_dollar_volume_pivot[f'cs_spot_kurtosis_dollar_volume_skew_{window}d'] = spot_kurtosis_dollar_volume_pivot.skew(axis=1)
                spot_kurtosis_dollar_volume_pivot[f'cs_spot_kurtosis_dollar_volume_kurtosis_{window}d'] = spot_kurtosis_dollar_volume_pivot.kurtosis(axis=1)
                spot_kurtosis_dollar_volume_pivot[f'cs_spot_kurtosis_dollar_volume_median_{window}d'] = spot_kurtosis_dollar_volume_pivot.median(axis=1)

                # Cross-sectional kurtosis buy dollar volume percentile for each symbol/day
                spot_kurtosis_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'kurtosis_buy_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_kurtosis_buy_dollar_volume_percentile = spot_kurtosis_buy_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_kurtosis_buy_dollar_volume_percentile.columns = [f'{col}cs_spot_kurtosis_buy_dollar_volume_percentile_{window}d' for col in spot_kurtosis_buy_dollar_volume_percentile.columns]

                # Cross-sectional z-score of kurtosis buy dollar volume for each symbol/day
                spot_kurtosis_buy_dollar_volume_pivot_mean = spot_kurtosis_buy_dollar_volume_pivot.mean(axis=1)
                spot_kurtosis_buy_dollar_volume_pivot_std = spot_kurtosis_buy_dollar_volume_pivot.std(axis=1)
                spot_kurtosis_buy_dollar_volume_zscore = spot_kurtosis_buy_dollar_volume_pivot.sub(spot_kurtosis_buy_dollar_volume_pivot_mean, axis=0).div(spot_kurtosis_buy_dollar_volume_pivot_std, axis=0)
                spot_kurtosis_buy_dollar_volume_zscore.columns = [f'{col}cs_spot_kurtosis_buy_dollar_volume_zscore_{window}d' for col in spot_kurtosis_buy_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional kurtosis buy dollar volume
                spot_kurtosis_buy_dollar_volume_pivot[f'cs_spot_kurtosis_buy_dollar_volume_mean_{window}d'] = spot_kurtosis_buy_dollar_volume_pivot.mean(axis=1)
                spot_kurtosis_buy_dollar_volume_pivot[f'cs_spot_kurtosis_buy_dollar_volume_std_{window}d'] = spot_kurtosis_buy_dollar_volume_pivot.std(axis=1)
                spot_kurtosis_buy_dollar_volume_pivot[f'cs_spot_kurtosis_buy_dollar_volume_skew_{window}d'] = spot_kurtosis_buy_dollar_volume_pivot.skew(axis=1)
                spot_kurtosis_buy_dollar_volume_pivot[f'cs_spot_kurtosis_buy_dollar_volume_kurtosis_{window}d'] = spot_kurtosis_buy_dollar_volume_pivot.kurtosis(axis=1)
                spot_kurtosis_buy_dollar_volume_pivot[f'cs_spot_kurtosis_buy_dollar_volume_median_{window}d'] = spot_kurtosis_buy_dollar_volume_pivot.median(axis=1)

                # Cross-sectional kurtosis sell dollar volume percentile for each symbol/day
                spot_kurtosis_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'kurtosis_sell_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_kurtosis_sell_dollar_volume_percentile = spot_kurtosis_sell_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_kurtosis_sell_dollar_volume_percentile.columns = [f'{col}cs_spot_kurtosis_sell_dollar_volume_percentile_{window}d' for col in spot_kurtosis_sell_dollar_volume_percentile.columns]

                # Cross-sectional z-score of kurtosis sell dollar volume for each symbol/day
                spot_kurtosis_sell_dollar_volume_pivot_mean = spot_kurtosis_sell_dollar_volume_pivot.mean(axis=1)
                spot_kurtosis_sell_dollar_volume_pivot_std = spot_kurtosis_sell_dollar_volume_pivot.std(axis=1)
                spot_kurtosis_sell_dollar_volume_zscore = spot_kurtosis_sell_dollar_volume_pivot.sub(spot_kurtosis_sell_dollar_volume_pivot_mean, axis=0).div(spot_kurtosis_sell_dollar_volume_pivot_std, axis=0)
                spot_kurtosis_sell_dollar_volume_zscore.columns = [f'{col}cs_spot_kurtosis_sell_dollar_volume_zscore_{window}d' for col in spot_kurtosis_sell_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional kurtosis sell dollar volume
                spot_kurtosis_sell_dollar_volume_pivot[f'cs_spot_kurtosis_sell_dollar_volume_mean_{window}d'] = spot_kurtosis_sell_dollar_volume_pivot.mean(axis=1)
                spot_kurtosis_sell_dollar_volume_pivot[f'cs_spot_kurtosis_sell_dollar_volume_std_{window}d'] = spot_kurtosis_sell_dollar_volume_pivot.std(axis=1)
                spot_kurtosis_sell_dollar_volume_pivot[f'cs_spot_kurtosis_sell_dollar_volume_skew_{window}d'] = spot_kurtosis_sell_dollar_volume_pivot.skew(axis=1)
                spot_kurtosis_sell_dollar_volume_pivot[f'cs_spot_kurtosis_sell_dollar_volume_kurtosis_{window}d'] = spot_kurtosis_sell_dollar_volume_pivot.kurtosis(axis=1)
                spot_kurtosis_sell_dollar_volume_pivot[f'cs_spot_kurtosis_sell_dollar_volume_median_{window}d'] = spot_kurtosis_sell_dollar_volume_pivot.median(axis=1)

                # Cross-sectional median dollar volume percentile for each symbol/day
                spot_median_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'median_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_median_dollar_volume_percentile = spot_median_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_median_dollar_volume_percentile.columns = [f'{col}cs_spot_median_dollar_volume_percentile_{window}d' for col in spot_median_dollar_volume_percentile.columns]

                # Cross-sectional z-score of median dollar volume for each symbol/day
                spot_median_dollar_volume_pivot_mean = spot_median_dollar_volume_pivot.mean(axis=1)
                spot_median_dollar_volume_pivot_std = spot_median_dollar_volume_pivot.std(axis=1)
                spot_median_dollar_volume_zscore = spot_median_dollar_volume_pivot.sub(spot_median_dollar_volume_pivot_mean, axis=0).div(spot_median_dollar_volume_pivot_std, axis=0)
                spot_median_dollar_volume_zscore.columns = [f'{col}cs_spot_median_dollar_volume_zscore_{window}d' for col in spot_median_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional median dollar volume
                spot_median_dollar_volume_pivot[f'cs_spot_median_dollar_volume_mean_{window}d'] = spot_median_dollar_volume_pivot.mean(axis=1)
                spot_median_dollar_volume_pivot[f'cs_spot_median_dollar_volume_std_{window}d'] = spot_median_dollar_volume_pivot.std(axis=1)
                spot_median_dollar_volume_pivot[f'cs_spot_median_dollar_volume_skew_{window}d'] = spot_median_dollar_volume_pivot.skew(axis=1)
                spot_median_dollar_volume_pivot[f'cs_spot_median_dollar_volume_kurtosis_{window}d'] = spot_median_dollar_volume_pivot.kurtosis(axis=1)
                spot_median_dollar_volume_pivot[f'cs_spot_median_dollar_volume_median_{window}d'] = spot_median_dollar_volume_pivot.median(axis=1)

                # Cross-sectional median buy dollar volume percentile for each symbol/day
                spot_median_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'median_buy_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_median_buy_dollar_volume_percentile = spot_median_buy_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_median_buy_dollar_volume_percentile.columns = [f'{col}cs_spot_median_buy_dollar_volume_percentile_{window}d' for col in spot_median_buy_dollar_volume_percentile.columns]

                # Cross-sectional z-score of median buy dollar volume for each symbol/day
                spot_median_buy_dollar_volume_pivot_mean = spot_median_buy_dollar_volume_pivot.mean(axis=1)
                spot_median_buy_dollar_volume_pivot_std = spot_median_buy_dollar_volume_pivot.std(axis=1)
                spot_median_buy_dollar_volume_zscore = spot_median_buy_dollar_volume_pivot.sub(spot_median_buy_dollar_volume_pivot_mean, axis=0).div(spot_median_buy_dollar_volume_pivot_std, axis=0)
                spot_median_buy_dollar_volume_zscore.columns = [f'{col}cs_spot_median_buy_dollar_volume_zscore_{window}d' for col in spot_median_buy_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional median buy dollar volume
                spot_median_buy_dollar_volume_pivot[f'cs_spot_median_buy_dollar_volume_mean_{window}d'] = spot_median_buy_dollar_volume_pivot.mean(axis=1)
                spot_median_buy_dollar_volume_pivot[f'cs_spot_median_buy_dollar_volume_std_{window}d'] = spot_median_buy_dollar_volume_pivot.std(axis=1)
                spot_median_buy_dollar_volume_pivot[f'cs_spot_median_buy_dollar_volume_skew_{window}d'] = spot_median_buy_dollar_volume_pivot.skew(axis=1)
                spot_median_buy_dollar_volume_pivot[f'cs_spot_median_buy_dollar_volume_kurtosis_{window}d'] = spot_median_buy_dollar_volume_pivot.kurtosis(axis=1)
                spot_median_buy_dollar_volume_pivot[f'cs_spot_median_buy_dollar_volume_median_{window}d'] = spot_median_buy_dollar_volume_pivot.median(axis=1)

                # Cross-sectional median sell dollar volume percentile for each symbol/day
                spot_median_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'median_sell_dollar_volume_{window}d_spot', dropna=False).sort_index()
                spot_median_sell_dollar_volume_percentile = spot_median_sell_dollar_volume_pivot.rank(axis=1, pct=True)
                spot_median_sell_dollar_volume_percentile.columns = [f'{col}cs_spot_median_sell_dollar_volume_percentile_{window}d' for col in spot_median_sell_dollar_volume_percentile.columns]

                # Cross-sectional z-score of median sell dollar volume for each symbol/day
                spot_median_sell_dollar_volume_pivot_mean = spot_median_sell_dollar_volume_pivot.mean(axis=1)
                spot_median_sell_dollar_volume_pivot_std = spot_median_sell_dollar_volume_pivot.std(axis=1)
                spot_median_sell_dollar_volume_zscore = spot_median_sell_dollar_volume_pivot.sub(spot_median_sell_dollar_volume_pivot_mean, axis=0).div(spot_median_sell_dollar_volume_pivot_std, axis=0)
                spot_median_sell_dollar_volume_zscore.columns = [f'{col}cs_spot_median_sell_dollar_volume_zscore_{window}d' for col in spot_median_sell_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional median sell dollar volume
                spot_median_sell_dollar_volume_pivot[f'cs_spot_median_sell_dollar_volume_mean_{window}d'] = spot_median_sell_dollar_volume_pivot.mean(axis=1)
                spot_median_sell_dollar_volume_pivot[f'cs_spot_median_sell_dollar_volume_std_{window}d'] = spot_median_sell_dollar_volume_pivot.std(axis=1)
                spot_median_sell_dollar_volume_pivot[f'cs_spot_median_sell_dollar_volume_skew_{window}d'] = spot_median_sell_dollar_volume_pivot.skew(axis=1)
                spot_median_sell_dollar_volume_pivot[f'cs_spot_median_sell_dollar_volume_kurtosis_{window}d'] = spot_median_sell_dollar_volume_pivot.kurtosis(axis=1)
                spot_median_sell_dollar_volume_pivot[f'cs_spot_median_sell_dollar_volume_median_{window}d'] = spot_median_sell_dollar_volume_pivot.median(axis=1)

                # Add the cross-sectional percentile features to the final features list
                final_features.extend([
                    spot_std_dollar_volume_percentile,
                    spot_std_buy_dollar_volume_percentile,
                    spot_std_sell_dollar_volume_percentile,
                    spot_skewness_dollar_volume_percentile,
                    spot_skewness_buy_dollar_volume_percentile,
                    spot_skewness_sell_dollar_volume_percentile,
                    spot_kurtosis_dollar_volume_percentile,
                    spot_kurtosis_buy_dollar_volume_percentile,
                    spot_kurtosis_sell_dollar_volume_percentile,
                    spot_median_dollar_volume_percentile,
                    spot_median_buy_dollar_volume_percentile,
                    spot_median_sell_dollar_volume_percentile,
                ])

                # Add the cross-sectional z-score features to the final features list
                final_features.extend([
                    spot_std_dollar_volume_zscore,
                    spot_std_buy_dollar_volume_zscore,
                    spot_std_sell_dollar_volume_zscore,
                    spot_skewness_dollar_volume_zscore,
                    spot_skewness_buy_dollar_volume_zscore,
                    spot_skewness_sell_dollar_volume_zscore,
                    spot_kurtosis_dollar_volume_zscore,
                    spot_kurtosis_buy_dollar_volume_zscore,
                    spot_kurtosis_sell_dollar_volume_zscore,
                    spot_median_dollar_volume_zscore,
                    spot_median_buy_dollar_volume_zscore,
                    spot_median_sell_dollar_volume_zscore,
                ])

                # Add the cross-sectional moments features to the final features list
                cs_moments = [
                    spot_std_dollar_volume_pivot,
                    spot_std_buy_dollar_volume_pivot,
                    spot_std_sell_dollar_volume_pivot,
                    spot_skewness_dollar_volume_pivot,
                    spot_skewness_buy_dollar_volume_pivot,
                    spot_skewness_sell_dollar_volume_pivot,
                    spot_kurtosis_dollar_volume_pivot,
                    spot_kurtosis_buy_dollar_volume_pivot,
                    spot_kurtosis_sell_dollar_volume_pivot,
                    spot_median_dollar_volume_pivot,
                    spot_median_buy_dollar_volume_pivot,
                    spot_median_sell_dollar_volume_pivot,
                ]
                # Only keep cross-sectional moments features in the pivot table
                for i, moment in enumerate(cs_moments):
                    valid_cols = [m for m in moment.columns if 'cs_' in m]
                    cs_moments[i] = moment[valid_cols]

                final_features.extend(cs_moments)

            # Futures
            # Cross-sectional total buy dollar volume percentile for each symbol/day
            futures_total_buy_dollar_volume_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'total_buy_dollar_volume_{window}d_futures', dropna=False).sort_index()
            futures_total_buy_dollar_volume_percentile = futures_total_buy_dollar_volume_pivot.rank(axis=1, pct=True)
            futures_total_buy_dollar_volume_percentile.columns = [f'{col}cs_futures_total_buy_dollar_volume_percentile_{window}d' for col in futures_total_buy_dollar_volume_percentile.columns]

            # Cross-sectional z-score of total buy dollar volume for each symbol/day
            futures_total_buy_dollar_volume_pivot_mean = futures_total_buy_dollar_volume_pivot.mean(axis=1)
            futures_total_buy_dollar_volume_pivot_std = futures_total_buy_dollar_volume_pivot.std(axis=1)
            futures_total_buy_dollar_volume_zscore = futures_total_buy_dollar_volume_pivot.sub(futures_total_buy_dollar_volume_pivot_mean, axis=0).div(futures_total_buy_dollar_volume_pivot_std, axis=0)
            futures_total_buy_dollar_volume_zscore.columns = [f'{col}cs_futures_total_buy_dollar_volume_zscore_{window}d' for col in futures_total_buy_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional total buy dollar volume
            futures_total_buy_dollar_volume_pivot[f'cs_futures_total_buy_dollar_volume_mean_{window}d'] = futures_total_buy_dollar_volume_pivot.mean(axis=1)
            futures_total_buy_dollar_volume_pivot[f'cs_futures_total_buy_dollar_volume_std_{window}d'] = futures_total_buy_dollar_volume_pivot.std(axis=1)
            futures_total_buy_dollar_volume_pivot[f'cs_futures_total_buy_dollar_volume_skew_{window}d'] = futures_total_buy_dollar_volume_pivot.skew(axis=1)
            futures_total_buy_dollar_volume_pivot[f'cs_futures_total_buy_dollar_volume_kurtosis_{window}d'] = futures_total_buy_dollar_volume_pivot.kurtosis(axis=1)
            futures_total_buy_dollar_volume_pivot[f'cs_futures_total_buy_dollar_volume_median_{window}d'] = futures_total_buy_dollar_volume_pivot.median(axis=1)

            # Cross-sectional total sell dollar volume percentile for each symbol/day
            futures_total_sell_dollar_volume_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'total_sell_dollar_volume_{window}d_futures', dropna=False).sort_index()
            futures_total_sell_dollar_volume_percentile = futures_total_sell_dollar_volume_pivot.rank(axis=1, pct=True)
            futures_total_sell_dollar_volume_percentile.columns = [f'{col}cs_futures_total_sell_dollar_volume_percentile_{window}d' for col in futures_total_sell_dollar_volume_percentile.columns]

            # Cross-sectional z-score of total sell dollar volume for each symbol/day
            futures_total_sell_dollar_volume_pivot_mean = futures_total_sell_dollar_volume_pivot.mean(axis=1)
            futures_total_sell_dollar_volume_pivot_std = futures_total_sell_dollar_volume_pivot.std(axis=1)
            futures_total_sell_dollar_volume_zscore = futures_total_sell_dollar_volume_pivot.sub(futures_total_sell_dollar_volume_pivot_mean, axis=0).div(futures_total_sell_dollar_volume_pivot_std, axis=0)
            futures_total_sell_dollar_volume_zscore.columns = [f'{col}cs_futures_total_sell_dollar_volume_zscore_{window}d' for col in futures_total_sell_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional total sell dollar volume
            futures_total_sell_dollar_volume_pivot[f'cs_futures_total_sell_dollar_volume_mean_{window}d'] = futures_total_sell_dollar_volume_pivot.mean(axis=1)
            futures_total_sell_dollar_volume_pivot[f'cs_futures_total_sell_dollar_volume_std_{window}d'] = futures_total_sell_dollar_volume_pivot.std(axis=1)
            futures_total_sell_dollar_volume_pivot[f'cs_futures_total_sell_dollar_volume_skew_{window}d'] = futures_total_sell_dollar_volume_pivot.skew(axis=1)
            futures_total_sell_dollar_volume_pivot[f'cs_futures_total_sell_dollar_volume_kurtosis_{window}d'] = futures_total_sell_dollar_volume_pivot.kurtosis(axis=1)
            futures_total_sell_dollar_volume_pivot[f'cs_futures_total_sell_dollar_volume_median_{window}d'] = futures_total_sell_dollar_volume_pivot.median(axis=1)

            # Cross-sectional total dollar volume percentile for each symbol/day
            futures_total_dollar_volume_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'total_dollar_volume_{window}d_futures', dropna=False).sort_index()
            futures_total_dollar_volume_percentile = futures_total_dollar_volume_pivot.rank(axis=1, pct=True)
            futures_total_dollar_volume_percentile.columns = [f'{col}cs_futures_total_dollar_volume_percentile_{window}d' for col in futures_total_dollar_volume_percentile.columns]

            # Cross-sectional z-score of total dollar volume for each symbol/day
            futures_total_dollar_volume_pivot_mean = futures_total_dollar_volume_pivot.mean(axis=1)
            futures_total_dollar_volume_pivot_std = futures_total_dollar_volume_pivot.std(axis=1)
            futures_total_dollar_volume_zscore = futures_total_dollar_volume_pivot.sub(futures_total_dollar_volume_pivot_mean, axis=0).div(futures_total_dollar_volume_pivot_std, axis=0)
            futures_total_dollar_volume_zscore.columns = [f'{col}cs_futures_total_dollar_volume_zscore_{window}d' for col in futures_total_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional total dollar volume
            futures_total_dollar_volume_pivot[f'cs_futures_total_dollar_volume_mean_{window}d'] = futures_total_dollar_volume_pivot.mean(axis=1)
            futures_total_dollar_volume_pivot[f'cs_futures_total_dollar_volume_std_{window}d'] = futures_total_dollar_volume_pivot.std(axis=1)
            futures_total_dollar_volume_pivot[f'cs_futures_total_dollar_volume_skew_{window}d'] = futures_total_dollar_volume_pivot.skew(axis=1)
            futures_total_dollar_volume_pivot[f'cs_futures_total_dollar_volume_kurtosis_{window}d'] = futures_total_dollar_volume_pivot.kurtosis(axis=1)
            futures_total_dollar_volume_pivot[f'cs_futures_total_dollar_volume_median_{window}d'] = futures_total_dollar_volume_pivot.median(axis=1)

            # Cross-sectional number of buys percentile for each symbol/day
            futures_num_buys_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'num_buys_{window}d_futures', dropna=False).sort_index()
            futures_num_buys_percentile = futures_num_buys_pivot.rank(axis=1, pct=True)
            futures_num_buys_percentile.columns = [f'{col}cs_futures_num_buys_percentile_{window}d' for col in futures_num_buys_percentile.columns]

            # Cross-sectional z-score of number of buys for each symbol/day
            futures_num_buys_pivot_mean = futures_num_buys_pivot.mean(axis=1)
            futures_num_buys_pivot_std = futures_num_buys_pivot.std(axis=1)
            futures_num_buys_zscore = futures_num_buys_pivot.sub(futures_num_buys_pivot_mean, axis=0).div(futures_num_buys_pivot_std, axis=0)
            futures_num_buys_zscore.columns = [f'{col}cs_futures_num_buys_zscore_{window}d' for col in futures_num_buys_zscore.columns]

            # 4 moments of cross-sectional number of buys
            futures_num_buys_pivot[f'cs_futures_num_buys_mean_{window}d'] = futures_num_buys_pivot.mean(axis=1)
            futures_num_buys_pivot[f'cs_futures_num_buys_std_{window}d'] = futures_num_buys_pivot.std(axis=1)
            futures_num_buys_pivot[f'cs_futures_num_buys_skew_{window}d'] = futures_num_buys_pivot.skew(axis=1)
            futures_num_buys_pivot[f'cs_futures_num_buys_kurtosis_{window}d'] = futures_num_buys_pivot.kurtosis(axis=1)
            futures_num_buys_pivot[f'cs_futures_num_buys_median_{window}d'] = futures_num_buys_pivot.median(axis=1)
            
            # Cross-sectional number of sells percentile for each symbol/day
            futures_num_sells_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'num_sells_{window}d_futures', dropna=False).sort_index()
            futures_num_sells_percentile = futures_num_sells_pivot.rank(axis=1, pct=True)
            futures_num_sells_percentile.columns = [f'{col}cs_futures_num_sells_percentile_{window}d' for col in futures_num_sells_percentile.columns]

            # Cross-sectional z-score of number of sells for each symbol/day
            futures_num_sells_pivot_mean = futures_num_sells_pivot.mean(axis=1)
            futures_num_sells_pivot_std = futures_num_sells_pivot.std(axis=1)
            futures_num_sells_zscore = futures_num_sells_pivot.sub(futures_num_sells_pivot_mean, axis=0).div(futures_num_sells_pivot_std, axis=0)
            futures_num_sells_zscore.columns = [f'{col}cs_futures_num_sells_zscore_{window}d' for col in futures_num_sells_zscore.columns]

            # 4 moments of cross-sectional number of sells
            futures_num_sells_pivot[f'cs_futures_num_sells_mean_{window}d'] = futures_num_sells_pivot.mean(axis=1)
            futures_num_sells_pivot[f'cs_futures_num_sells_std_{window}d'] = futures_num_sells_pivot.std(axis=1)
            futures_num_sells_pivot[f'cs_futures_num_sells_skew_{window}d'] = futures_num_sells_pivot.skew(axis=1)
            futures_num_sells_pivot[f'cs_futures_num_sells_kurtosis_{window}d'] = futures_num_sells_pivot.kurtosis(axis=1)
            futures_num_sells_pivot[f'cs_futures_num_sells_median_{window}d'] = futures_num_sells_pivot.median(axis=1)

            # Cross-sectional percentage buys percentile for each symbol/day
            futures_pct_buys_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'pct_buys_{window}d_futures', dropna=False).sort_index()
            futures_pct_buys_percentile = futures_pct_buys_pivot.rank(axis=1, pct=True)
            futures_pct_buys_percentile.columns = [f'{col}cs_futures_pct_buys_percentile_{window}d' for col in futures_pct_buys_percentile.columns]

            # Cross-sectional z-score of percentage buys for each symbol/day
            futures_pct_buys_pivot_mean = futures_pct_buys_pivot.mean(axis=1)
            futures_pct_buys_pivot_std = futures_pct_buys_pivot.std(axis=1)
            futures_pct_buys_zscore = futures_pct_buys_pivot.sub(futures_pct_buys_pivot_mean, axis=0).div(futures_pct_buys_pivot_std, axis=0)
            futures_pct_buys_zscore.columns = [f'{col}cs_futures_pct_buys_zscore_{window}d' for col in futures_pct_buys_zscore.columns]

            # 4 moments of cross-sectional percentage buys
            futures_pct_buys_pivot[f'cs_futures_pct_buys_mean_{window}d'] = futures_pct_buys_pivot.mean(axis=1)
            futures_pct_buys_pivot[f'cs_futures_pct_buys_std_{window}d'] = futures_pct_buys_pivot.std(axis=1)
            futures_pct_buys_pivot[f'cs_futures_pct_buys_skew_{window}d'] = futures_pct_buys_pivot.skew(axis=1)
            futures_pct_buys_pivot[f'cs_futures_pct_buys_kurtosis_{window}d'] = futures_pct_buys_pivot.kurtosis(axis=1)
            futures_pct_buys_pivot[f'cs_futures_pct_buys_median_{window}d'] = futures_pct_buys_pivot.median(axis=1)

            # Cross-sectional percentage sells percentile for each symbol/day
            futures_pct_sells_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'pct_sells_{window}d_futures', dropna=False).sort_index()
            futures_pct_sells_percentile = futures_pct_sells_pivot.rank(axis=1, pct=True)
            futures_pct_sells_percentile.columns = [f'{col}cs_futures_pct_sells_percentile_{window}d' for col in futures_pct_sells_percentile.columns]

            # Cross-sectional z-score of percentage sells for each symbol/day
            futures_pct_sells_pivot_mean = futures_pct_sells_pivot.mean(axis=1)
            futures_pct_sells_pivot_std = futures_pct_sells_pivot.std(axis=1)
            futures_pct_sells_zscore = futures_pct_sells_pivot.sub(futures_pct_sells_pivot_mean, axis=0).div(futures_pct_sells_pivot_std, axis=0)
            futures_pct_sells_zscore.columns = [f'{col}cs_futures_pct_sells_zscore_{window}d' for col in futures_pct_sells_zscore.columns]

            # 4 moments of cross-sectional percentage sells
            futures_pct_sells_pivot[f'cs_futures_pct_sells_mean_{window}d'] = futures_pct_sells_pivot.mean(axis=1)
            futures_pct_sells_pivot[f'cs_futures_pct_sells_std_{window}d'] = futures_pct_sells_pivot.std(axis=1)
            futures_pct_sells_pivot[f'cs_futures_pct_sells_skew_{window}d'] = futures_pct_sells_pivot.skew(axis=1)
            futures_pct_sells_pivot[f'cs_futures_pct_sells_kurtosis_{window}d'] = futures_pct_sells_pivot.kurtosis(axis=1)
            futures_pct_sells_pivot[f'cs_futures_pct_sells_median_{window}d'] = futures_pct_sells_pivot.median(axis=1)

            # Cross-sectional trade imbalance percentile for each symbol/day
            futures_trade_dollar_volume_imbalance_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'trade_imbalance_{window}d_futures', dropna=False).sort_index()
            futures_trade_dollar_volume_imbalance_percentile = futures_trade_dollar_volume_imbalance_pivot.rank(axis=1, pct=True)
            futures_trade_dollar_volume_imbalance_percentile.columns = [f'{col}cs_futures_trade_dollar_volume_imbalance_percentile_{window}d' for col in futures_trade_dollar_volume_imbalance_percentile.columns]

            # Cross-sectional z-score of trade imbalance for each symbol/day
            futures_trade_dollar_volume_imbalance_pivot_mean = futures_trade_dollar_volume_imbalance_pivot.mean(axis=1)
            futures_trade_dollar_volume_imbalance_pivot_std = futures_trade_dollar_volume_imbalance_pivot.std(axis=1)
            futures_trade_dollar_volume_imbalance_zscore = futures_trade_dollar_volume_imbalance_pivot.sub(futures_trade_dollar_volume_imbalance_pivot_mean, axis=0).div(futures_trade_dollar_volume_imbalance_pivot_std, axis=0)
            futures_trade_dollar_volume_imbalance_zscore.columns = [f'{col}cs_futures_trade_dollar_volume_imbalance_zscore_{window}d' for col in futures_trade_dollar_volume_imbalance_zscore.columns]

            # 4 moments of cross-sectional trade imbalance
            futures_trade_dollar_volume_imbalance_pivot[f'cs_futures_trade_dollar_volume_imbalance_mean_{window}d'] = futures_trade_dollar_volume_imbalance_pivot.mean(axis=1)
            futures_trade_dollar_volume_imbalance_pivot[f'cs_futures_trade_dollar_volume_imbalance_std_{window}d'] = futures_trade_dollar_volume_imbalance_pivot.std(axis=1)
            futures_trade_dollar_volume_imbalance_pivot[f'cs_futures_trade_dollar_volume_imbalance_skew_{window}d'] = futures_trade_dollar_volume_imbalance_pivot.skew(axis=1)
            futures_trade_dollar_volume_imbalance_pivot[f'cs_futures_trade_dollar_volume_imbalance_kurtosis_{window}d'] = futures_trade_dollar_volume_imbalance_pivot.kurtosis(axis=1)
            futures_trade_dollar_volume_imbalance_pivot[f'cs_futures_trade_dollar_volume_imbalance_median_{window}d'] = futures_trade_dollar_volume_imbalance_pivot.median(axis=1)

            # Cross-sectional percentage buy dollar volume percentile for each symbol/day
            futures_pct_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id',values=f'pct_buy_dollar_volume_{window}d_futures', dropna=False).sort_index()
            futures_pct_buy_dollar_volume_percentile = futures_pct_buy_dollar_volume_pivot.rank(axis=1, pct=True)
            futures_pct_buy_dollar_volume_percentile.columns = [f'{col}cs_futures_pct_buy_dollar_volume_percentile_{window}d' for col in futures_pct_buy_dollar_volume_percentile.columns]

            # Cross-sectional z-score of percentage buy dollar volume for each symbol/day
            futures_pct_buy_dollar_volume_pivot_mean = futures_pct_buy_dollar_volume_pivot.mean(axis=1)
            futures_pct_buy_dollar_volume_pivot_std = futures_pct_buy_dollar_volume_pivot.std(axis=1)
            futures_pct_buy_dollar_volume_zscore = futures_pct_buy_dollar_volume_pivot.sub(futures_pct_buy_dollar_volume_pivot_mean, axis=0).div(futures_pct_buy_dollar_volume_pivot_std, axis=0)
            futures_pct_buy_dollar_volume_zscore.columns = [f'{col}cs_futures_pct_buy_dollar_volume_zscore_{window}d' for col in futures_pct_buy_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional percentage buy dollar volume
            futures_pct_buy_dollar_volume_pivot[f'cs_futures_pct_buy_dollar_volume_mean_{window}d'] = futures_pct_buy_dollar_volume_pivot.mean(axis=1)
            futures_pct_buy_dollar_volume_pivot[f'cs_futures_pct_buy_dollar_volume_std_{window}d'] = futures_pct_buy_dollar_volume_pivot.std(axis=1)
            futures_pct_buy_dollar_volume_pivot[f'cs_futures_pct_buy_dollar_volume_skew_{window}d'] = futures_pct_buy_dollar_volume_pivot.skew(axis=1)
            futures_pct_buy_dollar_volume_pivot[f'cs_futures_pct_buy_dollar_volume_kurtosis_{window}d'] = futures_pct_buy_dollar_volume_pivot.kurtosis(axis=1)
            futures_pct_buy_dollar_volume_pivot[f'cs_futures_pct_buy_dollar_volume_median_{window}d'] = futures_pct_buy_dollar_volume_pivot.median(axis=1)

            # Cross-sectional percentage sell dollar volume percentile for each symbol/day
            futures_pct_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id',values=f'pct_sell_dollar_volume_{window}d_futures', dropna=False).sort_index()
            futures_pct_sell_dollar_volume_percentile = futures_pct_sell_dollar_volume_pivot.rank(axis=1, pct=True)
            futures_pct_sell_dollar_volume_percentile.columns = [f'{col}cs_futures_pct_sell_dollar_volume_percentile_{window}d' for col in futures_pct_sell_dollar_volume_percentile.columns]

            # Cross-sectional z-score of percentage sell dollar volume for each symbol/day
            futures_pct_sell_dollar_volume_pivot_mean = futures_pct_sell_dollar_volume_pivot.mean(axis=1)
            futures_pct_sell_dollar_volume_pivot_std = futures_pct_sell_dollar_volume_pivot.std(axis=1)
            futures_pct_sell_dollar_volume_zscore = futures_pct_sell_dollar_volume_pivot.sub(futures_pct_sell_dollar_volume_pivot_mean, axis=0).div(futures_pct_sell_dollar_volume_pivot_std, axis=0)
            futures_pct_sell_dollar_volume_zscore.columns = [f'{col}cs_futures_pct_sell_dollar_volume_zscore_{window}d' for col in futures_pct_sell_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional percentage sell dollar volume
            futures_pct_sell_dollar_volume_pivot[f'cs_futures_pct_sell_dollar_volume_mean_{window}d'] = futures_pct_sell_dollar_volume_pivot.mean(axis=1)
            futures_pct_sell_dollar_volume_pivot[f'cs_futures_pct_sell_dollar_volume_std_{window}d'] = futures_pct_sell_dollar_volume_pivot.std(axis=1)
            futures_pct_sell_dollar_volume_pivot[f'cs_futures_pct_sell_dollar_volume_skew_{window}d'] = futures_pct_sell_dollar_volume_pivot.skew(axis=1)
            futures_pct_sell_dollar_volume_pivot[f'cs_futures_pct_sell_dollar_volume_kurtosis_{window}d'] = futures_pct_sell_dollar_volume_pivot.kurtosis(axis=1)
            futures_pct_sell_dollar_volume_pivot[f'cs_futures_pct_sell_dollar_volume_median_{window}d'] = futures_pct_sell_dollar_volume_pivot.median(axis=1)

            for lookback_2 in self.lookback_windows:
                # Ema of cross-sectional total buy dollar volume for regime detection
                futures_total_buy_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_futures_total_buy_dollar_volume_mean_{window}d'] = futures_total_buy_dollar_volume_pivot[f'cs_futures_total_buy_dollar_volume_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional total sell dollar volume for regime detection
                futures_total_sell_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_futures_total_sell_dollar_volume_mean_{window}d'] = futures_total_sell_dollar_volume_pivot[f'cs_futures_total_sell_dollar_volume_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional total dollar volume for regime detection
                futures_total_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_futures_total_dollar_volume_mean_{window}d'] = futures_total_dollar_volume_pivot[f'cs_futures_total_dollar_volume_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional number of buys for regime detection
                futures_num_buys_pivot[f'rolling_ema_{lookback_2}_cs_futures_num_buys_mean_{window}d'] = futures_num_buys_pivot[f'cs_futures_num_buys_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional number of sells for regime detection
                futures_num_sells_pivot[f'rolling_ema_{lookback_2}_cs_futures_num_sells_mean_{window}d'] = futures_num_sells_pivot[f'cs_futures_num_sells_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional percentage buys for regime detection
                futures_pct_buys_pivot[f'rolling_ema_{lookback_2}_cs_futures_pct_buys_mean_{window}d'] = futures_pct_buys_pivot[f'cs_futures_pct_buys_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional percentage sells for regime detection
                futures_pct_sells_pivot[f'rolling_ema_{lookback_2}_cs_futures_pct_sells_mean_{window}d'] = futures_pct_sells_pivot[f'cs_futures_pct_sells_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional trade imbalance for regime detection
                futures_trade_dollar_volume_imbalance_pivot[f'rolling_ema_{lookback_2}_cs_futures_trade_dollar_volume_imbalance_mean_{window}d'] = futures_trade_dollar_volume_imbalance_pivot[f'cs_futures_trade_dollar_volume_imbalance_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional percentage buy dollar volume for regime detection
                futures_pct_buy_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_futures_pct_buy_dollar_volume_mean_{window}d'] = futures_pct_buy_dollar_volume_pivot[f'cs_futures_pct_buy_dollar_volume_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()
                # Ema of cross-sectional percentage sell dollar volume for regime detection
                futures_pct_sell_dollar_volume_pivot[f'rolling_ema_{lookback_2}_cs_futures_pct_sell_dollar_volume_mean_{window}d'] = futures_pct_sell_dollar_volume_pivot[f'cs_futures_pct_sell_dollar_volume_mean_{window}d'].ewm(span = lookback_2, min_periods = 7).mean()

                futures_total_buy_dollar_volume_pivot_ema = futures_total_buy_dollar_volume_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_total_buy_dollar_volume_mean_{window}d')
                futures_total_sell_dollar_volume_pivot_ema = futures_total_sell_dollar_volume_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_total_sell_dollar_volume_mean_{window}d')
                futures_total_dollar_volume_pivot_ema = futures_total_dollar_volume_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_total_dollar_volume_mean_{window}d')
                futures_num_buys_pivot_ema = futures_num_buys_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_num_buys_mean_{window}d')
                futures_num_sells_pivot_ema = futures_num_sells_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_num_sells_mean_{window}d')
                futures_pct_buys_pivot_ema = futures_pct_buys_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_pct_buys_mean_{window}d')
                futures_pct_sells_pivot_ema = futures_pct_sells_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_pct_sells_mean_{window}d')
                futures_trade_dollar_volume_imbalance_pivot_ema = futures_trade_dollar_volume_imbalance_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_trade_dollar_volume_imbalance_mean_{window}d')
                futures_pct_buy_dollar_volume_pivot_ema = futures_pct_buy_dollar_volume_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_pct_buy_dollar_volume_mean_{window}d')
                futures_pct_sell_dollar_volume_pivot_ema = futures_pct_sell_dollar_volume_pivot.filter(like = f'rolling_ema_{lookback_2}_cs_futures_pct_sell_dollar_volume_mean_{window}d')

                # Add the ema features to the final features list
                final_features.append(futures_total_buy_dollar_volume_pivot_ema)
                final_features.append(futures_total_sell_dollar_volume_pivot_ema)
                final_features.append(futures_total_dollar_volume_pivot_ema)
                final_features.append(futures_num_buys_pivot_ema)
                final_features.append(futures_num_sells_pivot_ema)
                final_features.append(futures_pct_buys_pivot_ema)
                final_features.append(futures_pct_sells_pivot_ema)
                final_features.append(futures_trade_dollar_volume_imbalance_pivot_ema)
                final_features.append(futures_pct_buy_dollar_volume_pivot_ema)
                final_features.append(futures_pct_sell_dollar_volume_pivot_ema)

            # Cross-sectional avg dollar volume percentile for each symbol/day
            futures_avg_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'avg_dollar_volume_{window}d_futures', dropna=False).sort_index()
            futures_avg_dollar_volume_percentile = futures_avg_dollar_volume_pivot.rank(axis=1, pct=True)
            futures_avg_dollar_volume_percentile.columns = [f'{col}cs_futures_avg_dollar_volume_percentile_{window}d' for col in futures_avg_dollar_volume_percentile.columns]

            # Cross-sectional z-score of avg dollar volume for each symbol/day
            futures_avg_dollar_volume_pivot_mean = futures_avg_dollar_volume_pivot.mean(axis=1)
            futures_avg_dollar_volume_pivot_std = futures_avg_dollar_volume_pivot.std(axis=1)
            futures_avg_dollar_volume_zscore = futures_avg_dollar_volume_pivot.sub(futures_avg_dollar_volume_pivot_mean, axis=0).div(futures_avg_dollar_volume_pivot_std, axis=0)
            futures_avg_dollar_volume_zscore.columns = [f'{col}cs_futures_avg_dollar_volume_zscore_{window}d' for col in futures_avg_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional avg dollar volume
            futures_avg_dollar_volume_pivot[f'cs_futures_avg_dollar_volume_mean_{window}d'] = futures_avg_dollar_volume_pivot.mean(axis=1)
            futures_avg_dollar_volume_pivot[f'cs_futures_avg_dollar_volume_std_{window}d'] = futures_avg_dollar_volume_pivot.std(axis=1)
            futures_avg_dollar_volume_pivot[f'cs_futures_avg_dollar_volume_skew_{window}d'] = futures_avg_dollar_volume_pivot.skew(axis=1)
            futures_avg_dollar_volume_pivot[f'cs_futures_avg_dollar_volume_kurtosis_{window}d'] = futures_avg_dollar_volume_pivot.kurtosis(axis=1)
            futures_avg_dollar_volume_pivot[f'cs_futures_avg_dollar_volume_median_{window}d'] = futures_avg_dollar_volume_pivot.median(axis=1)

            # Cross-sectional avg buy dollar volume percentile for each symbol/day
            futures_avg_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'avg_buy_dollar_volume_{window}d_futures', dropna=False).sort_index()
            futures_avg_buy_dollar_volume_percentile = futures_avg_buy_dollar_volume_pivot.rank(axis=1, pct=True)
            futures_avg_buy_dollar_volume_percentile.columns = [f'{col}cs_futures_avg_buy_dollar_volume_percentile_{window}d' for col in futures_avg_buy_dollar_volume_percentile.columns]

            # Cross-sectional z-score of avg buy dollar volume for each symbol/day
            futures_avg_buy_dollar_volume_pivot_mean = futures_avg_buy_dollar_volume_pivot.mean(axis=1)
            futures_avg_buy_dollar_volume_pivot_std = futures_avg_buy_dollar_volume_pivot.std(axis=1)
            futures_avg_buy_dollar_volume_zscore = futures_avg_buy_dollar_volume_pivot.sub(futures_avg_buy_dollar_volume_pivot_mean, axis=0).div(futures_avg_buy_dollar_volume_pivot_std, axis=0)
            futures_avg_buy_dollar_volume_zscore.columns = [f'{col}cs_futures_avg_buy_dollar_volume_zscore_{window}d' for col in futures_avg_buy_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional avg buy dollar volume
            futures_avg_buy_dollar_volume_pivot[f'cs_futures_avg_buy_dollar_volume_mean_{window}d'] = futures_avg_buy_dollar_volume_pivot.mean(axis=1)
            futures_avg_buy_dollar_volume_pivot[f'cs_futures_avg_buy_dollar_volume_std_{window}d'] = futures_avg_buy_dollar_volume_pivot.std(axis=1)
            futures_avg_buy_dollar_volume_pivot[f'cs_futures_avg_buy_dollar_volume_skew_{window}d'] = futures_avg_buy_dollar_volume_pivot.skew(axis=1)
            futures_avg_buy_dollar_volume_pivot[f'cs_futures_avg_buy_dollar_volume_kurtosis_{window}d'] = futures_avg_buy_dollar_volume_pivot.kurtosis(axis=1)
            futures_avg_buy_dollar_volume_pivot[f'cs_futures_avg_buy_dollar_volume_median_{window}d'] = futures_avg_buy_dollar_volume_pivot.median(axis=1)

            # Cross-sectional avg sell dollar volume percentile for each symbol/day
            futures_avg_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'avg_sell_dollar_volume_{window}d_futures', dropna=False).sort_index()
            futures_avg_sell_dollar_volume_percentile = futures_avg_sell_dollar_volume_pivot.rank(axis=1, pct=True)
            futures_avg_sell_dollar_volume_percentile.columns = [f'{col}cs_futures_avg_sell_dollar_volume_percentile_{window}d' for col in futures_avg_sell_dollar_volume_percentile.columns]

            # Cross-sectional z-score of avg sell dollar volume for each symbol/day
            futures_avg_sell_dollar_volume_pivot_mean = futures_avg_sell_dollar_volume_pivot.mean(axis=1)
            futures_avg_sell_dollar_volume_pivot_std = futures_avg_sell_dollar_volume_pivot.std(axis=1)
            futures_avg_sell_dollar_volume_zscore = futures_avg_sell_dollar_volume_pivot.sub(futures_avg_sell_dollar_volume_pivot_mean, axis=0).div(futures_avg_sell_dollar_volume_pivot_std, axis=0)
            futures_avg_sell_dollar_volume_zscore.columns = [f'{col}cs_futures_avg_sell_dollar_volume_zscore_{window}d' for col in futures_avg_sell_dollar_volume_zscore.columns]

            # 4 moments of cross-sectional avg sell dollar volume
            futures_avg_sell_dollar_volume_pivot[f'cs_futures_avg_sell_dollar_volume_mean_{window}d'] = futures_avg_sell_dollar_volume_pivot.mean(axis=1)
            futures_avg_sell_dollar_volume_pivot[f'cs_futures_avg_sell_dollar_volume_std_{window}d'] = futures_avg_sell_dollar_volume_pivot.std(axis=1)
            futures_avg_sell_dollar_volume_pivot[f'cs_futures_avg_sell_dollar_volume_skew_{window}d'] = futures_avg_sell_dollar_volume_pivot.skew(axis=1)
            futures_avg_sell_dollar_volume_pivot[f'cs_futures_avg_sell_dollar_volume_kurtosis_{window}d'] = futures_avg_sell_dollar_volume_pivot.kurtosis(axis=1)
            futures_avg_sell_dollar_volume_pivot[f'cs_futures_avg_sell_dollar_volume_median_{window}d'] = futures_avg_sell_dollar_volume_pivot.median(axis=1)

            if window == 1:
                # Cross-sectional std dollar volume percentile for each symbol/day
                futures_std_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'std_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_std_dollar_volume_percentile = futures_std_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_std_dollar_volume_percentile.columns = [f'{col}cs_futures_std_dollar_volume_percentile_{window}d' for col in futures_std_dollar_volume_percentile.columns]

                # Cross-sectional z-score of std dollar volume for each symbol/day
                futures_std_dollar_volume_pivot_mean = futures_std_dollar_volume_pivot.mean(axis=1)
                futures_std_dollar_volume_pivot_std = futures_std_dollar_volume_pivot.std(axis=1)
                futures_std_dollar_volume_zscore = futures_std_dollar_volume_pivot.sub(futures_std_dollar_volume_pivot_mean, axis=0).div(futures_std_dollar_volume_pivot_std, axis=0)
                futures_std_dollar_volume_zscore.columns = [f'{col}cs_futures_std_dollar_volume_zscore_{window}d' for col in futures_std_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional std dollar volume
                futures_std_dollar_volume_pivot[f'cs_futures_std_dollar_volume_mean_{window}d'] = futures_std_dollar_volume_pivot.mean(axis=1)
                futures_std_dollar_volume_pivot[f'cs_futures_std_dollar_volume_std_{window}d'] = futures_std_dollar_volume_pivot.std(axis=1)
                futures_std_dollar_volume_pivot[f'cs_futures_std_dollar_volume_skew_{window}d'] = futures_std_dollar_volume_pivot.skew(axis=1)
                futures_std_dollar_volume_pivot[f'cs_futures_std_dollar_volume_kurtosis_{window}d'] = futures_std_dollar_volume_pivot.kurtosis(axis=1)
                futures_std_dollar_volume_pivot[f'cs_futures_std_dollar_volume_median_{window}d'] = futures_std_dollar_volume_pivot.median(axis=1)

                # Cross-sectional std buy dollar volume percentile for each symbol/day
                futures_std_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'std_buy_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_std_buy_dollar_volume_percentile = futures_std_buy_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_std_buy_dollar_volume_percentile.columns = [f'{col}cs_futures_std_buy_dollar_volume_percentile_{window}d' for col in futures_std_buy_dollar_volume_percentile.columns]

                # Cross-sectional z-score of std buy dollar volume for each symbol/day
                futures_std_buy_dollar_volume_pivot_mean = futures_std_buy_dollar_volume_pivot.mean(axis=1)
                futures_std_buy_dollar_volume_pivot_std = futures_std_buy_dollar_volume_pivot.std(axis=1)
                futures_std_buy_dollar_volume_zscore = futures_std_buy_dollar_volume_pivot.sub(futures_std_buy_dollar_volume_pivot_mean, axis=0).div(futures_std_buy_dollar_volume_pivot_std, axis=0)
                futures_std_buy_dollar_volume_zscore.columns = [f'{col}cs_futures_std_buy_dollar_volume_zscore_{window}d' for col in futures_std_buy_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional std buy dollar volume
                futures_std_buy_dollar_volume_pivot[f'cs_futures_std_buy_dollar_volume_mean_{window}d'] = futures_std_buy_dollar_volume_pivot.mean(axis=1)
                futures_std_buy_dollar_volume_pivot[f'cs_futures_std_buy_dollar_volume_std_{window}d'] = futures_std_buy_dollar_volume_pivot.std(axis=1)
                futures_std_buy_dollar_volume_pivot[f'cs_futures_std_buy_dollar_volume_skew_{window}d'] = futures_std_buy_dollar_volume_pivot.skew(axis=1)
                futures_std_buy_dollar_volume_pivot[f'cs_futures_std_buy_dollar_volume_kurtosis_{window}d'] = futures_std_buy_dollar_volume_pivot.kurtosis(axis=1)
                futures_std_buy_dollar_volume_pivot[f'cs_futures_std_buy_dollar_volume_median_{window}d'] = futures_std_buy_dollar_volume_pivot.median(axis=1)

                # Cross-sectional std sell dollar volume percentile for each symbol/day
                futures_std_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'std_sell_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_std_sell_dollar_volume_percentile = futures_std_sell_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_std_sell_dollar_volume_percentile.columns = [f'{col}cs_futures_std_sell_dollar_volume_percentile_{window}d' for col in futures_std_sell_dollar_volume_percentile.columns]

                # Cross-sectional z-score of std sell dollar volume for each symbol/day
                futures_std_sell_dollar_volume_pivot_mean = futures_std_sell_dollar_volume_pivot.mean(axis=1)
                futures_std_sell_dollar_volume_pivot_std = futures_std_sell_dollar_volume_pivot.std(axis=1)
                futures_std_sell_dollar_volume_zscore = futures_std_sell_dollar_volume_pivot.sub(futures_std_sell_dollar_volume_pivot_mean, axis=0).div(futures_std_sell_dollar_volume_pivot_std, axis=0)
                futures_std_sell_dollar_volume_zscore.columns = [f'{col}cs_futures_std_sell_dollar_volume_zscore_{window}d' for col in futures_std_sell_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional std sell dollar volume
                futures_std_sell_dollar_volume_pivot[f'cs_futures_std_sell_dollar_volume_mean_{window}d'] = futures_std_sell_dollar_volume_pivot.mean(axis=1)
                futures_std_sell_dollar_volume_pivot[f'cs_futures_std_sell_dollar_volume_std_{window}d'] = futures_std_sell_dollar_volume_pivot.std(axis=1)
                futures_std_sell_dollar_volume_pivot[f'cs_futures_std_sell_dollar_volume_skew_{window}d'] = futures_std_sell_dollar_volume_pivot.skew(axis=1)
                futures_std_sell_dollar_volume_pivot[f'cs_futures_std_sell_dollar_volume_kurtosis_{window}d'] = futures_std_sell_dollar_volume_pivot.kurtosis(axis=1)
                futures_std_sell_dollar_volume_pivot[f'cs_futures_std_sell_dollar_volume_median_{window}d'] = futures_std_sell_dollar_volume_pivot.median(axis=1)

                # Cross-sectional skewness dollar volume percentile for each symbol/day
                futures_skewness_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'skew_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_skewness_dollar_volume_percentile = futures_skewness_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_skewness_dollar_volume_percentile.columns = [f'{col}cs_futures_skewness_dollar_volume_percentile_{window}d' for col in futures_skewness_dollar_volume_percentile.columns]

                # Cross-sectional z-score of skewness dollar volume for each symbol/day
                futures_skewness_dollar_volume_pivot_mean = futures_skewness_dollar_volume_pivot.mean(axis=1)
                futures_skewness_dollar_volume_pivot_std = futures_skewness_dollar_volume_pivot.std(axis=1)
                futures_skewness_dollar_volume_zscore = futures_skewness_dollar_volume_pivot.sub(futures_skewness_dollar_volume_pivot_mean, axis=0).div(futures_skewness_dollar_volume_pivot_std, axis=0)
                futures_skewness_dollar_volume_zscore.columns = [f'{col}cs_futures_skewness_dollar_volume_zscore_{window}d' for col in futures_skewness_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional skewness dollar volume
                futures_skewness_dollar_volume_pivot[f'cs_futures_skewness_dollar_volume_mean_{window}d'] = futures_skewness_dollar_volume_pivot.mean(axis=1)
                futures_skewness_dollar_volume_pivot[f'cs_futures_skewness_dollar_volume_std_{window}d'] = futures_skewness_dollar_volume_pivot.std(axis=1)
                futures_skewness_dollar_volume_pivot[f'cs_futures_skewness_dollar_volume_skew_{window}d'] = futures_skewness_dollar_volume_pivot.skew(axis=1)
                futures_skewness_dollar_volume_pivot[f'cs_futures_skewness_dollar_volume_kurtosis_{window}d'] = futures_skewness_dollar_volume_pivot.kurtosis(axis=1)
                futures_skewness_dollar_volume_pivot[f'cs_futures_skewness_dollar_volume_median_{window}d'] = futures_skewness_dollar_volume_pivot.median(axis=1)

                # Cross-sectional skewness buy dollar volume percentile for each symbol/day
                futures_skewness_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'skew_buy_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_skewness_buy_dollar_volume_percentile = futures_skewness_buy_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_skewness_buy_dollar_volume_percentile.columns = [f'{col}cs_futures_skewness_buy_dollar_volume_percentile_{window}d' for col in futures_skewness_buy_dollar_volume_percentile.columns]

                # Cross-sectional z-score of skewness buy dollar volume for each symbol/day
                futures_skewness_buy_dollar_volume_pivot_mean = futures_skewness_buy_dollar_volume_pivot.mean(axis=1)
                futures_skewness_buy_dollar_volume_pivot_std = futures_skewness_buy_dollar_volume_pivot.std(axis=1)
                futures_skewness_buy_dollar_volume_zscore = futures_skewness_buy_dollar_volume_pivot.sub(futures_skewness_buy_dollar_volume_pivot_mean, axis=0).div(futures_skewness_buy_dollar_volume_pivot_std, axis=0)
                futures_skewness_buy_dollar_volume_zscore.columns = [f'{col}cs_futures_skewness_buy_dollar_volume_zscore_{window}d' for col in futures_skewness_buy_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional skewness buy dollar volume
                futures_skewness_buy_dollar_volume_pivot[f'cs_futures_skewness_buy_dollar_volume_mean_{window}d'] = futures_skewness_buy_dollar_volume_pivot.mean(axis=1)
                futures_skewness_buy_dollar_volume_pivot[f'cs_futures_skewness_buy_dollar_volume_std_{window}d'] = futures_skewness_buy_dollar_volume_pivot.std(axis=1)
                futures_skewness_buy_dollar_volume_pivot[f'cs_futures_skewness_buy_dollar_volume_skew_{window}d'] = futures_skewness_buy_dollar_volume_pivot.skew(axis=1)
                futures_skewness_buy_dollar_volume_pivot[f'cs_futures_skewness_buy_dollar_volume_kurtosis_{window}d'] = futures_skewness_buy_dollar_volume_pivot.kurtosis(axis=1)
                futures_skewness_buy_dollar_volume_pivot[f'cs_futures_skewness_buy_dollar_volume_median_{window}d'] = futures_skewness_buy_dollar_volume_pivot.median(axis=1)

                # Cross-sectional skewness sell dollar volume percentile for each symbol/day
                futures_skewness_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'skew_sell_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_skewness_sell_dollar_volume_percentile = futures_skewness_sell_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_skewness_sell_dollar_volume_percentile.columns = [f'{col}cs_futures_skewness_sell_dollar_volume_percentile_{window}d' for col in futures_skewness_sell_dollar_volume_percentile.columns]

                # Cross-sectional z-score of skewness sell dollar volume for each symbol/day
                futures_skewness_sell_dollar_volume_pivot_mean = futures_skewness_sell_dollar_volume_pivot.mean(axis=1)
                futures_skewness_sell_dollar_volume_pivot_std = futures_skewness_sell_dollar_volume_pivot.std(axis=1)
                futures_skewness_sell_dollar_volume_zscore = futures_skewness_sell_dollar_volume_pivot.sub(futures_skewness_sell_dollar_volume_pivot_mean, axis=0).div(futures_skewness_sell_dollar_volume_pivot_std, axis=0)
                futures_skewness_sell_dollar_volume_zscore.columns = [f'{col}cs_futures_skewness_sell_dollar_volume_zscore_{window}d' for col in futures_skewness_sell_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional skewness sell dollar volume
                futures_skewness_sell_dollar_volume_pivot[f'cs_futures_skewness_sell_dollar_volume_mean_{window}d'] = futures_skewness_sell_dollar_volume_pivot.mean(axis=1)
                futures_skewness_sell_dollar_volume_pivot[f'cs_futures_skewness_sell_dollar_volume_std_{window}d'] = futures_skewness_sell_dollar_volume_pivot.std(axis=1)
                futures_skewness_sell_dollar_volume_pivot[f'cs_futures_skewness_sell_dollar_volume_skew_{window}d'] = futures_skewness_sell_dollar_volume_pivot.skew(axis=1)
                futures_skewness_sell_dollar_volume_pivot[f'cs_futures_skewness_sell_dollar_volume_kurtosis_{window}d'] = futures_skewness_sell_dollar_volume_pivot.kurtosis(axis=1)
                futures_skewness_sell_dollar_volume_pivot[f'cs_futures_skewness_sell_dollar_volume_median_{window}d'] = futures_skewness_sell_dollar_volume_pivot.median(axis=1)

                # Cross-sectional kurtosis dollar volume percentile for each symbol/day
                futures_kurtosis_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'kurtosis_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_kurtosis_dollar_volume_percentile = futures_kurtosis_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_kurtosis_dollar_volume_percentile.columns = [f'{col}cs_futures_kurtosis_dollar_volume_percentile_{window}d' for col in futures_kurtosis_dollar_volume_percentile.columns]

                # Cross-sectional z-score of kurtosis dollar volume for each symbol/day
                futures_kurtosis_dollar_volume_pivot_mean = futures_kurtosis_dollar_volume_pivot.mean(axis=1)
                futures_kurtosis_dollar_volume_pivot_std = futures_kurtosis_dollar_volume_pivot.std(axis=1)
                futures_kurtosis_dollar_volume_zscore = futures_kurtosis_dollar_volume_pivot.sub(futures_kurtosis_dollar_volume_pivot_mean, axis=0).div(futures_kurtosis_dollar_volume_pivot_std, axis=0)
                futures_kurtosis_dollar_volume_zscore.columns = [f'{col}cs_futures_kurtosis_dollar_volume_zscore_{window}d' for col in futures_kurtosis_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional kurtosis dollar volume
                futures_kurtosis_dollar_volume_pivot[f'cs_futures_kurtosis_dollar_volume_mean_{window}d'] = futures_kurtosis_dollar_volume_pivot.mean(axis=1)
                futures_kurtosis_dollar_volume_pivot[f'cs_futures_kurtosis_dollar_volume_std_{window}d'] = futures_kurtosis_dollar_volume_pivot.std(axis=1)
                futures_kurtosis_dollar_volume_pivot[f'cs_futures_kurtosis_dollar_volume_skew_{window}d'] = futures_kurtosis_dollar_volume_pivot.skew(axis=1)
                futures_kurtosis_dollar_volume_pivot[f'cs_futures_kurtosis_dollar_volume_kurtosis_{window}d'] = futures_kurtosis_dollar_volume_pivot.kurtosis(axis=1)
                futures_kurtosis_dollar_volume_pivot[f'cs_futures_kurtosis_dollar_volume_median_{window}d'] = futures_kurtosis_dollar_volume_pivot.median(axis=1)

                # Cross-sectional kurtosis buy dollar volume percentile for each symbol/day
                futures_kurtosis_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'kurtosis_buy_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_kurtosis_buy_dollar_volume_percentile = futures_kurtosis_buy_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_kurtosis_buy_dollar_volume_percentile.columns = [f'{col}cs_futures_kurtosis_buy_dollar_volume_percentile_{window}d' for col in futures_kurtosis_buy_dollar_volume_percentile.columns]

                # Cross-sectional z-score of kurtosis buy dollar volume for each symbol/day
                futures_kurtosis_buy_dollar_volume_pivot_mean = futures_kurtosis_buy_dollar_volume_pivot.mean(axis=1)
                futures_kurtosis_buy_dollar_volume_pivot_std = futures_kurtosis_buy_dollar_volume_pivot.std(axis=1)
                futures_kurtosis_buy_dollar_volume_zscore = futures_kurtosis_buy_dollar_volume_pivot.sub(futures_kurtosis_buy_dollar_volume_pivot_mean, axis=0).div(futures_kurtosis_buy_dollar_volume_pivot_std, axis=0)
                futures_kurtosis_buy_dollar_volume_zscore.columns = [f'{col}cs_futures_kurtosis_buy_dollar_volume_zscore_{window}d' for col in futures_kurtosis_buy_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional kurtosis buy dollar volume
                futures_kurtosis_buy_dollar_volume_pivot[f'cs_futures_kurtosis_buy_dollar_volume_mean_{window}d'] = futures_kurtosis_buy_dollar_volume_pivot.mean(axis=1)
                futures_kurtosis_buy_dollar_volume_pivot[f'cs_futures_kurtosis_buy_dollar_volume_std_{window}d'] = futures_kurtosis_buy_dollar_volume_pivot.std(axis=1)
                futures_kurtosis_buy_dollar_volume_pivot[f'cs_futures_kurtosis_buy_dollar_volume_skew_{window}d'] = futures_kurtosis_buy_dollar_volume_pivot.skew(axis=1)
                futures_kurtosis_buy_dollar_volume_pivot[f'cs_futures_kurtosis_buy_dollar_volume_kurtosis_{window}d'] = futures_kurtosis_buy_dollar_volume_pivot.kurtosis(axis=1)
                futures_kurtosis_buy_dollar_volume_pivot[f'cs_futures_kurtosis_buy_dollar_volume_median_{window}d'] = futures_kurtosis_buy_dollar_volume_pivot.median(axis=1)

                # Cross-sectional kurtosis sell dollar volume percentile for each symbol/day
                futures_kurtosis_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'kurtosis_sell_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_kurtosis_sell_dollar_volume_percentile = futures_kurtosis_sell_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_kurtosis_sell_dollar_volume_percentile.columns = [f'{col}cs_futures_kurtosis_sell_dollar_volume_percentile_{window}d' for col in futures_kurtosis_sell_dollar_volume_percentile.columns]

                # Cross-sectional z-score of kurtosis sell dollar volume for each symbol/day
                futures_kurtosis_sell_dollar_volume_pivot_mean = futures_kurtosis_sell_dollar_volume_pivot.mean(axis=1)
                futures_kurtosis_sell_dollar_volume_pivot_std = futures_kurtosis_sell_dollar_volume_pivot.std(axis=1)
                futures_kurtosis_sell_dollar_volume_zscore = futures_kurtosis_sell_dollar_volume_pivot.sub(futures_kurtosis_sell_dollar_volume_pivot_mean, axis=0).div(futures_kurtosis_sell_dollar_volume_pivot_std, axis=0)
                futures_kurtosis_sell_dollar_volume_zscore.columns = [f'{col}cs_futures_kurtosis_sell_dollar_volume_zscore_{window}d' for col in futures_kurtosis_sell_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional kurtosis sell dollar volume
                futures_kurtosis_sell_dollar_volume_pivot[f'cs_futures_kurtosis_sell_dollar_volume_mean_{window}d'] = futures_kurtosis_sell_dollar_volume_pivot.mean(axis=1)
                futures_kurtosis_sell_dollar_volume_pivot[f'cs_futures_kurtosis_sell_dollar_volume_std_{window}d'] = futures_kurtosis_sell_dollar_volume_pivot.std(axis=1)
                futures_kurtosis_sell_dollar_volume_pivot[f'cs_futures_kurtosis_sell_dollar_volume_skew_{window}d'] = futures_kurtosis_sell_dollar_volume_pivot.skew(axis=1)
                futures_kurtosis_sell_dollar_volume_pivot[f'cs_futures_kurtosis_sell_dollar_volume_kurtosis_{window}d'] = futures_kurtosis_sell_dollar_volume_pivot.kurtosis(axis=1)
                futures_kurtosis_sell_dollar_volume_pivot[f'cs_futures_kurtosis_sell_dollar_volume_median_{window}d'] = futures_kurtosis_sell_dollar_volume_pivot.median(axis=1)

                # Cross-sectional median dollar volume percentile for each symbol/day
                futures_median_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'median_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_median_dollar_volume_percentile = futures_median_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_median_dollar_volume_percentile.columns = [f'{col}cs_futures_median_dollar_volume_percentile_{window}d' for col in futures_median_dollar_volume_percentile.columns]

                # Cross-sectional z-score of median dollar volume for each symbol/day
                futures_median_dollar_volume_pivot_mean = futures_median_dollar_volume_pivot.mean(axis=1)
                futures_median_dollar_volume_pivot_std = futures_median_dollar_volume_pivot.std(axis=1)
                futures_median_dollar_volume_zscore = futures_median_dollar_volume_pivot.sub(futures_median_dollar_volume_pivot_mean, axis=0).div(futures_median_dollar_volume_pivot_std, axis=0)
                futures_median_dollar_volume_zscore.columns = [f'{col}cs_futures_median_dollar_volume_zscore_{window}d' for col in futures_median_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional median dollar volume
                futures_median_dollar_volume_pivot[f'cs_futures_median_dollar_volume_mean_{window}d'] = futures_median_dollar_volume_pivot.mean(axis=1)
                futures_median_dollar_volume_pivot[f'cs_futures_median_dollar_volume_std_{window}d'] = futures_median_dollar_volume_pivot.std(axis=1)
                futures_median_dollar_volume_pivot[f'cs_futures_median_dollar_volume_skew_{window}d'] = futures_median_dollar_volume_pivot.skew(axis=1)
                futures_median_dollar_volume_pivot[f'cs_futures_median_dollar_volume_kurtosis_{window}d'] = futures_median_dollar_volume_pivot.kurtosis(axis=1)
                futures_median_dollar_volume_pivot[f'cs_futures_median_dollar_volume_median_{window}d'] = futures_median_dollar_volume_pivot.median(axis=1)

                # Cross-sectional median buy dollar volume percentile for each symbol/day
                futures_median_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'median_buy_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_median_buy_dollar_volume_percentile = futures_median_buy_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_median_buy_dollar_volume_percentile.columns = [f'{col}cs_futures_median_buy_dollar_volume_percentile_{window}d' for col in futures_median_buy_dollar_volume_percentile.columns]

                # Cross-sectional z-score of median buy dollar volume for each symbol/day
                futures_median_buy_dollar_volume_pivot_mean = futures_median_buy_dollar_volume_pivot.mean(axis=1)
                futures_median_buy_dollar_volume_pivot_std = futures_median_buy_dollar_volume_pivot.std(axis=1)
                futures_median_buy_dollar_volume_zscore = futures_median_buy_dollar_volume_pivot.sub(futures_median_buy_dollar_volume_pivot_mean, axis=0).div(futures_median_buy_dollar_volume_pivot_std, axis=0)
                futures_median_buy_dollar_volume_zscore.columns = [f'{col}cs_futures_median_buy_dollar_volume_zscore_{window}d' for col in futures_median_buy_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional median buy dollar volume
                futures_median_buy_dollar_volume_pivot[f'cs_futures_median_buy_dollar_volume_mean_{window}d'] = futures_median_buy_dollar_volume_pivot.mean(axis=1)
                futures_median_buy_dollar_volume_pivot[f'cs_futures_median_buy_dollar_volume_std_{window}d'] = futures_median_buy_dollar_volume_pivot.std(axis=1)
                futures_median_buy_dollar_volume_pivot[f'cs_futures_median_buy_dollar_volume_skew_{window}d'] = futures_median_buy_dollar_volume_pivot.skew(axis=1)
                futures_median_buy_dollar_volume_pivot[f'cs_futures_median_buy_dollar_volume_kurtosis_{window}d'] = futures_median_buy_dollar_volume_pivot.kurtosis(axis=1)
                futures_median_buy_dollar_volume_pivot[f'cs_futures_median_buy_dollar_volume_median_{window}d'] = futures_median_buy_dollar_volume_pivot.median(axis=1)

                # Cross-sectional median sell dollar volume percentile for each symbol/day
                futures_median_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id', values=f'median_sell_dollar_volume_{window}d_futures', dropna=False).sort_index()
                futures_median_sell_dollar_volume_percentile = futures_median_sell_dollar_volume_pivot.rank(axis=1, pct=True)
                futures_median_sell_dollar_volume_percentile.columns = [f'{col}cs_futures_median_sell_dollar_volume_percentile_{window}d' for col in futures_median_sell_dollar_volume_percentile.columns]

                # Cross-sectional z-score of median sell dollar volume for each symbol/day
                futures_median_sell_dollar_volume_pivot_mean = futures_median_sell_dollar_volume_pivot.mean(axis=1)
                futures_median_sell_dollar_volume_pivot_std = futures_median_sell_dollar_volume_pivot.std(axis=1)
                futures_median_sell_dollar_volume_zscore = futures_median_sell_dollar_volume_pivot.sub(futures_median_sell_dollar_volume_pivot_mean, axis=0).div(futures_median_sell_dollar_volume_pivot_std, axis=0)
                futures_median_sell_dollar_volume_zscore.columns = [f'{col}cs_futures_median_sell_dollar_volume_zscore_{window}d' for col in futures_median_sell_dollar_volume_zscore.columns]

                # 4 moments of cross-sectional median sell dollar volume
                futures_median_sell_dollar_volume_pivot[f'cs_futures_median_sell_dollar_volume_mean_{window}d'] = futures_median_sell_dollar_volume_pivot.mean(axis=1)
                futures_median_sell_dollar_volume_pivot[f'cs_futures_median_sell_dollar_volume_std_{window}d'] = futures_median_sell_dollar_volume_pivot.std(axis=1)
                futures_median_sell_dollar_volume_pivot[f'cs_futures_median_sell_dollar_volume_skew_{window}d'] = futures_median_sell_dollar_volume_pivot.skew(axis=1)
                futures_median_sell_dollar_volume_pivot[f'cs_futures_median_sell_dollar_volume_kurtosis_{window}d'] = futures_median_sell_dollar_volume_pivot.kurtosis(axis=1)
                futures_median_sell_dollar_volume_pivot[f'cs_futures_median_sell_dollar_volume_median_{window}d'] = futures_median_sell_dollar_volume_pivot.median(axis=1)

                # Add the 1d features to the final features list
                final_features.extend([
                    futures_std_dollar_volume_percentile,
                    futures_std_buy_dollar_volume_percentile,
                    futures_std_sell_dollar_volume_percentile,
                    futures_skewness_dollar_volume_percentile,
                    futures_skewness_buy_dollar_volume_percentile,  
                    futures_skewness_sell_dollar_volume_percentile,
                    futures_kurtosis_dollar_volume_percentile,
                    futures_kurtosis_buy_dollar_volume_percentile,
                    futures_kurtosis_sell_dollar_volume_percentile,
                    futures_median_dollar_volume_percentile,
                    futures_median_buy_dollar_volume_percentile,
                    futures_median_sell_dollar_volume_percentile,
                ])     

                # Add the z-score features to the final features list
                final_features.extend([
                    futures_std_dollar_volume_zscore,
                    futures_std_buy_dollar_volume_zscore,
                    futures_std_sell_dollar_volume_zscore,
                    futures_skewness_dollar_volume_zscore,
                    futures_skewness_buy_dollar_volume_zscore,
                    futures_skewness_sell_dollar_volume_zscore,
                    futures_kurtosis_dollar_volume_zscore,
                    futures_kurtosis_buy_dollar_volume_zscore,
                    futures_kurtosis_sell_dollar_volume_zscore,
                    futures_median_dollar_volume_zscore,
                    futures_median_buy_dollar_volume_zscore,
                    futures_median_sell_dollar_volume_zscore,
                ])   

                # Add the 4 moments features to the final features list
                cs_moments = [
                    futures_std_dollar_volume_pivot,
                    futures_std_buy_dollar_volume_pivot,
                    futures_std_sell_dollar_volume_pivot,
                    futures_skewness_dollar_volume_pivot,
                    futures_skewness_buy_dollar_volume_pivot,
                    futures_skewness_sell_dollar_volume_pivot,
                    futures_kurtosis_dollar_volume_pivot,
                    futures_kurtosis_buy_dollar_volume_pivot,
                    futures_kurtosis_sell_dollar_volume_pivot,
                    futures_median_dollar_volume_pivot,
                    futures_median_buy_dollar_volume_pivot,
                    futures_median_sell_dollar_volume_pivot,
                ]
                for i, moment in enumerate(cs_moments):
                    valid_cols = [m for m in moment.columns if 'cs_' in m]
                    cs_moments[i] = moment[valid_cols]
                final_features.extend(cs_moments)   

            # Append the features to the final list
            final_features.extend([
                spot_total_buy_dollar_volume_percentile,
                spot_total_sell_dollar_volume_percentile,
                spot_total_dollar_volume_percentile,
                spot_num_buys_percentile,
                spot_num_sells_percentile,
                spot_pct_buys_percentile,
                spot_pct_sells_percentile,
                spot_trade_dollar_volume_imbalance_percentile,
                spot_pct_buy_dollar_volume_percentile,
                spot_pct_sell_dollar_volume_percentile,
                spot_avg_dollar_volume_percentile,
                spot_avg_buy_dollar_volume_percentile,
                spot_avg_sell_dollar_volume_percentile,
                futures_total_buy_dollar_volume_percentile,
                futures_total_sell_dollar_volume_percentile,
                futures_total_dollar_volume_percentile,
                futures_num_buys_percentile,
                futures_num_sells_percentile,
                futures_pct_buys_percentile,
                futures_pct_sells_percentile,
                futures_trade_dollar_volume_imbalance_percentile,
                futures_pct_buy_dollar_volume_percentile,
                futures_pct_sell_dollar_volume_percentile,
                futures_avg_dollar_volume_percentile,
                futures_avg_buy_dollar_volume_percentile,
                futures_avg_sell_dollar_volume_percentile
            ])

            # Add the z-score features to the final features list
            final_features.extend([
                spot_total_buy_dollar_volume_zscore,
                spot_total_sell_dollar_volume_zscore,
                spot_total_dollar_volume_zscore,
                spot_num_buys_zscore,
                spot_num_sells_zscore,
                spot_pct_buys_zscore,
                spot_pct_sells_zscore,
                spot_trade_dollar_volume_imbalance_zscore,
                spot_pct_buy_dollar_volume_zscore,
                spot_pct_sell_dollar_volume_zscore,
                spot_avg_dollar_volume_zscore,
                spot_avg_buy_dollar_volume_zscore,
                spot_avg_sell_dollar_volume_zscore,
                futures_total_buy_dollar_volume_zscore,
                futures_total_sell_dollar_volume_zscore,
                futures_total_dollar_volume_zscore,
                futures_num_buys_zscore,
                futures_num_sells_zscore,
                futures_pct_buys_zscore,
                futures_pct_sells_zscore,
                futures_trade_dollar_volume_imbalance_zscore,
                futures_pct_buy_dollar_volume_zscore,
                futures_pct_sell_dollar_volume_zscore,
                futures_avg_dollar_volume_zscore,
                futures_avg_buy_dollar_volume_zscore,
                futures_avg_sell_dollar_volume_zscore
            ])

            # Add the 4 moments features to the final features list
            cs_moments = [
                spot_total_buy_dollar_volume_pivot,
                spot_total_sell_dollar_volume_pivot,
                spot_total_dollar_volume_pivot,
                spot_num_buys_pivot,
                spot_num_sells_pivot,
                spot_pct_buys_pivot,
                spot_pct_sells_pivot,
                spot_trade_dollar_volume_imbalance_pivot,
                spot_pct_buy_dollar_volume_pivot,
                spot_pct_sell_dollar_volume_pivot,
                spot_avg_dollar_volume_pivot,
                spot_avg_buy_dollar_volume_pivot,
                spot_avg_sell_dollar_volume_pivot,
                futures_total_buy_dollar_volume_pivot,
                futures_total_sell_dollar_volume_pivot,
                futures_total_dollar_volume_pivot,
                futures_num_buys_pivot,
                futures_num_sells_pivot,
                futures_pct_buys_pivot,
                futures_pct_sells_pivot,
                futures_trade_dollar_volume_imbalance_pivot,
                futures_pct_buy_dollar_volume_pivot,
                futures_pct_sell_dollar_volume_pivot,
                futures_avg_dollar_volume_pivot,
                futures_avg_buy_dollar_volume_pivot,
                futures_avg_sell_dollar_volume_pivot
            ]
            for i, moment in enumerate(cs_moments):
                valid_cols = [m for m in moment.columns if 'cs_' in m]
                cs_moments[i] = moment[valid_cols]
            final_features.extend(cs_moments)

        trade_features = pd.concat(final_features, axis = 1)
        self.trade_features = trade_features.reset_index()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        asset_id_base, asset_id_quote, exchange_id = X['symbol_id'].iloc[0].split('_')
        symbol_id = f'{asset_id_base}_{asset_id_quote}_{exchange_id}'
        X_spot = QUERY(
            f"""
            SELECT 
            asset_id_base || '_' || asset_id_quote ||'_' || exchange_id as symbol_id,
            *
            FROM market_data.spot_trade_features_rolling
            WHERE
                asset_id_base = '{asset_id_base}' AND
                asset_id_quote = '{asset_id_quote}' AND
                exchange_id = '{exchange_id}'
            ORDER BY time_period_end 
            """
        )
        X_spot['time_period_end'] = pd.to_datetime(X_spot['time_period_end'])
        X_futures = QUERY(
            f"""
            SELECT 
            asset_id_base || '_' || asset_id_quote ||'_' || exchange_id as symbol_id,
            *
            FROM market_data.futures_trade_features_rolling
            WHERE
                asset_id_base = '{asset_id_base}' AND
                asset_id_quote = '{asset_id_quote}' AND
                exchange_id = '{exchange_id}'
            ORDER BY time_period_end 
            """
        )
        X_futures['time_period_end'] = pd.to_datetime(X_futures['time_period_end'])

        merged = pd.merge(
            X_spot,
            X_futures,
            on = 'time_period_end',
            how = 'outer',
            suffixes = ('_spot', '_futures')
        )
        merged.rename({
            'time_period_end_spot': 'time_period_end',
            'symbol_id_spot': 'symbol_id',
            'asset_id_base_spot': 'asset_id_base',
            'asset_id_quote_spot': 'asset_id_quote',
            'exchange_id_spot': 'exchange_id'
        }, axis = 1, inplace = True)
        merged.drop(columns = ['time_period_end_futures', 'symbol_id_futures', 'asset_id_base_futures', 'asset_id_quote_futures', 'exchange_id_futures'], errors = 'ignore', axis = 1, inplace = True)

        X = pd.merge(
            X,
            merged,
            on = ['time_period_end', 'symbol_id'],
            how = 'left'
        )

        for window in self.windows:
            for lookback in self.lookback_windows:
                for market_type in ['spot', 'futures']:
                    # Rolling 4 moments of total buy dollar volume
                    X[f'{market_type}_avg_total_buy_dollar_volume_{window}d_{lookback}d'] = X[f'total_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_buy_dollar_volume_{window}d_{lookback}d'] = X[f'total_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skew_buy_dollar_volume_{window}d_{lookback}d'] = X[f'total_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_buy_dollar_volume_{window}d_{lookback}d'] = X[f'total_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_total_buy_dollar_volume_{window}d_{lookback}d'] = X[f'total_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of total sell dollar volume
                    X[f'{market_type}_avg_total_sell_dollar_volume_{window}d_{lookback}d'] = X[f'total_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_sell_dollar_volume_{window}d_{lookback}d'] = X[f'total_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skew_sell_dollar_volume_{window}d_{lookback}d'] = X[f'total_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_sell_dollar_volume_{window}d_{lookback}d'] = X[f'total_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_total_sell_dollar_volume_{window}d_{lookback}d'] = X[f'total_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of total dollar volume
                    X[f'{market_type}_avg_total_dollar_volume_{window}d_{lookback}d'] = X[f'total_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_dollar_volume_{window}d_{lookback}d'] = X[f'total_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skew_dollar_volume_{window}d_{lookback}d'] = X[f'total_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_dollar_volume_{window}d_{lookback}d'] = X[f'total_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_total_dollar_volume_{window}d_{lookback}d'] = X[f'total_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of number of buys
                    X[f'{market_type}_avg_num_buys_{window}d_{lookback}d'] = X[f'num_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_num_buys_{window}d_{lookback}d'] = X[f'num_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_num_buys_{window}d_{lookback}d'] = X[f'num_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_num_buys_{window}d_{lookback}d'] = X[f'num_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_num_buys_{window}d_{lookback}d'] = X[f'num_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of number of sells
                    X[f'{market_type}_avg_num_sells_{window}d_{lookback}d'] = X[f'num_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_num_sells_{window}d_{lookback}d'] = X[f'num_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_num_sells_{window}d_{lookback}d'] = X[f'num_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_num_sells_{window}d_{lookback}d'] = X[f'num_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_num_sells_{window}d_{lookback}d'] = X[f'num_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of pct_buys
                    X[f'{market_type}_avg_pct_buys_{window}d_{lookback}d'] = X[f'pct_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_pct_buys_{window}d_{lookback}d'] = X[f'pct_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_pct_buys_{window}d_{lookback}d'] = X[f'pct_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_pct_buys_{window}d_{lookback}d'] = X[f'pct_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_pct_buys_{window}d_{lookback}d'] = X[f'pct_buys_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of pct_sells
                    X[f'{market_type}_avg_pct_sells_{window}d_{lookback}d'] = X[f'pct_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_pct_sells_{window}d_{lookback}d'] = X[f'pct_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_pct_sells_{window}d_{lookback}d'] = X[f'pct_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_pct_sells_{window}d_{lookback}d'] = X[f'pct_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_pct_sells_{window}d_{lookback}d'] = X[f'pct_sells_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of trade dollar volume imbalance
                    X[f'{market_type}_avg_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = X[f'trade_imbalance_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = X[f'trade_imbalance_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = X[f'trade_imbalance_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = X[f'trade_imbalance_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = X[f'trade_imbalance_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of pct buy dollar volume
                    X[f'{market_type}_avg_pct_buy_dollar_volume_{window}d_{lookback}d'] = X[f'pct_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_pct_buy_dollar_volume_{window}d_{lookback}d'] = X[f'pct_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_pct_buy_dollar_volume_{window}d_{lookback}d'] = X[f'pct_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_pct_buy_dollar_volume_{window}d_{lookback}d'] = X[f'pct_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_pct_buy_dollar_volume_{window}d_{lookback}d'] = X[f'pct_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of pct sell dollar volume
                    X[f'{market_type}_avg_pct_sell_dollar_volume_{window}d_{lookback}d'] = X[f'pct_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_pct_sell_dollar_volume_{window}d_{lookback}d'] = X[f'pct_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_pct_sell_dollar_volume_{window}d_{lookback}d'] = X[f'pct_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_pct_sell_dollar_volume_{window}d_{lookback}d'] = X[f'pct_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_pct_sell_dollar_volume_{window}d_{lookback}d'] = X[f'pct_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of avg dollar volume
                    X[f'{market_type}_avg_avg_dollar_volume_{window}d_{lookback}d'] = X[f'avg_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_avg_dollar_volume_{window}d_{lookback}d'] = X[f'avg_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_avg_dollar_volume_{window}d_{lookback}d'] = X[f'avg_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_avg_dollar_volume_{window}d_{lookback}d'] = X[f'avg_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_avg_dollar_volume_{window}d_{lookback}d'] = X[f'avg_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of avg buy dollar volume
                    X[f'{market_type}_avg_avg_buy_dollar_volume_{window}d_{lookback}d'] = X[f'avg_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_avg_buy_dollar_volume_{window}d_{lookback}d'] = X[f'avg_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_avg_buy_dollar_volume_{window}d_{lookback}d'] = X[f'avg_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_avg_buy_dollar_volume_{window}d_{lookback}d'] = X[f'avg_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_avg_buy_dollar_volume_{window}d_{lookback}d'] = X[f'avg_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    # Rolling 4 moments of avg sell dollar volume
                    X[f'{market_type}_avg_avg_sell_dollar_volume_{window}d_{lookback}d'] = X[f'avg_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                    X[f'{market_type}_std_avg_sell_dollar_volume_{window}d_{lookback}d'] = X[f'avg_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                    X[f'{market_type}_skewness_avg_sell_dollar_volume_{window}d_{lookback}d'] = X[f'avg_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                    X[f'{market_type}_kurtosis_avg_sell_dollar_volume_{window}d_{lookback}d'] = X[f'avg_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                    X[f'{market_type}_median_avg_sell_dollar_volume_{window}d_{lookback}d'] = X[f'avg_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                    if window == 1:
                        # Rolling 4 moments of standard deviation of total buy dollar volume
                        X[f'{market_type}_avg_std_buy_dollar_volume_{window}d_{lookback}d'] = X[f'std_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_std_buy_dollar_volume_{window}d_{lookback}d'] = X[f'std_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_std_buy_dollar_volume_{window}d_{lookback}d'] = X[f'std_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_std_buy_dollar_volume_{window}d_{lookback}d'] = X[f'std_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_std_buy_dollar_volume_{window}d_{lookback}d'] = X[f'std_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                        # Rolling 4 moments of standard deviation of total sell dollar volume
                        X[f'{market_type}_avg_std_sell_dollar_volume_{window}d_{lookback}d'] = X[f'std_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_std_sell_dollar_volume_{window}d_{lookback}d'] = X[f'std_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_std_sell_dollar_volume_{window}d_{lookback}d'] = X[f'std_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_std_sell_dollar_volume_{window}d_{lookback}d'] = X[f'std_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_std_sell_dollar_volume_{window}d_{lookback}d'] = X[f'std_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                        # Rolling 4 moments of standard deviation of total dollar volume
                        X[f'{market_type}_avg_std_dollar_volume_{window}d_{lookback}d'] = X[f'std_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_std_dollar_volume_{window}d_{lookback}d'] = X[f'std_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_std_dollar_volume_{window}d_{lookback}d'] = X[f'std_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_std_dollar_volume_{window}d_{lookback}d'] = X[f'std_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_std_dollar_volume_{window}d_{lookback}d'] = X[f'std_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                        # Rolling 4 moments of skewness of total buy dollar volume
                        X[f'{market_type}_avg_skew_buy_dollar_volume_{window}d_{lookback}d'] = X[f'skew_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_skew_buy_dollar_volume_{window}d_{lookback}d'] = X[f'skew_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_skew_buy_dollar_volume_{window}d_{lookback}d'] = X[f'skew_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_skew_buy_dollar_volume_{window}d_{lookback}d'] = X[f'skew_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_skew_buy_dollar_volume_{window}d_{lookback}d'] = X[f'skew_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()
                        # Rolling 4 moments of skewness of total sell dollar volume
                        X[f'{market_type}_avg_skew_sell_dollar_volume_{window}d_{lookback}d'] = X[f'skew_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_skew_sell_dollar_volume_{window}d_{lookback}d'] = X[f'skew_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_skew_sell_dollar_volume_{window}d_{lookback}d'] = X[f'skew_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_skew_sell_dollar_volume_{window}d_{lookback}d'] = X[f'skew_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_skew_sell_dollar_volume_{window}d_{lookback}d'] = X[f'skew_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                        # Rolling 4 moments of skewness of total dollar volume
                        X[f'{market_type}_avg_skew_dollar_volume_{window}d_{lookback}d'] = X[f'skew_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_skew_dollar_volume_{window}d_{lookback}d'] = X[f'skew_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_skew_dollar_volume_{window}d_{lookback}d'] = X[f'skew_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_skew_dollar_volume_{window}d_{lookback}d'] = X[f'skew_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_skew_dollar_volume_{window}d_{lookback}d'] = X[f'skew_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                        # Rolling 4 moments of kurtosis of total buy dollar volume
                        X[f'{market_type}_avg_kurtosis_buy_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_kurtosis_buy_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_kurtosis_buy_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_kurtosis_buy_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_kurtosis_buy_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_buy_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

                        # Rolling 4 moments of kurtosis of total sell dollar volume
                        X[f'{market_type}_avg_kurtosis_sell_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_kurtosis_sell_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_kurtosis_sell_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_kurtosis_sell_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_kurtosis_sell_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_sell_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()
                        # Rolling 4 moments of kurtosis of total dollar volume
                        X[f'{market_type}_avg_kurtosis_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).mean()
                        X[f'{market_type}_std_kurtosis_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).std()
                        X[f'{market_type}_skewness_kurtosis_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).skew()
                        X[f'{market_type}_kurtosis_kurtosis_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).kurt()
                        X[f'{market_type}_median_kurtosis_dollar_volume_{window}d_{lookback}d'] = X[f'kurtosis_dollar_volume_{window}d_{market_type}'].rolling(window = lookback, min_periods = 7).median()

        cross_sectional_ranking_cols = self.trade_features.filter(like = symbol_id).columns.tolist()
        cross_sectional_moments_cols = [col for col in self.trade_features.columns if col.startswith('cs_spot') or col.startswith('cs_futures')]
        cross_sectional_ema_cols = [col for col in self.trade_features.columns if 'rolling_ema' in col]
        valid_cols = cross_sectional_ranking_cols + cross_sectional_moments_cols + cross_sectional_ema_cols

        # Merge current token trade features with the cross-sectional features
        X = pd.merge(
            X, 
            self.trade_features[['time_period_end'] + valid_cols],
            on = 'time_period_end', 
            how = 'left', 
            suffixes = ('', '__remove')
        )
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)

        # Rename the cross-sectional ranking columns to remove the symbol_id from it
        for col in X.columns:
            new_col = col.replace(symbol_id, '')
            X.rename({col: new_col}, inplace = True, axis = 1)

        # Merge data to final features
        X = pd.merge(X, X, on = 'time_period_end', suffixes = ('', '__remove'), how = 'left')

        # Drop the columns with '__remove' suffix
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)

        return X

class RiskFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, windows, lookback_windows):
        self.windows = windows
        self.lookback_windows = lookback_windows

        # 1 day ml_dataset
        self.ml_dataset = QUERY(
            """
            SELECT *
            FROM market_data.ml_dataset_1d
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        self.ml_dataset['symbol_id'] = self.ml_dataset['asset_id_base'] + '_' + self.ml_dataset['asset_id_quote'] + '_' + self.ml_dataset['exchange_id']
        self.ml_dataset['time_period_end'] = pd.to_datetime(self.ml_dataset['time_period_end'])
        # Set index to time_period_end for consistent indexing when calculating rolling risk features
        self.ml_dataset = self.ml_dataset.set_index('time_period_end').sort_index()

        tokens = sorted(self.ml_dataset['symbol_id'].unique().tolist())

        # Calculate cross-sectional risk features
        final_features = []

        # N-day returns
        for window in self.windows:
            self.ml_dataset[f'returns_{window}'] = np.nan

        for token in tokens:
            filter = self.ml_dataset['symbol_id'] == token
            for window in self.windows:
                if window == 1:
                    clip_upper_bound = 0.57
                elif window == 7:
                    clip_upper_bound = 3.55
                elif window == 30:
                    clip_upper_bound = 9.44
                elif window == 180:
                    clip_upper_bound = 59
                    
                # Calculate returns
                self.ml_dataset.loc[filter, f'returns_{window}'] = self.ml_dataset.loc[filter, 'close'].pct_change(window)

        btc = self.ml_dataset[self.ml_dataset['symbol_id'] == 'BTC_USDT_BINANCE'][[f'returns_{window}' for window in self.windows]]
        eth = self.ml_dataset[self.ml_dataset['symbol_id'] == 'ETH_USDT_BINANCE'][[f'returns_{window}' for window in self.windows]]
        market = pd.merge(btc, eth, left_index = True, right_index = True, suffixes = ('_btc', '_eth'), how = 'outer')

        for window in self.windows:
            # Calculate market returns
            market[f'market_returns_{window}'] = market[[f'returns_{window}_btc', f'returns_{window}_eth']].mean(axis = 1)

        # Merge market returns with the ml_dataset for rolling beta and alpha calculations later
        self.ml_dataset = pd.merge(self.ml_dataset, market, left_index = True, right_index = True, how = 'left')

        for token in tokens:
            print(f'Calculating risk features for {token}')
            filter = self.ml_dataset['symbol_id'] == token
            for window in self.windows:
                for lookback_window in self.lookback_windows:
                    # Rolling beta
                    self.ml_dataset.loc[filter, f'beta_{window}d_{lookback_window}d'] = (
                        self.ml_dataset.loc[filter, f'returns_{window}']
                        .rolling(window = lookback_window)
                        .cov(self.ml_dataset.loc[filter, f'market_returns_{window}'])
                        / self.ml_dataset.loc[filter, f'market_returns_{window}'].rolling(window = lookback_window, min_periods = 7).var()
                    )

                    # Rolling alpha
                    self.ml_dataset.loc[filter, f'alpha_{window}d_{lookback_window}d'] = (
                        self.ml_dataset.loc[filter, f'returns_{window}']
                        .rolling(window = lookback_window)
                        .mean()
                        - self.ml_dataset.loc[filter, f'beta_{window}d_{lookback_window}d'] * self.ml_dataset.loc[filter, f'market_returns_{window}'].rolling(window = lookback_window, min_periods = 7).mean()
                    )

                    # Rolling alpha_div_beta
                    self.ml_dataset.loc[filter, f'alpha_div_beta_{window}d_{lookback_window}d'] = (
                        self.ml_dataset.loc[filter, f'alpha_{window}d_{lookback_window}d'] / self.ml_dataset.loc[filter, f'beta_{window}d_{lookback_window}d']
                    )

        # Cross-sectional risk features
        for window in self.windows:
            for lookback_window in self.lookback_windows:
                if lookback_window is None:
                    continue

                # Cross-sectional alpha percentile
                alpha_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'alpha_{window}d_{lookback_window}d', dropna = False)
                cs_alpha_percentile = alpha_pivot.rank(axis = 1, pct = True)
                cs_alpha_percentile.columns = [col + f'alpha_{window}d_{lookback_window}d_percentile' for col in cs_alpha_percentile.columns]

                # Cross-sectional alpha z-score
                alpha_pivot_mean = alpha_pivot.mean(axis = 1)
                alpha_pivot_std = alpha_pivot.std(axis = 1)
                alpha_pivot_zscore = alpha_pivot.sub(alpha_pivot_mean, axis = 0).div(alpha_pivot_std, axis = 0)
                alpha_pivot_zscore.columns = [col + f'alpha_{window}d_{lookback_window}d_zscore' for col in alpha_pivot_zscore.columns]

                # 4 moments of cross-sectional alpha
                alpha_pivot[f'cs_avg_alpha_{window}d_{lookback_window}d'] = alpha_pivot.mean(axis = 1)
                alpha_pivot[f'cs_std_alpha_{window}d_{lookback_window}d'] = alpha_pivot.std(axis = 1)
                alpha_pivot[f'cs_skewness_alpha_{window}d_{lookback_window}d'] = alpha_pivot.skew(axis = 1)
                alpha_pivot[f'cs_kurtosis_alpha_{window}d_{lookback_window}d'] = alpha_pivot.kurtosis(axis = 1)
                alpha_pivot[f'cs_median_alpha_{window}d_{lookback_window}d'] = alpha_pivot.median(axis = 1)

                # Append alpha percentile features
                final_features.append(cs_alpha_percentile)

                # Append alpha z-score features
                final_features.append(alpha_pivot_zscore)

                # Append alpha moments features
                valid_cols = [col for col in alpha_pivot.columns if col.startswith('cs_')]
                final_features.append(alpha_pivot[valid_cols])

                # Cross-sectional beta
                beta_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'beta_{window}d_{lookback_window}d', dropna = False)
                cs_beta_percentile = beta_pivot.rank(axis = 1, pct = True)
                cs_beta_percentile.columns = [col + f'beta_{window}d_{lookback_window}d_percentile' for col in cs_beta_percentile.columns]

                # Cross-sectional beta z-score
                beta_pivot_mean = beta_pivot.mean(axis = 1)
                beta_pivot_std = beta_pivot.std(axis = 1)
                beta_pivot_zscore = beta_pivot.sub(beta_pivot_mean, axis = 0).div(beta_pivot_std, axis = 0)
                beta_pivot_zscore.columns = [col + f'beta_{window}d_{lookback_window}d_zscore' for col in beta_pivot_zscore.columns]

                # 4 moments of cross-sectional beta
                beta_pivot[f'cs_avg_beta_{window}d_{lookback_window}d'] = beta_pivot.mean(axis = 1)
                beta_pivot[f'cs_std_beta_{window}d_{lookback_window}d'] = beta_pivot.std(axis = 1)
                beta_pivot[f'cs_skewness_beta_{window}d_{lookback_window}d'] = beta_pivot.skew(axis = 1)
                beta_pivot[f'cs_kurtosis_beta_{window}d_{lookback_window}d'] = beta_pivot.kurtosis(axis = 1)
                beta_pivot[f'cs_median_beta_{window}d_{lookback_window}d'] = beta_pivot.median(axis = 1)

                # Append beta percentile features
                final_features.append(cs_beta_percentile)

                # Append beta z-score features
                final_features.append(beta_pivot_zscore)

                # Append beta moments features
                valid_cols = [col for col in beta_pivot.columns if col.startswith('cs_')]
                final_features.append(beta_pivot[valid_cols])

                # Cross-sectional alpha_div_beta
                alpha_div_beta_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'alpha_div_beta_{window}d_{lookback_window}d', dropna = False)
                cs_alpha_div_beta_percentile = alpha_div_beta_pivot.rank(axis = 1, pct = True)
                cs_alpha_div_beta_percentile.columns = [col + f'alpha_div_beta_{window}d_{lookback_window}d_percentile' for col in cs_alpha_div_beta_percentile.columns]

                # Cross-sectional alpha_div_beta z-score
                alpha_div_beta_pivot_mean = alpha_div_beta_pivot.mean(axis = 1)
                alpha_div_beta_pivot_std = alpha_div_beta_pivot.std(axis = 1)
                alpha_div_beta_pivot_zscore = alpha_div_beta_pivot.sub(alpha_div_beta_pivot_mean, axis = 0).div(alpha_div_beta_pivot_std, axis = 0)
                alpha_div_beta_pivot_zscore.columns = [col + f'alpha_div_beta_{window}d_{lookback_window}d_zscore' for col in alpha_div_beta_pivot_zscore.columns]

                # 4 moments of cross-sectional alpha_div_beta
                alpha_div_beta_pivot[f'cs_avg_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot.mean(axis = 1)
                alpha_div_beta_pivot[f'cs_std_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot.std(axis = 1)
                alpha_div_beta_pivot[f'cs_skewness_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot.skew(axis = 1)
                alpha_div_beta_pivot[f'cs_kurtosis_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot.kurtosis(axis = 1)
                alpha_div_beta_pivot[f'cs_median_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot.median(axis = 1)
                # Append alpha_div_beta percentile features
                final_features.append(cs_alpha_div_beta_percentile)
                
                # Append alpha_div_beta z-score features
                final_features.append(alpha_div_beta_pivot_zscore)

                # Append alpha_div_beta moments features
                valid_cols = [col for col in alpha_div_beta_pivot.columns if col.startswith('cs_')]
                final_features.append(alpha_div_beta_pivot[valid_cols])

                for lookback_window_2 in self.lookback_windows:
                    # Rolling ema of cross-sectional alpha, beta and alpha_div_beta moments for regime detection
                    alpha_pivot[f'rolling_ema_{lookback_window_2}_cs_avg_alpha_{window}d_{lookback_window}d'] = alpha_pivot[f'cs_avg_alpha_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    alpha_pivot[f'rolling_ema_{lookback_window_2}_cs_std_alpha_{window}d_{lookback_window}d'] = alpha_pivot[f'cs_std_alpha_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    alpha_pivot[f'rolling_ema_{lookback_window_2}_cs_skewness_alpha_{window}d_{lookback_window}d'] = alpha_pivot[f'cs_skewness_alpha_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    alpha_pivot[f'rolling_ema_{lookback_window_2}_cs_kurtosis_alpha_{window}d_{lookback_window}d'] = alpha_pivot[f'cs_kurtosis_alpha_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    alpha_pivot[f'rolling_ema_{lookback_window_2}_cs_median_alpha_{window}d_{lookback_window}d'] = alpha_pivot[f'cs_median_alpha_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()

                    beta_pivot[f'rolling_ema_{lookback_window_2}_cs_avg_beta_{window}d_{lookback_window}d'] = beta_pivot[f'cs_avg_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    beta_pivot[f'rolling_ema_{lookback_window_2}_cs_std_beta_{window}d_{lookback_window}d'] = beta_pivot[f'cs_std_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    beta_pivot[f'rolling_ema_{lookback_window_2}_cs_skewness_beta_{window}d_{lookback_window}d'] = beta_pivot[f'cs_skewness_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    beta_pivot[f'rolling_ema_{lookback_window_2}_cs_kurtosis_beta_{window}d_{lookback_window}d'] = beta_pivot[f'cs_kurtosis_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    beta_pivot[f'rolling_ema_{lookback_window_2}_cs_median_beta_{window}d_{lookback_window}d'] = beta_pivot[f'cs_median_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()

                    alpha_div_beta_pivot[f'rolling_ema_{lookback_window_2}_cs_avg_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot[f'cs_avg_alpha_div_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    alpha_div_beta_pivot[f'rolling_ema_{lookback_window_2}_cs_std_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot[f'cs_std_alpha_div_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    alpha_div_beta_pivot[f'rolling_ema_{lookback_window_2}_cs_skewness_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot[f'cs_skewness_alpha_div_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    alpha_div_beta_pivot[f'rolling_ema_{lookback_window_2}_cs_kurtosis_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot[f'cs_kurtosis_alpha_div_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()
                    alpha_div_beta_pivot[f'rolling_ema_{lookback_window_2}_cs_median_alpha_div_beta_{window}d_{lookback_window}d'] = alpha_div_beta_pivot[f'cs_median_alpha_div_beta_{window}d_{lookback_window}d'].ewm(span = lookback_window_2, min_periods = 7).mean()

                    # Append rolling ema features
                    avg_alpha_ema = alpha_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_avg_alpha_{window}d_{lookback_window}d')
                    std_alpha_ema = alpha_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_std_alpha_{window}d_{lookback_window}d')
                    skewness_alpha_ema = alpha_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_skewness_alpha_{window}d_{lookback_window}d')
                    kurtosis_alpha_ema = alpha_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_kurtosis_alpha_{window}d_{lookback_window}d')
                    median_alpha_ema = alpha_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_median_alpha_{window}d_{lookback_window}d')

                    avg_beta_ema = beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_avg_beta_{window}d_{lookback_window}d')
                    std_beta_ema = beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_std_beta_{window}d_{lookback_window}d')
                    skewness_beta_ema = beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_skewness_beta_{window}d_{lookback_window}d')
                    kurtosis_beta_ema = beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_kurtosis_beta_{window}d_{lookback_window}d')
                    median_beta_ema = beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_median_beta_{window}d_{lookback_window}d')

                    avg_alpha_div_beta_ema = alpha_div_beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_avg_alpha_div_beta_{window}d_{lookback_window}d')
                    std_alpha_div_beta_ema = alpha_div_beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_std_alpha_div_beta_{window}d_{lookback_window}d')
                    skewness_alpha_div_beta_ema = alpha_div_beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_skewness_alpha_div_beta_{window}d_{lookback_window}d')
                    kurtosis_alpha_div_beta_ema = alpha_div_beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_kurtosis_alpha_div_beta_{window}d_{lookback_window}d')
                    median_alpha_div_beta_ema = alpha_div_beta_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_median_alpha_div_beta_{window}d_{lookback_window}d')

                    final_features.append(avg_alpha_ema)
                    final_features.append(std_alpha_ema)
                    final_features.append(skewness_alpha_ema)
                    final_features.append(kurtosis_alpha_ema)
                    final_features.append(median_alpha_ema)

                    final_features.append(avg_beta_ema)
                    final_features.append(std_beta_ema)
                    final_features.append(skewness_beta_ema)
                    final_features.append(kurtosis_beta_ema)
                    final_features.append(median_beta_ema)

                    final_features.append(avg_alpha_div_beta_ema)
                    final_features.append(std_alpha_div_beta_ema)
                    final_features.append(skewness_alpha_div_beta_ema)
                    final_features.append(kurtosis_alpha_div_beta_ema)
                    final_features.append(median_alpha_div_beta_ema)
        
        # Concatenate all final features
        final_features = pd.concat(final_features, axis = 1)
        self.final_features = final_features.reset_index()
        self.ml_dataset = self.ml_dataset.reset_index()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Cross-sectional rank features for current token and time period
        symbol_id = X['symbol_id'].iloc[0]

        alpha_cols = list(set([col for col in self.ml_dataset.columns if 'alpha' in col and 'div' not in col]))
        beta_cols = list(set([col for col in self.ml_dataset.columns if 'beta' in col and 'div' not in col]))
        alpha_div_beta_cols = list(set([col for col in self.ml_dataset.columns if 'alpha_div_beta' in col]))
        cross_sectional_ranking_cols = self.final_features.filter(like = symbol_id).columns.tolist()
        cross_sectional_moments_cols = [col for col in self.final_features.columns if col.startswith('cs_')]
        cross_sectional_ema_cols = [col for col in self.final_features.columns if 'rolling_ema' in col]

        # Filter the ml_dataset for the current token and columns of interest
        data = self.ml_dataset[self.ml_dataset['symbol_id'] == symbol_id][['time_period_end'] + alpha_cols + beta_cols + alpha_div_beta_cols].sort_values(by = 'time_period_end')

        # Merge X to ml_dataset for rolling alpha and beta features
        X = pd.merge(X, data, on = 'time_period_end', suffixes = ('', '__remove'), how = 'left')
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)

        # Merge data to cross-sectional ranking, moments, and ema features
        X = pd.merge(
            X, self.final_features[['time_period_end'] + cross_sectional_ranking_cols + cross_sectional_moments_cols + cross_sectional_ema_cols],
            on = 'time_period_end',
            suffixes = ('', '__remove'),
            how = 'left'
        )
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)
        
        # Rename the cross-sectional ranking columns to remove the symbol_id from it
        for col in cross_sectional_ranking_cols:
            new_col = col.replace(symbol_id, '')
            X.rename({col: new_col}, inplace = True, axis = 1)

        return X

class SpotFuturesInteractionFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, windows, lookback_windows):
        self.windows = windows
        self.lookback_windows = lookback_windows

        # Load the spot price data
        spot_price_data = QUERY(
            """
            SELECT *
            FROM market_data.ml_dataset_1d
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        spot_price_data['symbol_id'] = spot_price_data['asset_id_base'] + '_' + spot_price_data['asset_id_quote'] + '_' + spot_price_data['exchange_id']
        spot_price_data['time_period_end'] = pd.to_datetime(spot_price_data['time_period_end'])
        
        # Load the futures price data
        futures_price_data = QUERY(
            """
            SELECT *
            FROM market_data.ml_dataset_futures_1d
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        futures_price_data['symbol_id'] = futures_price_data['asset_id_base'] + '_' + futures_price_data['asset_id_quote'] + '_' + futures_price_data['exchange_id']
        futures_price_data['time_period_end'] = pd.to_datetime(futures_price_data['time_period_end'])

        tokens_spot = sorted(spot_price_data['symbol_id'].unique().tolist())
        tokens_futures = sorted(futures_price_data['symbol_id'].unique().tolist())

        # Returns for spot and futures prices
        for token in tokens_spot:
            for window in self.windows:
                # Filter the price data for the current token
                filter_spot = spot_price_data['symbol_id'] == token
                if window == 1:
                    clip_upper_bound = 0.57
                elif window == 7:
                    clip_upper_bound = 3.55
                elif window == 30:
                    clip_upper_bound = 9.44
                elif window == 180:
                    clip_upper_bound = 59
                
                # Calculate returns
                spot_price_data.loc[filter_spot, f'returns_{window}d'] = spot_price_data.loc[filter_spot, 'close'].pct_change(window)
        
        for token in tokens_futures:
            for window in self.windows:
                # Filter the price data for the current token
                filter_futures = futures_price_data['symbol_id'] == token
                if window == 1:
                    clip_upper_bound = 0.47
                elif window == 7:
                    clip_upper_bound = 2.05
                elif window == 30:
                    clip_upper_bound = 7.09
                elif window == 180:
                    clip_upper_bound = 22.7
                    
                futures_price_data.loc[filter_futures, f'returns_{window}d'] = futures_price_data.loc[filter_futures, 'close'].pct_change(window)

        # Basis (futures - spot) related features
        price_data = pd.merge(
            spot_price_data,
            futures_price_data,
            on = ['time_period_end', 'symbol_id'],
            how = 'outer',
            suffixes = ('_spot', '_futures')
        )
        price_data.rename({
            'time_period_end_spot': 'time_period_end',
            'asset_id_base_spot': 'asset_id_base',
            'asset_id_quote_spot': 'asset_id_quote',
            'exchange_id_spot': 'exchange_id',
            'symbol_id_spot': 'symbol_id',
        }, inplace = True, axis = 1)
        price_data.drop(columns = ['asset_id_base_futures', 'asset_id_quote_futures', 'exchange_id_futures', 'symbol_id_futures', 'time_period_end_futures'], axis = 1, errors = 'ignore', inplace = True)
        self.price_data = price_data
        self.price_data['basis'] = self.price_data['close_futures'] - self.price_data['close_spot']
        self.price_data['basis_pct'] = self.price_data['basis'] / self.price_data['close_spot']

        basis_features = []

        basis_pivot = self.price_data.pivot_table(
            index = 'time_period_end',
            columns = 'symbol_id',
            values = 'basis',
            dropna = False
        )

        # Cross-sectional percentiles of basis
        cs_basis_percentile = basis_pivot.rank(axis = 1, pct = True)
        cs_basis_percentile.columns = [col + 'cs_basis_percentile' for col in cs_basis_percentile.columns]

        # Cross-sectional z-scores of basis
        basis_pivot_mean = basis_pivot.mean(axis = 1)
        basis_pivot_std = basis_pivot.std(axis = 1)
        basis_pivot_zscore = basis_pivot.sub(basis_pivot_mean, axis = 0).div(basis_pivot_std, axis = 0)
        basis_pivot_zscore.columns = [col + 'cs_basis_zscore' for col in basis_pivot_zscore.columns]

        # 4 moments of basis
        basis_pivot['cs_avg_basis'] = basis_pivot.mean(axis = 1)
        basis_pivot['cs_std_basis'] = basis_pivot.std(axis = 1)
        basis_pivot['cs_skewness_basis'] = basis_pivot.skew(axis = 1)
        basis_pivot['cs_kurtosis_basis'] = basis_pivot.kurtosis(axis = 1)
        basis_pivot['cs_median_basis'] = basis_pivot.median(axis = 1)

        basis_pct_pivot = self.price_data.pivot_table(
            index = 'time_period_end',
            columns = 'symbol_id',
            values = 'basis_pct',
            dropna = False
        )

        # Cross-sectional percentiles of basis_pct
        cs_basis_pct_percentile = basis_pct_pivot.rank(axis = 1, pct = True)
        cs_basis_pct_percentile.columns = [col + 'cs_basis_pct_percentile' for col in cs_basis_pct_percentile.columns]

        # Cross-sectional z-scores of basis_pct
        basis_pct_pivot_mean = basis_pct_pivot.mean(axis = 1)
        basis_pct_pivot_std = basis_pct_pivot.std(axis = 1)
        basis_pct_pivot_zscore = basis_pct_pivot.sub(basis_pct_pivot_mean, axis = 0).div(basis_pct_pivot_std, axis = 0)
        basis_pct_pivot_zscore.columns = [col + 'cs_basis_pct_zscore' for col in basis_pct_pivot_zscore.columns]

        # 4 moments of basis_pct
        basis_pct_pivot['cs_avg_basis_pct'] = basis_pct_pivot.mean(axis = 1)
        basis_pct_pivot['cs_std_basis_pct'] = basis_pct_pivot.std(axis = 1)
        basis_pct_pivot['cs_skewness_basis_pct'] = basis_pct_pivot.skew(axis = 1)
        basis_pct_pivot['cs_kurtosis_basis_pct'] = basis_pct_pivot.kurtosis(axis = 1)
        basis_pct_pivot['cs_median_basis_pct'] = basis_pct_pivot.median(axis = 1)

        # Append basis percentile features to the list
        basis_features.append(cs_basis_percentile)
        basis_features.append(cs_basis_pct_percentile)

        # Append basis z-score features to the list
        basis_features.append(basis_pivot_zscore)
        basis_features.append(basis_pct_pivot_zscore)

        # Append basis moments features to the list
        basis_moments_features = [
            basis_pivot,
            basis_pct_pivot
        ]
        for i, moments_feature in enumerate(basis_moments_features):
            valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
            moments_feature = moments_feature[valid_cols]
            basis_features.append(moments_feature)

        # Rolling ema of cross-sectional basis and basis_pct moments for regime detection
        for lookback_window_2 in self.lookback_windows:
            basis_pivot[f'rolling_ema_{lookback_window_2}_cs_avg_basis'] = basis_pivot['cs_avg_basis'].ewm(span = lookback_window_2, min_periods = 7).mean()
            basis_pivot[f'rolling_ema_{lookback_window_2}_cs_std_basis'] = basis_pivot['cs_std_basis'].ewm(span = lookback_window_2, min_periods = 7).mean()
            basis_pivot[f'rolling_ema_{lookback_window_2}_cs_skewness_basis'] = basis_pivot['cs_skewness_basis'].ewm(span = lookback_window_2, min_periods = 7).mean()
            basis_pivot[f'rolling_ema_{lookback_window_2}_cs_kurtosis_basis'] = basis_pivot['cs_kurtosis_basis'].ewm(span = lookback_window_2, min_periods = 7).mean()
            basis_pivot[f'rolling_ema_{lookback_window_2}_cs_median_basis'] = basis_pivot['cs_median_basis'].ewm(span = lookback_window_2, min_periods = 7).mean()

            basis_pct_pivot[f'rolling_ema_{lookback_window_2}_cs_avg_basis_pct'] = basis_pct_pivot['cs_avg_basis_pct'].ewm(span = lookback_window_2, min_periods = 7).mean()
            basis_pct_pivot[f'rolling_ema_{lookback_window_2}_cs_std_basis_pct'] = basis_pct_pivot['cs_std_basis_pct'].ewm(span = lookback_window_2, min_periods = 7).mean()
            basis_pct_pivot[f'rolling_ema_{lookback_window_2}_cs_skewness_basis_pct'] = basis_pct_pivot['cs_skewness_basis_pct'].ewm(span = lookback_window_2, min_periods = 7).mean()
            basis_pct_pivot[f'rolling_ema_{lookback_window_2}_cs_kurtosis_basis_pct'] = basis_pct_pivot['cs_kurtosis_basis_pct'].ewm(span = lookback_window_2, min_periods = 7).mean()
            basis_pct_pivot[f'rolling_ema_{lookback_window_2}_cs_median_basis_pct'] = basis_pct_pivot['cs_median_basis_pct'].ewm(span = lookback_window_2, min_periods = 7).mean()

            basis_pivot_ema = basis_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_')
            basis_pct_pivot_ema = basis_pct_pivot.filter(like = f'rolling_ema_{lookback_window_2}_cs_')

            # Append rolling ema features
            basis_features.append(basis_pivot_ema)
            basis_features.append(basis_pct_pivot_ema)      

        # Load the spot trade features
        spot_trade_features = QUERY(
            """
            SELECT *
            FROM market_data.spot_trade_features_rolling
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        spot_trade_features['symbol_id'] = spot_trade_features['asset_id_base'] + '_' + spot_trade_features['asset_id_quote'] + '_' + spot_trade_features['exchange_id']
        spot_trade_features['time_period_end'] = pd.to_datetime(spot_trade_features['time_period_end'])

        # Load the futures trade features
        futures_trade_features = QUERY(
            """
            SELECT *
            FROM market_data.futures_trade_features_rolling
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        futures_trade_features['symbol_id'] = futures_trade_features['asset_id_base'] + '_' + futures_trade_features['asset_id_quote'] + '_' + futures_trade_features['exchange_id']
        futures_trade_features['time_period_end'] = pd.to_datetime(futures_trade_features['time_period_end'])

        # Trade-related features
        trade_data = pd.merge(
            spot_trade_features,
            futures_trade_features,
            on = ['time_period_end', 'symbol_id'],
            how = 'outer',
            suffixes = ('_spot', '_futures')
        )
        trade_data.rename({
            'time_period_end_spot': 'time_period_end',
            'asset_id_base_spot': 'asset_id_base',
            'asset_id_quote_spot': 'asset_id_quote',
            'exchange_id_spot': 'exchange_id',
            'symbol_id_spot': 'symbol_id',
        }, inplace = True, axis = 1)
        trade_data.drop(columns = ['asset_id_base_futures', 'asset_id_quote_futures', 'exchange_id_futures', 'symbol_id_futures', 'time_period_end_futures'], axis = 1, errors = 'ignore', inplace = True)
        self.trade_data = trade_data

        for window in self.windows:
            # Volume delta features
            trade_data[f'spot_futures_buy_dollar_volume_delta_{window}d'] = trade_data[f'total_buy_dollar_volume_{window}d_futures'] - trade_data[f'total_buy_dollar_volume_{window}d_spot']
            trade_data[f'spot_futures_sell_dollar_volume_delta_{window}d'] = trade_data[f'total_sell_dollar_volume_{window}d_futures'] - trade_data[f'total_sell_dollar_volume_{window}d_spot']

            # Buy volume delta percentiles
            buy_volume_delta_pivot = trade_data.pivot_table(
                index = 'time_period_end',
                columns = 'symbol_id',
                values = f'spot_futures_buy_dollar_volume_delta_{window}d',
                dropna = False
            )
            cs_buy_volume_delta_percentile = buy_volume_delta_pivot.rank(axis = 1, pct = True)
            cs_buy_volume_delta_percentile.columns = [col + f'cs_buy_volume_delta_{window}d_percentile' for col in cs_buy_volume_delta_percentile.columns]

            # Cross-sectional z-scores of buy volume delta
            buy_volume_delta_pivot_mean = buy_volume_delta_pivot.mean(axis = 1)
            buy_volume_delta_pivot_std = buy_volume_delta_pivot.std(axis = 1)
            buy_volume_delta_pivot_zscore = buy_volume_delta_pivot.sub(buy_volume_delta_pivot_mean, axis = 0).div(buy_volume_delta_pivot_std, axis = 0)
            buy_volume_delta_pivot_zscore.columns = [col + f'cs_buy_volume_delta_{window}d_zscore' for col in buy_volume_delta_pivot_zscore.columns]

            # 4 moments of buy volume delta
            buy_volume_delta_pivot[f'cs_avg_buy_volume_delta_{window}d'] = buy_volume_delta_pivot.mean(axis = 1)
            buy_volume_delta_pivot[f'cs_std_buy_volume_delta_{window}d'] = buy_volume_delta_pivot.std(axis = 1)
            buy_volume_delta_pivot[f'cs_skewness_buy_volume_delta_{window}d'] = buy_volume_delta_pivot.skew(axis = 1)
            buy_volume_delta_pivot[f'cs_kurtosis_buy_volume_delta_{window}d'] = buy_volume_delta_pivot.kurtosis(axis = 1)
            buy_volume_delta_pivot[f'cs_median_buy_volume_delta_{window}d'] = buy_volume_delta_pivot.median(axis = 1)

            # Append buy volume delta percentile features to the basis features
            basis_features.append(cs_buy_volume_delta_percentile)

            # Append buy volume delta z-score features to the basis features
            basis_features.append(buy_volume_delta_pivot_zscore)

            # Append buy volume delta moments features to the basis features
            buy_volume_delta_moments_features = [
                buy_volume_delta_pivot
            ]
            for i, moments_feature in enumerate(buy_volume_delta_moments_features):
                valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                moments_feature = moments_feature[valid_cols]
                basis_features.append(moments_feature)

            # Sell volume delta percentiles
            sell_volume_delta_pivot = trade_data.pivot_table(
                index = 'time_period_end',
                columns = 'symbol_id',
                values = f'spot_futures_sell_dollar_volume_delta_{window}d',
                dropna = False
            )
            cs_sell_volume_delta_percentile = sell_volume_delta_pivot.rank(axis = 1, pct = True)
            cs_sell_volume_delta_percentile.columns = [col + f'cs_sell_volume_delta_{window}d_percentile' for col in cs_sell_volume_delta_percentile.columns]

            # Cross-sectional z-scores of sell volume delta
            sell_volume_delta_pivot_mean = sell_volume_delta_pivot.mean(axis = 1)
            sell_volume_delta_pivot_std = sell_volume_delta_pivot.std(axis = 1)
            sell_volume_delta_pivot_zscore = sell_volume_delta_pivot.sub(sell_volume_delta_pivot_mean, axis = 0).div(sell_volume_delta_pivot_std, axis = 0)
            sell_volume_delta_pivot_zscore.columns = [col + f'cs_sell_volume_delta_{window}d_zscore' for col in sell_volume_delta_pivot_zscore.columns]

            # 4 moments of sell volume delta
            sell_volume_delta_pivot[f'cs_avg_sell_volume_delta_{window}d'] = sell_volume_delta_pivot.mean(axis = 1)
            sell_volume_delta_pivot[f'cs_std_sell_volume_delta_{window}d'] = sell_volume_delta_pivot.std(axis = 1)
            sell_volume_delta_pivot[f'cs_skewness_sell_volume_delta_{window}d'] = sell_volume_delta_pivot.skew(axis = 1)
            sell_volume_delta_pivot[f'cs_kurtosis_sell_volume_delta_{window}d'] = sell_volume_delta_pivot.kurtosis(axis = 1)
            sell_volume_delta_pivot[f'cs_median_sell_volume_delta_{window}d'] = sell_volume_delta_pivot.median(axis = 1)

            # Append sell volume delta percentile features to the basis features
            basis_features.append(cs_sell_volume_delta_percentile)

            # Append sell volume delta z-score features to the basis features
            basis_features.append(sell_volume_delta_pivot_zscore)

            # Append sell volume delta moments features to the basis features
            sell_volume_delta_moments_features = [
                sell_volume_delta_pivot
            ]
            for i, moments_feature in enumerate(sell_volume_delta_moments_features):
                valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                moments_feature = moments_feature[valid_cols]
                basis_features.append(moments_feature)

            # Trade imbalance delta features
            trade_data[f'spot_futures_trade_imbalance_delta_{window}d'] = trade_data[f'trade_imbalance_{window}d_futures'] - trade_data[f'trade_imbalance_{window}d_spot']

            # Trade imbalance delta percentiles
            trade_imbalance_pivot = trade_data.pivot_table(
                index = 'time_period_end',
                columns = 'symbol_id',
                values = f'spot_futures_trade_imbalance_delta_{window}d',
                dropna = False
            )
            cs_trade_imbalance_delta_percentile = trade_imbalance_pivot.rank(axis = 1, pct = True)
            cs_trade_imbalance_delta_percentile.columns = [col + f'cs_trade_imbalance_delta_{window}d_percentile' for col in cs_trade_imbalance_delta_percentile.columns]

            # Cross-sectional z-scores of trade imbalance delta
            trade_imbalance_pivot_mean = trade_imbalance_pivot.mean(axis = 1)
            trade_imbalance_pivot_std = trade_imbalance_pivot.std(axis = 1)
            trade_imbalance_pivot_zscore = trade_imbalance_pivot.sub(trade_imbalance_pivot_mean, axis = 0).div(trade_imbalance_pivot_std, axis = 0)
            trade_imbalance_pivot_zscore.columns = [col + f'cs_trade_imbalance_delta_{window}d_zscore' for col in trade_imbalance_pivot_zscore.columns]

            # 4 moments of trade imbalance delta
            trade_imbalance_pivot[f'cs_avg_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot.mean(axis = 1)
            trade_imbalance_pivot[f'cs_std_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot.std(axis = 1)
            trade_imbalance_pivot[f'cs_skewness_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot.skew(axis = 1)
            trade_imbalance_pivot[f'cs_kurtosis_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot.kurtosis(axis = 1)
            trade_imbalance_pivot[f'cs_median_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot.median(axis = 1)

            for window_2 in self.lookback_windows:
                # Rolling ema of trade imbalance delta moments for regime detection
                trade_imbalance_pivot[f'rolling_ema_{window_2}_cs_avg_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot[f'cs_avg_trade_imbalance_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                trade_imbalance_pivot[f'rolling_ema_{window_2}_cs_std_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot[f'cs_std_trade_imbalance_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                trade_imbalance_pivot[f'rolling_ema_{window_2}_cs_skewness_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot[f'cs_skewness_trade_imbalance_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                trade_imbalance_pivot[f'rolling_ema_{window_2}_cs_kurtosis_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot[f'cs_kurtosis_trade_imbalance_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                trade_imbalance_pivot[f'rolling_ema_{window_2}_cs_median_trade_imbalance_delta_{window}d'] = trade_imbalance_pivot[f'cs_median_trade_imbalance_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()

                # Rolling ema of buy volume delta moments for regime detection
                buy_volume_delta_pivot[f'rolling_ema_{window_2}_cs_avg_buy_volume_delta_{window}d'] = buy_volume_delta_pivot[f'cs_avg_buy_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                buy_volume_delta_pivot[f'rolling_ema_{window_2}_cs_std_buy_volume_delta_{window}d'] = buy_volume_delta_pivot[f'cs_std_buy_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                buy_volume_delta_pivot[f'rolling_ema_{window_2}_cs_skewness_buy_volume_delta_{window}d'] = buy_volume_delta_pivot[f'cs_skewness_buy_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                buy_volume_delta_pivot[f'rolling_ema_{window_2}_cs_kurtosis_buy_volume_delta_{window}d'] = buy_volume_delta_pivot[f'cs_kurtosis_buy_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                buy_volume_delta_pivot[f'rolling_ema_{window_2}_cs_median_buy_volume_delta_{window}d'] = buy_volume_delta_pivot[f'cs_median_buy_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()

                # Rolling ema of sell volume delta moments for regime detection
                sell_volume_delta_pivot[f'rolling_ema_{window_2}_cs_avg_sell_volume_delta_{window}d'] = sell_volume_delta_pivot[f'cs_avg_sell_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                sell_volume_delta_pivot[f'rolling_ema_{window_2}_cs_std_sell_volume_delta_{window}d'] = sell_volume_delta_pivot[f'cs_std_sell_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                sell_volume_delta_pivot[f'rolling_ema_{window_2}_cs_skewness_sell_volume_delta_{window}d'] = sell_volume_delta_pivot[f'cs_skewness_sell_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                sell_volume_delta_pivot[f'rolling_ema_{window_2}_cs_kurtosis_sell_volume_delta_{window}d'] = sell_volume_delta_pivot[f'cs_kurtosis_sell_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()
                sell_volume_delta_pivot[f'rolling_ema_{window_2}_cs_median_sell_volume_delta_{window}d'] = sell_volume_delta_pivot[f'cs_median_sell_volume_delta_{window}d'].ewm(span = window_2, min_periods = 7).mean()

                trade_imbalance_pivot_ema = trade_imbalance_pivot.filter(like = f'rolling_ema_{window_2}_cs_')
                buy_volume_delta_pivot_ema = buy_volume_delta_pivot.filter(like = f'rolling_ema_{window_2}_cs_')
                sell_volume_delta_pivot_ema = sell_volume_delta_pivot.filter(like = f'rolling_ema_{window_2}_cs_')

                #  Append rolling ema features for trade imbalance delta moments
                basis_features.append(trade_imbalance_pivot_ema)
                #  Append rolling ema features for buy volume delta moments
                basis_features.append(buy_volume_delta_pivot_ema)
                #  Append rolling ema features for sell volume delta moments
                basis_features.append(sell_volume_delta_pivot_ema)

            # Append trade imbalance delta percentile features to the basis features 
            basis_features.append(cs_trade_imbalance_delta_percentile)

            # Append trade imbalance delta z-score features to the basis features
            basis_features.append(trade_imbalance_pivot_zscore)

            # Append trade imbalance delta moments features to the basis features
            trade_imbalance_moments_features = [
                trade_imbalance_pivot
            ]
            for i, moments_feature in enumerate(trade_imbalance_moments_features):
                valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                moments_feature = moments_feature[valid_cols]
                basis_features.append(moments_feature)

            # 4-moments delta features
            # Avg. dollar volume delta features
            trade_data[f'spot_futures_avg_buy_dollar_volume_delta_{window}d'] = trade_data[f'avg_buy_dollar_volume_{window}d_futures'] - trade_data[f'avg_buy_dollar_volume_{window}d_spot']
            trade_data[f'spot_futures_avg_sell_dollar_volume_delta_{window}d'] = trade_data[f'avg_sell_dollar_volume_{window}d_futures'] - trade_data[f'avg_sell_dollar_volume_{window}d_spot']

            # Cross-sectional percentiles of average dollar volume delta (buy)
            avg_buy_volume_delta_pivot = trade_data.pivot_table(
                index = 'time_period_end',
                columns = 'symbol_id',
                values = f'spot_futures_avg_buy_dollar_volume_delta_{window}d',
                dropna = False
            )
            cs_avg_buy_volume_delta_percentile = avg_buy_volume_delta_pivot.rank(axis = 1, pct = True)
            cs_avg_buy_volume_delta_percentile.columns = [col + f'cs_avg_buy_volume_delta_{window}d_percentile' for col in cs_avg_buy_volume_delta_percentile.columns]

            # Cross-sectional z-scores of average dollar volume delta (buy)
            avg_buy_volume_delta_pivot_mean = avg_buy_volume_delta_pivot.mean(axis = 1)
            avg_buy_volume_delta_pivot_std = avg_buy_volume_delta_pivot.std(axis = 1)
            avg_buy_volume_delta_pivot_zscore = avg_buy_volume_delta_pivot.sub(avg_buy_volume_delta_pivot_mean, axis = 0).div(avg_buy_volume_delta_pivot_std, axis = 0)
            avg_buy_volume_delta_pivot_zscore.columns = [col + f'cs_avg_buy_volume_delta_{window}d_zscore' for col in avg_buy_volume_delta_pivot_zscore.columns]

            # 4 moments of average dollar volume delta (buy)
            avg_buy_volume_delta_pivot[f'cs_avg_avg_buy_volume_delta_{window}d'] = avg_buy_volume_delta_pivot.mean(axis = 1)
            avg_buy_volume_delta_pivot[f'cs_std_avg_buy_volume_delta_{window}d'] = avg_buy_volume_delta_pivot.std(axis = 1)
            avg_buy_volume_delta_pivot[f'cs_skewness_avg_buy_volume_delta_{window}d'] = avg_buy_volume_delta_pivot.skew(axis = 1)
            avg_buy_volume_delta_pivot[f'cs_kurtosis_avg_buy_volume_delta_{window}d'] = avg_buy_volume_delta_pivot.kurtosis(axis = 1)
            avg_buy_volume_delta_pivot[f'cs_median_avg_buy_volume_delta_{window}d'] = avg_buy_volume_delta_pivot.median(axis = 1)

            # Append average dollar volume delta percentile features to the basis features (buy)
            basis_features.append(cs_avg_buy_volume_delta_percentile)

            # Append average dollar volume delta z-score features to the basis features (buy)
            basis_features.append(avg_buy_volume_delta_pivot_zscore)

            # Append average dollar volume delta moments features to the basis features (buy)
            avg_buy_volume_delta_moments_features = [
                avg_buy_volume_delta_pivot
            ]
            for i, moments_feature in enumerate(avg_buy_volume_delta_moments_features):
                valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                moments_feature = moments_feature[valid_cols]
                basis_features.append(moments_feature)

            # Cross-sectional percentiles of average dollar volume delta (sell)
            avg_sell_volume_delta_pivot = trade_data.pivot_table(
                index = 'time_period_end',
                columns = 'symbol_id',
                values = f'spot_futures_avg_sell_dollar_volume_delta_{window}d',
                dropna = False
            )
            cs_avg_sell_volume_delta_percentile = avg_sell_volume_delta_pivot.rank(axis = 1, pct = True)
            cs_avg_sell_volume_delta_percentile.columns = [col + f'cs_avg_sell_volume_delta_{window}d_percentile' for col in cs_avg_sell_volume_delta_percentile.columns]

            # Cross-sectional z-scores of average dollar volume delta (sell)
            avg_sell_volume_delta_pivot_mean = avg_sell_volume_delta_pivot.mean(axis = 1)
            avg_sell_volume_delta_pivot_std = avg_sell_volume_delta_pivot.std(axis = 1)
            avg_sell_volume_delta_pivot_zscore = avg_sell_volume_delta_pivot.sub(avg_sell_volume_delta_pivot_mean, axis = 0).div(avg_sell_volume_delta_pivot_std, axis = 0)
            avg_sell_volume_delta_pivot_zscore.columns = [col + f'cs_avg_sell_volume_delta_{window}d_zscore' for col in avg_sell_volume_delta_pivot_zscore.columns]

            # 4 moments of average dollar volume delta (sell)
            avg_sell_volume_delta_pivot[f'cs_avg_avg_sell_volume_delta_{window}d'] = avg_sell_volume_delta_pivot.mean(axis = 1)
            avg_sell_volume_delta_pivot[f'cs_std_avg_sell_volume_delta_{window}d'] = avg_sell_volume_delta_pivot.std(axis = 1)
            avg_sell_volume_delta_pivot[f'cs_skewness_avg_sell_volume_delta_{window}d'] = avg_sell_volume_delta_pivot.skew(axis = 1)
            avg_sell_volume_delta_pivot[f'cs_kurtosis_avg_sell_volume_delta_{window}d'] = avg_sell_volume_delta_pivot.kurtosis(axis = 1)
            avg_sell_volume_delta_pivot[f'cs_median_avg_sell_volume_delta_{window}d'] = avg_sell_volume_delta_pivot.median(axis = 1)

            # Append average dollar volume delta percentile features to the basis features (sell)
            basis_features.append(cs_avg_sell_volume_delta_percentile)

            # Append average dollar volume delta z-score features to the basis features (sell)
            basis_features.append(avg_sell_volume_delta_pivot_zscore)

            # Append average dollar volume delta moments features to the basis features (sell)
            avg_sell_volume_delta_moments_features = [
                avg_sell_volume_delta_pivot
            ]
            for i, moments_feature in enumerate(avg_sell_volume_delta_moments_features):
                valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                moments_feature = moments_feature[valid_cols]
                basis_features.append(moments_feature)

            if window == 1:
                # Standard deviation delta features
                trade_data[f'spot_futures_std_buy_dollar_volume_delta_{window}d'] = trade_data[f'std_buy_dollar_volume_{window}d_futures'] - trade_data[f'std_buy_dollar_volume_{window}d_spot']
                trade_data[f'spot_futures_std_sell_dollar_volume_delta_{window}d'] = trade_data[f'std_sell_dollar_volume_{window}d_futures'] - trade_data[f'std_sell_dollar_volume_{window}d_spot']

                # Cross-sectional percentiles of standard deviation delta (buy)
                std_buy_volume_delta_pivot = trade_data.pivot_table(
                    index = 'time_period_end',
                    columns = 'symbol_id',
                    values = f'spot_futures_std_buy_dollar_volume_delta_{window}d',
                    dropna = False
                )
                cs_std_buy_volume_delta_percentile = std_buy_volume_delta_pivot.rank(axis = 1, pct = True)
                cs_std_buy_volume_delta_percentile.columns = [col + f'cs_std_buy_volume_delta_{window}d_percentile' for col in cs_std_buy_volume_delta_percentile.columns]

                # Cross-sectional z-scores of standard deviation delta (buy)
                std_buy_volume_delta_pivot_mean = std_buy_volume_delta_pivot.mean(axis = 1)
                std_buy_volume_delta_pivot_std = std_buy_volume_delta_pivot.std(axis = 1)
                std_buy_volume_delta_pivot_zscore = std_buy_volume_delta_pivot.sub(std_buy_volume_delta_pivot_mean, axis = 0).div(std_buy_volume_delta_pivot_std, axis = 0)
                std_buy_volume_delta_pivot_zscore.columns = [col + f'cs_std_buy_volume_delta_{window}d_zscore' for col in std_buy_volume_delta_pivot_zscore.columns]

                # 4 moments of standard deviation delta (buy)
                std_buy_volume_delta_pivot[f'cs_avg_std_buy_volume_delta_{window}d'] = std_buy_volume_delta_pivot.mean(axis = 1)
                std_buy_volume_delta_pivot[f'cs_std_std_buy_volume_delta_{window}d'] = std_buy_volume_delta_pivot.std(axis = 1)
                std_buy_volume_delta_pivot[f'cs_skewness_std_buy_volume_delta_{window}d'] = std_buy_volume_delta_pivot.skew(axis = 1)
                std_buy_volume_delta_pivot[f'cs_kurtosis_std_buy_volume_delta_{window}d'] = std_buy_volume_delta_pivot.kurtosis(axis = 1)
                std_buy_volume_delta_pivot[f'cs_median_std_buy_volume_delta_{window}d'] = std_buy_volume_delta_pivot.median(axis = 1)

                # Append standard deviation delta percentile features to the basis features (buy)
                basis_features.append(cs_std_buy_volume_delta_percentile)

                # Append standard deviation delta z-score features to the basis features (buy)
                basis_features.append(std_buy_volume_delta_pivot_zscore)

                # Append standard deviation delta moments features to the basis features (buy)
                std_buy_volume_delta_moments_features = [
                    std_buy_volume_delta_pivot
                ]
                for i, moments_feature in enumerate(std_buy_volume_delta_moments_features):
                    valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                    moments_feature = moments_feature[valid_cols]
                    basis_features.append(moments_feature)

                # Cross-sectional percentiles of standard deviation delta (sell)
                std_sell_volume_delta_pivot = trade_data.pivot_table(
                    index = 'time_period_end',
                    columns = 'symbol_id',
                    values = f'spot_futures_std_sell_dollar_volume_delta_{window}d',
                    dropna = False
                )
                cs_std_sell_volume_delta_percentile = std_sell_volume_delta_pivot.rank(axis = 1, pct = True)
                cs_std_sell_volume_delta_percentile.columns = [col + f'cs_std_sell_volume_delta_{window}d_percentile' for col in cs_std_sell_volume_delta_percentile.columns]

                # Cross-sectional z-scores of standard deviation delta (sell)
                std_sell_volume_delta_pivot_mean = std_sell_volume_delta_pivot.mean(axis = 1)
                std_sell_volume_delta_pivot_std = std_sell_volume_delta_pivot.std(axis = 1)
                std_sell_volume_delta_pivot_zscore = std_sell_volume_delta_pivot.sub(std_sell_volume_delta_pivot_mean, axis = 0).div(std_sell_volume_delta_pivot_std, axis = 0)
                std_sell_volume_delta_pivot_zscore.columns = [col + f'cs_std_sell_volume_delta_{window}d_zscore' for col in std_sell_volume_delta_pivot_zscore.columns]

                # 4 moments of standard deviation delta (sell)
                std_sell_volume_delta_pivot[f'cs_avg_std_sell_volume_delta_{window}d'] = std_sell_volume_delta_pivot.mean(axis = 1)
                std_sell_volume_delta_pivot[f'cs_std_std_sell_volume_delta_{window}d'] = std_sell_volume_delta_pivot.std(axis = 1)
                std_sell_volume_delta_pivot[f'cs_skewness_std_sell_volume_delta_{window}d'] = std_sell_volume_delta_pivot.skew(axis = 1)
                std_sell_volume_delta_pivot[f'cs_kurtosis_std_sell_volume_delta_{window}d'] = std_sell_volume_delta_pivot.kurtosis(axis = 1)
                std_sell_volume_delta_pivot[f'cs_median_std_sell_volume_delta_{window}d'] = std_sell_volume_delta_pivot.median(axis = 1)

                # Append standard deviation delta percentile features to the basis features (sell)
                basis_features.append(cs_std_sell_volume_delta_percentile)

                # Append standard deviation delta z-score features to the basis features (sell)
                basis_features.append(std_sell_volume_delta_pivot_zscore)

                # Append standard deviation delta moments features to the basis features (sell)
                std_sell_volume_delta_moments_features = [
                    std_sell_volume_delta_pivot
                ]
                for i, moments_feature in enumerate(std_sell_volume_delta_moments_features):
                    valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                    moments_feature = moments_feature[valid_cols]
                    basis_features.append(moments_feature)

                # Skewness delta features
                trade_data[f'spot_futures_skew_buy_dollar_volume_delta_{window}d'] = trade_data[f'skew_buy_dollar_volume_{window}d_futures'] - trade_data[f'skew_buy_dollar_volume_{window}d_spot']
                trade_data[f'spot_futures_skew_sell_dollar_volume_delta_{window}d'] = trade_data[f'skew_sell_dollar_volume_{window}d_futures'] - trade_data[f'skew_sell_dollar_volume_{window}d_spot']

                # Cross-sectional percentiles of skewness delta (buy)
                skew_buy_volume_delta_pivot = trade_data.pivot_table(
                    index = 'time_period_end',
                    columns = 'symbol_id',
                    values = f'spot_futures_skew_buy_dollar_volume_delta_{window}d',
                    dropna = False
                )
                cs_skew_buy_volume_delta_percentile = skew_buy_volume_delta_pivot.rank(axis = 1, pct = True)
                cs_skew_buy_volume_delta_percentile.columns = [col + f'cs_skew_buy_volume_delta_{window}d_percentile' for col in cs_skew_buy_volume_delta_percentile.columns]

                # Cross-sectional z-scores of skewness delta (buy)
                skew_buy_volume_delta_pivot_mean = skew_buy_volume_delta_pivot.mean(axis = 1)
                skew_buy_volume_delta_pivot_std = skew_buy_volume_delta_pivot.std(axis = 1)
                skew_buy_volume_delta_pivot_zscore = skew_buy_volume_delta_pivot.sub(skew_buy_volume_delta_pivot_mean, axis = 0).div(skew_buy_volume_delta_pivot_std, axis = 0)
                skew_buy_volume_delta_pivot_zscore.columns = [col + f'cs_skew_buy_volume_delta_{window}d_zscore' for col in skew_buy_volume_delta_pivot_zscore.columns]

                # 4 moments of skewness delta (buy)
                skew_buy_volume_delta_pivot[f'cs_avg_skew_buy_volume_delta_{window}d'] = skew_buy_volume_delta_pivot.mean(axis = 1)
                skew_buy_volume_delta_pivot[f'cs_std_skew_buy_volume_delta_{window}d'] = skew_buy_volume_delta_pivot.std(axis = 1)
                skew_buy_volume_delta_pivot[f'cs_skewness_skew_buy_volume_delta_{window}d'] = skew_buy_volume_delta_pivot.skew(axis = 1)
                skew_buy_volume_delta_pivot[f'cs_kurtosis_skew_buy_volume_delta_{window}d'] = skew_buy_volume_delta_pivot.kurtosis(axis = 1)
                skew_buy_volume_delta_pivot[f'cs_median_skew_buy_volume_delta_{window}d'] = skew_buy_volume_delta_pivot.median(axis = 1)

                # Append skewness delta percentile features to the basis features (buy)
                basis_features.append(cs_skew_buy_volume_delta_percentile)

                # Append skewness delta z-score features to the basis features (buy)
                basis_features.append(skew_buy_volume_delta_pivot_zscore)

                # Append skewness delta moments features to the basis features (buy)
                skew_buy_volume_delta_moments_features = [
                    skew_buy_volume_delta_pivot
                ]
                for i, moments_feature in enumerate(skew_buy_volume_delta_moments_features):
                    valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                    moments_feature = moments_feature[valid_cols]
                    basis_features.append(moments_feature)

                # Cross-sectional percentiles of skewness delta (sell)
                skew_sell_volume_delta_pivot = trade_data.pivot_table(
                    index = 'time_period_end',
                    columns = 'symbol_id',
                    values = f'spot_futures_skew_sell_dollar_volume_delta_{window}d',
                    dropna = False
                )
                cs_skew_sell_volume_delta_percentile = skew_sell_volume_delta_pivot.rank(axis = 1, pct = True)
                cs_skew_sell_volume_delta_percentile.columns = [col + f'cs_skew_sell_volume_delta_{window}d_percentile' for col in cs_skew_sell_volume_delta_percentile.columns]

                # Cross-sectional z-scores of skewness delta (sell)
                skew_sell_volume_delta_pivot_mean = skew_sell_volume_delta_pivot.mean(axis = 1)
                skew_sell_volume_delta_pivot_std = skew_sell_volume_delta_pivot.std(axis = 1)
                skew_sell_volume_delta_pivot_zscore = skew_sell_volume_delta_pivot.sub(skew_sell_volume_delta_pivot_mean, axis = 0).div(skew_sell_volume_delta_pivot_std, axis = 0)
                skew_sell_volume_delta_pivot_zscore.columns = [col + f'cs_skew_sell_volume_delta_{window}d_zscore' for col in skew_sell_volume_delta_pivot_zscore.columns]

                # 4 moments of skewness delta (sell)
                skew_sell_volume_delta_pivot[f'cs_avg_skew_sell_volume_delta_{window}d'] = skew_sell_volume_delta_pivot.mean(axis = 1)
                skew_sell_volume_delta_pivot[f'cs_std_skew_sell_volume_delta_{window}d'] = skew_sell_volume_delta_pivot.std(axis = 1)
                skew_sell_volume_delta_pivot[f'cs_skewness_skew_sell_volume_delta_{window}d'] = skew_sell_volume_delta_pivot.skew(axis = 1)
                skew_sell_volume_delta_pivot[f'cs_kurtosis_skew_sell_volume_delta_{window}d'] = skew_sell_volume_delta_pivot.kurtosis(axis = 1)
                skew_sell_volume_delta_pivot[f'cs_median_skew_sell_volume_delta_{window}d'] = skew_sell_volume_delta_pivot.median(axis = 1)

                # Append skewness delta percentile features to the basis features (sell)
                basis_features.append(cs_skew_sell_volume_delta_percentile)

                # Append skewness delta z-score features to the basis features (sell)
                basis_features.append(skew_sell_volume_delta_pivot_zscore)

                # Append skewness delta moments features to the basis features (sell)
                skew_sell_volume_delta_moments_features = [
                    skew_sell_volume_delta_pivot
                ]
                for i, moments_feature in enumerate(skew_sell_volume_delta_moments_features):
                    valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                    moments_feature = moments_feature[valid_cols]
                    basis_features.append(moments_feature)

                # Kurtosis delta features
                trade_data[f'spot_futures_kurtosis_buy_dollar_volume_delta_{window}d'] = trade_data[f'kurtosis_buy_dollar_volume_{window}d_futures'] - trade_data[f'kurtosis_buy_dollar_volume_{window}d_spot']
                trade_data[f'spot_futures_kurtosis_sell_dollar_volume_delta_{window}d'] = trade_data[f'kurtosis_sell_dollar_volume_{window}d_futures'] - trade_data[f'kurtosis_sell_dollar_volume_{window}d_spot']

                # Cross-sectional percentiles of kurtosis delta (buy)
                kurtosis_buy_volume_delta_pivot = trade_data.pivot_table(
                    index = 'time_period_end',
                    columns = 'symbol_id',
                    values = f'spot_futures_kurtosis_buy_dollar_volume_delta_{window}d',
                    dropna = False
                )
                cs_kurtosis_buy_volume_delta_percentile = kurtosis_buy_volume_delta_pivot.rank(axis = 1, pct = True)
                cs_kurtosis_buy_volume_delta_percentile.columns = [col + f'cs_kurtosis_buy_volume_delta_{window}d_percentile' for col in cs_kurtosis_buy_volume_delta_percentile.columns]

                # Cross-sectional z-scores of kurtosis delta (buy)
                kurtosis_buy_volume_delta_pivot_mean = kurtosis_buy_volume_delta_pivot.mean(axis = 1)
                kurtosis_buy_volume_delta_pivot_std = kurtosis_buy_volume_delta_pivot.std(axis = 1)
                kurtosis_buy_volume_delta_pivot_zscore = kurtosis_buy_volume_delta_pivot.sub(kurtosis_buy_volume_delta_pivot_mean, axis = 0).div(kurtosis_buy_volume_delta_pivot_std, axis = 0)
                kurtosis_buy_volume_delta_pivot_zscore.columns = [col + f'cs_kurtosis_buy_volume_delta_{window}d_zscore' for col in kurtosis_buy_volume_delta_pivot_zscore.columns]

                # 4 moments of kurtosis delta (buy)
                kurtosis_buy_volume_delta_pivot[f'cs_avg_kurtosis_buy_volume_delta_{window}d'] = kurtosis_buy_volume_delta_pivot.mean(axis = 1)
                kurtosis_buy_volume_delta_pivot[f'cs_std_kurtosis_buy_volume_delta_{window}d'] = kurtosis_buy_volume_delta_pivot.std(axis = 1)
                kurtosis_buy_volume_delta_pivot[f'cs_skewness_kurtosis_buy_volume_delta_{window}d'] = kurtosis_buy_volume_delta_pivot.skew(axis = 1)
                kurtosis_buy_volume_delta_pivot[f'cs_kurtosis_kurtosis_buy_volume_delta_{window}d'] = kurtosis_buy_volume_delta_pivot.kurtosis(axis = 1)
                kurtosis_buy_volume_delta_pivot[f'cs_median_kurtosis_buy_volume_delta_{window}d'] = kurtosis_buy_volume_delta_pivot.median(axis = 1)

                # Append kurtosis delta percentile features to the basis features (buy)
                basis_features.append(cs_kurtosis_buy_volume_delta_percentile)

                # Append kurtosis delta z-score features to the basis features (buy)
                basis_features.append(kurtosis_buy_volume_delta_pivot_zscore)

                # Append kurtosis delta moments features to the basis features (buy)
                kurtosis_buy_volume_delta_moments_features = [
                    kurtosis_buy_volume_delta_pivot
                ]
                for i, moments_feature in enumerate(kurtosis_buy_volume_delta_moments_features):
                    valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                    moments_feature = moments_feature[valid_cols]
                    basis_features.append(moments_feature)

                # Median delta features
                trade_data[f'spot_futures_median_buy_dollar_volume_delta_{window}d'] = trade_data[f'median_buy_dollar_volume_{window}d_futures'] - trade_data[f'median_buy_dollar_volume_{window}d_spot']
                trade_data[f'spot_futures_median_sell_dollar_volume_delta_{window}d'] = trade_data[f'median_sell_dollar_volume_{window}d_futures'] - trade_data[f'median_sell_dollar_volume_{window}d_spot']

                # Cross-sectional percentiles of median delta (buy)
                median_buy_volume_delta_pivot = trade_data.pivot_table(
                    index = 'time_period_end',
                    columns = 'symbol_id',
                    values = f'spot_futures_median_buy_dollar_volume_delta_{window}d',
                    dropna = False
                )
                cs_median_buy_volume_delta_percentile = median_buy_volume_delta_pivot.rank(axis = 1, pct = True)
                cs_median_buy_volume_delta_percentile.columns = [col + f'cs_median_buy_volume_delta_{window}d_percentile' for col in cs_median_buy_volume_delta_percentile.columns]

                # Cross-sectional z-scores of median delta (buy)
                median_buy_volume_delta_pivot_mean = median_buy_volume_delta_pivot.mean(axis = 1)
                median_buy_volume_delta_pivot_std = median_buy_volume_delta_pivot.std(axis = 1)
                median_buy_volume_delta_pivot_zscore = median_buy_volume_delta_pivot.sub(median_buy_volume_delta_pivot_mean, axis = 0).div(median_buy_volume_delta_pivot_std, axis = 0)
                median_buy_volume_delta_pivot_zscore.columns = [col + f'cs_median_buy_volume_delta_{window}d_zscore' for col in median_buy_volume_delta_pivot_zscore.columns]

                # 4 moments of median delta (buy)
                median_buy_volume_delta_pivot[f'cs_avg_median_buy_volume_delta_{window}d'] = median_buy_volume_delta_pivot.mean(axis = 1)
                median_buy_volume_delta_pivot[f'cs_std_median_buy_volume_delta_{window}d'] = median_buy_volume_delta_pivot.std(axis = 1)
                median_buy_volume_delta_pivot[f'cs_skewness_median_buy_volume_delta_{window}d'] = median_buy_volume_delta_pivot.skew(axis = 1)
                median_buy_volume_delta_pivot[f'cs_kurtosis_median_buy_volume_delta_{window}d'] = median_buy_volume_delta_pivot.kurtosis(axis = 1)
                median_buy_volume_delta_pivot[f'cs_median_median_buy_volume_delta_{window}d'] = median_buy_volume_delta_pivot.median(axis = 1)

                # Append median delta percentile features to the basis features (buy)
                basis_features.append(cs_median_buy_volume_delta_percentile)

                # Append median delta z-score features to the basis features (buy)
                basis_features.append(median_buy_volume_delta_pivot_zscore)

                # Append median delta moments features to the basis features (buy)
                median_buy_volume_delta_moments_features = [
                    median_buy_volume_delta_pivot
                ]
                for i, moments_feature in enumerate(median_buy_volume_delta_moments_features):
                    valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                    moments_feature = moments_feature[valid_cols]
                    basis_features.append(moments_feature)

                # Cross-sectional percentiles of median delta (sell)
                median_sell_volume_delta_pivot = trade_data.pivot_table(
                    index = 'time_period_end',
                    columns = 'symbol_id',
                    values = f'spot_futures_median_sell_dollar_volume_delta_{window}d',
                    dropna = False
                )
                cs_median_sell_volume_delta_percentile = median_sell_volume_delta_pivot.rank(axis = 1, pct = True)
                cs_median_sell_volume_delta_percentile.columns = [col + f'cs_median_sell_volume_delta_{window}d_percentile' for col in cs_median_sell_volume_delta_percentile.columns]

                # Cross-sectional z-scores of median delta (sell)
                median_sell_volume_delta_pivot_mean = median_sell_volume_delta_pivot.mean(axis = 1)
                median_sell_volume_delta_pivot_std = median_sell_volume_delta_pivot.std(axis = 1)
                median_sell_volume_delta_pivot_zscore = median_sell_volume_delta_pivot.sub(median_sell_volume_delta_pivot_mean, axis = 0).div(median_sell_volume_delta_pivot_std, axis = 0)
                median_sell_volume_delta_pivot_zscore.columns = [col + f'cs_median_sell_volume_delta_{window}d_zscore' for col in median_sell_volume_delta_pivot_zscore.columns]

                # 4 moments of median delta (sell)
                median_sell_volume_delta_pivot[f'cs_avg_median_sell_volume_delta_{window}d'] = median_sell_volume_delta_pivot.mean(axis = 1)
                median_sell_volume_delta_pivot[f'cs_std_median_sell_volume_delta_{window}d'] = median_sell_volume_delta_pivot.std(axis = 1)
                median_sell_volume_delta_pivot[f'cs_skewness_median_sell_volume_delta_{window}d'] = median_sell_volume_delta_pivot.skew(axis = 1)
                median_sell_volume_delta_pivot[f'cs_kurtosis_median_sell_volume_delta_{window}d'] = median_sell_volume_delta_pivot.kurtosis(axis = 1)
                median_sell_volume_delta_pivot[f'cs_median_median_sell_volume_delta_{window}d'] = median_sell_volume_delta_pivot.median(axis = 1)

                # Append median delta percentile features to the basis features (sell)
                basis_features.append(cs_median_sell_volume_delta_percentile)

                # Append median delta z-score features to the basis features (sell)
                basis_features.append(median_sell_volume_delta_pivot_zscore)

                # Append median delta moments features to the basis features (sell)
                median_sell_volume_delta_moments_features = [
                    median_sell_volume_delta_pivot
                ]
                for i, moments_feature in enumerate(median_sell_volume_delta_moments_features):
                    valid_cols = [col for col in moments_feature.columns if col.startswith('cs_')]
                    moments_feature = moments_feature[valid_cols]
                    basis_features.append(moments_feature)

        # Concatenate all basis features
        basis_features = pd.concat(basis_features, axis = 1)
        self.basis_features = basis_features
        self.basis_features = basis_features.reset_index()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        asset_id_base, asset_id_quote, exchange_id = X['symbol_id'].iloc[0].split('_')
        symbol_id = f'{asset_id_base}_{asset_id_quote}_{exchange_id}'

        # Get the basis features for the current token
        X = pd.merge(
            X,
            self.price_data,
            how='left',
            left_on=['time_period_end', 'symbol_id'],
            right_on=['time_period_end', 'symbol_id'],
            suffixes=('', '__remove')
        )
        X = X.drop([col for col in X.columns if col.endswith('__remove')], axis=1)

        # Get the trade features for the current token
        X = pd.merge(
            X,
            self.trade_data,
            how='left',
            left_on=['time_period_end', 'symbol_id'],
            right_on=['time_period_end', 'symbol_id'],
            suffixes=('', '__remove')
        )
        X = X.drop([col for col in X.columns if col.endswith('__remove')], axis=1)

        for lookback in self.lookback_windows:
            # Rolling 4 moments of the basis features
            X[f'avg_basis_{lookback}d'] = X['basis'].rolling(window = lookback).mean()
            X[f'std_basis_{lookback}d'] = X['basis'].rolling(window = lookback).std()
            X[f'skewness_basis_{lookback}d'] = X['basis'].rolling(window = lookback).skew()
            X[f'kurtosis_basis_{lookback}d'] = X['basis'].rolling(window = lookback).kurt()
            X[f'median_basis_{lookback}d'] = X['basis'].rolling(window = lookback).median()
            # Rolling 4 moments of the basis pct features
            X[f'avg_basis_pct_{lookback}d'] = X['basis_pct'].rolling(window = lookback).mean()
            X[f'std_basis_pct_{lookback}d'] = X['basis_pct'].rolling(window = lookback).std()
            X[f'skewness_basis_pct_{lookback}d'] = X['basis_pct'].rolling(window = lookback).skew()
            X[f'kurtosis_basis_pct_{lookback}d'] = X['basis_pct'].rolling(window = lookback).kurt()
            X[f'median_basis_pct_{lookback}d'] = X['basis_pct'].rolling(window = lookback).median()
        for window in self.windows:
            for lookback in self.lookback_windows:
                # Rolling 4 moments of the volume delta features (buy)
                X[f'avg_spot_futures_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                X[f'std_spot_futures_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                X[f'skewness_spot_futures_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                X[f'kurtosis_spot_futures_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                X[f'median_spot_futures_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                # Rolling 4 moments of the volume delta features (sell)
                X[f'avg_spot_futures_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                X[f'std_spot_futures_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                X[f'skewness_spot_futures_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                X[f'kurtosis_spot_futures_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                X[f'median_spot_futures_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                # Rolling 4 moments of the trade imbalance features
                X[f'avg_spot_futures_trade_imbalance_delta_{window}d_{lookback}d'] = X[f'spot_futures_trade_imbalance_delta_{window}d'].rolling(window = lookback).mean()
                X[f'std_spot_futures_trade_imbalance_delta_{window}d_{lookback}d'] = X[f'spot_futures_trade_imbalance_delta_{window}d'].rolling(window = lookback).std()
                X[f'skewness_spot_futures_trade_imbalance_delta_{window}d_{lookback}d'] = X[f'spot_futures_trade_imbalance_delta_{window}d'].rolling(window = lookback).skew()
                X[f'kurtosis_spot_futures_trade_imbalance_delta_{window}d_{lookback}d'] = X[f'spot_futures_trade_imbalance_delta_{window}d'].rolling(window = lookback).kurt()
                X[f'median_spot_futures_trade_imbalance_delta_{window}d_{lookback}d'] = X[f'spot_futures_trade_imbalance_delta_{window}d'].rolling(window = lookback).median()

                # Rolling 4 moments of the avg dollar volume delta feature (buy)
                X[f'avg_spot_futures_avg_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                X[f'std_spot_futures_avg_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                X[f'skewness_spot_futures_avg_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                X[f'kurtosis_spot_futures_avg_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                X[f'median_spot_futures_avg_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                # Rolling 4 moments of the avg dollar volume delta feature (sell)
                X[f'avg_spot_futures_avg_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                X[f'std_spot_futures_avg_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                X[f'skewness_spot_futures_avg_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                X[f'kurtosis_spot_futures_avg_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                X[f'median_spot_futures_avg_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_avg_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                if window == 1:
                    # Rolling 4 moments of the std dollar volume delta feature (buy)
                    X[f'avg_spot_futures_std_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                    X[f'std_spot_futures_std_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                    X[f'skewness_spot_futures_std_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                    X[f'kurtosis_spot_futures_std_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                    X[f'median_spot_futures_std_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                    # Rolling 4 moments of the std dollar volume delta feature (sell)
                    X[f'avg_spot_futures_std_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                    X[f'std_spot_futures_std_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                    X[f'skewness_spot_futures_std_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                    X[f'kurtosis_spot_futures_std_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                    X[f'median_spot_futures_std_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_std_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                    # Rolling 4 moments of the skew dollar volume delta feature (buy)
                    X[f'avg_spot_futures_skew_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                    X[f'std_spot_futures_skew_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                    X[f'skewness_spot_futures_skew_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                    X[f'kurtosis_spot_futures_skew_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                    X[f'median_spot_futures_skew_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                    # Rolling 4 moments of the skew dollar volume delta feature (sell)
                    X[f'avg_spot_futures_skew_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                    X[f'std_spot_futures_skew_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                    X[f'skewness_spot_futures_skew_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                    X[f'kurtosis_spot_futures_skew_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                    X[f'median_spot_futures_skew_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_skew_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                    # Rolling 4 moments of the kurtosis dollar volume delta feature (buy)
                    X[f'avg_spot_futures_kurtosis_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                    X[f'std_spot_futures_kurtosis_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                    X[f'skewness_spot_futures_kurtosis_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                    X[f'kurtosis_spot_futures_kurtosis_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                    X[f'median_spot_futures_kurtosis_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                    # Rolling 4 moments of the kurtosis dollar volume delta feature (sell)
                    X[f'avg_spot_futures_kurtosis_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                    X[f'std_spot_futures_kurtosis_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                    X[f'skewness_spot_futures_kurtosis_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                    X[f'kurtosis_spot_futures_kurtosis_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                    X[f'median_spot_futures_kurtosis_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_kurtosis_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                    # Rolling 4 moments of the median dollar volume delta feature (buy)
                    X[f'avg_spot_futures_median_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                    X[f'std_spot_futures_median_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                    X[f'skewness_spot_futures_median_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                    X[f'kurtosis_spot_futures_median_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                    X[f'median_spot_futures_median_buy_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_buy_dollar_volume_delta_{window}d'].rolling(window = lookback).median()

                    # Rolling 4 moments of the median dollar volume delta feature (sell)
                    X[f'avg_spot_futures_median_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).mean()
                    X[f'std_spot_futures_median_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).std()
                    X[f'skewness_spot_futures_median_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).skew()
                    X[f'kurtosis_spot_futures_median_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).kurt()
                    X[f'median_spot_futures_median_sell_dollar_volume_delta_{window}d_{lookback}d'] = X[f'spot_futures_median_sell_dollar_volume_delta_{window}d'].rolling(window = lookback).median()
        
        cross_sectional_ranking_cols = self.basis_features.filter(like = symbol_id).columns.to_list()
        cross_sectional_moments_cols = [col for col in self.basis_features.columns if col.startswith('cs_')]
        cross_sectional_ema_cols = [col for col in self.basis_features.columns if col.startswith('rolling_ema_')]
        valid_cols = cross_sectional_ranking_cols + cross_sectional_moments_cols + cross_sectional_ema_cols

        # Merge the cross-sectional features into the main DataFrame
        X = pd.merge(
            X, 
            self.basis_features[['time_period_end'] + valid_cols],
            how='left',
            on='time_period_end',
            suffixes=('', '__remove')
        )
        X = X.drop(columns=[col for col in X.columns if '__remove' in col])

        # Rename the cross-sectional ranking columns to remove symbol_id
        for col in cross_sectional_ranking_cols:
            new_col_name = col.replace(symbol_id, '')
            X.rename({col: new_col_name}, inplace=True, axis=1)

        return X

