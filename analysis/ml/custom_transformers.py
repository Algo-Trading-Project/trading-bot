from sklearn.base import BaseEstimator, TransformerMixin
from utils.db_utils import QUERY
from analysis.ml.labeling import calculate_triple_barrier_labels

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

        for col in X.columns:
            if col in ('symbol_id', 'asset_id_base', 'asset_id_quote', 'exchange_id', 'time_period_end') \
            or col in triple_barrier_label_cols \
            or col in ('hour', 'day_of_week', 'day_of_month', 'month', 'year', 'is_holiday') \
            or col in trade_returns_cols:
                continue

            for window_size in self.window_sizes:
                col_name = col + '_rz_' + str(window_size)
                rolling_mean = X[col].shift(1).rolling(window = window_size, min_window = 7).mean()
                rolling_std = X[col].shift(1).rolling(window = window_size, min_window = 7).std()
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

        # 1 day ml_dataset
        self.ml_dataset = QUERY(
            """
            SELECT *
            FROM market_data.ml_dataset
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        self.ml_dataset['symbol_id'] = self.ml_dataset['asset_id_base'] + '_' + self.ml_dataset['asset_id_quote'] + '_' + self.ml_dataset['exchange_id']
        self.ml_dataset['time_period_end'] = pd.to_datetime(self.ml_dataset['time_period_end'])
        tokens = sorted(self.ml_dataset['symbol_id'].unique().tolist())

        # Calculate cross-sectional 1d returns features
        final_features = []

        # N-day returns
        for window in self.window_sizes:
            self.ml_dataset[f'returns_{window}'] = np.nan
            self.ml_dataset[f'log_returns_{window}'] = np.nan

        for token in tokens:
            filter = self.ml_dataset['symbol_id'] == token
            for window in self.window_sizes:
                if window == 1:
                    clip_upper_bound = 1
                elif window == 7:
                    clip_upper_bound = 3
                else:
                    clip_upper_bound = 5

                self.ml_dataset.loc[filter, f'returns_{window}'] = self.ml_dataset.loc[filter, 'close'].pct_change(window).clip(-1, clip_upper_bound)
                self.ml_dataset.loc[filter, f'log_returns_{window}'] = np.log(self.ml_dataset.loc[filter, 'close'] / self.ml_dataset.loc[filter, 'close'].shift(window))

                for lookback in self.lookback_windows:
                    # Avg returns
                    self.ml_dataset.loc[filter, f'avg_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'returns_{window}'].rolling(lookback, min_window = 7).mean()
                    # Std returns
                    self.ml_dataset.loc[filter, f'std_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'returns_{window}'].rolling(lookback, min_window = 7).std()
                    # Sharpe ratio
                    self.ml_dataset.loc[filter, f'sharpe_ratio_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'avg_returns_{window}_{lookback}'] / self.ml_dataset.loc[filter, f'std_returns_{window}_{lookback}']
                    # Std negative returns
                    self.ml_dataset.loc[filter, f'std_negative_returns_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'returns_{window}'].rolling(lookback, min_window = 7).apply(lambda x: np.std(x[x < 0]), raw = False)
                    # Sortino ratio
                    self.ml_dataset.loc[filter, f'sortino_ratio_{window}_{lookback}'] = self.ml_dataset.loc[filter, f'avg_returns_{window}_{lookback}'] / self.ml_dataset.loc[filter, f'std_negative_returns_{window}_{lookback}']

        for window in self.window_sizes:
            print(f'Calculating cross-sectional returns for window size {window}')

            returns_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'returns_{window}', dropna = False)
            returns_pivot_percentile = returns_pivot.rank(axis = 1, pct = True)
            returns_pivot_percentile.columns = [col + f'_returns_percentile_{window}' for col in returns_pivot.columns]

            final_features.append(returns_pivot_percentile)

        for lookback in self.lookback_windows:
            # Cross-sectional sharpe ratios for each symbol for each time period
            pivot_sharpe = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'sharpe_ratio_1_{lookback}', dropna = False)
            pivot_sharpe_decile = pivot_sharpe.rank(axis = 1, pct = True)
            pivot_sharpe_decile.columns = [col + f'_sharpe_percentile_{lookback}' for col in pivot_sharpe_decile.columns]
            final_features.append(pivot_sharpe_decile)

            # Cross-sectional sortino ratios for each symbol for each time period
            pivot_sortino = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'sortino_ratio_1_{lookback}', dropna = False)
            pivot_sortino_decile = pivot_sortino.rank(axis = 1, pct = True)
            pivot_sortino_decile.columns = [col + f'_sortino_percentile_{lookback}' for col in pivot_sortino_decile.columns]
            final_features.append(pivot_sortino_decile)

        self.final_features = pd.concat(final_features, axis = 1)
        self.final_features = self.final_features.reset_index()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['dollar_volume'] = X['close'] * X['volume']
        for window_size in self.window_sizes:
            if window_size == 1:
                clip_upper_bound = 1
            elif window_size == 7:
                clip_upper_bound = 3
            else:
                clip_upper_bound = 5

            X[f'returns_{window_size}'] = X['close'].pct_change(window_size).clip(-1, clip_upper_bound)
            X[f'returns_high_{window_size}'] = X['high'].pct_change(window_size).clip(-1, clip_upper_bound)
            X[f'returns_low_{window_size}'] = X['low'].pct_change(window_size).clip(-1, clip_upper_bound)
            X[f'returns_open_{window_size}'] = X['open'].pct_change(window_size).clip(-1, clip_upper_bound)
            X[f'returns_volume_{window_size}'] = X['volume'].pct_change(window_size).clip(-1, clip_upper_bound)
            X[f'returns_dollar_volume_{window_size}'] = X['dollar_volume'].pct_change(window_size).clip(-1, clip_upper_bound)
            X[f'forward_returns_{window_size}'] = X[f'returns_{window_size}'].shift(-window_size)

            # Returns per dollar volume
            X[f'returns_per_dollar_volume_{window_size}'] = X[f'returns_{window_size}'] / X['dollar_volume']

            # Absolute returns per dollar volume
            X[f'abs_returns_per_dollar_volume_{window_size}'] = X[f'returns_{window_size}'].abs() / X['dollar_volume']

            for lookback_window in self.lookback_windows:
                # Calculate returns distributional features
                X[f'avg_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).mean()
                X[f'10th_percentile_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).quantile(0.1)
                X[f'median_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).median()
                X[f'90th_percentile_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).quantile(0.9)
                X[f'std_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).std()
                X[f'skewness_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).skew()
                X[f'kurtosis_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).kurt()

                # Sharpe ratio
                X[f'sharpe_ratio_{window_size}_{lookback_window}'] = X[f'avg_returns_{window_size}_{lookback_window}'] / X[f'std_returns_{window_size}_{lookback_window}']

                # Sortino ratio
                X[f'std_negative_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).apply(lambda x: np.std(x[x < 0]), raw = False)
                X[f'sortino_ratio_{window_size}_{lookback_window}'] = X[f'avg_returns_{window_size}_{lookback_window}'] / X[f'std_negative_returns_{window_size}_{lookback_window}']

        # Cross-sectional rank features for current token and time period
        symbol_id = X.iloc[0]['symbol_id']

        # Get cross_sectional decile columns for the current token
        cross_sectional_decile_cols = [col for col in self.final_features.columns if col.startswith(symbol_id)]
        valid_cols = cross_sectional_decile_cols

        # Merge data to final features
        X = pd.merge(X, self.final_features[['time_period_end'] + valid_cols], on = 'time_period_end', how = 'left', suffixes = ('', '__remove'))

        # Drop the columns with '__remove' suffix
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)

        # Rename the cross-sectional decile columns to remove the symbol_id from it
        for col in cross_sectional_decile_cols:
            new_col = col.replace(symbol_id + '_', '')
            X = X.rename(columns = {col: new_col})

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
            eth_usd_coinbase[f'returns_{window_size}'] = eth_usd_coinbase['close'].pct_change(window_size)
            btc_usd_coinbase[f'returns_{window_size}'] = btc_usd_coinbase['close'].pct_change(window_size)

        X.time_period_end = pd.to_datetime(X.time_period_end)

        merged = pd.merge(X, eth_usd_coinbase, on = 'time_period_end', suffixes = ('', '_ETH'), how = 'left',)
        merged = pd.merge(merged, btc_usd_coinbase, on = 'time_period_end', suffixes = ('', '_BTC'), how = 'left')

        for window_size in self.window_sizes:
            for lookback_window in self.lookback_windows:
                # Calculate autocorrelation of returns
                X[f'autocorrelation_returns_{window_size}_{lookback_window}'] = X[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7).corr(X[f'returns_{window_size}'])

                merged_rolling = merged[f'returns_{window_size}'].rolling(window = lookback_window, min_window = 7)

                # Calculate cross-correlation of returns with ETH
                X[f'cross_correlation_returns_{window_size}_{lookback_window}_ETH'] = merged_rolling.corr(merged[f'returns_{window_size}_ETH'])

                # Calculate cross-correlation of returns with BTC
                X[f'cross_correlation_returns_{window_size}_{lookback_window}_BTC'] = merged_rolling.corr(merged[f'returns_{window_size}_BTC'])

        return X
        
class OrderBookFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

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

        # Normalize the spread by the mid-price
        ob_1d['normalize_quoted_spread'] = (ob_1d[ask_cols[0]] - ob_1d[bid_cols[0]]) / ((ob_1d[ask_cols[0]] + ob_1d[bid_cols[0]]) / 2)

        # Drop unnecessary columns
        ob_1d = ob_1d.drop(columns = bid_cols + ask_cols + bid_size_cols + ask_size_cols, axis = 1)

        # Merge the order book features with the original data
        X = pd.merge(X, ob_1d, on = 'time_period_end', suffixes = ('', '__remove'), how = 'left')
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)
        
        return X

class FillNaTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col == 'time_period_end':
                continue
            elif X[col].dtype in ('O', 'object', 'category'):
                if X[col].isnull().sum() > 0:
                    mode = X[col].mode().loc[0]
                    X[col] = X[col].fillna(mode)
            else:
                if X[col].isnull().sum() > 0:
                    # Fill missing values with the rolling mean
                    X[col] = X[col].fillna(X[col].rolling(window = 7).mean().fillna(0))

        return X

class TradeFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, windows, lookback_windows = (30, 60, 90, 180)):
        self.windows = windows
        self.lookback_windows = lookback_windows

        trade_features = QUERY(
            """
            SELECT *
            FROM market_data.trade_features_rolling
            ORDER BY asset_id_base, asset_id_quote, exchange_id, time_period_end
            """
        )
        trade_features['symbol_id'] = trade_features['asset_id_base'] + '_' + trade_features['asset_id_quote'] + '_' + trade_features['exchange_id']
        trade_features['time_period_end'] = pd.to_datetime(trade_features['time_period_end'])

        final_features = []

        for window in self.windows:
            # Cross-sectional features
            print(f'Calculating cross-sectional features for window size {window}...')
            print()
            total_buy_dollar_volume_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'total_buy_dollar_volume_{window}d', dropna=False).sort_index()
            total_buy_dollar_volume_percentile = total_buy_dollar_volume_pivot.rank(axis=1, pct=True)
            total_buy_dollar_volume_percentile.columns = [f'{col}_total_buy_dollar_volume_percentile_{window}d' for col in total_buy_dollar_volume_percentile.columns]

            total_sell_dollar_volume_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'total_sell_dollar_volume_{window}d', dropna=False).sort_index()
            total_sell_dollar_volume_percentile = total_sell_dollar_volume_pivot.rank(axis=1, pct=True)
            total_sell_dollar_volume_percentile.columns = [f'{col}_total_sell_dollar_volume_percentile_{window}d' for col in total_sell_dollar_volume_percentile.columns]

            num_buys_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'num_buys_{window}d', dropna=False).sort_index()
            num_buys_percentile = num_buys_pivot.rank(axis=1, pct=True)
            num_buys_percentile.columns = [f'{col}_num_buys_percentile_{window}d' for col in num_buys_percentile.columns]

            num_sells_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'num_sells_{window}d', dropna=False).sort_index()
            num_sells_percentile = num_sells_pivot.rank(axis=1, pct=True)
            num_sells_percentile.columns = [f'{col}_num_sells_percentile_{window}d' for col in num_sells_percentile.columns]

            pct_buys_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'pct_buys_{window}d', dropna=False).sort_index()
            pct_buys_percentile = pct_buys_pivot.rank(axis=1, pct=True)
            pct_buys_percentile.columns = [f'{col}_pct_buys_percentile_{window}d' for col in pct_buys_percentile.columns]

            pct_sells_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'pct_sells_{window}d', dropna=False).sort_index()
            pct_sells_percentile = pct_sells_pivot.rank(axis=1, pct=True)
            pct_sells_percentile.columns = [f'{col}_pct_sells_percentile_{window}d' for col in pct_sells_percentile.columns]

            trade_dollar_volume_imbalance_pivot = trade_features.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'trade_imbalance_{window}d', dropna=False).sort_index()
            trade_dollar_volume_imbalance_percentile = trade_dollar_volume_imbalance_pivot.rank(axis=1, pct=True)
            trade_dollar_volume_imbalance_percentile.columns = [f'{col}_trade_dollar_volume_imbalance_percentile_{window}d' for col in trade_dollar_volume_imbalance_percentile.columns]

            pct_buy_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id',values=f'pct_buy_dollar_volume_{window}d', dropna=False).sort_index()
            pct_buy_dollar_volume_percentile = pct_buy_dollar_volume_pivot.rank(axis=1, pct=True)
            pct_buy_dollar_volume_percentile.columns = [f'{col}_pct_buy_dollar_volume_percentile_{window}d' for col in pct_buy_dollar_volume_percentile.columns]

            pct_sell_dollar_volume_pivot = trade_features.pivot_table(index='time_period_end', columns='symbol_id',values=f'pct_sell_dollar_volume_{window}d', dropna=False).sort_index()
            pct_sell_dollar_volume_percentile = pct_sell_dollar_volume_pivot.rank(axis=1, pct=True)
            pct_sell_dollar_volume_percentile.columns = [f'{col}_pct_sell_dollar_volume_percentile_{window}d' for col in pct_sell_dollar_volume_percentile.columns]

            # Append the features to the final list
            final_features.append(total_buy_dollar_volume_percentile)
            final_features.append(total_sell_dollar_volume_percentile)
            final_features.append(num_buys_percentile)
            final_features.append(num_sells_percentile)
            final_features.append(pct_buys_percentile)
            final_features.append(pct_sells_percentile)
            final_features.append(trade_dollar_volume_imbalance_percentile)
            final_features.append(pct_buy_dollar_volume_percentile)
            final_features.append(pct_sell_dollar_volume_percentile)

        trade_features = pd.concat(final_features, axis = 1)
        self.trade_features = trade_features.reset_index()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Cross-sectional rank features for current token and time period
        asset_id_base, asset_id_quote, exchange_id = X['symbol_id'].iloc[0].split('_')
        symbol_id = f'{asset_id_base}_{asset_id_quote}_{exchange_id}'
        curr_token_trade_features = QUERY(
            f"""
            SELECT *
            FROM market_data.trade_features_rolling
            WHERE
                asset_id_base = '{asset_id_base}' AND
                asset_id_quote = '{asset_id_quote}' AND
                exchange_id = '{exchange_id}'
            ORDER BY time_period_end 
            """
        )
        curr_token_trade_features['time_period_end'] = pd.to_datetime(curr_token_trade_features['time_period_end'])

        for window in self.windows:
            for lookback in self.lookback_windows:
                # Rolling five moments of pct_buys
                curr_token_trade_features[f'avg_pct_buys_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buys_{window}d'].rolling(window = lookback, min_window = 7).mean()
                curr_token_trade_features[f'std_pct_buys_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buys_{window}d'].rolling(window = lookback, min_window = 7).std()
                curr_token_trade_features[f'skewness_pct_buys_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buys_{window}d'].rolling(window = lookback, min_window = 7).skew()
                curr_token_trade_features[f'kurtosis_pct_buys_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buys_{window}d'].rolling(window = lookback, min_window = 7).kurt()
                curr_token_trade_features[f'median_pct_buys_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buys_{window}d'].rolling(window = lookback, min_window = 7).median()
                curr_token_trade_features[f'10th_percentile_pct_buys_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buys_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.1)
                curr_token_trade_features[f'90th_percentile_pct_buys_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buys_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.9)

                # Rolling five moments of pct_sells
                curr_token_trade_features[f'avg_pct_sells_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sells_{window}d'].rolling(window = lookback, min_window = 7).mean()
                curr_token_trade_features[f'std_pct_sells_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sells_{window}d'].rolling(window = lookback, min_window = 7).std()
                curr_token_trade_features[f'skewness_pct_sells_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sells_{window}d'].rolling(window = lookback, min_window = 7).skew()
                curr_token_trade_features[f'kurtosis_pct_sells_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sells_{window}d'].rolling(window = lookback, min_window = 7).kurt()
                curr_token_trade_features[f'median_pct_sells_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sells_{window}d'].rolling(window = lookback, min_window = 7).median()
                curr_token_trade_features[f'10th_percentile_pct_sells_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sells_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.1)
                curr_token_trade_features[f'90th_percentile_pct_sells_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sells_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.9)

                # Rolling five moments of trade dollar volume imbalance
                curr_token_trade_features[f'avg_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = curr_token_trade_features[f'trade_imbalance_{window}d'].rolling(window = lookback, min_window = 7).mean()
                curr_token_trade_features[f'std_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = curr_token_trade_features[f'trade_imbalance_{window}d'].rolling(window = lookback, min_window = 7).std()
                curr_token_trade_features[f'skewness_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = curr_token_trade_features[f'trade_imbalance_{window}d'].rolling(window = lookback, min_window = 7).skew()
                curr_token_trade_features[f'kurtosis_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = curr_token_trade_features[f'trade_imbalance_{window}d'].rolling(window = lookback, min_window = 7).kurt()
                curr_token_trade_features[f'median_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = curr_token_trade_features[f'trade_imbalance_{window}d'].rolling(window = lookback, min_window = 7).median()
                curr_token_trade_features[f'10th_percentile_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = curr_token_trade_features[f'trade_imbalance_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.1)
                curr_token_trade_features[f'90th_percentile_trade_dollar_volume_imbalance_{window}d_{lookback}d'] = curr_token_trade_features[f'trade_imbalance_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.9)

                # Rolling five moments of pct buy dollar volume
                curr_token_trade_features[f'avg_pct_buy_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buy_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).mean()
                curr_token_trade_features[f'std_pct_buy_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buy_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).std()
                curr_token_trade_features[f'skewness_pct_buy_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buy_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).skew()
                curr_token_trade_features[f'kurtosis_pct_buy_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buy_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).kurt()
                curr_token_trade_features[f'median_pct_buy_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buy_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).median()
                curr_token_trade_features[f'10th_percentile_pct_buy_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buy_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.1)
                curr_token_trade_features[f'90th_percentile_pct_buy_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_buy_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.9)

                # Rolling five moments of pct sell dollar volume
                curr_token_trade_features[f'avg_pct_sell_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sell_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).mean()
                curr_token_trade_features[f'std_pct_sell_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sell_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).std()
                curr_token_trade_features[f'skewness_pct_sell_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sell_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).skew()
                curr_token_trade_features[f'kurtosis_pct_sell_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sell_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).kurt()
                curr_token_trade_features[f'median_pct_sell_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sell_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).median()
                curr_token_trade_features[f'10th_percentile_pct_sell_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sell_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.1)
                curr_token_trade_features[f'90th_percentile_pct_sell_dollar_volume_{window}d_{lookback}d'] = curr_token_trade_features[f'pct_sell_dollar_volume_{window}d'].rolling(window = lookback, min_window = 7).quantile(0.9)

        cross_sectional_percentile_cols = [col for col in self.trade_features.columns if col.startswith(symbol_id)]

        # Merge current token trade features with the cross-sectional percentile features
        curr_token_trade_features = pd.merge(curr_token_trade_features, self.trade_features[['time_period_end'] + cross_sectional_percentile_cols], on = 'time_period_end', how = 'left', suffixes = ('', '__remove'))
        curr_token_trade_features = curr_token_trade_features.drop(columns = [col for col in curr_token_trade_features.columns if '__remove' in col], axis = 1)

        # Rename the cross-sectional percentile columns to remove the symbol_id from it
        for col in cross_sectional_percentile_cols:
            new_col = col.replace(symbol_id + '_', '')
            curr_token_trade_features = curr_token_trade_features.rename(columns = {col: new_col})

        # Merge data to final features
        X = pd.merge(X, curr_token_trade_features, on = 'time_period_end', suffixes = ('', '__remove'), how = 'left')

        # Drop the columns with '__remove' suffix
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)

        # # Drop duplicate columns
        X = X.loc[:,~X.columns.duplicated()].copy()

        return X

class RiskFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, windows, lookback_windows):
        self.windows = windows
        self.lookback_windows = lookback_windows

        # 1 day ml_dataset
        self.ml_dataset = QUERY(
            """
            SELECT *
            FROM market_data.ml_dataset
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
                    clip_upper_bound = 1
                elif window == 7:
                    clip_upper_bound = 3
                else:
                    clip_upper_bound = 5

                # Calculate returns
                self.ml_dataset.loc[filter, f'returns_{window}'] = self.ml_dataset.loc[filter, 'close'].pct_change(window).clip(-1, clip_upper_bound)

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
                        / self.ml_dataset.loc[filter, f'market_returns_{window}'].rolling(window = lookback_window, min_windows = 7).var()
                    )

                    # Rolling alpha
                    self.ml_dataset.loc[filter, f'alpha_{window}d_{lookback_window}d'] = (
                        self.ml_dataset.loc[filter, f'returns_{window}']
                        .rolling(window = lookback_window)
                        .mean()
                        - self.ml_dataset.loc[filter, f'beta_{window}d_{lookback_window}d'] * self.ml_dataset.loc[filter, f'market_returns_{window}'].rolling(window = lookback_window, min_windows = 7).mean()
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

                # Cross-sectional alpha
                alpha_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'alpha_{window}d_{lookback_window}d', dropna = False)
                cs_alpha_percentile = alpha_pivot.rank(axis = 1, pct = True)
                cs_alpha_percentile.columns = [col + f'_alpha_percentile_{lookback_window}' for col in cs_alpha_percentile.columns]

                alpha_pivot[f'cs_avg_alpha_{window}d_{lookback_window}d'] = alpha_pivot.mean(axis = 1)
                alpha_pivot[f'avg_cs_avg_alpha_{window}d_{lookback_window}d'] = alpha_pivot[f'cs_avg_alpha_{window}d_{lookback_window}d'].rolling(window = lookback_window, min_windows = 7).mean()

                # Cross-sectional beta
                beta_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'beta_{window}d_{lookback_window}d', dropna = False)
                cs_beta_percentile = beta_pivot.rank(axis = 1, pct = True)
                cs_beta_percentile.columns = [col + f'_beta_percentile_{lookback_window}' for col in cs_beta_percentile.columns]

                beta_pivot[f'cs_avg_beta_{window}d_{lookback_window}d'] = beta_pivot.mean(axis = 1)
                beta_pivot[f'avg_cs_avg_beta_{window}d_{lookback_window}d'] = beta_pivot[f'cs_avg_beta_{window}d_{lookback_window}d'].rolling(window = lookback_window, min_windows = 7).mean()

                # Cross-sectional alpha_div_beta
                alpha_div_beta_pivot = self.ml_dataset.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = f'alpha_div_beta_{window}d_{lookback_window}d', dropna = False)
                cs_alpha_div_beta_percentile = alpha_div_beta_pivot.rank(axis = 1, pct = True)
                cs_alpha_div_beta_percentile.columns = [col + f'_alpha_div_beta_percentile_{lookback_window}' for col in cs_alpha_div_beta_percentile.columns]

                final_features.append(alpha_pivot)
                final_features.append(beta_pivot)
                final_features.append(alpha_div_beta_pivot)
                final_features.append(cs_alpha_div_beta_percentile)
                final_features.append(cs_alpha_percentile)
                final_features.append(cs_beta_percentile)

        final_features = pd.concat(final_features, axis = 1)
        self.risk_features = final_features.reset_index()
        self.ml_dataset = self.ml_dataset.reset_index()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Cross-sectional rank features for current token and time period
        symbol_id = X['symbol_id'].iloc[0]

        alpha_cols = list(set([col for col in self.ml_dataset.columns if 'alpha' in col]))
        beta_cols = list(set([col for col in self.ml_dataset.columns if 'beta' in col]))
        alpha_div_beta_cols = list(set([col for col in self.ml_dataset.columns if 'alpha_div_beta' in col]))
        risk_cols = [col for col in self.risk_features.columns if col.startswith(symbol_id + '_')]

        # Filter the ml_dataset for the current token and columns of interest
        data = self.ml_dataset[self.ml_dataset['symbol_id'] == symbol_id][['time_period_end'] + alpha_cols + beta_cols + alpha_div_beta_cols]

        # Merge X to ml_dataset for rolling alpha and beta features
        X = pd.merge(X, data, on = 'time_period_end', suffixes = ('', '__remove'), how = 'left')

        # Merge data to risk_features for cross-sectional risk features
        X = pd.merge(X, self.risk_features[['time_period_end'] + risk_cols], on = 'time_period_end', suffixes = ('', '__remove'), how = 'left')

        # Drop the columns with '__remove' suffix
        X = X.drop(columns = [col for col in X.columns if '__remove' in col], axis = 1)

        # Rename the cross-sectional decile columns to remove the symbol_id from it
        for col in risk_cols:
            new_col = col.replace(symbol_id + '_', '')
            X = X.rename(columns = {col: new_col})

        # Drop duplicate columns
        X = X.loc[:,~X.columns.duplicated()].copy()

        return X
