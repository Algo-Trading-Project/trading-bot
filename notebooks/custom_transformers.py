from sklearn.base import BaseEstimator, TransformerMixin
from helper import execute_query
import pandas as pd

class RollingMinMaxScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        new_cols = []
        for col in X_copy.columns:
            if '_rz_' in col:
                continue

            for window_size in self.window_sizes:
                col_name = col + '_rmm_' + str(window_size)
                rolling_min = X_copy[col].rolling(window = window_size).min()
                rolling_max = X_copy[col].rolling(window = window_size).max()
                new_col = pd.Series((X_copy[col] - rolling_min) / (rolling_max - rolling_min), name = col_name)

                new_cols.append(new_col)
        
        X_copy = pd.concat([X_copy] + new_cols, axis = 1)
        return X_copy

class RollingZScoreScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        new_cols = []

        for col in X_copy.columns:
            if '_rmm_' in col:
                continue

            for window_size in self.window_sizes:
                col_name = col + '_rz_' + str(window_size)
                rolling_mean = X_copy[col].rolling(window = window_size).mean()
                rolling_std = X_copy[col].rolling(window = window_size).std()
                new_col = pd.Series((X_copy[col] - rolling_mean) / rolling_std, name = col_name)

                new_cols.append(new_col)   

        X_copy = pd.concat([X_copy] + new_cols, axis = 1)
        
        return X_copy

class LagFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, lags):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = []
        for col in X_copy.columns:
            for lag in self.lags:
                col_name = col + '_lag_' + str(lag)
                cols.append(pd.Series(X_copy[col].shift(lag), name = col_name))

        X_copy = pd.concat([X_copy] + cols, axis = 1)

        return X_copy.dropna()

class BlockFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        query = """
        SELECT *
        FROM administrator.metrics.block_metrics
        ORDER BY timestamp ASC
        """
        block_cols = [
            'timestamp', 'avg_block_size', 'block_count', 'avg_block_reward',
            'avg_difficulty', 'num_unique_miners', 'difficulty_adjustment', 
            'pct_gas_limit_used'
        ]
        
        block_data = execute_query(
            query = query,
            cols = block_cols,
            date_col = 'timestamp'
        )

        self.block_data = block_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        merged = pd.merge(X, self.block_data, how = 'inner', left_index = True, right_index = True)
        return merged

class TransactionFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        query = """
        SELECT *
        FROM administrator.metrics.transaction_metrics
        ORDER BY "timestamp" ASC
        """
        transaction_cols = [
            'timestamp', 'transaction_volume', 'avg_transaction_size', 'num_transactions', 
            'num_transactions_gt_1000_eth', 'num_transactions_gt_10000_eth', 'num_transactions_gt_100000_eth',
            'transaction_failure_rate', 'avg_gas_used', 'total_gas_used', 'avg_gas_price', 'avg_gas_fee',
            'total_gas_fees'
            ]
        
        transaction_data = execute_query(
            query = query,
            cols = transaction_cols,
            date_col = 'timestamp'
        )
        self.transaction_data = transaction_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):    
        merged = pd.merge(X, self.transaction_data, how = 'inner', left_index = True, right_index = True)
        return merged
    
class TransferFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        query = """
        SELECT *
        FROM metrics.eth.transfer_metrics
        ORDER BY "timestamp" ASC
        """
        transfer_cols = []
        transfer_data = execute_query(
            query = query,
            cols = transfer_cols,
            date_col = 'timestamp'
        )
        self.transfer_data = transfer_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        merged = pd.merge(X, self.transfer_data, how = 'inner', left_index = True, right_index = True)
        return merged
  
class OrderBookFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, symbol_id):
        base, quote, exchange = symbol_id.split('_')
        query = f"""
        SELECT *
        FROM token_price.coinapi.order_book_data_1h
        WHERE
            asset_id_base = '{base}' AND
            asset_id_quote = '{quote}' AND
            exchange_id = '{exchange}'
        ORDER BY time_exchange ASC
        """
        order_book_cols = ['symbol_id', 'time_exchange', 'time_coinapi', 'asks', 'bids', 'asset_id_base', 'asset_id_quote', 'exchange_id']
        order_book_data = execute_query(
            query = query,
            cols = order_book_cols,
            date_col = 'time_exchange'
        )
        self.snapshot = OrderBookSnapshot(order_book_data)
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        merged = pd.merge(X, self.transaction_data, how = 'inner', left_index = True, right_index = True)
        return merged
     
class NetworkFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        query = f"""
        SELECT *
        FROM metrics.eth.network_metrics
        ORDER BY "timestamp" ASC
        """
        network_cols = ['timestamp', 'liquidity_ratio']
        network_data = execute_query(
            query = query,
            cols = network_cols,
            date_col = 'timestamp'
        )
        self.network_data = network_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        merged = pd.merge(X, self.network_data, how = 'inner', left_index = True, right_index = True)
        return merged

class TickFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, symbol_id):
        base, quote, exchange = symbol_id.split('_')
        query = f"""
        SELECT
            "timestamp", 
            buy_to_sell_ratio, 
            buy_to_sell_volume_ratio, 
            avg_trade_size,
            vwap, 
            buy_trade_size_std, 
            sell_trade_size_std, 
            avg_buy_trade_size,
            avg_sell_trade_size, 
            avg_seconds_between_sell_trades, 
            avg_seconds_between_buy_trades,
            std_seconds_between_sell_trades, 
            std_seconds_between_buy_trades
        FROM token_price.metrics.tick_metrics
        WHERE
            asset_id_base = '{base}' AND
            asset_id_quote = '{quote}' AND
            exchange_id = '{exchange}'
        ORDER BY timestamp ASC
        """
        tick_cols = [
            'timestamp', 'buy_to_sell_ratio', 'buy_to_sell_volume_ratio', 'avg_trade_size',
            'vwap', 'buy_trade_size_std', 'sell_trade_size_std', 'avg_buy_trade_size',
            'avg_sell_trade_size', 'avg_seconds_between_sell_trades', 'avg_seconds_between_buy_trades',
            'std_seconds_between_sell_trades', 'std_seconds_between_buy_trades'
        ]
        tick_data = execute_query(
            query = query,
            cols = tick_cols,
            date_col = 'timestamp'
        )
        self.tick_data = tick_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        merged = pd.merge(X, self.tick_data, how = 'inner', left_index = True, right_index = True)
        return merged
    
class WalletFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass
    
class PriceFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, symbol_id):
        self.symbol_id = symbol_id

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass

class TimeFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Extract various time-based features from datetime index
        X_copy = X.copy()
        X_copy['hour'] = X_copy.index.hour        
        X_copy['day'] = X_copy.index.day
        X_copy['month'] = X_copy.index.month
        X_copy['week'] = X_copy.index.isocalendar().week
        X_copy['quarter'] = X_copy.index.quarter
        X_copy['year'] = X_copy.index.year
        X_copy['is_month_start'] = X_copy.index.is_month_start
        X_copy['is_month_end'] = X_copy.index.is_month_end
        X_copy['is_quarter_start'] = X_copy.index.is_quarter_start
        X_copy['is_quarter_end'] = X_copy.index.is_quarter_end
        X_copy['is_year_start'] = X_copy.index.is_year_start
        X_copy['is_year_end'] = X_copy.index.is_year_end
        X_copy['is_leap_year'] = X_copy.index.is_leap_year

        return X_copy

class DropColumns(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # cols_to_drop = [col for col in X.columns if not ('_rmm_' in col or '_rz_' in col)]
        # X_copy = X_copy.drop(columns = cols_to_drop, axis = 1)

        # Fill nan values for each column with the rolling mean of that column
        for col in X_copy.columns:
            try:
                X_copy[col] = X_copy[col].fillna(X_copy[col].rolling(window = 24).mean().fillna(0))
            except:
                continue

        return X_copy