from sklearn.base import BaseEstimator, TransformerMixin
from .helper import execute_query
import pandas as pd
import ta
import numpy as np

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
                cols.append(pd.Series(X_copy[col].shift(lag), name = col_name).fillna(0))

        X_copy = pd.concat([X_copy] + cols, axis = 1)

        return X_copy

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
            'timestamp', 'hourly_transaction_volume', 'daily_transaction_volume', 'weekly_transaction_volume', 'monthly_transaction_volume', 
            'avg_transaction_size', 'num_transactions', 'num_transactions_gt_100_eth', 'num_transactions_gt_1000_eth',
            'num_transactions_gt_10000_eth', 'num_transactions_gt_100000_eth', 'transaction_failure_rate', 'avg_gas_used', 
            'total_gas_used', 'avg_gas_price', 'avg_gas_fee', 'total_gas_fees'
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
        FROM token_price.metrics.order_book_metrics
        WHERE
            asset_id_base = '{base}' AND
            asset_id_quote = '{quote}' AND
            exchange_id = '{exchange}'
        ORDER BY timestamp ASC
        """
        order_book_cols = [
            'timestamp', 'asset_id_base', 'asset_id_quote', 'exchange_id', 
            'bid_ask_spread', 'bid_volume_std', 'ask_volume_std', 'bid_volume_skew',
            'ask_volume_skew', 'bid_volume_kurtosis', 'ask_volume_kurtosis', 'mid_price',
            'vwap_bids', 'vwap_asks', 'vwap_diff_bids_asks', 'order_book_imbalance'
        ]

        bid_depth_cols = [f'bid_depth_level_{i}' for i in range(1, 21)]
        ask_depth_cols = [f'ask_depth_level_{i}' for i in range(1, 21)]
        cumulative_bid_depth_cols = [f'cumulative_bid_depth_level_{i}' for i in range(1, 21)]
        cumulative_ask_depth_cols = [f'cumulative_ask_depth_level_{i}' for i in range(1, 21)]

        order_book_cols += bid_depth_cols + ask_depth_cols + cumulative_bid_depth_cols + cumulative_ask_depth_cols

        order_book_data = execute_query(
            query = query,
            cols = order_book_cols,
            date_col = 'timestamp'
        )

        # Drop asset_id_base, asset_id_quote, and exchange_id columns
        order_book_data = order_book_data.drop(['asset_id_base', 'asset_id_quote', 'exchange_id'], axis = 1)
        
        self.order_book_data = order_book_data
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        merged = pd.merge(X, self.order_book_data, how = 'inner', left_index = True, right_index = True)
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
        merged['vwap_deviation_pct'] = (merged['vwap'] - merged['price_close']) / merged['price_close']
        return merged
    
class WalletFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        query = """
        SELECT *
        FROM administrator.metrics.active_addresses
        ORDER BY "timestamp" ASC
        """
        wallet_cols = [
            'timestamp', 'hourly_active_addresses', 'daily_active_addresses', 
            'weekly_active_addresses', 'monthly_active_addresses'
        ]

        wallet_data = execute_query(
            query = query,
            cols = wallet_cols,
            date_col = 'timestamp'
        )
        self.wallet_data = wallet_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        merged = pd.merge(X, self.wallet_data, how = 'inner', left_index = True, right_index = True)

        merged['hourly_liquidity_ratio'] = merged['hourly_active_addresses'] / merged['hourly_transaction_volume']
        merged['daily_liquidity_ratio'] = merged['daily_active_addresses'] / merged['daily_transaction_volume']
        merged['weekly_liquidity_ratio'] = merged['weekly_active_addresses'] / merged['weekly_transaction_volume']
        merged['monthly_liquidity_ratio'] = merged['monthly_active_addresses'] / merged['monthly_transaction_volume']
        
        return merged
    
class PriceFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Calculate TA features
        X_copy = X.copy()
        X_copy = ta.add_all_ta_features(X_copy, open = 'price_open', high = 'price_high', low = 'price_low', close = 'price_close', volume = 'volume_traded', fillna = True)
        
        # Drop columns that are not needed and replace inf values with nan
        X_copy = X_copy.drop(['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded'], axis = 1)
        X_copy = X_copy.replace([np.inf, -np.inf], 0)

        return X_copy

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

class FillNaN(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Fill nan or infinity values for each column with the rolling mean of that column
        for col in X_copy.columns:
            try:
                # Replace inf values with nan
                X_copy[col] = X_copy[col].replace([np.inf, -np.inf], np.nan)
                X_copy[col] = X_copy[col].fillna(X_copy[col].rolling(window = 24).mean().fillna(0))
            except:
                continue

        return X_copy