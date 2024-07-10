import pandas as pd
import numpy as np
import duckdb
from numba import njit, prange
import os

@njit
def calculate_labels(high, low, close, std, max_holding_time, transaction_cost_multiplier):
    n = len(close)
    labels = np.zeros(n)
    trade_returns = np.zeros(n)

    for i in prange(n - max_holding_time):
        # Upper barrier is 2 standard deviations above the close price plus the transaction cost
        upper_barrier = (close[i] + 2 * std[i]) * transaction_cost_multiplier
        # Lower barrier is 1 standard deviation below the close price plus the transaction cost
        lower_barrier = (close[i] - std[i]) * transaction_cost_multiplier
        
        high_slice = high[i+1:i+max_holding_time+1]
        low_slice = low[i+1:i+max_holding_time+1]
        
        if np.any(high_slice >= upper_barrier):
            labels[i] = 1
            # Find the earliest index where the high price is at or above the upper barrier
            first_touch_idx = np.argmax(high_slice >= upper_barrier)
            # Returns of the trade startting at i using high_slice
            trade_returns[i] = (high_slice[first_touch_idx] - close[i]) / close[i]

        elif np.any(low_slice <= lower_barrier):
            labels[i] = -1
            # Find the earliest index where the low price is at or below the lower barrier
            first_touch_idx = np.argmax(low_slice <= lower_barrier)
            # Returns of the trade startting at i using low_slice
            trade_returns[i] = (low_slice[first_touch_idx] - close[i]) / close[i]

        else:
            labels[i] = 0
            # Returns of the trade startting at i using close
            trade_returns[i] = (close[i + max_holding_time] - close[i]) / close[i]

    return labels, trade_returns

def calculate_triple_barrier_labels(ohlcv_df, window, max_holding_time, transaction_cost_percent=0.29):
    high = np.array(ohlcv_df['price_high'].values)
    low = np.array(ohlcv_df['price_low'].values)
    close = np.array(ohlcv_df['price_close'].values)

    if not (len(high) == len(low) == len(close)):
        raise ValueError("Length of high, low, and close arrays must be the same")

    std = ohlcv_df['price_close'].rolling(window).std().values
    transaction_cost_multiplier = 1 + transaction_cost_percent / 100
    labels, trade_returns = calculate_labels(high, low, close, std, max_holding_time, transaction_cost_multiplier)
    
    label_series = pd.Series(labels, index=ohlcv_df.index)
    trade_returns_series = pd.Series(trade_returns, index=ohlcv_df.index)

    return label_series, trade_returns_series

def QUERY(query):
    # Connect to DuckDB
    with duckdb.connect(
        database = '/Users/louisspencer/Desktop/Trading-Bot-Data-Pipelines/data/database.db',
        read_only = False
    ) as conn:
        # Execute the query
        result = conn.sql(query).df()
        # Return the result
        return result

def construct_dataset_for_ml():
    # Check if the dataset already exists
    try:
        dataset = pd.read_csv('/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset.csv', index_col = 0)
        return dataset
    except FileNotFoundError:
        pass

    with duckdb.connect(
        database = '/Users/louisspencer/Desktop/Trading-Bot-Data-Pipelines/data/database.db',
        read_only = False
    ) as conn:
        
        # Get all distinct assets in the database
        assets = conn.sql(
        """
        SELECT DISTINCT 
            asset_id_base, 
            asset_id_quote, 
            exchange_id 
        FROM market_data.price_data_1m 
        ORDER BY asset_id_base
        """).df()

        # Create an empty DataFrame to store the final dataset
        dataset = []

        # Keep track of the number of assets skipped due to missing data
        total_skipped = 0

        # Loop through each asset
        for i in range(len(assets)):
            print(f"Processing asset {i + 1} of {len(assets)} ({assets.iloc[i]['asset_id_base']}/{assets.iloc[i]['asset_id_quote']} on {assets.iloc[i]['exchange_id']})...")
            # Get the asset
            asset = assets.iloc[i]

            # Get the asset data
            data = conn.sql(
                f"""
                SELECT 
                    time_period_end,
                    asset_id_base || '_' || asset_id_quote || '_' || exchange_id AS symbol_id,
                    price_open,
                    price_high,
                    price_low,
                    price_close,
                    volume_traded,
                    trades_count
                FROM market_data.price_data_1m
                WHERE
                    asset_id_base = '{asset['asset_id_base']}' AND
                    asset_id_quote = '{asset['asset_id_quote']}' AND
                    exchange_id = '{asset['exchange_id']}'
                ORDER BY time_period_start
                """
            ).df().set_index('time_period_end').resample('1min').agg({
                'symbol_id': 'last',
                'price_open': 'first',
                'price_high': 'max',
                'price_low': 'min',
                'price_close': 'last',
                'volume_traded': 'sum',
                'trades_count': 'sum'
            })

            # Skip over tokens with more than 25% missing data
            pct_missing_price_close = data.loc[:,'price_close'].isna().mean() * 100
            
            if pct_missing_price_close > 25:
                print(f"Skipping asset due to {pct_missing_price_close:.2f}% missing price_close data")
                total_skipped += 1
                continue
            
            # Interpolate missing values
            numeric_cols = [col for col in data.columns if col not in ('symbol_id')]
            categorical_cols = ['symbol_id']

            data.loc[:,numeric_cols] = data.loc[:,numeric_cols].interpolate(method = 'time')

            for col in categorical_cols:
                mode = data.loc[:,col].mode().iloc[0]
                data.loc[:,col] = data.loc[:,col].fillna(mode)
                            
            # Downsample to 30 minutes
            data = data.resample('30min', label = 'right', closed = 'right').agg({
                'symbol_id': 'last',
                'price_close': 'last',
                'price_high': 'max',
                'price_low': 'min',
                'volume_traded': 'sum',
                'trades_count': 'sum'
            })

            data.index = pd.to_datetime(data.index, utc = True)

            # windows = [2 * 12, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]
            # max_holding_times = [2 * 12, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]

            # for window in windows:
            #     for max_holding_time in max_holding_times:
            #         # Calculate the triple barrier labels
            #         labels, trade_returns = calculate_triple_barrier_labels(data, window = window, max_holding_time = max_holding_time)
            #         data.loc[:,f'triple_barrier_label_w{window}_h{max_holding_time}'] = labels
            #         data.loc[:,f'trade_returns_w{window}_h{max_holding_time}'] = trade_returns

            # Add the asset data to the dataset
            dataset.append(data)

        dataset = pd.concat(dataset)
        dataset = dataset.reset_index()

        print(f"Total assets skipped due to missing data: {total_skipped}")
        print(f'Percentage of assets skipped: {total_skipped / len(assets) * 100:.2f}%')

        # Save if the dataset file does not exist
        if not os.path.exists('/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset.csv'):
            dataset.to_csv('/Users/louisspencer/Desktop/Trading-Bot/data/ml_dataset.csv')

        # Return the dataset
        return dataset

