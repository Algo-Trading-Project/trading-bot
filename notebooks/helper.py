import pandas as pd
import numpy as np
import duckdb
from numba import njit, prange

@njit
def calculate_rolling_std(close, window):
    n = len(close)
    std = np.zeros(n)
    
    cum_sum = np.cumsum(close)
    cum_sum_sq = np.cumsum(close**2)
    
    for i in range(window, n):
        window_sum = cum_sum[i] - cum_sum[i - window]
        window_sum_sq = cum_sum_sq[i] - cum_sum_sq[i - window]
        
        mean = window_sum / window
        mean_sq = window_sum_sq / window
        variance = mean_sq - mean**2
        std[i] = np.sqrt(variance)
    
    return std

@njit(parallel=True)
def calculate_labels(high, low, close, std, max_holding_time, transaction_cost_multiplier):
    n = len(close)
    labels = np.zeros(n)

    for i in prange(n - max_holding_time):
        upper_barrier = close[i] * transaction_cost_multiplier + 2 * std[i]
        lower_barrier = close[i] - std[i]
        
        high_slice = high[i+1:i+max_holding_time+1]
        low_slice = low[i+1:i+max_holding_time+1]
        
        if np.any(high_slice >= upper_barrier):
            labels[i] = 1
        elif np.any(low_slice <= lower_barrier):
            labels[i] = -1
        else:
            final_price = close[min(i + max_holding_time, n - 1)]
            labels[i] = 1 if final_price > close[i] else -1
    
    labels[labels == 0] = -1
    return labels

def calculate_triple_barrier_labels(ohlcv_df, window, max_holding_time, transaction_cost_percent=0.29):
    high = np.array(ohlcv_df['price_high'].values)
    low = np.array(ohlcv_df['price_low'].values)
    close = np.array(ohlcv_df['price_close'].values)

    if not (len(high) == len(low) == len(close)):
        raise ValueError("Length of high, low, and close arrays must be the same")

    std = calculate_rolling_std(close, window)
    transaction_cost_multiplier = 1 + transaction_cost_percent / 100
    labels = calculate_labels(high, low, close, std, max_holding_time, transaction_cost_multiplier)
    
    label_series = pd.Series(labels, index=ohlcv_df.index)
    return label_series

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
        dataset = pd.read_csv('/Users/louisspencer/Desktop/Trading-Bot-Data-Pipelines/data/ml_dataset.csv')
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
        dataset = pd.DataFrame()

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
                    asset_id_base,
                    asset_id_quote,
                    exchange_id,
                    price_close,
                    price_high,
                    price_low,
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
                'asset_id_base': 'last',
                'asset_id_quote': 'last',
                'exchange_id': 'last',
                'price_close': 'last',
                'price_high': 'max',
                'price_low': 'min',
                'volume_traded': 'sum',
                'trades_count': 'sum'
            })

            # Skip over tokens with more than 50% missing data
            pct_missing_price_close = data.loc[:,'price_close'].isna().mean() * 100
            
            if pct_missing_price_close > 50:
                print(f"Skipping asset due to {pct_missing_price_close:.2f}% missing price_close data")
                total_skipped += 1
                continue
            
            # Interpolate missing values
            numeric_cols = [col for col in data.columns if col not in ('asset_id_base', 'asset_id_quote', 'exchange_id')]
            categorical_cols = ['asset_id_base', 'asset_id_quote', 'exchange_id']

            data.loc[:,numeric_cols] = data.loc[:,numeric_cols].interpolate(method = 'time')

            for col in categorical_cols:
                mode = data.loc[:,col].mode().iloc[0]
                data.loc[:,col] = data.loc[:,col].fillna(mode)
                            
            # Downsample to 30 minutes
            data = data.resample('30min').agg({
                'asset_id_base': 'last',
                'asset_id_quote': 'last',
                'exchange_id': 'last',
                'price_close': 'last',
                'price_high': 'max',
                'price_low': 'min',
                'volume_traded': 'sum',
                'trades_count': 'sum'
            })
            
            # Calculate the returns
            data.loc[:,'returns'] = data.loc[:,'price_close'].pct_change()

            # Calculate the triple barrier labels
            data.loc[:,'triple_barrier_label'] = calculate_triple_barrier_labels(data, window = 2 * 24 * 7, max_holding_time = 2 * 24)

            # Create a symbol_id column and drop the asset_id_base, asset_id_quote, and exchange_id columns
            data.loc[:,'symbol_id'] = data.loc[:,'asset_id_base'] + '_' + data.loc[:,'asset_id_quote'] + '_' + data.loc[:,'exchange_id']
            data = data.drop(['asset_id_base', 'asset_id_quote', 'exchange_id'], axis = 1)

            # Add the asset data to the dataset
            dataset = pd.concat([dataset, data])

        print(f"Total assets skipped due to missing data: {total_skipped}")
        print(f'Percentage of assets skipped: {total_skipped / len(assets) * 100:.2f}%')

        # Save the dataset to file
        dataset.to_csv('/Users/louisspencer/Desktop/Trading-Bot-Data-Pipelines/data/ml_dataset.csv')

        # Return the dataset
        return dataset
    
def cusum_filter_positive_breaks_diff(price_series, threshold=None, volatility_lookback = 60 * 24 * 7, volatility_multiplier=2):
    """
    CUSUM Filter to detect significant positive structural breaks in a price series using price differences, 
    with an option to dynamically set the threshold based on recent volatility.

    :param price_series: pd.Series, series of prices
    :param threshold: float or None, the fixed threshold for detecting a significant positive change. If None,
                      the threshold is set dynamically based on recent volatility.
    :param volatility_lookback: int, the lookback period for volatility calculation if the threshold is set dynamically
    :param volatility_multiplier: float, multiplier for the dynamic threshold based on recent volatility
    :return: pd.DatetimeIndex, indices of the detected significant positive structural breaks
    """
    # Calculate price differences
    price_diff = price_series.diff().dropna()

    # Determine the dynamic threshold based on volatility if not provided
    if threshold is None:
        rolling_volatility = price_diff.rolling(window=volatility_lookback, min_periods=1).std()
        dynamic_threshold = rolling_volatility * volatility_multiplier
    else:
        dynamic_threshold = pd.Series([threshold] * len(price_diff), index=price_diff.index)

    # Initialize the positive CUSUM series
    cusum_pos = pd.Series(0.0, index=price_diff.index)

    # List to store indices of significant positive structural breaks
    events = []

    for i in range(1, len(price_diff)):
        # Update positive CUSUM series with price differences
        cusum_pos.iloc[i] = max(0, cusum_pos.iloc[i-1] + price_diff.iloc[i])

        # Check if the positive CUSUM series exceeds the threshold
        if cusum_pos.iloc[i] > dynamic_threshold.iloc[i]:
            events.append(price_diff.index[i])
            cusum_pos.iloc[i] = 0  # Reset positive CUSUM series

    return pd.DatetimeIndex(events)
