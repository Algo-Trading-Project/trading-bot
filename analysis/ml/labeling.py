import pandas as pd
import numpy as np
from numba import njit, prange

@njit
def __calculate_labels(high, low, close, std_returns, max_holding_time):
    n = len(close)
    labels = np.zeros(n)
    trade_returns = np.zeros(n)
    trade_drawdowns = np.zeros(n)
    start_date_indices = np.full(n, np.nan)
    end_date_indices = np.full(n, np.nan)

    for i in prange(n - max_holding_time):
        # Upper barrier is 1 standard deviation of the returns above the close price
        upper_barrier = close[i] * (1 + std_returns[i] * 1)
        # Lower barrier is 1 standard deviation of the returns below the close price
        lower_barrier = close[i] * (1 - std_returns[i] * 1)

        # Store the start date of the trade
        start_date_indices[i] = i

        # Slice of the high and low prices for the next max_holding_time periods
        high_slice = high[i+1:i+max_holding_time+1]
        low_slice = low[i+1:i+max_holding_time+1]

        # Initialize touch indices to max_holding_time (i.e., barrier not touched)
        upper_touch_idx = max_holding_time
        lower_touch_idx = max_holding_time

        # Find the earliest index where the high price crosses the upper barrier
        upper_touch_indices = np.where(high_slice >= upper_barrier)[0]
        if len(upper_touch_indices) > 0:
            upper_touch_idx = upper_touch_indices[0]

        # Find the earliest index where the low price crosses the lower barrier
        lower_touch_indices = np.where(low_slice <= lower_barrier)[0]
        if len(lower_touch_indices) > 0:
            lower_touch_idx = lower_touch_indices[0]
            
        # Determine which barrier is touched first
        if upper_touch_idx < lower_touch_idx:
            labels[i] = 1

            # Store the end date of the trade
            end_date_indices[i] = i + upper_touch_idx + 1 # +1 because the look-ahead period starts at i+1

            try:
                # Returns of the trade starting at i and hitting the upper barrier
                trade_returns[i] = (upper_barrier - close[i]) / close[i]
            except:
                trade_returns[i] = 0

        elif lower_touch_idx < upper_touch_idx:
            labels[i] = -1

            # Store the end date of the trade
            end_date_indices[i] = i + lower_touch_idx + 1 # +1 because the look-ahead period starts at i+1

            try:
                # Returns of the trade starting at i and hitting the lower barrier
                trade_returns[i] = (lower_barrier - close[i]) / close[i]
            except:
                trade_returns[i] = 0

        else:
            labels[i] = 0

            # Store the end date of the trade
            # We don't use +1 here because we know neither horizontal barrier was touched, 
            # thus the trade ends at i + max_holding_time
            end_date_indices[i] = i + max_holding_time

            try:
                # Returns of the trade starting at i and ending at the end of the max_holding_time period
                trade_returns[i] = (close[i + max_holding_time] - close[i]) / close[i]
            except:
                trade_returns[i] = 0

    return labels, trade_returns, start_date_indices, end_date_indices

def calculate_triple_barrier_labels(ohlcv_df, max_holding_time, std_lookback_window):
    high = np.array(ohlcv_df['high'].values)
    low = np.array(ohlcv_df['low'].values)
    close = np.array(ohlcv_df['close'].values)

    # Calculate the rolling standard deviation of the returns
    std_returns = np.array(ohlcv_df['close'].pct_change().rolling(std_lookback_window).std().values)

    if not (len(high) == len(low) == len(close) == len(std_returns)):
        raise ValueError("Length of high, low, std, and close arrays must be the same")

    labels, trade_returns, start_date_indices, end_date_indices = __calculate_labels(high, low, close, std_returns, max_holding_time)
    
    label_series = pd.Series(labels, index=ohlcv_df.index)
    trade_returns_series = pd.Series(trade_returns, index=ohlcv_df.index)
    start_date_indices_series = pd.Series(start_date_indices, index=ohlcv_df.index)
    end_date_indices_series = pd.Series(end_date_indices, index=ohlcv_df.index)

    return label_series, trade_returns_series, start_date_indices_series, end_date_indices_series