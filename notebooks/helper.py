import pandas as pd
import numpy as np
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
        upper_barrier = close[i] * transaction_cost_multiplier + std[i]
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
    high = ohlcv_df['price_high'].values
    low = ohlcv_df['price_low'].values
    close = ohlcv_df['price_close'].values

    if not (len(high) == len(low) == len(close)):
        raise ValueError("Length of high, low, and close arrays must be the same")

    std = calculate_rolling_std(close, window)
    transaction_cost_multiplier = 1 + transaction_cost_percent / 100
    labels = calculate_labels(high, low, close, std, max_holding_time, transaction_cost_multiplier)
    
    label_series = pd.Series(labels, index=ohlcv_df.index)
    return label_series
