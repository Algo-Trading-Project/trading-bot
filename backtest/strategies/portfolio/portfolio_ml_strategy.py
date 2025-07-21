from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
from utils.db_utils import QUERY
from vectorbtpro import *
from numba import njit
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import joblib
import numpy as np
import vectorbtpro as vbt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

@njit
def post_segment_func_nb(c):
    is_trading = c.in_outputs.is_trading
    g = c.group
    total_return = ((c.last_value[g] - c.init_cash) / c.init_cash)[0]
    if total_return <= -0.1:
        # If total return is less than or equal to -10%, we stop trading for the rest of the backtest period
        is_trading[g] = False

@njit
def signal_func_nb(c, long_entry_signals, long_exit_signals, short_entry_signals, short_exit_signals):
    i = c.i
    g = c.group
    col = c.col

    # If strategy has stopped trading, we do not enter any new positions
    # sl_predicate = not c.in_outputs.is_trading[g]
    # if sl_predicate:
    #     return False, True, False, True

    # Check for long/short entry signals
    long_entry_signal = long_entry_signals[i, col]
    long_exit_signal = long_exit_signals[i, col]
    short_entry_signal = short_entry_signals[i, col]
    short_exit_signal = short_exit_signals[i, col]

    # If long exit signal is present, we exit the long position
    if long_exit_signal:
        return False, True, False, False

    # If short exit signal is present, we exit the short position
    if short_exit_signal:
        return False, False, False, True

    # If long entry signal is present, we enter a long position
    if long_entry_signal:
        return long_entry_signal, False, False, False
    # If short entry signal is present, we enter a short position
    elif short_entry_signal:
        return False, False, short_entry_signal, False
    # If no entry signals are present, we do nothing
    else:
        return False, False, False, False

class PortfolioMLStrategy(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'PortfolioMLStrategy',
    }

    optimize_dict = {
        # Maximum portfolio loss before exiting all positions and
        # preventing new positions from being entered for the rest of backtest period
        'max_loss': vbt.Param(np.array([
            0.1
        ])),

        # Expected value threshold for entering long positions
        'long_threshold': vbt.Param(np.array([
            0.0,
        ])),

        # Expected value threshold for entering short positions
        'short_threshold': vbt.Param(np.array([
            0.0,
        ])),

        # Number of positions per side
        'max_positions': vbt.Param(np.array([
            5
        ])),
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.0075,  # 0.75% fees
        'slippage': 0.00,
        'sl_stop': 'std',
        'tp_stop': 'std',
        'size': 0.05,
        'size_type': "valuepercent",,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load the ML features from the database
        ml_features = pd.read_parquet('/Users/louisspencer/Desktop/Trading-Bot/data/ml_features.parquet')

        ml_features.rename(columns={'time_period_end': 'time_period_open'}, inplace=True)
        ml_features['symbol_id'] = ml_features['symbol_id'].astype('category')
        ml_features['day_of_week'] = ml_features['day_of_week'].astype('category')
        ml_features['month'] = ml_features['month'].astype('category')
        ml_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        ml_features = ml_features.dropna(subset=['forward_returns_7'])
        ml_features.set_index('time_period_open', inplace=True)
        ml_features.sort_index(inplace=True)

        # Downcast numeric columns to reduce memory usage
        for col in ml_features.select_dtypes(include=['float64']).columns:
            ml_features[col] = pd.to_numeric(ml_features[col], downcast='float')
        for col in ml_features.select_dtypes(include=['int64']).columns:
            ml_features[col] = pd.to_numeric(ml_features[col], downcast='integer')

        # Columns we need to drop before training the model
        forward_returns_cols = [col for col in ml_features if 'forward_returns' in col]

        non_numeric_cols = [
            'asset_id_base','asset_id_base_x','asset_id_base_y', 
            'asset_id_quote','asset_id_quote_x', 'asset_id_quote_y', 
            'exchange_id','exchange_id_x','exchange_id_y', 
        ]

        other_cols = [
            'open_spot', 'high_spot', 'low_spot', 'close_spot',
            'open_futures', 'high_futures', 'low_futures', 'close_futures',
            'time_period_open', 'y_pred'
        ]

        num_cols = [col for col in ml_features if 'num' in col and 'rz' not in col and 'zscore' not in col and 'percentile' not in col]

        dollar_cols = [col for col in ml_features if 'dollar' in col and 'rz' not in col and 'zscore' not in col and 'percentile' not in col]

        delta_cols = [col for col in ml_features if 'delta' in col and 'rz' not in col and 'zscore' not in col and 'percentile' not in col]

        other = [col for col in ml_features if '10th_percentile' in col or '90th_percentile' in col]

        cols_to_drop = (
            forward_returns_cols +
            non_numeric_cols +
            other_cols +
            num_cols +
            dollar_cols +
            delta_cols +
            other
        )

        # Columns to include in the model
        returns_cols = [col for col in ml_features if ('spot_returns' in col or 'futures_returns' in col) and 'cs_' not in col]
        returns_cs_cols = [col for col in ml_features if ('spot_returns' in col or 'futures_returns' in col) and 'cs_' in col and 'kurtosis' not in col]

        alpha_beta_cols = [col for col in ml_features if ('alpha' in col or 'beta' in col) and 'cs_' not in col]
        alpha_beta_cs_cols = [col for col in ml_features if ('alpha' in col or 'beta' in col) and 'cs_' in col and 'kurtosis' not in col]

        basis_pct_cols = [col for col in ml_features if 'basis_pct' in col and 'cs_' not in col]
        basis_pct_cs_cols = [col for col in ml_features if 'basis_pct' in col and 'cs_' in col and 'kurtosis' not in col]

        trade_imbalance_cols = [col for col in ml_features if 'trade_imbalance' in col and 'cs_' not in col]
        trade_imbalance_cs_cols = [col for col in ml_features if 'trade_imbalance' in col and 'cs_' in col and 'kurtosis' not in col]

        valid_cols = (
            returns_cols +
            returns_cs_cols +
            alpha_beta_cols +
            alpha_beta_cs_cols +
            basis_pct_cols +
            basis_pct_cs_cols +
            trade_imbalance_cols +
            trade_imbalance_cs_cols
        )
        rz_cols = [col for col in ml_features if ('rz' in col or 'zscore' in col or ('percentile' in col and '10th_percentile' not in col and '90th_percentile' not in col) or col in valid_cols) and 'forward_returns' not in col]        
        
        self.cols_to_include = rz_cols
        self.cols_to_drop = cols_to_drop
        self.ml_features = ml_features
        self.start_date = None

    def __get_model(self, min_date: pd.Timestamp): 
        year = min_date.year
        month = min_date.month
        model_path_long = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/regression/lgbm_long_model_{year}_{month}_7d_2.pkl'
        model_path_short = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/regression/lgbm_short_model_{year}_{month}_7d_2.pkl'
        model_long = joblib.load(model_path_long)
        model_long.set_params(verbosity=-1)
        try:
            model_short = joblib.load(model_path_short)
            model_short.set_params(verbosity=-1)
        except FileNotFoundError:
            model_short = None
        return model_long, model_short

    def calculate_size(self, volatilities):
        # Volatility targeting
        token_ann_vol = np.sqrt(365) * volatilities

        # Position size is volatility weighted
        ann_vol_target = 0.15  # Target annualized volatility of the portfolio
        max_positions = 20  # Maximum number of positions in the portfolio
        size = ann_vol_target / token_ann_vol / max_positions

        # Replace NaNs or infinite values with 0
        size = size.replace([np.inf, -np.inf, np.nan], 0)
        size = size.clip(lower = 0, upper = 0.05)
        return size

    def run_strategy_with_parameter_combination(
            self,
            universe: pd.DataFrame,
            max_loss: float,
            long_threshold: float,
            short_threshold: float,
            max_positions: int,
    ):
        universe.index = pd.to_datetime(universe.index)
        min_date = universe.index.min()
        max_date = universe.index.max()

        # Get models
        model_long, model_short = self.__get_model(min_date)

        # Get features for this month
        feature_filter = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date) &
            (~self.ml_features['close_spot'].isna()) &
            (~self.ml_features['asset_id_base'].str.contains('USD'))
        )
        feature_filter_futures = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date) &
            (~self.ml_features['close_futures'].isna()) &
            (~self.ml_features['asset_id_base'].str.contains('USD')) 
        )

        spot_data = self.ml_features.loc[feature_filter]
        futures_data = self.ml_features.loc[feature_filter_futures]

        # If there are no futures features for this month, create an empty DataFrame with the same index as spot_data
        # and columns suffixed with '_futures'
        if futures_data.empty:
            futures_data = pd.DataFrame(index=spot_data.index, columns=[col for col in spot_data.columns])

        # Get LightGBM model features from each model
        features_long = model_long.feature_names_in_
        if model_short is not None:
            features_short = model_short.feature_names_in_

        # Predictions on this month
        y_pred_long = model_long.predict(spot_data[features_long])
        if model_short is None:
            y_pred_short = np.zeros_like(len(futures_data))
        else:
            y_pred_short = model_short.predict(futures_data[features_short])
        
        # Add the predictions to the DataFrame
        spot_data['expected_value_long'] = y_pred_long
        futures_data['expected_value_short'] = y_pred_short

        # Pivot futures model's predictions for ranking the models' signals by date
        long_model_probs = (
            spot_data
            .reset_index()[['time_period_open', 'symbol_id', 'expected_value_long', 'rolling_ema_30_cs_spot_returns_mean_7', 'rolling_ema_30_cs_futures_returns_mean_7']]
            .pivot_table(index='time_period_open', columns='symbol_id', values=['expected_value_long', 'rolling_ema_30_cs_spot_returns_mean_7', 'rolling_ema_30_cs_futures_returns_mean_7'], dropna=False)
        )
        long_model_probs = long_model_probs[long_model_probs.index.isin(universe.index)]
        short_model_probs = (
            futures_data
            .reset_index()[['time_period_open', 'symbol_id', 'expected_value_short', 'rolling_ema_30_cs_spot_returns_mean_7']]
            .pivot_table(index='time_period_open', columns='symbol_id', values=['expected_value_short', 'rolling_ema_30_cs_spot_returns_mean_7'], dropna=False)
        )
        short_model_probs = short_model_probs[short_model_probs.index.isin(universe.index)]

        short_entry_symbols = pd.Series(index = short_model_probs.index, dtype=object)
        long_entry_symbols = pd.Series(index = long_model_probs.index, dtype=object)
        for date in long_model_probs.index:
            long_universe = long_model_probs.loc[date, 'expected_value_long']
            regime = long_model_probs.loc[date, 'rolling_ema_30_cs_spot_returns_mean_7'].mean()
            is_futures_available = not np.isnan(long_model_probs.loc[date, 'rolling_ema_30_cs_futures_returns_mean_7'].mean())             
            if regime > 0:
                bottom_symbols = []
                top_symbols = long_universe.nlargest(10).index.to_list()
            elif regime < 0:
                if not is_futures_available:
                    bottom_symbols = long_universe.nsmallest(5).index.to_list()
                    top_symbols = long_universe.nlargest(5).index.to_list()
                else:
                    short_universe = short_model_probs.loc[date, 'expected_value_short']
                    bottom_symbols = short_universe.nsmallest(5).index.to_list()
                    top_symbols = long_universe.nlargest(5).index.to_list()

            long_entry_symbols.loc[date] = list(top_symbols)
            short_entry_symbols.loc[date] = list(bottom_symbols)

        short_entries = short_model_probs['expected_value_short'].copy() * 0
        short_entries = short_entries.astype(bool)
        long_entries = long_model_probs['expected_value_long'].copy() * 0
        long_entries = long_entries.astype(bool)

        for date in long_entry_symbols.index:
            long_symbols = long_entry_symbols.loc[date]
            long_entries.loc[date, long_symbols] = True
            try:
                short_symbols = short_entry_symbols.loc[date]
            except:
                short_symbols = []
            short_entries.loc[date, short_symbols] = True

        # long_entries = long_entries & (long_model_probs['expected_value_long'] > 0) 
        # short_entries = short_entries & (short_model_probs['expected_value_short'] < 0)

        empty_long = pd.DataFrame(
            np.zeros((long_model_probs['expected_value_long'].shape[0], long_model_probs['expected_value_long'].shape[1])),
            index=long_model_probs.index,
            columns=[col for col in long_model_probs['expected_value_long'].columns]
        )
        empty_short = pd.DataFrame(
            np.zeros((short_model_probs['expected_value_short'].shape[0], short_model_probs['expected_value_short'].shape[1])),
            index=short_model_probs.index,
            columns=[col for col in short_model_probs['expected_value_short'].columns]
        )

        # Merge the long and short entries w/ the empty DataFrames 
        # to have a consistent universe structure for VBT
        long_entries = pd.merge(
            long_entries,
            empty_short,
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('', '_futures')
        ).fillna(False)
        long_entries = long_entries[long_entries.index.isin(universe.index)]
        short_entries = pd.merge(
            empty_long,
            short_entries,
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('', '_futures')
        ).fillna(False)
        short_entries = short_entries[short_entries.index.isin(universe.index)]

        # Make the last exit in the long and short exits DataFrame True
        # to ensure exiting all positions at the end of the backtest
        long_exits = (long_entries.copy() * 0).astype(bool)
        long_exits.iloc[-1] = True
        long_exits = long_exits[long_exits.index.isin(universe.index)]
        short_exits = (short_entries.copy() * 0).astype(bool)
        if not short_entries.empty:
            short_exits.iloc[-1] = True
        short_exits = short_exits[short_exits.index.isin(universe.index)]

        # Intersection of columns between the universe and the signals
        open_spot = universe.open
        open_futures = universe.open_futures
        open_futures.columns = [f'{col}_futures' for col in open_futures.columns]

        high_spot = universe.high
        high_futures = universe.high_futures
        high_futures.columns = [f'{col}_futures' for col in high_futures.columns]

        low_spot = universe.low
        low_futures = universe.low_futures
        low_futures.columns = [f'{col}_futures' for col in low_futures.columns]
        
        close_spot = universe.close
        close_futures = universe.close_futures
        close_futures.columns = [f'{col}_futures' for col in close_futures.columns]

        # Save these parameters temporarily
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        open = pd.merge(
            open_spot,
            open_futures,
            left_index=True,
            right_index=True,
            how='outer',
        )
        high = pd.merge(
            high_spot,
            high_futures,
            left_index=True,
            right_index=True,
            how='outer',
        )
        low = pd.merge(
            low_spot,
            low_futures,
            left_index=True,
            right_index=True,
            how='outer',
        )
        close = pd.merge(
            close_spot,
            close_futures,
            left_index=True,
            right_index=True,
            how='outer',
        )
        
        long_entries.index = long_entries.index.astype('datetime64[ns]')
        long_exits.index = long_exits.index.astype('datetime64[ns]')
        short_entries.index = short_entries.index.astype('datetime64[ns]')
        short_exits.index = short_exits.index.astype('datetime64[ns]')
        open.index = open.index.astype('datetime64[ns]')
        high.index = high.index.astype('datetime64[ns]')
        low.index = low.index.astype('datetime64[ns]')
        close.index = close.index.astype('datetime64[ns]')

        long_entries = long_entries.astype('bool')
        short_entries = short_entries.astype('bool')

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            signal_func_nb=signal_func_nb,
            signal_args=(
                long_entries.values,
                long_exits.values,
                short_entries.values, 
                short_exits.values
            ),
            in_outputs=dict(
                is_trading=vbt.RepEval(
                    "np.full(len(cs_group_lens), True)"
                ),
            ),
            open=open,
            high=high,
            low=low,
            close=close,
            price = -np.inf,
            size=0.05,
            sl_stop=1, 
            td_stop=pd.Timedelta(days=7),
            cash_sharing=True,
            accumulate=True,
            freq='D',
            **self.backtest_params
        ) 

        # Restore the parameters
        self.backtest_params['sl_stop'] = temp_sl_stop
        self.backtest_params['tp_stop'] = temp_tp_stop
        self.backtest_params['size'] = temp_size

        return portfolio