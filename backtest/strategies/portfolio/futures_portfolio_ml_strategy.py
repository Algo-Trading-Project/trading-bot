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
def signal_func_nb(c, long_entry_signals, short_entry_signals):
    i = c.i
    g = c.group
    col = c.col

    # If strategy has stopped trading, we do not enter any new positions
    sl_predicate = not c.in_outputs.is_trading[g]
    if sl_predicate:
        return False, True, False, True

    # Check for long/short entry signals
    long_entry_signal = long_entry_signals[i, col]
    short_entry_signal = short_entry_signals[i, col]

    # If long entry signal is present, we enter a long position
    if long_entry_signal:
        return long_entry_signal, False, False, False
    # If short entry signal is present, we enter a short position
    elif short_entry_signal:
        return False, False, short_entry_signal, False
    # If no entry signals are present, we do nothing
    else: 
        return False, False, False, False

class FuturesPortfolioMLStrategy(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'FuturesPortfolioMLStrategy',
    }

    optimize_dict = {
        # Maximum portfolio loss before exiting all positions and
        # preventing new positions from being entered for the rest of backtest period
        'max_loss': vbt.Param(np.array([
            0.1,  # 10% max loss per month
        ])),
        'expected_value_threshold': vbt.Param(np.array([
            0.0,
        ])),
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.005,
        'slippage': 0.00,
        'sl_stop': 'std',
        'tp_stop': 'std',
        'size': 0.05,
        'size_type': "valuepercent",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load the ML features from the database
        ml_features = pd.read_parquet('/Users/louisspencer/Desktop/Trading-Bot/data/ml_features.parquet')
        ml_features_cols = ml_features.columns.to_list()

        ml_features.rename(columns={'time_period_end': 'time_period_open'}, inplace=True)
        ml_features['symbol_id'] = ml_features['symbol_id'].astype('category')
        ml_features['day_of_week'] = ml_features['day_of_week'].astype('category')
        ml_features['month'] = ml_features['month'].astype('category')
        ml_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        ml_features = ml_features.dropna(subset=['forward_returns_7'])
        ml_features.set_index('time_period_open', inplace=True)
        ml_features.sort_index(inplace=True)

        # Columns we need to drop before training the model
        forward_returns_cols = [col for col in ml_features_cols if 'forward_returns' in col]

        non_numeric_cols = [
            'asset_id_base','asset_id_base_x','asset_id_base_y', 
            'asset_id_quote','asset_id_quote_x', 'asset_id_quote_y', 
            'exchange_id','exchange_id_x','exchange_id_y', 'symbol_id'
        ]

        other_cols = [
            'open_spot', 'high_spot', 'low_spot', 'close_spot', 
            'open_futures', 'high_futures', 'low_futures', 'close_futures', 
            'time_period_open',
        ]

        num_cols = [col for col in ml_features_cols if 'num' in col and 'rz' not in col and 'zscore' not in col and 'percentile' not in col]

        dollar_cols = [col for col in ml_features_cols if 'dollar' in col and 'rz' not in col and 'zscore' not in col and 'percentile' not in col]

        delta_cols = [col for col in ml_features_cols if 'delta' in col and 'rz' not in col and 'zscore' not in col and 'percentile' not in col]

        other = [col for col in ml_features_cols if '10th_percentile' in col or '90th_percentile' in col]

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
        returns_cols = [col for col in ml_features_cols if ('spot_returns' in col or 'futures_returns' in col) and 'cs_' not in col]
        returns_cs_cols = [col for col in ml_features_cols if ('spot_returns' in col or 'futures_returns' in col) and 'cs_' in col and 'kurtosis' not in col]

        alpha_beta_cols = [col for col in ml_features_cols if ('alpha' in col or 'beta' in col) and 'cs_' not in col]
        alpha_beta_cs_cols = [col for col in ml_features_cols if ('alpha' in col or 'beta' in col) and 'cs_' in col and 'kurtosis' not in col]

        basis_pct_cols = [col for col in ml_features_cols if 'basis_pct' in col and 'cs_' not in col]
        basis_pct_cs_cols = [col for col in ml_features_cols if 'basis_pct' in col and 'cs_' in col and 'kurtosis' not in col]

        trade_imbalance_cols = [col for col in ml_features_cols if 'trade_imbalance' in col and 'cs_' not in col]
        trade_imbalance_cs_cols = [col for col in ml_features_cols if 'trade_imbalance' in col and 'cs_' in col and 'kurtosis' not in col]

        ema_cols = [col for col in ml_features_cols if 'ema' in col and not (col.endswith('_basis') or 'volume' in col or 'num' in col or 'kurtosis' in col)]

        valid_cols = (
            returns_cols +
            returns_cs_cols +
            alpha_beta_cols +
            alpha_beta_cs_cols +
            basis_pct_cols +
            basis_pct_cs_cols +
            trade_imbalance_cols +
            trade_imbalance_cs_cols +
            ema_cols
        )

        rz_cols = [col for col in ml_features_cols if ('rz' in col or 'zscore' in col or ('percentile' in col and '10th_percentile' not in col and '90th_percentile' not in col or col in valid_cols)) and 'forward_returns' not in col]        
        self.cols_to_include = rz_cols
        self.cols_to_drop = cols_to_drop
        self.ml_features = ml_features
        self.start_date = None

    def __get_model(self, min_date: pd.Timestamp):
        try:
            year = min_date.year
            month = min_date.month
            model_path = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/regression/lgbm_short_model_{year}_{month}_7d.pkl'
            model = joblib.load(model_path)
            model.set_params(verbosity=-1)  # Suppress LightGBM warnings
            return model
        except FileNotFoundError:
            return None

    def run_strategy_with_parameter_combination(
            self,
            universe: pd.DataFrame,
            max_loss: float,
            expected_value_threshold: float,
    ):
        universe.index = pd.to_datetime(universe.index)
        min_date = universe.index.min()
        max_date = universe.index.max()

        # Turn the universe into futures only
        universe_copy = self.ml_features[['symbol_id', 'asset_id_base', 'asset_id_quote', 'exchange_id', 'open_futures', 'high_futures', 'low_futures', 'close_futures']].copy()
        universe_copy = universe_copy[
            (universe_copy.index >= min_date) &
            (universe_copy.index <= max_date) &
            (~universe_copy['close_futures'].isna()) &
            (~universe_copy['asset_id_base'].str.contains('USD'))
        ].reset_index(drop=False)
        universe_copy = universe_copy.pivot_table(
            index='time_period_open',
            columns='symbol_id',
            values=['open_futures', 'high_futures', 'low_futures', 'close_futures'],
            dropna=False
        ).ffill().sort_index()

        # Get model
        model = self.__get_model(min_date)

        feature_filter_futures = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date) &
            (~self.ml_features['close_futures'].isna()) &
            (~self.ml_features['asset_id_base'].str.contains('USD'))
        )
        futures_data = self.ml_features.loc[feature_filter_futures]

        # Predictions on this month
        features = model.feature_names_in_
        y_pred = model.predict(futures_data[features])
        futures_data['expected_value'] = y_pred

        # Cross-sectional predictions
        short_model_preds = (
            futures_data
            .reset_index()[['time_period_open', 'symbol_id', 'expected_value', 'rolling_ema_30_cs_futures_returns_mean_7']]
            .pivot_table(index='time_period_open', columns='symbol_id', values=['expected_value', 'rolling_ema_30_cs_futures_returns_mean_7'], dropna=False)
        )
        short_model_preds = short_model_preds[short_model_preds.index.isin(universe.index)]

        # Get the top 10 symbols by expected value to enter short positions
        short_entry_symbols = pd.Series(index = short_model_preds.index, dtype=object)
        long_entry_symbols = pd.Series(index = short_model_preds.index, dtype=object)
        for date in short_model_preds.index:
            universe_for_day = short_model_preds.loc[date, 'expected_value'].dropna()
            regime = short_model_preds.loc[date, 'rolling_ema_30_cs_futures_returns_mean_7'].mean()
            is_futures_available = not np.isnan(short_model_preds.loc[date, 'rolling_ema_30_cs_futures_returns_mean_7'].mean())
            # If there are no futures available, we do not enter any positions
            if not is_futures_available:
                top_10 = []
                bottom_10 = []
                long_entry_symbols.loc[date] = list(top_10)
                short_entry_symbols.loc[date] = list(bottom_10)
            # If the regime is positive, we only enter long positions on the top 10 symbols
            elif regime > 0:
                top_10 = universe_for_day.nlargest(10).index.to_list()
                bottom_10 = []
                long_entry_symbols.loc[date] = list(top_10)
                short_entry_symbols.loc[date] = list(bottom_10)
            # If the regime is negative, we enter short positions on the bottom 5 symbols and long positions on the top 5 symbols
            else:
                top_5 = universe_for_day.nlargest(5).index.to_list()
                bottom_5 = universe_for_day.nsmallest(5).index.to_list()
                long_entry_symbols.loc[date] = list(top_5)
                short_entry_symbols.loc[date] = list(bottom_5)

        short_entries = short_model_preds['expected_value'].copy() * 0
        short_entries = short_entries.astype(bool)
        long_entries = short_model_preds['expected_value'].copy() * 0
        long_entries = long_entries.astype(bool)

        for date in short_entry_symbols.index:
            symbols_short = short_entry_symbols.loc[date]
            symbols_long = long_entry_symbols.loc[date]
            short_entries.loc[date, symbols_short] = True
            long_entries.loc[date, symbols_long] = True

        # Filter entries based on the expected value and rolling mean of universe-wide returns
        short_entries = short_entries & (short_model_preds['rolling_ema_30_cs_futures_returns_mean_7'] < 0) & (short_model_preds['expected_value'] < 0)
        long_entries = long_entries & (short_model_preds['expected_value'] > 0)

        # Save these parameters temporarily
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            signal_func_nb=signal_func_nb,
            signal_args=(
                long_entries.values,
                short_entries.values
            ),
            post_segment_func_nb=post_segment_func_nb,
            in_outputs=dict(
                is_trading=vbt.RepEval(
                    "np.full(len(cs_group_lens), True)"
                ),
            ),
            open=universe_copy.open_futures,
            high=universe_copy.high_futures,
            low=universe_copy.low_futures,
            close=universe_copy.close_futures,
            size=0.1,
            td_stop=pd.Timedelta(days=7),
            sl_stop=1,
            cash_sharing=True,
            accumulate=False,
            freq='D',
            **self.backtest_params
        )

        # Restore the parameters
        self.backtest_params['sl_stop'] = temp_sl_stop
        self.backtest_params['tp_stop'] = temp_tp_stop
        self.backtest_params['size'] = temp_size

        return portfolio