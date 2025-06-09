from quantstats.stats import expected_return

from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
from utils.db_utils import QUERY
from vectorbtpro import *
from numba import njit

import pandas as pd
import joblib
import numpy as np
import vectorbtpro as vbt

@njit
def post_segment_func_nb(c):
    max_value = c.in_outputs.max_value
    g = c.group
    max_value[g] = max(c.last_value[g], np.nan_to_num(max_value[g], nan = 0))

@njit
def signal_func_nb(c, long_entry_signals, short_entry_signals, max_loss, max_gain):
    i = c.i
    g = c.group
    col = c.col

    long_entry_signal = long_entry_signals[i, col]
    short_entry_signal = short_entry_signals[i, col]
    max_loss_ = max_loss[i, col]
    max_gain_ = max_gain[i, col]

    max_value = c.in_outputs.max_value[g]
    curr_value = c.last_value[g]
    curr_dd = (curr_value - max_value) / max_value

    tp_predicate = curr_value / c.init_cash[g] - 1 >= max_gain_
    sl_predicate = curr_dd <= -max_loss_

    if tp_predicate or sl_predicate:
        return False, True, False, True

    # If both long and short entry signals are present, we ignore them
    # as we can't be long and short at the same time
    if long_entry_signal and short_entry_signal:
        return False, False, False, False

    # If long entry signal is present, we enter a long position
    if long_entry_signal:
        return long_entry_signal, False, False, False

    # If short entry signal is present, we enter a short position
    elif short_entry_signal:
        return False, False, short_entry_signal, False

    # If neither long nor short entry signals are present, we do nothing
    else:
        return False, False, False, False

class PortfolioMLStrategy(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'PortfolioMLStrategy'
    }

    optimize_dict = {
        # Maximum portfolio drawdown before exiting all positions and
        # preventing new positions from being entered for the rest of backtest period
        'max_loss': vbt.Param(np.array([
            0.1
        ])),

        # Maximum portfolio gain before exiting all positions and
        # preventing new positions from being entered for the rest of backtest period
        'max_gain': vbt.Param(np.array([
            np.inf
        ])),

        # Prediction threshold for entering a position
        'prediction_threshold': vbt.Param(np.array([
            0.5
        ]))
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.00, # 0.1% maker/taker fees (Binance) + 0.25% spread each way
        'slippage': 0.00, # 0.25% slippage each way
        'sl_stop': 'std',
        'tp_stop': 'std',
        'size': 0.05,
        'size_type': "valuepercent", # for example, 2 if 'size' is 'Fixed Percent' and 0 otherwise,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = None

        ml_features = QUERY(
            """
            SELECT * FROM market_data.ml_features
            """
        )
        ml_features['symbol_id'] = ml_features['symbol_id'].astype('category')
        ml_features['time_period_end'] = pd.to_datetime(ml_features['time_period_end'], unit = 'ns')

        # Columns we need to drop before training the model
        triple_barrier_label_cols = [
            col for col in ml_features if 'triple_barrier_label' in col
        ]

        trade_returns_cols = [
            col for col in ml_features if 'trade_returns' in col
        ]

        non_numeric_cols = [
            'asset_id_base', 'asset_id_quote', 'exchange_id', 'Unnamed: 0', 'sample_weight', 'y_true', 'y_pred'
        ]

        forward_returns_cols = [
            'open', 'high', 'low', 'close', 'start_date_triple_barrier_label_h7', 'start_date_triple_barrier_label_h1',
            'end_date_triple_barrier_label_h1'
            'end_date_triple_barrier_label_h7', 'avg_uniqueness', 'time_period_end', 'forward_returns_1',
            'forward_returns_7'
        ]

        self.cols_to_drop = (
                triple_barrier_label_cols +
                trade_returns_cols +
                non_numeric_cols +
                forward_returns_cols
        )

        ml_features = ml_features.set_index('time_period_end').sort_index()

        self.ml_features = ml_features
        self.is_train = True
        self.long_model = None
        self.short_model = None

    def __get_model(self, min_date: pd.Timestamp):
        max_year = min_date.year
        max_month = min_date.month

        prev_month = max_month - 1
        if prev_month == 0:
            prev_month = 12
            max_year -= 1

        long_model_path = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/classification/xgboost_model_{max_year}_{prev_month}_p.pkl'
        short_model_path = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/classification/xgboost_short_model_{max_year}_{prev_month}.pkl'

        long_model = joblib.load(long_model_path)
        short_model = joblib.load(short_model_path)

        self.long_model = long_model
        self.short_model = short_model

        return long_model, short_model

    def calculate_size(self, volatilities, betas):
        """
        Calculate position sizes from the tokens' volatilities and betas

        :param volatilities: DataFrame of volatilities for each token (columns) and date (index)
        :param betas: DataFrame of betas for each token (columns) and date (index)
        :return: DataFrame of position sizes for each token (columns) and date (index)
        """
        # Volatility targeting
        token_ann_vol = np.sqrt(365) * volatilities

        # Position size is volatility weighted
        ann_vol_target = 0.2  # Target annualized volatility of the portfolio
        size = ann_vol_target / token_ann_vol

        # Replace NaNs or infinite values with 0
        size = size.replace([np.inf, -np.inf, np.nan], 0)

        # Ensure that the position sizes are between 0 and 0.1
        size = size.clip(0, 0.05)

        return size

    def run_strategy_with_parameter_combination(
            self,
            universe: pd.DataFrame,
            max_loss: float,
            max_gain: float,
            prediction_threshold: float
    ):
        universe.index = pd.to_datetime(universe.index)
        min_date = universe.index.min()
        max_date = universe.index.max()

        self.ml_features['y_pred_long'] = 0
        self.ml_features['y_pred_short'] = 0

        # Get model
        long_model, short_model = self.__get_model(min_date)

        # Get features for this month
        feature_filter = (
                (self.ml_features.index >= min_date) &
                (self.ml_features.index <= max_date)
        )
        X = self.ml_features.loc[feature_filter].drop(self.cols_to_drop, axis=1, errors='ignore')

        # Predictions on this month
        y_pred_long = long_model.predict_proba(X.drop(['y_pred_long', 'y_pred_short'], axis=1))[:, 1]
        y_pred_short = short_model.predict_proba(X.drop(['y_pred_long', 'y_pred_short'], axis=1))[:, 1]

        self.ml_features.loc[feature_filter, 'y_pred_long'] = y_pred_long
        self.ml_features.loc[feature_filter, 'y_pred_short'] = y_pred_short
        self.ml_features['dollar_volume'] = self.ml_features['close'] * self.ml_features['volume']

        # Pivot model's predictions for ranking the models' signals by date
        model_probs = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'y_pred_long', 'y_pred_short', 'dollar_volume']]
            .pivot_table(index='time_period_end', columns='symbol_id', values=['y_pred_long', 'y_pred_short', 'dollar_volume'], dropna=False)
        )
        model_probs = model_probs.sort_index().fillna(0)
        model_probs = model_probs[model_probs.index.isin(universe.index)]

        # Long entries
        long_entries = (
            (model_probs['y_pred_long'] > prediction_threshold) &
            (model_probs['dollar_volume'] >= 2000000)
        ).astype(int)

        # Short entries
        short_entries = (
            (model_probs['y_pred_short'] > prediction_threshold) &
            (model_probs['dollar_volume'] >= 2000000)
        ).astype(int)

        # Drop dollar_volume from ml_features as we don't need it anymore
        self.ml_features = self.ml_features.drop(columns=['dollar_volume'])

        # Pivot volatilities and betas for calculating position sizes
        volatilities = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'volatility_1_180']]
            .pivot_table(index='time_period_end', columns='symbol_id', values='volatility_1_180', dropna=False)
        )
        volatilities = volatilities.sort_index()

        betas = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'beta_1d_180d']]
            .pivot_table(index='time_period_end', columns='symbol_id', values='beta_1d_180d', dropna=False)
        )
        betas = betas.sort_index()

        # Position sizes
        size = self.calculate_size(volatilities, betas)

        # Ensure that the signals are aligned with the universe
        size = size[size.index.isin(universe.index)]

        # Save these parameters temporarily
        sl_stop = self.calculate_sl(universe=universe, backtest_params=self.backtest_params, window=7)
        tp_stop = self.calculate_tp(universe=universe, backtest_params=self.backtest_params, window=7)

        default_size = self.backtest_params['size']
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        # Intersection of columns between the universe and the signals
        common_cols = universe.close.columns.intersection(long_entries.columns)
        open = universe.open[common_cols]
        high = universe.high[common_cols]
        low = universe.low[common_cols]
        close = universe.close[common_cols]
        long_entries = long_entries[common_cols]
        short_entries = short_entries[common_cols]
        size = size[common_cols]
        sl_stop = sl_stop[common_cols]
        tp_stop = tp_stop[common_cols]

        # Make indices all the same datatype
        open.index = pd.to_datetime(open.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        close.index = pd.to_datetime(close.index)
        long_entries.index = long_entries.index.astype('datetime64[ns]')
        short_entries.index = short_entries.index.astype('datetime64[ns]')
        size.index = size.index.astype('datetime64[ns]')
        sl_stop.index = pd.to_datetime(sl_stop.index)
        tp_stop.index = pd.to_datetime(tp_stop.index)

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            signal_func_nb=signal_func_nb,
            signal_args=(
                long_entries.values,
                short_entries.values,
                np.full(long_entries.shape, max_loss),
                np.full(long_entries.shape, max_gain)
            ),
            post_segment_func_nb=post_segment_func_nb,
            in_outputs=dict(
                total_return=vbt.RepEval(
                    "np.full(len(cs_group_lens), np.nan)"
                ),
                max_value=vbt.RepEval(
                    "np.full(len(cs_group_lens), np.nan)"
                )
            ),
            open=open,
            high=high,
            low=low,
            close=close,
            size=size,
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            td_stop=pd.Timedelta(days=7),
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

# For Debugging purposes

# ML-based strategy on a portfolio
# from backtest.strategies.portfolio.portfolio_ml_strategy import PortfolioMLStrategy
#
# # Backtesters for single token and portfolio strategies
# from backtest.backtester import BackTester
#
# import time
#
#
# if __name__ == '__main__':
#     is_backtest = True
#
#     # Execute a walk-forward optimization across input strategies
#     # and log the results to DuckDB/Redshift
#     if is_backtest:
#
#         # Initialize a BackTester instance w/ the intended strategies to backtest and
#         # a performance metric to optimize on
#         b = BackTester(
#             strategies = [
#                 PortfolioMLStrategy(optimization_metric = 'Sharpe Ratio'),
#             ],
#             resample_period = '1d',
#             use_dollar_bars = False,
#             start_date = '2018-12-01',
#             end_date = '2024-12-31',
#         )
#
#         backtest_start = time.time()
#         b.execute()
#         backtest_end = time.time()
#         mins_elapsed = (backtest_end - backtest_start) / 60
#
#         print()
#         print(f'Backtests completed in {mins_elapsed:.2f} minutes')
#         print()
#     else:
#         pass
#
