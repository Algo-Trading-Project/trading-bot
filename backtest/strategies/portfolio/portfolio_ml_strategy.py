from quantstats.stats import expected_return

from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
from utils.db_utils import QUERY
from vectorbtpro import *
from numba import njit
from sklearn.preprocessing import OneHotEncoder

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
def signal_func_nb(c, long_entry_signals, max_loss, max_gain):
    i = c.i
    g = c.group
    col = c.col

    long_entry_signal = long_entry_signals[i, col]
    max_loss_ = max_loss[i, col]
    max_gain_ = max_gain[i, col]

    max_value = c.in_outputs.max_value[g]
    curr_value = c.last_value[g]
    curr_dd = (curr_value - max_value) / max_value

    tp_predicate = curr_value / c.init_cash[g] - 1 >= max_gain_
    sl_predicate = curr_dd <= -max_loss_

    if tp_predicate or sl_predicate:
        return False, True, False, False

    # If long entry signal is present, we enter a long position
    if long_entry_signal:
        return long_entry_signal, False, False, False
    else:
        return False, False, False, False

class PortfolioMLStrategy(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'PortfolioMLStrategy (vt)',
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
            0.7
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
        
        # Load the ML features from the database
        ml_features = QUERY(
            """
            SELECT * FROM market_data.ml_features
            """
        )
        ml_features['symbol_id'] = ml_features['symbol_id'].astype('category')
        ml_features['day_of_week'] = ml_features['day_of_week'].astype('category')
        ml_features['month'] = ml_features['month'].astype('category')
        ml_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        ml_features = ml_features.dropna(subset=['forward_returns_7'])
        ml_features.set_index('time_period_end', inplace=True)
        ml_features.sort_index(inplace=True)

        futures_ml_features = ml_features[~ml_features['close_futures'].isna()]

        # Columns we need to drop before training the model
        forward_returns_cols = [col for col in ml_features if 'forward_returns' in col]

        non_numeric_cols = [
            'asset_id_base','asset_id_base_x','asset_id_base_y', 
            'asset_id_quote','asset_id_quote_x', 'asset_id_quote_y', 
            'exchange_id','exchange_id_x','exchange_id_y', 
            'day_of_week', 'month', 'symbol_id'
        ]

        other_cols = [
            'open_spot', 'high_spot', 'low_spot', 'close_spot', 'volume_spot', 'trades_spot',
            'open_futures', 'high_futures', 'low_futures', 'close_futures', 'volume_futures', 'trades_futures',
            'time_period_end', 'y_pred'
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
        self.futures_ml_features = futures_ml_features

    def __get_model(self, min_date: pd.Timestamp):
        year = min_date.year
        month = min_date.month
        long_model_path = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/classification/xgb_model_{year}_{month}.pkl'
        short_model_path = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/classification/xgb_short_model_{year}_{month}_short.pkl'
        long_model = joblib.load(long_model_path)
        try:
            short_model = joblib.load(short_model_path)
        except FileNotFoundError:
            short_model = None
        return long_model, short_model

    def calculate_size(self, volatilities):
        """
        Calculate position sizes from the tokens' volatilities and betas

        :param volatilities: DataFrame of volatilities for each token (columns) and date (index)
        :param betas: DataFrame of betas for each token (columns) and date (index)
        :return: DataFrame of position sizes for each token (columns) and date (index)
        """
        # Volatility targeting
        token_ann_vol = np.sqrt(365) * volatilities

        # Position size is volatility weighted
        ann_vol_target = 0.3  # Target annualized volatility of the portfolio
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
            max_gain: float,
            prediction_threshold: float
    ):
        universe.index = pd.to_datetime(universe.index)
        min_date = universe.index.min()
        max_date = universe.index.max()

        self.ml_features['y_pred'] = 0

        # Get models
        long_model, short_model = self.__get_model(min_date)

        # Get features for this month
        feature_filter = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date)
        )
        feature_filter_futures = (
            (self.futures_ml_features.index >= min_date) &
            (self.futures_ml_features.index <= max_date)
        )
        spot_data = self.ml_features.loc[feature_filter]
        futures_data = self.futures_ml_features.loc[feature_filter_futures]
        if futures_data.empty:
            futures_data = pd.DataFrame(index=spot_data.index, columns=[f'{col}_futures' for col in spot_data.columns])

        print(f"Running strategy from {min_date} to {max_date}...")
        print()

        # Combine encoded features with the rest of the features
        X = spot_data.drop(columns=self.cols_to_drop, errors='ignore', axis=1)
        X_futures = futures_data.drop(columns=self.cols_to_drop, errors='ignore', axis=1)

        # Predictions on this month
        y_pred_long = (long_model.predict_proba(X)[:, 1] >= prediction_threshold).astype(int)
        if short_model is None:
            y_pred_short = np.zeros_like(y_pred_long)
        else:
            y_pred_short = (short_model.predict_proba(X_futures)[:, 1] >= prediction_threshold).astype(int)
        
        self.ml_features.loc[feature_filter, 'y_pred_long'] = y_pred_long
        self.futures_ml_features.loc[feature_filter_futures, 'y_pred_short'] = y_pred_short

        # Pivot model's predictions for ranking the models' signals by date
        long_model_probs = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'y_pred_long']]
            .pivot_table(index='time_period_end', columns='symbol_id', values=['y_pred_long'], dropna=False)
        )
        long_model_probs = long_model_probs.fillna(0)
        long_model_probs = long_model_probs[long_model_probs.index.isin(universe.index)]

        # Pivot futures model's predictions for ranking the models' signals by date
        short_model_probs = (
            self.futures_ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'y_pred_short']]
            .pivot_table(index='time_period_end', columns='symbol_id', values=['y_pred_short'], dropna=False)
        )
        short_model_probs = short_model_probs.fillna(0)
        short_model_probs = short_model_probs[short_model_probs.index.isin(universe.index)]
        short_model_probs.columns = [f'{col}_futures' for col in short_model_probs.columns]

        # Merge the long and short model probabilities
        empty_long = pd.DataFrame(
            np.zeros((long_model_probs.shape[0], long_model_probs.shape[1])),
            index=long_model_probs.index,
            columns=[col for col in long_model_probs.columns]
        )   
        empty_short = pd.DataFrame(
            np.zeros((short_model_probs.shape[0], short_model_probs.shape[1])),
            index=short_model_probs.index,
            columns=[col for col in short_model_probs.columns]
        )
        long_model_probs = pd.merge(
            long_model_probs,
            empty_short,
            left_index=True,
            right_index=True,
            how='outer'
        ).fillna(0)
        short_model_probs = pd.merge(
            empty_long,
            short_model_probs,
            left_index=True,
            right_index=True,
            how='outer'
        ).fillna(0)         

        # Long entries
        long_entries = long_model_probs == 1
        long_entries = long_entries[long_entries.index.isin(universe.index)]    

        # Short entries
        short_entries = short_model_probs == 1
        short_entries = short_entries[short_entries.index.isin(universe.index)]

        # Pivot volatilities for calculating position sizes
        volatilities = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'std_spot_returns_1_30']]
            .pivot_table(index='time_period_end', columns='symbol_id', values='std_spot_returns_1_30', dropna=False)
        )
        volatilities = volatilities.sort_index()

        # Position sizes
        # size = self.calculate_size(volatilities)

        # Ensure that the signals are aligned with the universe
        # size = size[size.index.isin(universe.index)]

        # Save these parameters temporarily
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        # Intersection of columns between the universe and the signals
        open_spot = universe.open
        open_futures = universe.open_futures
        open_futures.columns = [f'{col}_futures' for col in open_futures.columns]
        open = pd.merge(
            open_spot,
            open_futures,
            left_index=True,
            right_index=True,
            how='outer'
        )

        high_spot = universe.high
        high_futures = universe.high_futures
        high_futures.columns = [f'{col}_futures' for col in high_futures.columns]
        high = pd.merge(
            high_spot,
            high_futures,
            left_index=True,
            right_index=True,
            how='outer'
        )

        low_spot = universe.low
        low_futures = universe.low_futures
        low_futures.columns = [f'{col}_futures' for col in low_futures.columns]
        low = pd.merge(
            low_spot,
            low_futures,
            left_index=True,
            right_index=True,
            how='outer'
        )
        
        close = universe.close
        close_futures = universe.close_futures
        close_futures.columns = [f'{col}_futures' for col in close_futures.columns]
        close = pd.merge(
            close,
            close_futures,
            left_index=True,
            right_index=True,
            how='outer'
        )

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            # signal_func_nb=signal_func_nb,
            # signal_args=(
            #     long_entries.values,
            #     np.full(long_entries.shape, max_loss),
            #     np.full(long_entries.shape, max_gain)
            # ),
            # post_segment_func_nb=post_segment_func_nb,
            # in_outputs=dict(
            #     total_return=vbt.RepEval(
            #         "np.full(len(cs_group_lens), np.nan)"
            #     ),
            #     max_value=vbt.RepEval(
            #         "np.full(len(cs_group_lens), np.nan)"
            #     )
            # ),
            long_entries=long_entries,
            open=open,
            high=high,
            low=low,
            close=close,
            size=0.05,
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