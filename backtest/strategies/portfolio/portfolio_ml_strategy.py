from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
from utils.db_utils import QUERY
from vectorbtpro import *

import pandas as pd
import joblib
import numpy as np
import vectorbtpro as vbt

class PortfolioMLStrategy(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'PortfolioMLStrategy'
    }

    optimize_dict = {
        # Minimum predicted probability for entering a trade
        'prediction_threshold': vbt.Param(np.array([0.5, 0.55, 0.6, 0.65, 0.7])),
        
        # Only take positions in tokens with predicted probabilities 
        # that are above the prediction threshold and in the top p% of predicted probabilities
        'target_prediction_percentile': vbt.Param(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.0025,
        'sl_stop': 'std',
        'tp_stop': 'std',
        'size': 0.05,
        'size_type': 2, # e.g. 2 if 'size' is 'Fixed Percent' and 0 otherwise,
    }

    def __init__(self):
        self.month_to_quarter_map = {
            1: 'oct_to_dec',
            2: 'oct_to_dec',
            3: 'oct_to_dec',
            4: 'jan_to_march',
            5: 'jan_to_march',
            6: 'jan_to_march',
            7: 'april_to_june',
            8: 'april_to_june',
            9: 'april_to_june',
            10: 'july_to_sept',
            11: 'july_to_sept',
            12: 'july_to_sept'
        }

        ml_features = QUERY(
            """
            SELECT *
            FROM market_data.ml_features
            """
        )
        ml_features['symbol_id'] = ml_features['symbol_id'].astype('category')

        # Columns we need to drop before training the model
        triple_barrier_label_cols = [
            col for col in ml_features if 'triple_barrier_label_h' in col
        ]

        trade_returns_cols = [
            col for col in ml_features if 'trade_returns' in col
        ]

        non_numeric_cols = [
            'asset_id_base', 'asset_id_quote', 'exchange_id', 'Unnamed: 0',
        ]

        forward_returns_cols = [
            'open', 'high', 'low', 'close', 'start_date_triple_barrier_label_h7', 
            'end_date_triple_barrier_label_h7', 'avg_uniqueness', 'time_period_end'
        ]

        cols_to_drop = (
            triple_barrier_label_cols + 
            trade_returns_cols + 
            non_numeric_cols +
            forward_returns_cols
        )

        ml_features = ml_features.set_index('time_period_end')
        ml_features = ml_features.drop(cols_to_drop, axis=1, errors='ignore')
        self.ml_features = ml_features
    
    def __get_model(self, min_date: pd.Timestamp):
        min_year = min_date.year
        min_month = min_date.month

        # Get the previous quarter
        quarter = self.month_to_quarter_map[min_month]

        # If the quarter is the first quarter of the year, then the previous quarter is the last quarter of the previous year
        if quarter == 'oct_to_dec':
            min_year -= 1

        path = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/classification/xgboost_model_and_metrics_{min_year}_{quarter}.pkl'

        # Load the model
        model, model_performance = joblib.load(path)
        return model

    def calculate_size(self, model_probs, prediction_threshold, target_prediction_percentile):
        """
        Calculate position sizes from the model's cross-sectional predicted probabilities
        """
        
        def get_col_indices(row):
            return row[row].index.tolist()
        
        size = pd.DataFrame(np.full(model_probs.shape, 0), index = model_probs.index, columns = model_probs.columns)

        # Get all predicted probabilities that are above the threshold
        above_threshold = model_probs >= prediction_threshold

        # Get all column (tokens) indices for each row (date) where the predicted probability is above the threshold
        col_indices = above_threshold.apply(get_col_indices, axis=1)

        # For each date
        for date in col_indices.index:
            # Rank the predicted probabilities of the assets that are above the threshold
            rank = model_probs.loc[date, col_indices.loc[date]].rank(ascending=True, pct=True)

            # Get tokens that are in the top (1 - p)% of the predicted probabilities
            top_p = (rank >= target_prediction_percentile).astype(int)

            # Number of tokens in the top p% of predicted probabilities
            num_tokens = top_p.sum()

            # Take an equal position in each token in the top (1 - p)% of predicted probabilities
            top_p /= num_tokens

            # Update the size DataFrame with the position sizes for this date
            size.loc[date, col_indices.loc[date]] = top_p

        return size, col_indices

    def run_strategy_with_parameter_combination(
        self, 
        universe: pd.DataFrame, 
        prediction_threshold: float,
        target_prediction_percentile: float
    ):
        universe.index = pd.to_datetime(universe.index)
        min_date = universe.index.min()
        max_date = universe.index.max()
        
        self.ml_features['y_pred_prob'] = 0.0

        # Get model
        model = self.__get_model(min_date)

        # Get features for this quarter
        feature_filter = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date)
        )
        X = self.ml_features.loc[feature_filter]

        # Predictions on this quarter
        y_pred_prob = model.predict_proba(X.drop(['y_pred_prob', 'symbol_id'], axis=1))[:, 1]
        self.ml_features.loc[feature_filter, 'y_pred_prob'] = y_pred_prob

        # Pivot model's predictions for ranking signals
        model_probs = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'y_pred_prob']]
            .pivot_table(index='time_period_end', columns='symbol_id', values='y_pred_prob')
        )
        model_probs = model_probs.sort_index().fillna(0)
        model_probs = model_probs.loc[universe.index]

        size, col_indices = self.calculate_size(model_probs, prediction_threshold, target_prediction_percentile)
        entries = size > 0
        exits = pd.DataFrame(np.full(size.shape, False), index = entries.index, columns = entries.columns)
        tp = self.calculate_tp(universe, self.backtest_params, 7)
        sl = self.calculate_sl(universe, self.backtest_params, 7)
        td_stop = pd.DataFrame(pd.Timedelta(days=7), index = universe.index, columns = universe.columns)

        # Ensure that the signals are aligned with the universe
        entries = entries.loc[universe.index]
        exits = exits.loc[universe.index]
        tp = tp.loc[universe.index]
        sl = sl.loc[universe.index]
        size = size.loc[universe.index]
        td_stop = td_stop.loc[universe.index]

        # Save these parameters temporarily
        sl_stop = self.backtest_params['sl_stop']
        tp_stop = self.backtest_params['tp_stop']
        size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they are not passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            close = universe,
            entries = entries,
            exits = exits,
            size = size,
            sl_stop = sl,
            tp_stop = tp,
            td_stop = td_stop,
            group_by = True,
            cash_sharing = True,
            **self.backtest_params
        )

        # Restore the parameters
        self.backtest_params['sl_stop'] = sl_stop
        self.backtest_params['tp_stop'] = tp_stop
        self.backtest_params['size'] = size

        return portfolio