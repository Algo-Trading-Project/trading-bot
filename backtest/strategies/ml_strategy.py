import vectorbt as vbt
import numpy as np
import pandas as pd
import os
import duckdb

from .base_strategy import BaseStrategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from notebooks.custom_transformers import *
from notebooks.helper import calculate_triple_barrier_labels
from notebooks.helper import QUERY

class MLStrategy(BaseStrategy):

    indicator_factory_dict = {
        'class_name':'MLStrategy',
        'short_name':'ml_strategy',
        'input_names':['open', 'high', 'low', 'close', 'volume', 'trades_count'],
        'param_names':['prediction_threshold', 'trade_size_multiplier'],
        'output_names':['entries', 'exits', 'tp', 'sl', 'size']
    }

    optimize_dict = {
        # Minimum predicted probability for entering a trade
        'prediction_threshold': [0.5],

        # Multiplier for position size based on predicted class probabilities
        'trade_size_multiplier': [0.3]
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.002,
        'sl_stop': 'std',
        'sl_trail': False,
        'tp_stop': 'std',
        'size_type': 2
    }

    def __init__(self):
        self.model = RandomForestClassifier(
            max_depth = 10,
            max_features = 0.2,
            bootstrap = True,
            random_state = 9 + 10,
            n_jobs = -1,
            class_weight = 'balanced'
        )

        self.is_train = True
        self.symbol_id = ''
            
    def indicator_func(self, open, high, low, close, volume, trades_count,
                       prediction_threshold, trade_size_multiplier):
        
        # Get backtest parameters
        backtest_params = MLStrategy.backtest_params

        # Create OHLCV dataframe
        price_data = pd.DataFrame({
            'price_close': close, 
            'price_high': high, 
            'price_low': low, 
            'volume_traded': volume,
            'trades_count': trades_count
        })
        price_data = price_data.astype(float)


        # If is_train is True, fit the model and generate predictions for the training set
        if self.is_train:
            print('Training...')
            print()

            start = pd.to_datetime(price_data.index[0])
            end = pd.to_datetime(price_data.index[-1])

            query = f"""
            SELECT *
            FROM market_data.ml_features
            WHERE
                time_period_end <= '{end}'
            ORDER BY symbol_id, time_period_end
            """            
            
            X_train = QUERY(query).set_index('time_period_end')

            if len(X_train) > 1_000_000:
                X_train = X_train.sample(n = 1_000_000, replace = False, random_state = 9 + 10)
            
            windows_triple_barrier_label = [2 * 12, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]
            max_holding_times_triple_barrier_label = [2 * 12, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]

            triple_barrier_label_cols = [
                f'triple_barrier_label_w{w}_h{h}' 
                for w in windows_triple_barrier_label 
                for h in max_holding_times_triple_barrier_label
            ]

            cols_to_drop = triple_barrier_label_cols

            X_train = X_train.drop(columns = cols_to_drop)            
            y_train = []
            
            for symbol_id in X_train.symbol_id.unique():
                token = X_train.loc[X_train.symbol_id == symbol_id,:]
                labels = calculate_triple_barrier_labels(token, window = 2 * 24 * 7, max_holding_time = 2 * 24 * 7)
                y_train.append(labels)

            X_train = X_train.drop(['symbol_id'], axis = 1)
            X_train = X_train.reindex(sorted(X_train.columns), axis = 1)
            y_train = pd.concat(y_train)

            # Train the model
            self.model.fit(X_train, y_train, sample_weight = X_train['returns_1'].abs())

            # Generate predictions for the training set
            query = f"""
            SELECT *
            FROM market_data.ml_features
            WHERE
                time_period_end BETWEEN '{start}' AND '{end}' AND
                symbol_id = '{self.symbol_id}' 
            ORDER BY time_period_end
            """

            in_sample_data = QUERY(query).set_index('time_period_end')
            in_sample_data = in_sample_data.drop(columns = cols_to_drop)
            in_sample_data = in_sample_data.drop(['symbol_id'], axis = 1)
            in_sample_data = in_sample_data.reindex(sorted(in_sample_data.columns), axis = 1)

            class_1_index = self.model.classes_.tolist().index(1)
            y_pred_proba = self.model.predict_proba(in_sample_data)[:, class_1_index]

            # Turn the predictions into entry signals
            entries = pd.Series(np.where(y_pred_proba >= prediction_threshold, 1, 0), index = in_sample_data.index)
            # entries.loc[in_sample_data[in_sample_data['returns_1_rz_48'] < 1.5].index] = 0

            exits = pd.Series(np.zeros(len(entries)), index = entries.index)

            # Exit signals reflect a 1 week maximum holding time
            i = 0
            while i < len(entries):
                # If the entry signal is 1, set the exit signal to 1 after 1 week
                if entries.iloc[i] == 1:
                    exit_index = min(i + 2 * 24 * 7, len(entries) - 1)
                    exits.iloc[exit_index] = 1

                    # No other exit signals in between
                    exits.iloc[i + 1:exit_index] = 0

                    # Skip the next 1 week
                    i += ((2 * 24 * 7) + 1)
                
                i += 1

            # Calculate the take-profit and stop-loss levels
            tp = self.calculate_tp(
                price_data.price_close, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 2 * 24 * 7
            )

            sl = self.calculate_sl(
                price_data.price_close, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 2 * 24 * 7
            )

            # Calculate the position size based on predicted class probabilities
            size = pd.Series(y_pred_proba * trade_size_multiplier, index = in_sample_data.index)
                
            return entries, exits, tp, sl, size

        # If is_train is False, generate predictions for the test set
        else:
            print('Testing...')
            print()

            # OOS date is 2 * 24 * 30 * 6 periods after the start
            start_date_oos = price_data.index[0]
            end_date_oos = price_data.index[-1]

            query = f"""
            SELECT *
            FROM market_data.ml_features
            WHERE
                time_period_end BETWEEN '{start_date_oos}' AND '{end_date_oos}' AND
                symbol_id = '{self.symbol_id}'
            ORDER BY time_period_end
            """

            out_of_sample_data = QUERY(query).set_index('time_period_end')
            
            windows_triple_barrier_label = [2 * 12, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]
            max_holding_times_triple_barrier_label = [2 * 12, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]

            triple_barrier_label_cols = [
                f'triple_barrier_label_w{w}_h{h}' 
                for w in windows_triple_barrier_label 
                for h in max_holding_times_triple_barrier_label
            ]

            cols_to_drop = triple_barrier_label_cols
            out_of_sample_data = out_of_sample_data.drop(columns = cols_to_drop)
            out_of_sample_data = out_of_sample_data.drop(['symbol_id'], axis = 1)
            out_of_sample_data = out_of_sample_data.reindex(sorted(out_of_sample_data.columns), axis = 1)

            # Generate predictions for the test set
            class_1_index = self.model.classes_.tolist().index(1)
            y_pred_proba = self.model.predict_proba(out_of_sample_data)[:, class_1_index]
            
            # Turn the predictions into entry signals
            entries = pd.Series(np.where(y_pred_proba >= prediction_threshold, 1, 0), index = out_of_sample_data.index)
            # entries.loc[out_of_sample_data[out_of_sample_data['returns_1_rz_48'] < 1.5].index] = 0

            exits = pd.Series(np.zeros(len(entries)), index = entries.index)

            # Exit signals reflect a 1 week maximum holding time
            i = 0
            while i < len(entries):
                # If the entry signal is 1, set the exit signal to 1 after 1 week
                if entries.iloc[i] == 1:
                    exit_index = min(i + 2 * 24 * 7, len(entries) - 1)
                    exits.iloc[exit_index] = 1

                    # No other exit signals in between
                    exits.iloc[i + 1:exit_index] = 0

                    # Skip the next 1 week
                    i += ((2 * 24 * 7) + 1)

                i += 1

            # Calculate the take-profit and stop-loss levels
            tp = self.calculate_tp(
                price_data.price_close, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 2 * 24 * 7
            )

            sl = self.calculate_sl(
                price_data.price_close, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 2 * 24 * 7
            )

            # Calculate the position size based on predicted class probabilities
            size = pd.Series(y_pred_proba * trade_size_multiplier, index = out_of_sample_data.index)

            return entries, exits, tp, sl, size