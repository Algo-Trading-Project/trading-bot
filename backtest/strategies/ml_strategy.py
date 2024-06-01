import vectorbt as vbt
import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from notebooks.custom_transformers import *
from notebooks.helper import calculate_triple_barrier_labels
from notebooks.helper import construct_dataset_for_ml

class MLStrategy(BaseStrategy):
    
    indicator_factory_dict = {
        'class_name':'MLStrategy',
        'short_name':'ml_strategy',
        'input_names':['open', 'high', 'low', 'close', 'volume', 'trades_count'],
        'param_names':['prediction_threshold', 'trade_size_multiplier'],
        'output_names':['entries', 'exits', 'tp', 'sl', 'size']
    }

    optimize_dict = {
        'prediction_threshold': [0.6],
        'trade_size_multiplier': [0.2]
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.0029,
        'sl_stop': 'std',
        'sl_trail': True,
        'tp_stop': 'std',
        'size': 0.1,
        'size_type': 2
    }

    def __init__(self):
        MLStrategy.is_train = True
        
        MLStrategy.model = RandomForestClassifier(
            n_estimators = 25,
            bootstrap = False, 
            random_state = 9 + 10, 
            n_jobs = -1,
            max_depth = 10
        )

        # Specify window sizes for rolling min-max and z-score scaling
        window_sizes_scaling = [2 * 24, 2 * 24 * 7, 2 * 24 * 30]

        # Specify window sizes for returns-based features
        window_sizes_returns = [1, 2 * 24, 2 * 24 * 7, 2 * 24 * 30]
       
        # Pipeline for feature engineering
        MLStrategy.feature_engineering_pipeline = Pipeline([

            # Add returns-based features to the dataset
            ('returns_features', ReturnsFeatures(window_sizes_returns)),

            # Add rolling min-max scaled features to the dataset
            ('rolling_min_max_scaler', RollingMinMaxScaler(window_sizes_scaling)),

            # Add rolling z-score scaled features to the dataset
            ('rolling_z_score_scaler', RollingZScoreScaler(window_sizes_scaling)),

            # Add price-based features to the dataset
            # ('price_features', PriceFeatures()),

            # Add more feature engineering steps here
            # ...
            # ...

            # Clean NaN/infinity values from the dataset
            ('fill_nan', FillNaN()),

            # Add lagged features to the dataset
            ('lag_features', LagFeatures(lags = [1, 2, 3])),

            # Add time-based features to the dataset
            # ('time_features', TimeFeatures()),

        ])

        # Feature engineering
        MLStrategy.data = construct_dataset_for_ml()
        MLStrategy.data = MLStrategy.__get_ml_dataset()
        
    def __get_ml_dataset():
        df = MLStrategy.data
        X, y = [], []
        i = 1
        n = len(df.symbol_id.unique())

        for symbol_id in df.symbol_id.unique():
            print(f'Processing symbol_id: {symbol_id} ({i}/{n})')
            i += 1

            token = df.loc[df.symbol_id == symbol_id,:]
            labels = token.loc[:,'triple_barrier_label']
            features = MLStrategy.feature_engineering_pipeline.fit_transform(token.drop(['triple_barrier_label'], axis = 1))

            X.append(features)
            y.append(labels)

        X = pd.concat(X)
        y = pd.concat(y)

        X.loc[:,'triple_barrier_label'] = y
        return X
    
    @staticmethod
    def indicator_func(open, high, low, close, volume, trades_count,
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
        price_data.loc[:,'returns'] = price_data.loc[:,'price_close'].pct_change()

        df = MLStrategy.data

        # If is_train is True, fit the model and generate predictions for the training set
        if MLStrategy.is_train:

            in_sample_data = MLStrategy.feature_engineering_pipeline.fit_transform(price_data)
            
            # Training data is all data available by the end of the training period
            end = close.index[-1]
            curr_data = df.loc[df.index <= end,:]
            X_train = curr_data.drop(['triple_barrier_label'], axis = 1)

            # Recalculate the triple barrier labels for each token in X_train to avoid lookahead bias/data leakage
            y_train = []
            
            for symbol_id in X_train.symbol_id.unique():
                token = X_train.loc[X_train.symbol_id == symbol_id,:]
                labels = calculate_triple_barrier_labels(token, window = 2 * 24 * 7, max_holding_time = 2 * 24)
                y_train.append(labels)

            X_train = X_train.drop(['symbol_id'], axis = 1)
            y_train = pd.concat(y_train)

            # Train the model
            MLStrategy.model.fit(X_train, y_train)

            # Generate predictions for the training set
            y_pred_proba = MLStrategy.model.predict_proba(in_sample_data)[:,1]

            # Turn the predictions into entry signals
            entries = pd.Series(np.where(y_pred_proba >= prediction_threshold, 1, 0), index = in_sample_data.index)
            
            # Exit signals reflect a 24 period maximum holding time
            exits = entries.shift(2 * 24, fill_value = 0)

            # Calculate the take-profit and stop-loss levels
            tp = MLStrategy.calculate_tp(
                price_data.price_close, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 2 * 24 * 7
            )

            sl = MLStrategy.calculate_sl(
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
            out_of_sample_data = MLStrategy.feature_engineering_pipeline.fit_transform(price_data)

            # Generate predictions for the test set
            y_pred_proba = MLStrategy.model.predict_proba(out_of_sample_data)[:,1]
            
            # Turn the predictions into entry signals
            entries = pd.Series(np.where(y_pred_proba >= prediction_threshold, 1, 0), index = out_of_sample_data.index)

            # Exit signals reflect a 24 period maximum holding time
            exits = entries.shift(2 * 24, fill_value = 0)

            # Calculate the take-profit and stop-loss levels
            tp = MLStrategy.calculate_tp(
                price_data.price_close, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 2 * 24 * 7
            )

            sl = MLStrategy.calculate_sl(
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