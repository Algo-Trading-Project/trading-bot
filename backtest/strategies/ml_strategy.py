import vectorbt as vbt
import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score

from notebooks.custom_transformers import *
from notebooks.helper import calculate_triple_barrier_labels

class MLStrategy(BaseStrategy):
    
    indicator_factory_dict = {
        'class_name':'MLStrategy',
        'short_name':'ml_strategy',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['prediction_threshold', 'kelly_fraction'],
        'output_names':['entries', 'exits', 'tp', 'sl', 'size']
    }

    optimize_dict = {
        'prediction_threshold': [0.5, 0.6, 0.7, 0.8, 0.9],
        'kelly_fraction': [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    backtest_params = {
        'init_cash': 100_000,
        'fees': 0.0029,
        'sl_stop': 'atr',
        'sl_trail': False,
        'tp_stop': 'atr',
        'size': 0.05,
        'size_type': 2
    }

    def __init__(self):
        MLStrategy.is_train = True
        
        MLStrategy.model = RandomForestClassifier(
            bootstrap = False, 
            random_state = 9 + 10,
            n_estimators = 50,
            max_depth = 10,
            min_samples_split = 2,
            min_samples_leaf = 2,
            max_features = 'sqrt'
        )

        # Dataframe to save calculated features across all iterations of the walk-forward optimization
        MLStrategy.historical_features = pd.DataFrame()

        # Series to save calculated labels across all iterations of the walk-forward optimization
        MLStrategy.historical_labels = pd.Series()

        # Specify window sizes for rolling min-max and z-score scaling
        window_sizes = [12, 24, 24 * 7]

        # Specify the symbol ID
        symbol_id = 'BTC_USD_COINBASE'

        # Pipeline for feature engineering
        MLStrategy.feature_engineering_pipeline = Pipeline([

            # Add block-based features to the dataset
            ('block_features', BlockFeatures()),

            # Add transaction-based features to the dataset
            # ('transaction_features', TransactionFeatures()),

            # Add transfer-based features to the dataset
            # ('transfer_features', TransferFeatures()),

            # Add tick-based features to the dataset
            ('tick_features', TickFeatures(symbol_id = symbol_id)),

            # Add order book-based features to the dataset
            # ('order_book_features', OrderBookFeatures(symbol_id = symbol_id)),

            # Add wallet-based features to the dataset
            # ('wallet_features', WalletFeatures()),

            # Add rolling min-max scaled features to the dataset
            ('rolling_min_max_scaler', RollingMinMaxScaler(window_sizes)),

            # Add rolling z-score scaled features to the dataset
            ('rolling_z_score_scaler', RollingZScoreScaler(window_sizes)),

            # Add more feature engineering steps here
            # ...
            # ...

            # Clean NaN/infinity values from the dataset
            ('fill_nan', FillNaN()),

            # Add lagged features to the dataset
            ('lag_features', LagFeatures(lags = [1, 2, 3])),
        ])

    @staticmethod
    def indicator_func(open, high, low, close, volume,
                       prediction_threshold, kelly_fraction):
        
        # Get backtest parameters
        backtest_params = MLStrategy.backtest_params

        # Create OHLCV dataframe
        price_data = pd.DataFrame({'price_open': open, 'price_high': high, 'price_low': low, 'price_close': close, 'volume_traded': volume}, index = open.index)
        price_data = price_data.astype(float)

        # Calculate features
        X = MLStrategy.feature_engineering_pipeline.fit_transform(price_data)
        
        # calculate triple-barrier labels
        y = calculate_triple_barrier_labels(price_data, atr_window = 24, max_holding_time = 24)

        # Align X and y
        X = X[X.index.isin(y.index)]
        y = y[y.index.isin(X.index)]

        # If is_train is True, fit the model and save the calculated features
        if MLStrategy.is_train:
            
            # If historical_features is empty, set it to X, else append X to it and do the same for historical_labels
            if MLStrategy.historical_features.empty:
                MLStrategy.historical_features = X
                MLStrategy.historical_labels = y
            else:
                MLStrategy.historical_features = pd.concat([MLStrategy.historical_features, X])
                MLStrategy.historical_labels = pd.concat([MLStrategy.historical_labels, y])

                # Ensure there are unique indices
                MLStrategy.historical_features = MLStrategy.historical_features[~MLStrategy.historical_features.index.duplicated(keep = 'first')].sort_index()
                MLStrategy.historical_labels = MLStrategy.historical_labels[~MLStrategy.historical_labels.index.duplicated(keep = 'first')].sort_index()

            # Train the model
            MLStrategy.model.fit(MLStrategy.historical_features, MLStrategy.historical_labels)

            # Generate predictions for the training set
            y_pred = MLStrategy.model.predict_proba(MLStrategy.historical_features)[:,1]
            y_pred = np.where(y_pred >= prediction_threshold, 1, 0)

            # Turn the predictions into entry signals
            entries = pd.Series(y_pred, index = MLStrategy.historical_features.index)
            
            # Initialize a counter for periods since the last entry
            periods_since_last_entry = 24  # Start with 24 to allow an entry at the first period

            # Iterate through the entries to enforce the 24-period gap
            for i in range(len(entries)):
                if entries.iloc[i] == 1 and periods_since_last_entry >= 24:
                    # If an entry signal is found and at least 24 periods have passed since the last entry
                    periods_since_last_entry = 0  # Reset the counter
                else:
                    # If it's not time for an entry, or if the current signal is not an entry
                    if entries.iloc[i] == 1:
                        # If the current signal is an entry but it's not time yet, suppress it
                        entries.iloc[i] = 0
                    
                    periods_since_last_entry += 1  # Increment the counter

            # Exit signals reflect a 24 period maximum holding time
            exits = pd.Series(entries.shift(24, fill_value = 0), index = entries.index)

            # Calculate the take-profit and stop-loss levels
            tp = MLStrategy.calculate_tp(
                price_data.price_open, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 24
            )

            # Increase the take-profit level by the fees
            tp = tp * (1 + backtest_params['fees'])

            sl = MLStrategy.calculate_sl(
                price_data.price_open, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 24
            )

            # Align tp and sl with the entries and exits
            tp = tp[tp.index.isin(entries.index)]
            sl = sl[sl.index.isin(entries.index)]

            # Calculate the position size based on a fraction of the Kelly criterion
            
            # Net odds for the wager
            b = 1

            # Probability of winning for the positive class
            p = precision_score(MLStrategy.historical_labels, y_pred, pos_label = 1)

            # Probability of losing for the positive class
            q = 1 - p

            # Fraction of the Kelly criterion
            f = kelly_fraction * ((b * p - q) / b)

            # Create a series of the same length as entries and exits with the position size
            size = pd.Series(f, index = entries.index)

            # Switch is_train after each iteration
            MLStrategy.is_train = not MLStrategy.is_train

            # Match the indices of entries, exits, tp, sl, and size with the input data. Fill missing indices with 0
            entries = entries.reindex(price_data.index, fill_value = 0)
            exits = exits.reindex(price_data.index, fill_value = 0)
            tp = tp.reindex(price_data.index, fill_value = 0)
            sl = sl.reindex(price_data.index, fill_value = 0)
            size = size.reindex(price_data.index, fill_value = 0)

            return entries, exits, tp, sl, size

        # If is_train is False, generate predictions for the test set
        else:
            # Generate predictions for the test set
            y_pred = MLStrategy.model.predict_proba(X)[:,1]
            y_pred = np.where(y_pred >= prediction_threshold, 1, 0)

            # Turn the predictions into entry signals
            entries = pd.Series(y_pred, index = X.index)

            # Initialize a counter for periods since the last entry
            periods_since_last_entry = 24  # Start with 24 to allow an entry at the first period

            # Iterate through the entries to enforce the 24-period gap
            for i in range(len(entries)):
                if entries.iloc[i] == 1 and periods_since_last_entry >= 24:
                    # If an entry signal is found and at least 24 periods have passed since the last entry
                    periods_since_last_entry = 0  # Reset the counter
                else:
                    # If it's not time for an entry, or if the current signal is not an entry
                    if entries.iloc[i] == 1:
                        # If the current signal is an entry but it's not time yet, suppress it
                        entries.iloc[i] = 0
                    
                    periods_since_last_entry += 1  # Increment the counter

            # Exit signals reflect a 24 period maximum holding time
            exits = pd.Series(entries.shift(24, fill_value = 0), index = entries.index)

            # Calculate the take-profit and stop-loss levels
            tp = MLStrategy.calculate_tp(
                price_data.price_open, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 24
            )

            # Increase the take-profit level by the fees
            tp = tp * (1 + backtest_params['fees'])

            sl = MLStrategy.calculate_sl(
                price_data.price_open, 
                price_data.price_high, 
                price_data.price_low, 
                price_data.price_close, 
                price_data.volume_traded, 
                backtest_params,
                window = 24
            )

            # Align tp and sl with the entries and exits
            tp = tp[tp.index.isin(entries.index)]
            sl = sl[sl.index.isin(entries.index)]

            # Calculate the position size based on a fraction of the Kelly criterion
            
            # Net odds for the wager
            b = 1

            # Probability of winning for the positive class
            p = precision_score(y, y_pred, pos_label = 1)

            # Probability of losing for the positive class
            q = 1 - p

            # Fraction of the Kelly criterion
            f = kelly_fraction * ((b * p - q) / b)

            # Create a series of the same length as entries and exits with the position size
            size = pd.Series(f, index = entries.index)

            # Switch is_train after each iteration
            MLStrategy.is_train = not MLStrategy.is_train

            # Append X to historical_features
            MLStrategy.historical_features = pd.concat([MLStrategy.historical_features, X]).sort_index()

            # Append y to historical_labels
            MLStrategy.historical_labels = pd.concat([MLStrategy.historical_labels, y]).sort_index()

            # Match the indices of entries, exits, tp, sl, and size with the input data. Fill missing indices with 0
            entries = entries.reindex(price_data.index, fill_value = 0)
            exits = exits.reindex(price_data.index, fill_value = 0)
            tp = tp.reindex(price_data.index, fill_value = 0)
            sl = sl.reindex(price_data.index, fill_value = 0)
            size = size.reindex(price_data.index, fill_value = 0)

            return entries, exits, tp, sl, size