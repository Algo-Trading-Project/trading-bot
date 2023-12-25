from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import ta
import joblib

class WalkForwardModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_filename = 'walk_forward_model.pkl'

    def train_or_predict(self, features, close_prices):
        if not self.is_trained:
            self._train_model(features, close_prices)
            self.save_model()
            self.is_trained = True
        else:
            self.load_model()
            self.is_trained = False

        return self.predict(features)

    def _train_model(self, features, close_prices):
        next_step_returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
        target = (next_step_returns >= 0.01).astype(int)  # Binary target
        features = features[:-1, :]  # Align features with target

        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        self.model = LogisticRegression()
        self.model.fit(features_scaled, target)

    def predict(self, features):
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def save_model(self):
        joblib.dump((self.model, self.scaler), self.model_filename)

    def load_model(self):
        self.model, self.scaler = joblib.load(self.model_filename)

class LogisticRegressionStrategy:
    model_instance = WalkForwardModel()  # Class variable to hold the model instance

    indicator_factory_dict = {
        'class_name':'LinearRegressionStrategy',
        'short_name':'linearregression',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':[],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {}

    def calculate_holding_time(entries):
        holding_time = np.zeros_like(entries, dtype=int)
        for i in range(1, len(entries)):
            if entries[i]:
                holding_time[i] = 0  # Reset holding time on a new entry
            else:
                holding_time[i] = holding_time[i - 1] + 1 if entries[i - 1] else 0
        return holding_time

    def add_technical_indicators(data):
        data['sma'] = ta.trend.sma_indicator(close=data['close'], window=5)
        data['ema'] = ta.trend.ema_indicator(close=data['close'], window=5)
        data['rsi'] = ta.momentum.rsi(close=data['close'], window=14)
        data['macd'] = ta.trend.macd_diff(close=data['close'])
        data['upper_band'], data['lower_band'] = ta.volatility.bollinger_hband_indicator(close=data['close']), ta.volatility.bollinger_lband_indicator(close=data['close'])
        data['atr'] = ta.volatility.average_true_range(high=data['high'], low=data['low'], close=data['close'])
        data['obv'] = ta.volume.on_balance_volume(close=data['close'], volume=data['volume'])
        return data

    def indicator_func(open, high, low, close, volume):
        data = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume})
        data = LogisticRegressionStrategy.add_technical_indicators(data)
        data.fillna(0, inplace = True)
        features = data.drop(columns=['close']).values

        try:
            predictions = LogisticRegressionStrategy.model_instance.train_or_predict(features, data['close'].values)
        except:
            predictions = np.zeros_like(data['close'].values, dtype=int)
            
        entries = predictions == 1  # Entry signal when prediction is positive
        holding_time = LogisticRegressionStrategy.calculate_holding_time(entries)
        exits = holding_time >= 6  # Exit signal after holding for 6 or more hour

        return entries, exits